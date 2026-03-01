"""Quality-report workflow for lyrics + Whisper processing."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from .helpers import (
    _apply_timing_to_lines,
    _create_lines_from_plain_text,
    _create_no_lyrics_placeholder,
    _extract_text_lines_from_lrc,
    _load_lyrics_file,
    _romanize_lines,
)
from .lyrics_whisper_map import _map_lrc_lines_to_whisper_segments
from .lyrics_whisper_pipeline import (
    should_auto_enable_whisper,
    should_keep_lrc_timings_for_trailing_outro_padding,
)
from ...models import Line, SongMetadata

logger = logging.getLogger(__name__)


def _clip_lines_to_target_duration(
    lines: List[Line],
    target_duration: Optional[int],
    issues: List[str],
    grace_seconds: float = 0.4,
) -> List[Line]:
    """Trim/drop lines that extend beyond known track duration."""
    if not target_duration or target_duration <= 0:
        return lines

    max_time = float(target_duration) + grace_seconds
    clipped_lines: List[Line] = []
    dropped_lines = 0
    trimmed_words = 0

    for line in lines:
        if not line.words:
            continue
        if line.words[0].start_time >= max_time:
            dropped_lines += 1
            continue

        new_words = []
        for word in line.words:
            if word.start_time >= max_time:
                trimmed_words += 1
                break
            capped_end = min(word.end_time, max_time)
            if capped_end < word.start_time:
                capped_end = word.start_time
            if capped_end < word.end_time:
                trimmed_words += 1
            new_words.append(
                type(word)(
                    text=word.text,
                    start_time=word.start_time,
                    end_time=capped_end,
                    singer=word.singer,
                )
            )

        if not new_words:
            dropped_lines += 1
            continue
        clipped_lines.append(Line(words=new_words, singer=line.singer))

    if dropped_lines:
        issues.append(
            f"Dropped {dropped_lines} line(s) beyond track duration ({target_duration}s)"
        )
    if trimmed_words:
        issues.append(
            f"Trimmed {trimmed_words} word timing(s) past track duration ({target_duration}s)"
        )
    return clipped_lines


def _apply_lrc_timing_trust_policy(
    *,
    line_timings: Optional[List[Tuple[float, str]]],
    lrc_text: Optional[str],
    target_duration: Optional[int],
    drop_lrc_line_timings: bool,
    vocals_path: Optional[str],
    use_whisper: bool,
    whisper_map_lrc: bool,
    issues_list: List[str],
    quality_report: dict,
) -> Optional[List[Tuple[float, str]]]:
    if drop_lrc_line_timings and line_timings:
        issues_list.append(
            "Configured to ignore provider LRC line timings; deriving timing from audio"
        )
        quality_report["lrc_timing_trust"] = "dropped_configured"
        return None

    if target_duration and lrc_text:
        from .sync import get_lrc_duration

        lrc_duration = get_lrc_duration(lrc_text)
        if lrc_duration and abs(target_duration - lrc_duration) > 8:
            lrc_duration_mismatch_sec = abs(target_duration - lrc_duration)
            issues_list.append(
                f"LRC duration mismatch: LRC={lrc_duration}s vs audio={target_duration}s"
            )
            quality_report["lrc_timing_trust"] = "degraded_duration_mismatch"
            can_recover_with_audio_alignment = bool(vocals_path) and bool(
                use_whisper or whisper_map_lrc
            )
            likely_outro_padding = should_keep_lrc_timings_for_trailing_outro_padding(
                line_timings=line_timings,
                lrc_duration=lrc_duration,
                target_duration=target_duration,
            )
            if can_recover_with_audio_alignment and lrc_duration_mismatch_sec >= 12.0:
                if likely_outro_padding:
                    issues_list.append(
                        "Detected likely trailing instrumental outro padding; "
                        "keeping provider LRC timestamps"
                    )
                    quality_report["lrc_timing_trust"] = "degraded_outro_padding"
                    return line_timings
                issues_list.append(
                    "Ignoring provider LRC timestamps due to severe duration mismatch; "
                    "using audio/Whisper timing alignment instead"
                )
                quality_report["lrc_timing_trust"] = "dropped_duration_mismatch"
                return None
    return line_timings


def get_lyrics_with_quality(  # noqa: C901
    title: str,
    artist: str,
    vocals_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    lyrics_offset: Optional[float] = None,
    romanize: bool = True,
    filter_promos: bool = True,
    target_duration: Optional[int] = None,
    evaluate_sources: bool = False,
    use_whisper: bool = False,
    whisper_only: bool = False,
    whisper_map_lrc: bool = False,
    whisper_map_lrc_dtw: bool = False,
    lyrics_file: Optional[Path] = None,
    drop_lrc_line_timings: bool = False,
    whisper_language: Optional[str] = None,
    whisper_model: Optional[str] = None,
    whisper_force_dtw: bool = False,
    whisper_aggressive: bool = False,
    whisper_temperature: float = 0.0,
    lenient_vocal_activity_threshold: float = 0.3,
    lenient_activity_bonus: float = 0.4,
    low_word_confidence_threshold: float = 0.5,
    offline: bool = False,
) -> Tuple[List[Line], Optional[SongMetadata], dict]:
    from .lyrics_whisper import (
        _apply_singer_info,
        _calculate_quality_score,
        _detect_offset_with_issues,
        _fetch_genius_with_quality_tracking,
        _fetch_lrc_text_and_timings_for_state,
        _refine_timing_with_quality,
        create_lines_from_lrc,
        create_lines_from_lrc_timings,
        get_lyrics_simple,
    )
    from .sync import get_lyrics_quality_report

    quality_report = {
        "lyrics_quality": {},
        "alignment_method": "none",
        "lrc_timing_trust": "normal",
        "whisper_used": False,
        "whisper_auto_enabled": False,
        "whisper_corrections": 0,
        "whisper_requested": use_whisper or whisper_only or whisper_map_lrc,
        "whisper_force_dtw": whisper_force_dtw,
        "total_lines": 0,
        "overall_score": 0.0,
        "issues": [],
        "source": "",
    }
    issues_list: List[str] = quality_report["issues"]  # type: ignore[assignment]

    if whisper_only:
        lines, metadata = get_lyrics_simple(
            title=title,
            artist=artist,
            vocals_path=vocals_path,
            cache_dir=cache_dir,
            lyrics_offset=lyrics_offset,
            romanize=romanize,
            filter_promos=filter_promos,
            target_duration=target_duration,
            evaluate_sources=evaluate_sources,
            use_whisper=use_whisper,
            whisper_only=True,
            whisper_map_lrc=whisper_map_lrc,
            whisper_map_lrc_dtw=whisper_map_lrc_dtw,
            lyrics_file=lyrics_file,
            whisper_language=whisper_language,
            whisper_model=whisper_model,
            whisper_force_dtw=whisper_force_dtw,
            whisper_aggressive=whisper_aggressive,
            whisper_temperature=whisper_temperature,
            lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
            lenient_activity_bonus=lenient_activity_bonus,
            low_word_confidence_threshold=low_word_confidence_threshold,
            offline=offline,
        )
        quality_report["alignment_method"] = "whisper_only"
        quality_report["whisper_used"] = bool(lines)
        quality_report["total_lines"] = len(lines)
        quality_report["overall_score"] = 50.0 if lines else 0.0
        if not lines:
            issues_list.append("Whisper-only mode produced no lines")
        return lines, metadata, quality_report

    if whisper_map_lrc:
        quality_report["alignment_method"] = (
            "whisper_map_lrc_dtw" if whisper_map_lrc_dtw else "whisper_map_lrc"
        )

    file_lines: List[str] = []
    file_lrc_text: Optional[str] = None
    file_line_timings: Optional[List[Tuple[float, str]]] = None
    if lyrics_file:
        file_lrc_text, file_line_timings, file_lines = _load_lyrics_file(
            lyrics_file, filter_promos
        )
        if file_lrc_text or file_lines:
            quality_report["source"] = f"lyrics_file:{lyrics_file}"

    lrc_text, line_timings, source = _fetch_lrc_text_and_timings_for_state(
        title=title,
        artist=artist,
        target_duration=target_duration,
        vocals_path=vocals_path,
        evaluate_sources=evaluate_sources,
        filter_promos=filter_promos,
        offline=offline,
    )
    if file_lrc_text and file_line_timings:
        lrc_text = file_lrc_text
        line_timings = file_line_timings
        source = "lyrics_file_lrc"
    if not quality_report["source"]:
        quality_report["source"] = source
    line_timings = _apply_lrc_timing_trust_policy(
        line_timings=line_timings,
        lrc_text=lrc_text,
        target_duration=target_duration,
        drop_lrc_line_timings=drop_lrc_line_timings,
        vocals_path=vocals_path,
        use_whisper=use_whisper,
        whisper_map_lrc=whisper_map_lrc,
        issues_list=issues_list,
        quality_report=quality_report,
    )

    if (lrc_text or file_lines) and not line_timings:
        if offline:
            genius_lines, metadata = None, None
        else:
            from .genius import fetch_genius_lyrics_with_singers

            genius_lines, metadata = fetch_genius_lyrics_with_singers(title, artist)
    else:
        if offline:
            genius_lines, metadata = None, None
        else:
            genius_lines, metadata = _fetch_genius_with_quality_tracking(
                line_timings, title, artist, quality_report
            )
            if genius_lines is None and not line_timings:
                lines, meta = _create_no_lyrics_placeholder(title, artist)
                return lines, meta, quality_report

    if line_timings and lrc_text:
        quality_report["lyrics_quality"] = get_lyrics_quality_report(
            lrc_text, source, target_duration
        )

    if vocals_path and line_timings:
        line_timings, _ = _detect_offset_with_issues(
            vocals_path, line_timings, lyrics_offset, issues_list
        )

    if should_auto_enable_whisper(
        vocals_path=vocals_path,
        line_timings=line_timings,
        use_whisper=use_whisper,
        whisper_only=whisper_only,
        whisper_map_lrc=whisper_map_lrc,
    ):
        use_whisper = True
        quality_report["whisper_auto_enabled"] = True
        quality_report["whisper_requested"] = True
        issues_list.append(
            "No reliable line-level timings available; auto-enabling Whisper alignment"
        )

    has_lrc_timing = bool(line_timings)
    if lrc_text or file_lines:
        if line_timings and file_lines:
            lines = create_lines_from_lrc_timings(line_timings, file_lines)
            quality_report["alignment_method"] = "lrc_only"
        elif line_timings and lrc_text:
            lines = create_lines_from_lrc(
                lrc_text,
                romanize=False,
                title=title,
                artist=artist,
                filter_promos=filter_promos,
            )
            quality_report["alignment_method"] = "lrc_only"
            _apply_timing_to_lines(lines, line_timings)
        else:
            text_lines = file_lines or _extract_text_lines_from_lrc(lrc_text or "")
            lines = _create_lines_from_plain_text(text_lines)

        if (
            vocals_path
            and line_timings
            and len(line_timings) > 1
            and not whisper_map_lrc
        ):
            lines, method = _refine_timing_with_quality(
                lines,
                vocals_path,
                line_timings,
                lrc_text or "",
                target_duration,
                issues_list,
            )
            quality_report["alignment_method"] = method

        if vocals_path and use_whisper:
            lines, quality_report = _apply_whisper_with_quality(
                lines,
                vocals_path,
                whisper_language,
                whisper_model,
                whisper_force_dtw,
                whisper_aggressive,
                quality_report,
                whisper_temperature=whisper_temperature,
                prefer_whisper_timing_map=not has_lrc_timing,
                lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
                lenient_activity_bonus=lenient_activity_bonus,
                low_word_confidence_threshold=low_word_confidence_threshold,
            )
        elif vocals_path and whisper_map_lrc:
            try:
                if whisper_map_lrc_dtw:
                    from ..whisper.whisper_integration import (
                        align_lrc_text_to_whisper_timings,
                    )

                    model_size = whisper_model or "large"
                    lines, alignments, metrics = align_lrc_text_to_whisper_timings(
                        lines,
                        vocals_path,
                        language=whisper_language,
                        model_size=model_size,
                        aggressive=whisper_aggressive,
                        temperature=whisper_temperature,
                        lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
                        lenient_activity_bonus=lenient_activity_bonus,
                        low_word_confidence_threshold=low_word_confidence_threshold,
                    )
                    quality_report["alignment_method"] = "whisper_map_lrc_dtw"
                    quality_report["whisper_used"] = True
                    quality_report["whisper_corrections"] = len(alignments)
                    if metrics:
                        quality_report["dtw_metrics"] = metrics
                else:
                    from ...phonetic_utils import _whisper_lang_to_epitran
                    from ..whisper.whisper_integration import transcribe_vocals

                    model_size = whisper_model or "large"
                    transcription, _, detected_lang, _model = transcribe_vocals(
                        vocals_path,
                        whisper_language,
                        model_size,
                        whisper_aggressive,
                        whisper_temperature,
                    )
                    if transcription:
                        lang = _whisper_lang_to_epitran(detected_lang)
                        lrc_starts = (
                            [ts for ts, _ in line_timings]
                            if line_timings and not whisper_map_lrc
                            else None
                        )
                        lines, mapped, issues = _map_lrc_lines_to_whisper_segments(
                            lines, transcription, lang, lrc_line_starts=lrc_starts
                        )
                        if mapped:
                            quality_report["alignment_method"] = "whisper_map_lrc"
                            quality_report["whisper_used"] = True
                            quality_report["whisper_corrections"] = mapped
                        for issue in issues:
                            issues_list.append(issue)
            except Exception as e:
                logger.warning(f"Whisper LRC mapping failed: {e}")
                issues_list.append(f"Whisper LRC mapping failed: {e}")
        elif vocals_path and not has_lrc_timing:
            lines, quality_report = _apply_whisper_with_quality(
                lines,
                vocals_path,
                whisper_language,
                whisper_model,
                whisper_force_dtw,
                whisper_aggressive,
                quality_report,
                whisper_temperature=whisper_temperature,
                prefer_whisper_timing_map=True,
                lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
                lenient_activity_bonus=lenient_activity_bonus,
                low_word_confidence_threshold=low_word_confidence_threshold,
            )
    else:
        if genius_lines:
            text_lines = [text for text, _ in genius_lines if text.strip()]
            fallback_text = "\n".join(text_lines)
        else:
            fallback_text = ""
        lines = create_lines_from_lrc(
            fallback_text,
            romanize=romanize,
            title=title,
            artist=artist,
            filter_promos=filter_promos,
        )

    if romanize:
        _romanize_lines(lines)

    if metadata and metadata.is_duet and genius_lines:
        _apply_singer_info(lines, genius_lines, metadata)

    lines = _clip_lines_to_target_duration(lines, target_duration, issues_list)

    quality_report["total_lines"] = len(lines)
    quality_report["overall_score"] = _calculate_quality_score(quality_report)

    logger.debug(
        f"Returning {len(lines)} lines (quality: {quality_report['overall_score']:.0f})"
    )
    return lines, metadata, quality_report


def _fetch_genius_with_quality_tracking_impl(
    line_timings: Optional[List[Tuple[float, str]]],
    title: str,
    artist: str,
    quality_report: dict,
) -> Tuple[Optional[List[Tuple[str, str]]], Optional[SongMetadata]]:
    from .genius import fetch_genius_lyrics_with_singers

    if not line_timings:
        logger.debug("No LRC found, fetching lyrics from Genius...")
        genius_lines, metadata = fetch_genius_lyrics_with_singers(title, artist)
        quality_report["alignment_method"] = "genius_fallback"
        quality_report["issues"].append("No synced LRC found, using Genius text")
        if not genius_lines:
            logger.warning("No lyrics found from any source, using placeholder")
            quality_report["issues"].append("No lyrics found from any source")
            quality_report["overall_score"] = 0.0
            return None, None
        return genius_lines, metadata
    else:
        return fetch_genius_lyrics_with_singers(title, artist)


def _fetch_genius_with_quality_tracking(
    line_timings: Optional[List[Tuple[float, str]]],
    title: str,
    artist: str,
    quality_report: dict,
) -> Tuple[Optional[List[Tuple[str, str]]], Optional[SongMetadata]]:
    """Backwards-compatible alias for direct imports."""
    return _fetch_genius_with_quality_tracking_impl(
        line_timings, title, artist, quality_report
    )


def _apply_whisper_with_quality(
    lines: List[Line],
    vocals_path: str,
    whisper_language: Optional[str],
    whisper_model: Optional[str],
    whisper_force_dtw: bool,
    whisper_aggressive: bool = False,
    quality_report: Optional[dict] = None,
    whisper_temperature: float = 0.0,
    prefer_whisper_timing_map: bool = False,
    lenient_vocal_activity_threshold: float = 0.3,
    lenient_activity_bonus: float = 0.4,
    low_word_confidence_threshold: float = 0.5,
) -> Tuple[List[Line], dict]:
    from . import lyrics_whisper as lw

    if quality_report is None:
        quality_report = {"issues": []}
    try:
        lines, whisper_fixes, whisper_metrics = lw._apply_whisper_alignment_for_state(
            lines,
            vocals_path,
            whisper_language,
            whisper_model,
            whisper_force_dtw,
            whisper_aggressive,
            whisper_temperature=whisper_temperature,
            prefer_whisper_timing_map=prefer_whisper_timing_map,
            lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
            lenient_activity_bonus=lenient_activity_bonus,
            low_word_confidence_threshold=low_word_confidence_threshold,
        )
        no_evidence_fallback = bool(
            float(whisper_metrics.get("no_evidence_fallback", 0.0))
            if whisper_metrics
            else 0.0
        )
        quality_report["whisper_used"] = not no_evidence_fallback
        quality_report["whisper_corrections"] = (
            0 if no_evidence_fallback else len(whisper_fixes)
        )
        if whisper_metrics:
            quality_report["dtw_metrics"] = whisper_metrics
        if no_evidence_fallback:
            quality_report["alignment_method"] = "lrc_only"
        elif whisper_fixes:
            quality_report["alignment_method"] = "whisper_hybrid"
    except Exception as e:
        logger.warning(f"Whisper alignment failed: {e}")
        quality_report["issues"].append(f"Whisper alignment failed: {e}")
    return lines, quality_report
