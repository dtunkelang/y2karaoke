"""Quality-report workflow for lyrics + Whisper processing."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from .components.lyrics.helpers import (
    _apply_timing_to_lines,
    _create_lines_from_plain_text,
    _create_no_lyrics_placeholder,
    _extract_text_lines_from_lrc,
    _load_lyrics_file,
    _romanize_lines,
)
from .lyrics_whisper_map import _map_lrc_lines_to_whisper_segments
from .models import Line, SongMetadata

logger = logging.getLogger(__name__)


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
        _fetch_lrc_text_and_timings,
        _refine_timing_with_quality,
        create_lines_from_lrc,
        create_lines_from_lrc_timings,
        get_lyrics_simple,
    )
    from .sync import get_lyrics_quality_report

    quality_report = {
        "lyrics_quality": {},
        "alignment_method": "none",
        "whisper_used": False,
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

    lrc_text, line_timings, source = _fetch_lrc_text_and_timings(
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
    if target_duration and lrc_text:
        from .sync import get_lrc_duration

        lrc_duration = get_lrc_duration(lrc_text)
        if lrc_duration and abs(target_duration - lrc_duration) > 8:
            issues_list.append(
                "LRC duration mismatch: keeping text but ignoring LRC timings"
            )
            line_timings = None

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
                    from .whisper_integration import align_lrc_text_to_whisper_timings

                    model_size = whisper_model or "small"
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
                    from .phonetic_utils import _whisper_lang_to_epitran
                    from .whisper_integration import transcribe_vocals

                    model_size = whisper_model or "small"
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
        lines, whisper_fixes, whisper_metrics = lw._apply_whisper_alignment(
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
        quality_report["whisper_used"] = True
        quality_report["whisper_corrections"] = len(whisper_fixes)
        if whisper_metrics:
            quality_report["dtw_metrics"] = whisper_metrics
        if whisper_fixes:
            quality_report["alignment_method"] = "whisper_hybrid"
    except Exception as e:
        logger.warning(f"Whisper alignment failed: {e}")
        quality_report["issues"].append(f"Whisper alignment failed: {e}")
    return lines, quality_report
