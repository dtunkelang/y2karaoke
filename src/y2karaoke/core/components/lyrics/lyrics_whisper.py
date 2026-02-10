"""Whisper-related lyrics processing and refinement."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from ...models import Line, SongMetadata
from .lrc import (
    parse_lrc_with_timing,
    create_lines_from_lrc,
    create_lines_from_lrc_timings,
)
from .helpers import (
    _romanize_lines,
    _detect_and_apply_offset,
    _refine_timing_with_audio,
    _apply_timing_to_lines,
    _apply_whisper_alignment,
    _create_no_lyrics_placeholder,
    _load_lyrics_file,
    _extract_text_lines_from_lrc,
    _create_lines_from_plain_text,
)
from .lyrics_whisper_map import (
    _create_lines_from_whisper,
    _map_lrc_lines_to_whisper_segments,
)

logger = logging.getLogger(__name__)

__all__ = ["get_lyrics_simple", "get_lyrics_with_quality"]


def _apply_singer_info(
    lines: List[Line],
    genius_lines: List[Tuple[str, str]],
    metadata: SongMetadata,
) -> None:
    """Apply singer info from Genius to lines for duets."""
    for i, line in enumerate(lines):
        if i < len(genius_lines):
            _, singer_name = genius_lines[i]
            singer_id = metadata.get_singer_id(singer_name)
            line.singer = singer_id
            for word in line.words:
                word.singer = singer_id


def _detect_offset_with_issues(
    vocals_path: str,
    line_timings: List[Tuple[float, str]],
    lyrics_offset: Optional[float],
    issues: List[str],
) -> Tuple[List[Tuple[float, str]], float]:
    """Detect vocal offset, track issues for quality report.

    Returns (updated_line_timings, offset_applied).
    """
    from ..alignment.alignment import detect_song_start

    detected_vocal_start = detect_song_start(vocals_path)
    first_lrc_time = line_timings[0][0]
    delta = detected_vocal_start - first_lrc_time

    logger.info(
        f"Vocal timing: audio_start={detected_vocal_start:.2f}s, "
        f"LRC_start={first_lrc_time:.2f}s, delta={delta:+.2f}s"
    )

    offset = 0.0
    if lyrics_offset is not None:
        offset = lyrics_offset
    elif abs(delta) > 0.3 and abs(delta) <= 30.0:
        if abs(delta) > 10.0:
            logger.warning(f"Large vocal offset ({delta:+.2f}s) - audio may have intro")
            issues.append(f"Large vocal offset ({delta:+.2f}s)")
        offset = delta
        logger.info(f"Auto-applying vocal offset: {offset:+.2f}s")
    elif abs(delta) > 30.0:
        logger.warning(f"Large timing delta ({delta:+.2f}s) - not auto-applying.")
        issues.append(f"Large timing delta ({delta:+.2f}s) not applied")

    if offset != 0.0:
        line_timings = [(ts + offset, text) for ts, text in line_timings]

    return line_timings, offset


def _refine_timing_with_quality(
    lines: List[Line],
    vocals_path: str,
    line_timings: List[Tuple[float, str]],
    lrc_text: str,
    target_duration: Optional[int],
    issues: List[str],
) -> Tuple[List[Line], str]:
    """Refine timing and track issues. Returns (lines, alignment_method)."""
    from ...refine import refine_word_timing
    from ..alignment.alignment import adjust_timing_for_duration_mismatch
    from .sync import get_lrc_duration

    lines = refine_word_timing(lines, vocals_path)
    alignment_method = "onset_refined"
    logger.debug("Word-level timing refined using vocals")

    lrc_duration = get_lrc_duration(lrc_text)
    if target_duration and lrc_duration and abs(target_duration - lrc_duration) > 8:
        logger.info(f"Duration mismatch: LRC={lrc_duration}s, audio={target_duration}s")
        issues.append(
            f"Duration mismatch: LRC={lrc_duration}s vs audio={target_duration}s"
        )
        lines = adjust_timing_for_duration_mismatch(
            lines,
            line_timings,
            vocals_path,
            lrc_duration=lrc_duration,
            audio_duration=target_duration,
        )

    return lines, alignment_method


def _calculate_quality_score(quality_report: dict) -> float:
    """Calculate overall quality score from report components."""
    # Base score on lyrics quality if available
    if quality_report["lyrics_quality"]:
        base_score = quality_report["lyrics_quality"].get("quality_score", 50.0)
    elif quality_report.get("dtw_metrics"):
        base_score = _score_from_dtw_metrics(quality_report["dtw_metrics"])
    else:
        base_score = 30.0  # Genius fallback

    # Adjust for alignment method
    method_bonus = {
        "whisper_hybrid": 10,
        "onset_refined": 5,
        "lrc_only": 0,
        "genius_fallback": -20,
        "none": -50,
    }
    base_score += method_bonus.get(quality_report["alignment_method"], 0)

    # Adjust for issues
    base_score -= len(quality_report["issues"]) * 5

    return max(0.0, min(100.0, base_score))


def _score_from_dtw_metrics(metrics: dict) -> float:
    """Heuristic score derived from Whisper/DTW alignment metrics."""
    matched_ratio = float(metrics.get("matched_ratio", 0.0))
    avg_similarity = float(metrics.get("avg_similarity", 0.0))
    line_coverage = float(metrics.get("line_coverage", 0.0))

    score = 40.0
    score += matched_ratio * 25.0
    score += avg_similarity * 20.0
    score += line_coverage * 10.0
    return max(20.0, min(100.0, score))


def _fetch_lrc_text_and_timings(
    title: str,
    artist: str,
    target_duration: Optional[int] = None,
    vocals_path: Optional[str] = None,
    evaluate_sources: bool = False,
    filter_promos: bool = True,
    offline: bool = False,
) -> Tuple[Optional[str], Optional[List[Tuple[float, str]]], str]:
    """Fetch raw LRC text and parsed timings from available sources.

    Args:
        title: Song title
        artist: Artist name
        target_duration: Expected track duration in seconds (for validation)
        vocals_path: Path to vocals audio (for timing evaluation)
        evaluate_sources: If True, compare all sources and select best based on timing

    Returns:
        Tuple of (lrc_text, parsed_timings, source_name)
    """
    try:
        # If evaluation is requested and we have vocals, compare all sources
        if evaluate_sources and vocals_path and not offline:
            from ..alignment.timing_evaluator import select_best_source

            lrc_text, source, report = select_best_source(
                title, artist, vocals_path, target_duration
            )
            if lrc_text and source:
                lines = parse_lrc_with_timing(
                    lrc_text, title, artist, filter_promos=filter_promos
                )
                score_str = f" (score: {report.overall_score:.1f})" if report else ""
                logger.info(f"Selected best source: {source}{score_str}")
                return lrc_text, lines, source
            # Fall through to standard fetch if evaluation fails

        if target_duration:
            # Use duration-aware fetch to find LRC matching target
            from .sync import fetch_lyrics_for_duration

            lrc_text, is_synced, source, lrc_duration = fetch_lyrics_for_duration(
                title, artist, target_duration, tolerance=8, offline=offline
            )
            if lrc_text and is_synced:
                lines = parse_lrc_with_timing(
                    lrc_text, title, artist, filter_promos=filter_promos
                )
                logger.debug(
                    f"Got {len(lines)} LRC lines from {source} (duration: {lrc_duration}s)"
                )
                return lrc_text, lines, source
            else:
                logger.debug("No duration-matched LRC available")
                return None, None, ""
        else:
            # Fallback to standard fetch without duration validation
            from .sync import fetch_lyrics_multi_source

            lrc_text, is_synced, source = fetch_lyrics_multi_source(
                title, artist, offline=offline
            )
            if lrc_text and is_synced:
                lines = parse_lrc_with_timing(
                    lrc_text, title, artist, filter_promos=filter_promos
                )
                logger.debug(f"Got {len(lines)} LRC lines from {source}")
                return lrc_text, lines, source
            else:
                logger.debug(f"No synced LRC available from {source}")
                return None, None, ""
    except Exception as e:
        logger.warning(f"LRC fetch failed: {e}")
        return None, None, ""


def get_lyrics_simple(  # noqa: C901
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
) -> Tuple[List[Line], Optional[SongMetadata]]:
    """Simplified lyrics pipeline favoring LRC over Genius.

    Pipeline:
    1. Try to fetch LRC lyrics with timing (preferred source)
    2. If no LRC, fall back to Genius lyrics
    3. Detect vocal offset and align timing
    4. Create Line objects with word-level timing
    5. Refine timing using audio onset detection
    6. Optionally align to Whisper transcription for severely broken LRC
    7. Apply romanization if needed

    Args:
        title: Song title
        artist: Artist name
        vocals_path: Path to vocals audio (for timing refinement)
        cache_dir: Cache directory (unused, for API compatibility)
        lyrics_offset: Manual timing offset in seconds (auto-detected if None)
        romanize: Whether to romanize non-Latin scripts
        target_duration: Expected track duration in seconds (for LRC validation)
        evaluate_sources: If True, compare all lyrics sources and select best
                         based on timing alignment with audio
        use_whisper: If True, use Whisper transcription to align lyrics timing
        whisper_only: If True, generate lines directly from Whisper (no LRC/Genius)
        whisper_map_lrc: If True, map LRC text onto Whisper timing without shifting segments
        whisper_map_lrc_dtw: If True, map LRC text onto Whisper timing using phonetic DTW
        lyrics_file: Optional local lyrics file (plain text or .lrc)
        whisper_language: Language code for Whisper (auto-detected if None)
        whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        whisper_temperature: Temperature for Whisper transcription
        lenient_vocal_activity_threshold: Threshold for vocal activity
        lenient_activity_bonus: Bonus for phonetic cost under leniency
        low_word_confidence_threshold: Threshold for whisper word confidence

    Returns:
        Tuple of (lines, metadata)
    """
    from .genius import fetch_genius_lyrics_with_singers

    # Whisper-only mode: generate lines directly from transcription.
    if whisper_only:
        if not vocals_path:
            logger.warning(
                "Whisper-only mode requires vocals; using placeholder lyrics"
            )
            return _create_no_lyrics_placeholder(title, artist)
        from ..whisper.whisper_integration import transcribe_vocals

        model_size = whisper_model or "base"
        transcription, _, detected_lang, _model = transcribe_vocals(
            vocals_path,
            whisper_language,
            model_size,
            whisper_aggressive,
            whisper_temperature,
        )
        if not transcription:
            logger.warning(
                "No Whisper transcription available; using placeholder lyrics"
            )
            return _create_no_lyrics_placeholder(title, artist)
        lines = _create_lines_from_whisper(transcription)
        whisper_metadata = SongMetadata(
            singers=[],
            is_duet=False,
            title=title,
            artist=artist,
        )
        if romanize:
            _romanize_lines(lines)
        logger.debug(f"Returning {len(lines)} lines from Whisper-only mode")
        return lines, whisper_metadata

    file_lines: List[str] = []
    file_lrc_text: Optional[str] = None
    file_line_timings: Optional[List[Tuple[float, str]]] = None
    if lyrics_file:
        file_lrc_text, file_line_timings, file_lines = _load_lyrics_file(
            lyrics_file, filter_promos
        )
        if file_lines or file_lrc_text:
            logger.info(f"Using lyrics from file: {lyrics_file}")

    # 1. Try LRC first (preferred source), with duration validation if provided
    logger.debug(
        f"Fetching LRC lyrics... (target_duration={target_duration}, "
        f"evaluate={evaluate_sources})"
    )
    lrc_text, line_timings, _source = _fetch_lrc_text_and_timings(
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
    if target_duration and lrc_text:
        from .sync import get_lrc_duration

        lrc_duration = get_lrc_duration(lrc_text)
        if lrc_duration and abs(target_duration - lrc_duration) > 8:
            logger.warning(
                "LRC duration mismatch: keeping text but ignoring LRC timings"
            )
            line_timings = None

    # 2. Fetch Genius as fallback or for singer info
    genius_lines: Optional[List[Tuple[str, str]]] = None
    metadata: Optional[SongMetadata] = None
    if not line_timings and not lrc_text and not file_lines:
        if offline:
            logger.warning("Offline mode: no cached lyrics available")
            return _create_no_lyrics_placeholder(title, artist)
        logger.debug("No LRC found, fetching lyrics from Genius...")
        genius_lines, metadata = fetch_genius_lyrics_with_singers(title, artist)
        if not genius_lines:
            logger.warning("No lyrics found from any source, using placeholder")
            return _create_no_lyrics_placeholder(title, artist)
    else:
        # Fetch Genius for singer/duet metadata only
        genius_lines, metadata = fetch_genius_lyrics_with_singers(title, artist)

    # 3. Apply vocal offset if available
    if vocals_path and line_timings:
        line_timings, _ = _detect_and_apply_offset(
            vocals_path, line_timings, lyrics_offset
        )

    # 4. Create Line objects
    has_lrc_timing = bool(line_timings)
    if lrc_text or file_lines:
        if line_timings and file_lines:
            lines = create_lines_from_lrc_timings(line_timings, file_lines)
        elif line_timings and lrc_text:
            lines = create_lines_from_lrc(
                lrc_text,
                romanize=False,
                title=title,
                artist=artist,
                filter_promos=filter_promos,
            )
            _apply_timing_to_lines(lines, line_timings)
        else:
            text_lines = file_lines or _extract_text_lines_from_lrc(lrc_text or "")
            lines = _create_lines_from_plain_text(text_lines)

        # 5. Refine word timing using audio
        if vocals_path and line_timings and len(line_timings) > 1:
            lines = _refine_timing_with_audio(
                lines, vocals_path, line_timings, lrc_text or "", target_duration
            )

        # 5b. Optionally use Whisper for more accurate alignment
        if vocals_path and use_whisper:
            try:
                lines, _, _ = _apply_whisper_alignment(
                    lines,
                    vocals_path,
                    whisper_language,
                    whisper_model,
                    whisper_force_dtw,
                    whisper_aggressive,
                    whisper_temperature=whisper_temperature,
                    prefer_whisper_timing_map=not has_lrc_timing,
                    lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
                    lenient_activity_bonus=lenient_activity_bonus,
                    low_word_confidence_threshold=low_word_confidence_threshold,
                )
            except Exception as e:
                logger.warning(f"Whisper alignment failed: {e}")
        elif vocals_path and whisper_map_lrc:
            try:
                if whisper_map_lrc_dtw:
                    from ..whisper.whisper_integration import (
                        align_lrc_text_to_whisper_timings,
                    )

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
                    logger.info(
                        f"DTW-mapped {len(alignments)} LRC line(s) onto Whisper timing"
                    )
                    if metrics:
                        logger.debug(f"DTW metrics: {metrics}")
                else:
                    from ...phonetic_utils import _whisper_lang_to_epitran
                    from ..whisper.whisper_integration import transcribe_vocals

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
                            logger.info(
                                f"Mapped {mapped} LRC line(s) onto Whisper timing"
                            )
                        for issue in issues:
                            logger.debug(issue)
            except Exception as e:
                logger.warning(f"Whisper LRC mapping failed: {e}")
    else:
        # Fallback: use Genius text with evenly spaced lines
        if genius_lines:
            text_lines = [text for text, _ in genius_lines if text.strip()]
            lrc_text = "\n".join(text_lines)
        else:
            lrc_text = ""
        lines = create_lines_from_lrc(
            lrc_text,
            romanize=romanize,
            title=title,
            artist=artist,
            filter_promos=filter_promos,
        )

    # 6. Romanize
    if romanize:
        _romanize_lines(lines)

    # Apply singer info for duets (from Genius metadata)
    if metadata and metadata.is_duet and genius_lines:
        _apply_singer_info(lines, genius_lines, metadata)

    logger.debug(f"Returning {len(lines)} lines")
    return lines, metadata


def get_lyrics_with_quality(*args, **kwargs):
    """Compatibility wrapper for quality-aware lyrics flow."""
    from .lyrics_whisper_quality import get_lyrics_with_quality as _impl

    return _impl(*args, **kwargs)


def _fetch_genius_with_quality_tracking(*args, **kwargs):
    """Compatibility wrapper."""
    from .lyrics_whisper_quality import (
        _fetch_genius_with_quality_tracking_impl as _impl,
    )

    return _impl(*args, **kwargs)


def _apply_whisper_with_quality(*args, **kwargs):
    """Compatibility wrapper."""
    from .lyrics_whisper_quality import _apply_whisper_with_quality as _impl

    return _impl(*args, **kwargs)
