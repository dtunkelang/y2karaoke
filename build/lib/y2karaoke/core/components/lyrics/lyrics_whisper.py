"""Whisper-related lyrics processing and refinement."""

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

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
from .lyrics_whisper_pipeline import get_lyrics_simple_impl

logger = logging.getLogger(__name__)

__all__ = ["get_lyrics_simple", "get_lyrics_with_quality"]


@dataclass
class LyricsWhisperHooks:
    """Optional runtime overrides for lyrics-whisper collaborators."""

    fetch_lrc_text_and_timings_fn: Optional[
        Callable[..., Tuple[Optional[str], Optional[List[Tuple[float, str]]], str]]
    ] = None
    detect_and_apply_offset_fn: Optional[
        Callable[..., Tuple[List[Tuple[float, str]], float]]
    ] = None
    refine_timing_with_audio_fn: Optional[Callable[..., List[Line]]] = None
    apply_whisper_alignment_fn: Optional[
        Callable[..., Tuple[List[Line], List[str], dict]]
    ] = None
    fetch_genius_lyrics_with_singers_fn: Optional[
        Callable[..., Tuple[Optional[List[Tuple[str, str]]], Optional[SongMetadata]]]
    ] = None
    transcribe_vocals_fn: Optional[Callable[..., Tuple[list, list, str, str]]] = None
    whisper_lang_to_epitran_fn: Optional[Callable[..., str]] = None
    align_lrc_text_to_whisper_timings_fn: Optional[
        Callable[..., Tuple[List[Line], list, dict]]
    ] = None


_ACTIVE_HOOKS = LyricsWhisperHooks()


@contextmanager
def use_lyrics_whisper_hooks(
    *,
    fetch_lrc_text_and_timings_fn: Optional[
        Callable[..., Tuple[Optional[str], Optional[List[Tuple[float, str]]], str]]
    ] = None,
    detect_and_apply_offset_fn: Optional[
        Callable[..., Tuple[List[Tuple[float, str]], float]]
    ] = None,
    refine_timing_with_audio_fn: Optional[Callable[..., List[Line]]] = None,
    apply_whisper_alignment_fn: Optional[
        Callable[..., Tuple[List[Line], List[str], dict]]
    ] = None,
    fetch_genius_lyrics_with_singers_fn: Optional[
        Callable[..., Tuple[Optional[List[Tuple[str, str]]], Optional[SongMetadata]]]
    ] = None,
    transcribe_vocals_fn: Optional[Callable[..., Tuple[list, list, str, str]]] = None,
    whisper_lang_to_epitran_fn: Optional[Callable[..., str]] = None,
    align_lrc_text_to_whisper_timings_fn: Optional[
        Callable[..., Tuple[List[Line], list, dict]]
    ] = None,
) -> Iterator[None]:
    """Temporarily override lyrics-whisper collaborators for tests."""
    global _ACTIVE_HOOKS

    previous = _ACTIVE_HOOKS
    _ACTIVE_HOOKS = LyricsWhisperHooks(
        fetch_lrc_text_and_timings_fn=(
            fetch_lrc_text_and_timings_fn
            if fetch_lrc_text_and_timings_fn is not None
            else previous.fetch_lrc_text_and_timings_fn
        ),
        detect_and_apply_offset_fn=(
            detect_and_apply_offset_fn
            if detect_and_apply_offset_fn is not None
            else previous.detect_and_apply_offset_fn
        ),
        refine_timing_with_audio_fn=(
            refine_timing_with_audio_fn
            if refine_timing_with_audio_fn is not None
            else previous.refine_timing_with_audio_fn
        ),
        apply_whisper_alignment_fn=(
            apply_whisper_alignment_fn
            if apply_whisper_alignment_fn is not None
            else previous.apply_whisper_alignment_fn
        ),
        fetch_genius_lyrics_with_singers_fn=(
            fetch_genius_lyrics_with_singers_fn
            if fetch_genius_lyrics_with_singers_fn is not None
            else previous.fetch_genius_lyrics_with_singers_fn
        ),
        transcribe_vocals_fn=(
            transcribe_vocals_fn
            if transcribe_vocals_fn is not None
            else previous.transcribe_vocals_fn
        ),
        whisper_lang_to_epitran_fn=(
            whisper_lang_to_epitran_fn
            if whisper_lang_to_epitran_fn is not None
            else previous.whisper_lang_to_epitran_fn
        ),
        align_lrc_text_to_whisper_timings_fn=(
            align_lrc_text_to_whisper_timings_fn
            if align_lrc_text_to_whisper_timings_fn is not None
            else previous.align_lrc_text_to_whisper_timings_fn
        ),
    )
    try:
        yield
    finally:
        _ACTIVE_HOOKS = previous


def _fetch_lrc_text_and_timings_for_state(*args, **kwargs):
    fn = _ACTIVE_HOOKS.fetch_lrc_text_and_timings_fn
    if fn is not None:
        return fn(*args, **kwargs)
    return _fetch_lrc_text_and_timings(*args, **kwargs)


def _detect_and_apply_offset_for_state(*args, **kwargs):
    fn = _ACTIVE_HOOKS.detect_and_apply_offset_fn
    if fn is not None:
        return fn(*args, **kwargs)
    return _detect_and_apply_offset(*args, **kwargs)


def _refine_timing_with_audio_for_state(*args, **kwargs):
    fn = _ACTIVE_HOOKS.refine_timing_with_audio_fn
    if fn is not None:
        return fn(*args, **kwargs)
    return _refine_timing_with_audio(*args, **kwargs)


def _apply_whisper_alignment_for_state(*args, **kwargs):
    fn = _ACTIVE_HOOKS.apply_whisper_alignment_fn
    if fn is not None:
        return fn(*args, **kwargs)
    return _apply_whisper_alignment(*args, **kwargs)


def _fetch_genius_lyrics_with_singers_for_state(
    title: str, artist: str
) -> Tuple[Optional[List[Tuple[str, str]]], Optional[SongMetadata]]:
    fn = _ACTIVE_HOOKS.fetch_genius_lyrics_with_singers_fn
    if fn is not None:
        return fn(title, artist)
    from .genius import fetch_genius_lyrics_with_singers

    return fetch_genius_lyrics_with_singers(title, artist)


def _transcribe_vocals_for_state(*args, **kwargs):
    fn = _ACTIVE_HOOKS.transcribe_vocals_fn
    if fn is not None:
        return fn(*args, **kwargs)
    from ..whisper.whisper_integration import transcribe_vocals

    return transcribe_vocals(*args, **kwargs)


def _whisper_lang_to_epitran_for_state(detected_lang: str) -> str:
    fn = _ACTIVE_HOOKS.whisper_lang_to_epitran_fn
    if fn is not None:
        return fn(detected_lang)
    from ...phonetic_utils import _whisper_lang_to_epitran

    return _whisper_lang_to_epitran(detected_lang)


def _align_lrc_text_to_whisper_timings_for_state(*args, **kwargs):
    fn = _ACTIVE_HOOKS.align_lrc_text_to_whisper_timings_fn
    if fn is not None:
        return fn(*args, **kwargs)
    from ..whisper.whisper_integration import align_lrc_text_to_whisper_timings

    return align_lrc_text_to_whisper_timings(*args, **kwargs)


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

    AUTO_OFFSET_MAX_ABS_SEC = 5.0

    offset = 0.0
    if lyrics_offset is not None:
        offset = lyrics_offset
    elif abs(delta) > 2.5 and abs(delta) <= AUTO_OFFSET_MAX_ABS_SEC:
        logger.warning(
            f"Detected vocal offset ({delta:+.2f}s) matches suspicious range (2.5-5.0s) - NOT auto-applying."
        )
    elif abs(delta) > 0.3 and abs(delta) <= 2.5:
        offset = delta
        logger.info(f"Auto-applying vocal offset: {offset:+.2f}s")
    elif abs(delta) > AUTO_OFFSET_MAX_ABS_SEC:
        logger.warning(
            "Large timing delta (%+.2fs) exceeds auto-offset clamp (%.1fs) - "
            "not auto-applying.",
            delta,
            AUTO_OFFSET_MAX_ABS_SEC,
        )
        issues.append(
            f"Large timing delta ({delta:+.2f}s) exceeded auto-offset clamp and was not applied"
        )

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
    phonetic_similarity_coverage = float(
        metrics.get("phonetic_similarity_coverage", matched_ratio * avg_similarity)
    )
    high_similarity_ratio = float(metrics.get("high_similarity_ratio", avg_similarity))
    exact_match_ratio = float(metrics.get("exact_match_ratio", 0.0))

    score = 40.0
    score += matched_ratio * 20.0
    score += avg_similarity * 15.0
    score += line_coverage * 10.0
    score += phonetic_similarity_coverage * 10.0
    score += high_similarity_ratio * 3.0
    score += exact_match_ratio * 2.0
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
    """Simplified lyrics pipeline favoring LRC over Genius."""
    from .sync import get_lrc_duration

    _ = cache_dir
    return get_lyrics_simple_impl(
        title,
        artist,
        vocals_path,
        lyrics_offset,
        romanize,
        filter_promos,
        target_duration,
        evaluate_sources,
        use_whisper,
        whisper_only,
        whisper_map_lrc,
        whisper_map_lrc_dtw,
        lyrics_file,
        whisper_language,
        whisper_model,
        whisper_force_dtw,
        whisper_aggressive,
        whisper_temperature,
        lenient_vocal_activity_threshold,
        lenient_activity_bonus,
        low_word_confidence_threshold,
        offline,
        create_no_lyrics_placeholder_fn=_create_no_lyrics_placeholder,
        transcribe_vocals_for_state_fn=_transcribe_vocals_for_state,
        create_lines_from_whisper_fn=_create_lines_from_whisper,
        romanize_lines_fn=_romanize_lines,
        load_lyrics_file_fn=_load_lyrics_file,
        fetch_lrc_text_and_timings_for_state_fn=_fetch_lrc_text_and_timings_for_state,
        get_lrc_duration_fn=get_lrc_duration,
        fetch_genius_lyrics_with_singers_for_state_fn=(
            _fetch_genius_lyrics_with_singers_for_state
        ),
        detect_and_apply_offset_for_state_fn=_detect_and_apply_offset_for_state,
        create_lines_from_lrc_timings_fn=create_lines_from_lrc_timings,
        create_lines_from_lrc_fn=create_lines_from_lrc,
        apply_timing_to_lines_fn=_apply_timing_to_lines,
        extract_text_lines_from_lrc_fn=_extract_text_lines_from_lrc,
        create_lines_from_plain_text_fn=_create_lines_from_plain_text,
        refine_timing_with_audio_for_state_fn=_refine_timing_with_audio_for_state,
        apply_whisper_alignment_for_state_fn=_apply_whisper_alignment_for_state,
        align_lrc_text_to_whisper_timings_for_state_fn=(
            _align_lrc_text_to_whisper_timings_for_state
        ),
        whisper_lang_to_epitran_for_state_fn=_whisper_lang_to_epitran_for_state,
        map_lrc_lines_to_whisper_segments_fn=_map_lrc_lines_to_whisper_segments,
        apply_singer_info_fn=_apply_singer_info,
        logger=logger,
    )


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
