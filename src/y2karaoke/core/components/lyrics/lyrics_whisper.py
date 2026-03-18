"""Whisper-related lyrics processing and refinement."""

from contextlib import contextmanager
from contextvars import ContextVar
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
from . import lyrics_offset_quality as _offset_quality_policy
from .lyrics_whisper_map import (
    _create_lines_from_whisper,
    _map_lrc_lines_to_whisper_segments,
)
from .lyrics_whisper_pipeline import get_lyrics_simple_impl
from .lyrics_source_routing import (
    _initialize_routing_diagnostics,
    _select_disagreement_source_if_needed,
)
from .runtime_config import LyricsRuntimeConfig, load_lyrics_runtime_config

logger = logging.getLogger(__name__)

__all__ = ["get_lyrics_simple", "get_lyrics_with_quality"]

_detect_offset_with_issues = _offset_quality_policy._detect_offset_with_issues
_refine_timing_with_quality = _offset_quality_policy._refine_timing_with_quality
_calculate_quality_score = _offset_quality_policy._calculate_quality_score
_score_from_dtw_metrics = _offset_quality_policy._score_from_dtw_metrics


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


_ACTIVE_HOOKS: ContextVar[LyricsWhisperHooks] = ContextVar(
    "lyrics_whisper_hooks",
    default=LyricsWhisperHooks(),
)


def resolve_lyrics_whisper_hooks(
    hooks: Optional[LyricsWhisperHooks] = None,
) -> LyricsWhisperHooks:
    current = _ACTIVE_HOOKS.get()
    if hooks is None:
        return current
    return LyricsWhisperHooks(
        fetch_lrc_text_and_timings_fn=(
            hooks.fetch_lrc_text_and_timings_fn or current.fetch_lrc_text_and_timings_fn
        ),
        detect_and_apply_offset_fn=(
            hooks.detect_and_apply_offset_fn or current.detect_and_apply_offset_fn
        ),
        refine_timing_with_audio_fn=(
            hooks.refine_timing_with_audio_fn or current.refine_timing_with_audio_fn
        ),
        apply_whisper_alignment_fn=(
            hooks.apply_whisper_alignment_fn or current.apply_whisper_alignment_fn
        ),
        fetch_genius_lyrics_with_singers_fn=(
            hooks.fetch_genius_lyrics_with_singers_fn
            or current.fetch_genius_lyrics_with_singers_fn
        ),
        transcribe_vocals_fn=hooks.transcribe_vocals_fn or current.transcribe_vocals_fn,
        whisper_lang_to_epitran_fn=(
            hooks.whisper_lang_to_epitran_fn or current.whisper_lang_to_epitran_fn
        ),
        align_lrc_text_to_whisper_timings_fn=(
            hooks.align_lrc_text_to_whisper_timings_fn
            or current.align_lrc_text_to_whisper_timings_fn
        ),
    )


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
    merged_hooks = resolve_lyrics_whisper_hooks(
        LyricsWhisperHooks(
            fetch_lrc_text_and_timings_fn=fetch_lrc_text_and_timings_fn,
            detect_and_apply_offset_fn=detect_and_apply_offset_fn,
            refine_timing_with_audio_fn=refine_timing_with_audio_fn,
            apply_whisper_alignment_fn=apply_whisper_alignment_fn,
            fetch_genius_lyrics_with_singers_fn=fetch_genius_lyrics_with_singers_fn,
            transcribe_vocals_fn=transcribe_vocals_fn,
            whisper_lang_to_epitran_fn=whisper_lang_to_epitran_fn,
            align_lrc_text_to_whisper_timings_fn=(align_lrc_text_to_whisper_timings_fn),
        )
    )
    token = _ACTIVE_HOOKS.set(merged_hooks)
    try:
        yield
    finally:
        _ACTIVE_HOOKS.reset(token)


def _fetch_lrc_text_and_timings_for_state(
    *args, hooks: Optional[LyricsWhisperHooks] = None, **kwargs
):
    fn = resolve_lyrics_whisper_hooks(hooks).fetch_lrc_text_and_timings_fn
    if fn is not None:
        return fn(*args, **kwargs)
    return _fetch_lrc_text_and_timings(*args, **kwargs)


def _detect_and_apply_offset_for_state(
    *args, hooks: Optional[LyricsWhisperHooks] = None, **kwargs
):
    fn = resolve_lyrics_whisper_hooks(hooks).detect_and_apply_offset_fn
    if fn is not None:
        return fn(*args, **kwargs)
    return _detect_and_apply_offset(*args, **kwargs)


def _refine_timing_with_audio_for_state(
    *args, hooks: Optional[LyricsWhisperHooks] = None, **kwargs
):
    fn = resolve_lyrics_whisper_hooks(hooks).refine_timing_with_audio_fn
    if fn is not None:
        return fn(*args, **kwargs)
    return _refine_timing_with_audio(*args, **kwargs)


def _apply_whisper_alignment_for_state(
    *args, hooks: Optional[LyricsWhisperHooks] = None, **kwargs
):
    fn = resolve_lyrics_whisper_hooks(hooks).apply_whisper_alignment_fn
    if fn is not None:
        return fn(*args, **kwargs)
    return _apply_whisper_alignment(*args, **kwargs)


def _fetch_genius_lyrics_with_singers_for_state(
    title: str, artist: str, *, hooks: Optional[LyricsWhisperHooks] = None
) -> Tuple[Optional[List[Tuple[str, str]]], Optional[SongMetadata]]:
    fn = resolve_lyrics_whisper_hooks(hooks).fetch_genius_lyrics_with_singers_fn
    if fn is not None:
        return fn(title, artist)
    from .genius import fetch_genius_lyrics_with_singers

    return fetch_genius_lyrics_with_singers(title, artist)


def _transcribe_vocals_for_state(
    *args, hooks: Optional[LyricsWhisperHooks] = None, **kwargs
):
    fn = resolve_lyrics_whisper_hooks(hooks).transcribe_vocals_fn
    if fn is not None:
        return fn(*args, **kwargs)
    from ..whisper.whisper_integration import transcribe_vocals

    return transcribe_vocals(*args, **kwargs)


def _whisper_lang_to_epitran_for_state(
    detected_lang: str, *, hooks: Optional[LyricsWhisperHooks] = None
) -> str:
    fn = resolve_lyrics_whisper_hooks(hooks).whisper_lang_to_epitran_fn
    if fn is not None:
        return fn(detected_lang)
    from ...phonetic_utils import _whisper_lang_to_epitran

    return _whisper_lang_to_epitran(detected_lang)


def _align_lrc_text_to_whisper_timings_for_state(
    *args, hooks: Optional[LyricsWhisperHooks] = None, **kwargs
):
    fn = resolve_lyrics_whisper_hooks(hooks).align_lrc_text_to_whisper_timings_fn
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


def _fetch_lrc_text_and_timings(
    title: str,
    artist: str,
    target_duration: Optional[int] = None,
    vocals_path: Optional[str] = None,
    evaluate_sources: bool = False,
    filter_promos: bool = True,
    offline: bool = False,
    routing_diagnostics: Optional[dict] = None,
    runtime_config: Optional[LyricsRuntimeConfig] = None,
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
        runtime_config = runtime_config or load_lyrics_runtime_config()
        duration_tolerance = runtime_config.lrc_duration_tolerance_sec
        _initialize_routing_diagnostics(
            routing_diagnostics,
            target_duration=target_duration,
            vocals_path=vocals_path,
            evaluate_sources=evaluate_sources,
            offline=offline,
        )

        disagreement_selection = _select_disagreement_source_if_needed(
            title=title,
            artist=artist,
            target_duration=target_duration,
            vocals_path=vocals_path,
            evaluate_sources=evaluate_sources,
            filter_promos=filter_promos,
            offline=offline,
            routing_diagnostics=routing_diagnostics,
            runtime_config=runtime_config,
            logger=logger,
        )
        if disagreement_selection is not None:
            return disagreement_selection

        # If evaluation is requested and we have vocals, compare all sources
        if evaluate_sources and vocals_path and not offline:
            from ..alignment.timing_evaluator import select_best_source

            if routing_diagnostics is not None:
                routing_diagnostics["lyrics_source_audio_scoring_used"] = True
                routing_diagnostics["lyrics_source_selection_mode"] = (
                    "audio_scored_explicit"
                )
                routing_diagnostics["lyrics_source_routing_skip_reason"] = "none"
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
                title,
                artist,
                target_duration,
                tolerance=duration_tolerance,
                offline=offline,
                runtime_config=runtime_config,
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
                title, artist, offline=offline, runtime_config=runtime_config
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
    hooks: Optional[LyricsWhisperHooks] = None,
    runtime_config: Optional[LyricsRuntimeConfig] = None,
) -> Tuple[List[Line], Optional[SongMetadata]]:
    """Simplified lyrics pipeline favoring LRC over Genius."""
    from .sync import get_lrc_duration

    _ = cache_dir
    resolved_hooks = resolve_lyrics_whisper_hooks(hooks)
    runtime_config = runtime_config or load_lyrics_runtime_config()
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
        transcribe_vocals_for_state_fn=lambda *a, **k: _transcribe_vocals_for_state(
            *a, hooks=resolved_hooks, **k
        ),
        create_lines_from_whisper_fn=_create_lines_from_whisper,
        romanize_lines_fn=_romanize_lines,
        load_lyrics_file_fn=_load_lyrics_file,
        fetch_lrc_text_and_timings_for_state_fn=lambda *a, **k: _fetch_lrc_text_and_timings_for_state(
            *a, hooks=resolved_hooks, runtime_config=runtime_config, **k
        ),
        get_lrc_duration_fn=get_lrc_duration,
        fetch_genius_lyrics_with_singers_for_state_fn=(
            lambda *a, **k: _fetch_genius_lyrics_with_singers_for_state(
                *a, hooks=resolved_hooks, **k
            )
        ),
        detect_and_apply_offset_for_state_fn=lambda *a, **k: _detect_and_apply_offset_for_state(
            *a, hooks=resolved_hooks, **k
        ),
        create_lines_from_lrc_timings_fn=create_lines_from_lrc_timings,
        create_lines_from_lrc_fn=create_lines_from_lrc,
        apply_timing_to_lines_fn=_apply_timing_to_lines,
        extract_text_lines_from_lrc_fn=_extract_text_lines_from_lrc,
        create_lines_from_plain_text_fn=_create_lines_from_plain_text,
        refine_timing_with_audio_for_state_fn=lambda *a, **k: _refine_timing_with_audio_for_state(
            *a, hooks=resolved_hooks, **k
        ),
        apply_whisper_alignment_for_state_fn=lambda *a, **k: _apply_whisper_alignment_for_state(
            *a, hooks=resolved_hooks, **k
        ),
        align_lrc_text_to_whisper_timings_for_state_fn=(
            lambda *a, **k: _align_lrc_text_to_whisper_timings_for_state(
                *a, hooks=resolved_hooks, **k
            )
        ),
        whisper_lang_to_epitran_for_state_fn=lambda *a, **k: _whisper_lang_to_epitran_for_state(
            *a, hooks=resolved_hooks, **k
        ),
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
