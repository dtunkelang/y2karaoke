"""Quality-report workflow for lyrics + Whisper processing."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from .helpers import _romanize_lines
from . import lyrics_quality_alignment as _alignment
from . import lyrics_quality_sources as _sources
from . import lyrics_quality_tail_guardrail as _tail_guardrail
from .lyrics_whisper_pipeline import should_auto_enable_whisper
from ...models import Line, SongMetadata

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .lyrics_whisper import LyricsWhisperHooks
    from .runtime_config import LyricsRuntimeConfig

TailGuardrailSnapshot = _tail_guardrail.TailGuardrailSnapshot
_line_set_end = _tail_guardrail._line_set_end
_tail_guardrail_snapshot = _tail_guardrail._tail_guardrail_snapshot
_tail_guardrail_should_accept_retry = (
    _tail_guardrail._tail_guardrail_should_accept_retry
)
_apply_tail_guardrail_metrics = _tail_guardrail._apply_tail_guardrail_metrics
_maybe_retry_tail_guardrail = _tail_guardrail._maybe_retry_tail_guardrail
_clip_lines_to_target_duration = _tail_guardrail._clip_lines_to_target_duration

_build_quality_report = _sources._build_quality_report
_resolve_lrc_inputs = _sources._resolve_lrc_inputs
_resolve_genius_lines_and_metadata = _sources._resolve_genius_lines_and_metadata
_detect_line_timing_offset = _sources._detect_line_timing_offset
_should_suppress_disagreement_negative_offset = (
    _sources._should_suppress_disagreement_negative_offset
)
_fetch_genius_with_quality_tracking_impl = (
    _sources._fetch_genius_with_quality_tracking_impl
)
_fetch_genius_with_quality_tracking = _sources._fetch_genius_with_quality_tracking

_build_lines_from_lyrics_source = _alignment._build_lines_from_lyrics_source
_build_fallback_lines = _alignment._build_fallback_lines
_apply_audio_alignment = _alignment._apply_audio_alignment
_apply_whisper_with_quality = _alignment._apply_whisper_with_quality


def _get_whisper_only_quality_result(
    *,
    title: str,
    artist: str,
    vocals_path: Optional[str],
    cache_dir: Optional[str],
    lyrics_offset: Optional[float],
    romanize: bool,
    filter_promos: bool,
    target_duration: Optional[int],
    evaluate_sources: bool,
    use_whisper: bool,
    whisper_map_lrc: bool,
    whisper_map_lrc_dtw: bool,
    lyrics_file: Optional[Path],
    audio_start: float,
    whisper_language: Optional[str],
    whisper_model: Optional[str],
    whisper_force_dtw: bool,
    whisper_aggressive: bool,
    whisper_temperature: float,
    lenient_vocal_activity_threshold: float,
    lenient_activity_bonus: float,
    low_word_confidence_threshold: float,
    offline: bool,
    quality_report: dict,
    hooks: Optional["LyricsWhisperHooks"],
    runtime_config: Optional["LyricsRuntimeConfig"],
) -> Tuple[List[Line], Optional[SongMetadata], dict]:
    from .lyrics_whisper import get_lyrics_simple

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
        hooks=hooks,
        runtime_config=runtime_config,
    )
    quality_report["alignment_method"] = "whisper_only"
    quality_report["whisper_used"] = bool(lines)
    quality_report["total_lines"] = len(lines)
    quality_report["overall_score"] = 50.0 if lines else 0.0
    if not lines:
        quality_report["issues"].append("Whisper-only mode produced no lines")
    return lines, metadata, quality_report


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
    audio_start: float = 0.0,
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
    hooks: Optional["LyricsWhisperHooks"] = None,
    runtime_config: Optional["LyricsRuntimeConfig"] = None,
) -> Tuple[List[Line], Optional[SongMetadata], dict]:
    from .lyrics_whisper import (
        _apply_singer_info,
        _calculate_quality_score,
    )
    from .sync import get_lyrics_quality_report

    quality_report = _build_quality_report(
        use_whisper=use_whisper,
        whisper_only=whisper_only,
        whisper_map_lrc=whisper_map_lrc,
        whisper_force_dtw=whisper_force_dtw,
    )
    issues_list: List[str] = quality_report["issues"]  # type: ignore[assignment]

    if whisper_only:
        return _get_whisper_only_quality_result(
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
            whisper_map_lrc=whisper_map_lrc,
            whisper_map_lrc_dtw=whisper_map_lrc_dtw,
            lyrics_file=lyrics_file,
            audio_start=audio_start,
            whisper_language=whisper_language,
            whisper_model=whisper_model,
            whisper_force_dtw=whisper_force_dtw,
            whisper_aggressive=whisper_aggressive,
            whisper_temperature=whisper_temperature,
            lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
            lenient_activity_bonus=lenient_activity_bonus,
            low_word_confidence_threshold=low_word_confidence_threshold,
            offline=offline,
            quality_report=quality_report,
            hooks=hooks,
            runtime_config=runtime_config,
        )

    if whisper_map_lrc:
        quality_report["alignment_method"] = (
            "whisper_map_lrc_dtw" if whisper_map_lrc_dtw else "whisper_map_lrc"
        )

    lrc_text, line_timings, source, file_lines = _resolve_lrc_inputs(
        title=title,
        artist=artist,
        lyrics_file=lyrics_file,
        audio_start=audio_start,
        filter_promos=filter_promos,
        target_duration=target_duration,
        vocals_path=vocals_path,
        evaluate_sources=evaluate_sources,
        offline=offline,
        quality_report=quality_report,
        issues_list=issues_list,
        drop_lrc_line_timings=drop_lrc_line_timings,
        use_whisper=use_whisper,
        whisper_map_lrc=whisper_map_lrc,
        hooks=hooks,
        runtime_config=runtime_config,
    )

    genius_lines, metadata, early_placeholder = _resolve_genius_lines_and_metadata(
        lrc_text=lrc_text,
        file_lines=file_lines,
        line_timings=line_timings,
        offline=offline,
        title=title,
        artist=artist,
        quality_report=quality_report,
    )
    if early_placeholder is not None:
        lines, meta = early_placeholder
        return lines, meta, quality_report

    if line_timings and lrc_text:
        quality_report["lyrics_quality"] = get_lyrics_quality_report(
            lrc_text, source, target_duration
        )

    line_timings = _detect_line_timing_offset(
        vocals_path=vocals_path,
        line_timings=line_timings,
        lyrics_offset=lyrics_offset,
        quality_report=quality_report,
        issues_list=issues_list,
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

    if lrc_text or file_lines:
        lines, has_lrc_timing, _pre_whisper_lines, alignment_method = (
            _build_lines_from_lyrics_source(
                lrc_text=lrc_text,
                file_lines=file_lines,
                line_timings=line_timings,
                title=title,
                artist=artist,
                romanize=romanize,
                filter_promos=filter_promos,
                vocals_path=vocals_path,
                whisper_map_lrc=whisper_map_lrc,
                target_duration=target_duration,
                issues_list=issues_list,
            )
        )
        if alignment_method:
            quality_report["alignment_method"] = alignment_method
        lines = _apply_audio_alignment(
            lines=lines,
            vocals_path=vocals_path,
            use_whisper=use_whisper,
            whisper_map_lrc=whisper_map_lrc,
            whisper_map_lrc_dtw=whisper_map_lrc_dtw,
            has_lrc_timing=has_lrc_timing,
            line_timings=line_timings,
            quality_report=quality_report,
            issues_list=issues_list,
            whisper_language=whisper_language,
            whisper_model=whisper_model,
            whisper_force_dtw=whisper_force_dtw,
            whisper_aggressive=whisper_aggressive,
            whisper_temperature=whisper_temperature,
            lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
            lenient_activity_bonus=lenient_activity_bonus,
            low_word_confidence_threshold=low_word_confidence_threshold,
            target_duration=target_duration,
            hooks=hooks,
        )
    else:
        lines = _build_fallback_lines(
            genius_lines=genius_lines,
            title=title,
            artist=artist,
            romanize=romanize,
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
