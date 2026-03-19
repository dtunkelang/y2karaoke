"""Alignment helpers for lyrics quality workflows."""

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from ...models import Line
from .lyrics_quality_tail_guardrail import (
    _apply_tail_guardrail_metrics,
    _maybe_retry_tail_guardrail,
    _tail_guardrail_should_accept_retry,
    _tail_guardrail_snapshot,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .lyrics_whisper import LyricsWhisperHooks


def _serialize_line_timing_snapshot(lines: List[Line]) -> list[dict[str, Any]]:
    snapshot: list[dict[str, Any]] = []
    for idx, line in enumerate(lines, start=1):
        if not line.words:
            continue
        snapshot.append(
            {
                "index": idx,
                "text": line.text,
                "start": round(line.start_time, 3),
                "end": round(line.end_time, 3),
            }
        )
    return snapshot


def _build_lines_from_lyrics_source(
    *,
    lrc_text: Optional[str],
    file_lines: List[str],
    line_timings: Optional[List[Tuple[float, str]]],
    title: str,
    artist: str,
    romanize: bool,
    filter_promos: bool,
    vocals_path: Optional[str],
    whisper_map_lrc: bool,
    target_duration: Optional[int],
    issues_list: List[str],
) -> Tuple[List[Line], bool, Optional[List[dict[str, Any]]], Optional[str]]:
    from .helpers import (
        _apply_timing_to_lines,
        _anchor_plain_text_lines_to_audio_window,
        _create_lines_from_plain_text,
        _extract_text_lines_from_lrc,
    )
    from .lyrics_whisper import (
        _refine_timing_with_quality,
        create_lines_from_lrc,
        create_lines_from_lrc_timings,
    )

    pre_whisper_lines: Optional[List[dict[str, Any]]] = None
    alignment_method: Optional[str] = None
    has_lrc_timing = bool(line_timings)

    if line_timings and file_lines:
        lines = create_lines_from_lrc_timings(line_timings, file_lines)
        alignment_method = "lrc_only"
    elif line_timings and lrc_text:
        lines = create_lines_from_lrc(
            lrc_text,
            romanize=False,
            title=title,
            artist=artist,
            filter_promos=filter_promos,
        )
        alignment_method = "lrc_only"
        _apply_timing_to_lines(lines, line_timings)
    else:
        text_lines = file_lines or _extract_text_lines_from_lrc(lrc_text or "")
        lines = _create_lines_from_plain_text(text_lines)
        if target_duration:
            lines = _anchor_plain_text_lines_to_audio_window(
                lines,
                target_duration,
                vocals_path,
            )

    if vocals_path and line_timings and len(line_timings) > 1 and not whisper_map_lrc:
        lines, alignment_method = _refine_timing_with_quality(
            lines,
            vocals_path,
            line_timings,
            lrc_text or "",
            target_duration,
            issues_list,
        )

    return lines, has_lrc_timing, pre_whisper_lines, alignment_method


def _build_fallback_lines(
    *,
    genius_lines: Optional[List[Tuple[str, str]]],
    title: str,
    artist: str,
    romanize: bool,
    filter_promos: bool,
) -> List[Line]:
    from .lyrics_whisper import create_lines_from_lrc

    if genius_lines:
        text_lines = [text for text, _ in genius_lines if text.strip()]
        fallback_text = "\n".join(text_lines)
    else:
        fallback_text = ""
    return create_lines_from_lrc(
        fallback_text,
        romanize=romanize,
        title=title,
        artist=artist,
        filter_promos=filter_promos,
    )


def _apply_whisper_map_lrc_dtw(
    *,
    lines: List[Line],
    vocals_path: str,
    whisper_language: Optional[str],
    whisper_model: Optional[str],
    whisper_aggressive: bool,
    whisper_temperature: float,
    lenient_vocal_activity_threshold: float,
    lenient_activity_bonus: float,
    low_word_confidence_threshold: float,
    target_duration: Optional[int],
    issues_list: List[str],
    quality_report: dict,
) -> List[Line]:
    from ..whisper.whisper_integration import align_lrc_text_to_whisper_timings

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
    guard = _tail_guardrail_snapshot(
        lines,
        target_duration=target_duration,
        metrics=metrics,
    )
    metrics = _apply_tail_guardrail_metrics(metrics, guard)
    if guard["flagged"]:
        issues_list.append(
            "Tail completeness guardrail flagged possible truncated lyric ending"
        )
    if guard["flagged"] and not whisper_aggressive:
        metrics["tail_guardrail_fallback_attempted"] = 1.0
        try:
            retry_lines, retry_alignments, retry_metrics = (
                align_lrc_text_to_whisper_timings(
                    lines,
                    vocals_path,
                    language=whisper_language,
                    model_size=model_size,
                    aggressive=True,
                    temperature=whisper_temperature,
                    lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
                    lenient_activity_bonus=lenient_activity_bonus,
                    low_word_confidence_threshold=low_word_confidence_threshold,
                )
            )
            retry_guard = _tail_guardrail_snapshot(
                retry_lines,
                target_duration=target_duration,
                metrics=retry_metrics,
            )
            if _tail_guardrail_should_accept_retry(
                baseline_guard=guard,
                retry_guard=retry_guard,
                baseline_metrics=metrics,
                retry_metrics=retry_metrics,
            ):
                lines = retry_lines
                alignments = retry_alignments
                metrics = _apply_tail_guardrail_metrics(
                    retry_metrics,
                    retry_guard,
                    fallback_attempted=True,
                    fallback_applied=True,
                )
                issues_list.append(
                    "Tail completeness guardrail applied aggressive DTW retry"
                )
            else:
                issues_list.append(
                    "Tail completeness guardrail retry rejected (insufficient gain)"
                )
        except Exception as retry_err:
            issues_list.append(f"Tail completeness guardrail retry failed: {retry_err}")

    quality_report["alignment_method"] = "whisper_map_lrc_dtw"
    quality_report["whisper_used"] = True
    quality_report["whisper_corrections"] = len(alignments)
    if metrics:
        quality_report["dtw_metrics"] = metrics
    return lines


def _apply_whisper_map_lrc_transcription(
    *,
    lines: List[Line],
    vocals_path: str,
    whisper_language: Optional[str],
    whisper_model: Optional[str],
    whisper_aggressive: bool,
    whisper_temperature: float,
    line_timings: Optional[List[Tuple[float, str]]],
    whisper_map_lrc: bool,
    quality_report: dict,
    issues_list: List[str],
) -> List[Line]:
    from ...phonetic_utils import _whisper_lang_to_epitran
    from .lyrics_whisper_map import _map_lrc_lines_to_whisper_segments
    from ..whisper.whisper_integration import transcribe_vocals

    model_size = whisper_model or "large"
    transcription, _, detected_lang, _model = transcribe_vocals(
        vocals_path,
        whisper_language,
        model_size,
        whisper_aggressive,
        whisper_temperature,
    )
    if not transcription:
        return lines

    lang = _whisper_lang_to_epitran(detected_lang)
    lrc_starts = (
        [ts for ts, _ in line_timings] if line_timings and not whisper_map_lrc else None
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
    return lines


def _apply_audio_alignment(
    *,
    lines: List[Line],
    vocals_path: Optional[str],
    use_whisper: bool,
    whisper_map_lrc: bool,
    whisper_map_lrc_dtw: bool,
    has_lrc_timing: bool,
    line_timings: Optional[List[Tuple[float, str]]],
    quality_report: dict,
    issues_list: List[str],
    whisper_language: Optional[str],
    whisper_model: Optional[str],
    whisper_force_dtw: bool,
    whisper_aggressive: bool,
    whisper_temperature: float,
    lenient_vocal_activity_threshold: float,
    lenient_activity_bonus: float,
    low_word_confidence_threshold: float,
    target_duration: Optional[int],
    hooks: Optional["LyricsWhisperHooks"],
) -> List[Line]:
    if not vocals_path:
        return lines

    if use_whisper or whisper_map_lrc:
        pre_whisper_lines = _serialize_line_timing_snapshot(lines)
        quality_report["pre_whisper_lines"] = pre_whisper_lines
        quality_report["pre_whisper_line_count"] = len(pre_whisper_lines)

    if use_whisper:
        from . import lyrics_whisper as lw

        lines, quality_report = lw._apply_whisper_with_quality(
            lines,
            vocals_path,
            whisper_language,
            whisper_model,
            whisper_force_dtw,
            whisper_aggressive=whisper_aggressive,
            quality_report=quality_report,
            whisper_temperature=whisper_temperature,
            prefer_whisper_timing_map=not has_lrc_timing,
            lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
            lenient_activity_bonus=lenient_activity_bonus,
            low_word_confidence_threshold=low_word_confidence_threshold,
            target_duration=target_duration,
            hooks=hooks,
        )
        return lines

    if whisper_map_lrc:
        try:
            if whisper_map_lrc_dtw:
                return _apply_whisper_map_lrc_dtw(
                    lines=lines,
                    vocals_path=vocals_path,
                    whisper_language=whisper_language,
                    whisper_model=whisper_model,
                    whisper_aggressive=whisper_aggressive,
                    whisper_temperature=whisper_temperature,
                    lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
                    lenient_activity_bonus=lenient_activity_bonus,
                    low_word_confidence_threshold=low_word_confidence_threshold,
                    target_duration=target_duration,
                    issues_list=issues_list,
                    quality_report=quality_report,
                )
            return _apply_whisper_map_lrc_transcription(
                lines=lines,
                vocals_path=vocals_path,
                whisper_language=whisper_language,
                whisper_model=whisper_model,
                whisper_aggressive=whisper_aggressive,
                whisper_temperature=whisper_temperature,
                line_timings=line_timings,
                whisper_map_lrc=whisper_map_lrc,
                quality_report=quality_report,
                issues_list=issues_list,
            )
        except Exception as e:
            logger.warning(f"Whisper LRC mapping failed: {e}")
            issues_list.append(f"Whisper LRC mapping failed: {e}")
        return lines

    if not has_lrc_timing:
        from . import lyrics_whisper as lw

        lines, quality_report = lw._apply_whisper_with_quality(
            lines,
            vocals_path,
            whisper_language,
            whisper_model,
            whisper_force_dtw,
            whisper_aggressive=whisper_aggressive,
            quality_report=quality_report,
            whisper_temperature=whisper_temperature,
            prefer_whisper_timing_map=True,
            lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
            lenient_activity_bonus=lenient_activity_bonus,
            low_word_confidence_threshold=low_word_confidence_threshold,
            target_duration=target_duration,
            hooks=hooks,
        )
    return lines


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
    target_duration: Optional[int] = None,
    hooks: Optional["LyricsWhisperHooks"] = None,
) -> Tuple[List[Line], dict]:
    from . import lyrics_whisper as lw

    resolved_hooks = lw.resolve_lyrics_whisper_hooks(hooks)

    if quality_report is None:
        quality_report = {"issues": []}
    try:
        aligned_lines, whisper_fixes, whisper_metrics = (
            lw._apply_whisper_alignment_for_state(
                lines,
                vocals_path,
                whisper_language,
                whisper_model,
                whisper_force_dtw,
                whisper_aggressive,
                hooks=resolved_hooks,
                whisper_temperature=whisper_temperature,
                prefer_whisper_timing_map=prefer_whisper_timing_map,
                lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
                lenient_activity_bonus=lenient_activity_bonus,
                low_word_confidence_threshold=low_word_confidence_threshold,
            )
        )
        baseline_guard = _tail_guardrail_snapshot(
            aligned_lines,
            target_duration=target_duration,
            metrics=whisper_metrics,
        )
        whisper_metrics = _apply_tail_guardrail_metrics(whisper_metrics, baseline_guard)

        if baseline_guard["flagged"]:
            quality_report["issues"].append(
                "Tail completeness guardrail flagged possible truncated lyric ending"
            )
        should_retry_tail_guard = (
            baseline_guard["flagged"]
            and not prefer_whisper_timing_map
            and not bool(float(whisper_metrics.get("no_evidence_fallback", 0.0) or 0.0))
        )
        if should_retry_tail_guard:
            whisper_metrics["tail_guardrail_fallback_attempted"] = 1.0
            (
                retried_lines,
                retried_fixes,
                retried_metrics,
                retry_issue,
            ) = _maybe_retry_tail_guardrail(
                align_fn=lambda *a, **k: lw._apply_whisper_alignment_for_state(
                    *a, hooks=resolved_hooks, **k
                ),
                source_lines=lines,
                vocals_path=vocals_path,
                whisper_language=whisper_language,
                whisper_model=whisper_model,
                whisper_force_dtw=whisper_force_dtw,
                whisper_aggressive=whisper_aggressive,
                whisper_temperature=whisper_temperature,
                lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
                lenient_activity_bonus=lenient_activity_bonus,
                low_word_confidence_threshold=low_word_confidence_threshold,
                target_duration=target_duration,
                baseline_guard=baseline_guard,
                baseline_metrics=whisper_metrics,
            )
            if retried_fixes:
                aligned_lines = retried_lines
                whisper_fixes = retried_fixes
                whisper_metrics = retried_metrics
            quality_report["issues"].append(retry_issue)

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
        lines = aligned_lines
    except Exception as e:
        logger.warning(f"Whisper alignment failed: {e}")
        quality_report["issues"].append(f"Whisper alignment failed: {e}")
    return lines, quality_report
