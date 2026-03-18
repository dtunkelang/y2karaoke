"""Tail guardrail and duration clipping helpers for lyrics quality flows."""

from typing import Any, List, Optional

from ...models import Line, Word

TailGuardrailSnapshot = dict[str, Any]


def _line_set_end(lines: List[Line]) -> float:
    return max(
        (w.end_time for line in lines for w in line.words),
        default=0.0,
    )


def _tail_guardrail_snapshot(
    lines: List[Line],
    *,
    target_duration: Optional[int],
    metrics: Optional[dict],
) -> TailGuardrailSnapshot:
    line_end = _line_set_end(lines)
    snapshot: TailGuardrailSnapshot = {
        "line_end_sec": float(line_end),
        "target_coverage_ratio": None,
        "target_shortfall_sec": None,
        "whisper_timeline_ratio": None,
        "flagged": False,
        "reasons": [],
    }
    reasons: List[str] = []

    if isinstance(target_duration, (int, float)) and float(target_duration) > 0.0:
        duration = float(target_duration)
        shortfall = max(0.0, duration - line_end)
        coverage = line_end / duration if duration > 0.0 else 0.0
        snapshot["target_coverage_ratio"] = coverage
        snapshot["target_shortfall_sec"] = shortfall
        shortfall_threshold = max(14.0, duration * 0.08)
        if shortfall > shortfall_threshold:
            reasons.append(
                f"shortfall {shortfall:.1f}s exceeds tail threshold {shortfall_threshold:.1f}s"
            )

    if isinstance(metrics, dict):
        ratio = metrics.get("aligned_timeline_ratio")
        if isinstance(ratio, (int, float)):
            snapshot["whisper_timeline_ratio"] = float(ratio)
            if float(ratio) < 0.88:
                reasons.append(
                    f"aligned timeline ratio {float(ratio):.3f} below guardrail 0.880"
                )

    snapshot["flagged"] = bool(reasons)
    snapshot["reasons"] = reasons
    return snapshot


def _tail_guardrail_should_accept_retry(
    *,
    baseline_guard: dict,
    retry_guard: dict,
    baseline_metrics: Optional[dict],
    retry_metrics: Optional[dict],
) -> bool:
    base_cov = (
        float(baseline_guard["target_coverage_ratio"])
        if isinstance(baseline_guard.get("target_coverage_ratio"), (int, float))
        else 0.0
    )
    retry_cov = (
        float(retry_guard["target_coverage_ratio"])
        if isinstance(retry_guard.get("target_coverage_ratio"), (int, float))
        else 0.0
    )
    if retry_cov < base_cov + 0.04:
        return False

    base_match = (
        float((baseline_metrics or {}).get("matched_ratio", 0.0) or 0.0)
        if isinstance((baseline_metrics or {}).get("matched_ratio"), (int, float))
        else 0.0
    )
    retry_match = (
        float((retry_metrics or {}).get("matched_ratio", 0.0) or 0.0)
        if isinstance((retry_metrics or {}).get("matched_ratio"), (int, float))
        else 0.0
    )
    if retry_match < base_match - 0.08:
        return False

    base_line_cov = (
        float((baseline_metrics or {}).get("line_coverage", 0.0) or 0.0)
        if isinstance((baseline_metrics or {}).get("line_coverage"), (int, float))
        else 0.0
    )
    retry_line_cov = (
        float((retry_metrics or {}).get("line_coverage", 0.0) or 0.0)
        if isinstance((retry_metrics or {}).get("line_coverage"), (int, float))
        else 0.0
    )
    if retry_line_cov < base_line_cov - 0.08:
        return False
    return True


def _apply_tail_guardrail_metrics(
    metrics: Optional[dict],
    guard: TailGuardrailSnapshot,
    *,
    fallback_attempted: bool = False,
    fallback_applied: bool = False,
) -> dict:
    out = dict(metrics or {})
    out["tail_guardrail_flagged"] = 1.0 if bool(guard.get("flagged")) else 0.0
    out["tail_guardrail_fallback_attempted"] = 1.0 if fallback_attempted else 0.0
    out["tail_guardrail_fallback_applied"] = 1.0 if fallback_applied else 0.0
    if isinstance(guard.get("target_coverage_ratio"), (int, float)):
        out["tail_guardrail_target_coverage_ratio"] = float(
            guard["target_coverage_ratio"]
        )
    if isinstance(guard.get("target_shortfall_sec"), (int, float)):
        out["tail_guardrail_target_shortfall_sec"] = float(
            guard["target_shortfall_sec"]
        )
    if isinstance(guard.get("whisper_timeline_ratio"), (int, float)):
        out["tail_guardrail_whisper_timeline_ratio"] = float(
            guard["whisper_timeline_ratio"]
        )
    return out


def _maybe_retry_tail_guardrail(
    *,
    align_fn: Any,
    source_lines: List[Line],
    vocals_path: str,
    whisper_language: Optional[str],
    whisper_model: Optional[str],
    whisper_force_dtw: bool,
    whisper_aggressive: bool,
    whisper_temperature: float,
    lenient_vocal_activity_threshold: float,
    lenient_activity_bonus: float,
    low_word_confidence_threshold: float,
    target_duration: Optional[int],
    baseline_guard: TailGuardrailSnapshot,
    baseline_metrics: dict,
) -> tuple[List[Line], List[str], dict, str]:
    retry_lines, retry_fixes, retry_metrics = align_fn(
        source_lines,
        vocals_path,
        whisper_language,
        whisper_model,
        whisper_force_dtw,
        whisper_aggressive,
        whisper_temperature=whisper_temperature,
        prefer_whisper_timing_map=True,
        lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
        lenient_activity_bonus=lenient_activity_bonus,
        low_word_confidence_threshold=low_word_confidence_threshold,
    )
    retry_guard = _tail_guardrail_snapshot(
        retry_lines,
        target_duration=target_duration,
        metrics=retry_metrics,
    )
    if not _tail_guardrail_should_accept_retry(
        baseline_guard=baseline_guard,
        retry_guard=retry_guard,
        baseline_metrics=baseline_metrics,
        retry_metrics=retry_metrics,
    ):
        return (
            source_lines,
            [],
            baseline_metrics,
            ("Tail completeness guardrail retry rejected (insufficient gain)"),
        )
    updated_metrics = _apply_tail_guardrail_metrics(
        retry_metrics,
        retry_guard,
        fallback_attempted=True,
        fallback_applied=True,
    )
    return (
        retry_lines,
        retry_fixes,
        updated_metrics,
        "Tail completeness guardrail applied fallback timing-map retry",
    )


def _clip_lines_to_target_duration(
    lines: List[Line],
    target_duration: Optional[int],
    issues: List[str],
    grace_seconds: float = 0.4,
) -> List[Line]:
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

        new_words, clipped_word_count = _clip_line_words_to_max_time(line, max_time)
        trimmed_words += clipped_word_count

        if not new_words:
            dropped_lines += 1
            continue
        clipped_lines.append(Line(words=new_words, singer=line.singer))

    _append_clip_duration_issues(
        issues,
        target_duration=target_duration,
        dropped_lines=dropped_lines,
        trimmed_words=trimmed_words,
    )
    return clipped_lines


def _clip_line_words_to_max_time(line: Line, max_time: float) -> tuple[List[Word], int]:
    new_words: List[Word] = []
    trimmed_words = 0
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
    return new_words, trimmed_words


def _append_clip_duration_issues(
    issues: List[str],
    *,
    target_duration: Optional[int],
    dropped_lines: int,
    trimmed_words: int,
) -> None:
    if dropped_lines:
        issues.append(
            f"Dropped {dropped_lines} line(s) beyond track duration ({target_duration}s)"
        )
    if trimmed_words:
        issues.append(
            f"Trimmed {trimmed_words} word timing(s) past track duration ({target_duration}s)"
        )
