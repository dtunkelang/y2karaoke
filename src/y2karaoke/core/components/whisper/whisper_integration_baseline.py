"""Baseline/rollback helpers for Whisper integration."""

from __future__ import annotations

import json
import os
from typing import Any, List, Tuple

from ... import models


def _clone_lines_for_fallback(lines: List[models.Line]) -> List[models.Line]:
    """Deep-copy lines so rollback logic can safely restore original timing."""
    return [
        models.Line(
            words=[
                models.Word(
                    text=w.text,
                    start_time=w.start_time,
                    end_time=w.end_time,
                    singer=w.singer,
                )
                for w in line.words
            ],
            singer=line.singer,
        )
        for line in lines
    ]


def _is_implausibly_short_multiword_line(line: models.Line) -> bool:
    word_count = len(line.words)
    if word_count < 3:
        return False
    duration = line.end_time - line.start_time
    if duration <= 0:
        return True
    per_word = duration / max(word_count, 1)
    if duration < 0.5 and per_word < 0.14:
        return True
    return word_count >= 6 and duration < 1.0 and per_word < 0.09


def _is_severely_collapsed_compact_line(
    original: models.Line,
    aligned: models.Line,
    *,
    max_words: int = 3,
    min_original_duration_sec: float = 0.9,
    max_duration_ratio: float = 0.45,
    min_duration_delta_sec: float = 0.5,
) -> bool:
    word_count = len(aligned.words)
    if word_count == 0 or word_count > max_words or len(original.words) != word_count:
        return False

    original_duration = original.end_time - original.start_time
    aligned_duration = aligned.end_time - aligned.start_time
    if original_duration < min_original_duration_sec or aligned_duration <= 0.0:
        return False
    if aligned_duration > original_duration * max_duration_ratio:
        return False
    return (original_duration - aligned_duration) >= min_duration_delta_sec


def _implausibly_short_multiword_count(lines: List[models.Line]) -> int:
    """Count suspiciously compressed multi-word lines."""
    count = 0
    for line in lines:
        if _is_implausibly_short_multiword_line(line):
            count += 1
    return count


def _restore_implausibly_short_lines(
    original_lines: List[models.Line],
    aligned_lines: List[models.Line],
) -> Tuple[List[models.Line], int]:
    """Restore baseline timing only for newly implausibly short aligned lines."""
    repaired: List[models.Line] = []
    restored = 0
    total = max(len(original_lines), len(aligned_lines))
    for idx in range(total):
        original = original_lines[idx] if idx < len(original_lines) else None
        aligned = aligned_lines[idx] if idx < len(aligned_lines) else None
        if original is None and aligned is not None:
            repaired.append(aligned)
            continue
        if aligned is None and original is not None:
            repaired.append(original)
            continue
        if original is None or aligned is None:
            continue
        original_duration = original.end_time - original.start_time
        aligned_duration = aligned.end_time - aligned.start_time
        severe_duration_collapse = (
            len(aligned.words) >= 3
            and original_duration >= 3.5
            and aligned_duration > 0.0
            and (
                (
                    len(aligned.words) >= 5
                    and aligned_duration <= original_duration * 0.6
                    and (original_duration - aligned_duration) >= 1.2
                )
                or (
                    len(aligned.words) < 5
                    and aligned_duration <= original_duration * 0.35
                    and (original_duration - aligned_duration) >= 1.5
                )
            )
        )
        if _is_implausibly_short_multiword_line(
            aligned
        ) and not _is_implausibly_short_multiword_line(original):
            repaired.append(original)
            restored += 1
            continue
        if severe_duration_collapse:
            repaired.append(original)
            restored += 1
            continue
        if _is_severely_collapsed_compact_line(original, aligned):
            repaired.append(original)
            restored += 1
            continue
        repaired.append(aligned)
    return repaired, restored


def _should_rollback_short_line_degradation(
    original_lines: List[models.Line],
    aligned_lines: List[models.Line],
) -> Tuple[bool, int, int]:
    """Detect when Whisper introduces widespread short-line compression artifacts."""
    before = _implausibly_short_multiword_count(original_lines)
    after = _implausibly_short_multiword_count(aligned_lines)
    added = after - before
    min_added = max(3, int(0.06 * max(len(aligned_lines), 1)))
    should_rollback = added >= min_added and after >= max(4, before * 2)
    return should_rollback, before, after


def _has_start_inversion(lines: List[models.Line], tolerance: float = 0.01) -> bool:
    prev_start: float | None = None
    for line in lines:
        if not line.words:
            continue
        if prev_start is not None and line.start_time < prev_start - tolerance:
            return True
        prev_start = line.start_time
    return False


def _copy_line(line: models.Line) -> models.Line:
    return _clone_lines_for_fallback([line])[0]


def _maybe_write_baseline_constraint_trace(rows: list[dict]) -> None:
    trace_path = os.environ.get("Y2K_TRACE_BASELINE_CONSTRAINT_JSON", "").strip()
    if not trace_path:
        return
    with open(trace_path, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows}, fh, indent=2)


def _find_next_baseline_start(
    baseline_lines: List[models.Line], start_idx: int
) -> float | None:
    for nxt in baseline_lines[start_idx:]:
        if nxt.words:
            return nxt.start_time
    return None


def _compress_line_to_fit(
    line: models.Line, target_start: float, next_start: float, min_gap: float
) -> models.Line:
    available = max(0.1, (next_start - min_gap) - target_start)
    current = max(0.1, line.end_time - target_start)
    scale = min(1.0, available / current)
    compressed_words = []
    for word in line.words:
        ws = target_start + (word.start_time - target_start) * scale
        we = target_start + (word.end_time - target_start) * scale
        if we < ws:
            we = ws
        compressed_words.append(
            models.Word(
                text=word.text,
                start_time=ws,
                end_time=we,
                singer=word.singer,
            )
        )
    return models.Line(words=compressed_words, singer=line.singer)


def _constrain_line_starts_to_baseline(
    mapped_lines: List[models.Line],
    baseline_lines: List[models.Line],
    *,
    min_gap: float = 0.01,
    max_shift_sec: float = 2.5,
) -> List[models.Line]:
    """Force mapped line starts to baseline while preserving within-line shape."""
    constrained: List[models.Line] = []
    trace_rows: list[dict[str, Any]] = []
    prev_output_start: float | None = None
    unstable_sequence = _has_start_inversion(mapped_lines)
    for idx, line in enumerate(mapped_lines):
        row: dict[str, Any] = {
            "line_index": idx + 1,
            "mapped_start": round(line.start_time, 3) if line.words else None,
            "mapped_end": round(line.end_time, 3) if line.words else None,
        }
        if idx >= len(baseline_lines) or not line.words:
            row["decision"] = "keep_no_baseline_or_words"
            trace_rows.append(row)
            constrained.append(line)
            if line.words:
                prev_output_start = line.start_time
            continue

        baseline = baseline_lines[idx]
        if not baseline.words:
            row["decision"] = "keep_baseline_empty"
            trace_rows.append(row)
            constrained.append(line)
            if line.words:
                prev_output_start = line.start_time
            continue

        target_start = baseline.start_time
        shift = target_start - line.start_time
        row["baseline_start"] = round(target_start, 3)
        row["shift"] = round(shift, 3)
        if abs(shift) > max_shift_sec:
            if unstable_sequence and (
                prev_output_start is None
                or target_start >= (prev_output_start + min_gap)
            ):
                row["decision"] = "copy_baseline_unstable_sequence"
                constrained.append(_copy_line(baseline))
                trace_rows.append(row)
                prev_output_start = target_start
                continue
            row["decision"] = "keep_shift_too_large"
            trace_rows.append(row)
            constrained.append(line)
            prev_output_start = line.start_time
            continue
        if prev_output_start is not None and target_start < (
            prev_output_start + min_gap
        ):
            row["decision"] = "keep_prev_output_guard"
            trace_rows.append(row)
            constrained.append(line)
            prev_output_start = line.start_time
            continue
        shifted_words = [
            models.Word(
                text=w.text,
                start_time=w.start_time + shift,
                end_time=w.end_time + shift,
                singer=w.singer,
            )
            for w in line.words
        ]
        shifted_line = models.Line(words=shifted_words, singer=line.singer)

        next_baseline_start = _find_next_baseline_start(baseline_lines, idx + 1)

        if next_baseline_start is not None and shifted_line.end_time > (
            next_baseline_start - min_gap
        ):
            row["compressed_to_next_baseline"] = round(next_baseline_start, 3)
            shifted_line = _compress_line_to_fit(
                shifted_line, target_start, next_baseline_start, min_gap
            )

        row["decision"] = "shift_to_baseline"
        row["new_start"] = round(shifted_line.start_time, 3)
        row["new_end"] = round(shifted_line.end_time, 3)
        trace_rows.append(row)
        constrained.append(shifted_line)
        prev_output_start = shifted_line.start_time

    _maybe_write_baseline_constraint_trace(trace_rows)
    return constrained
