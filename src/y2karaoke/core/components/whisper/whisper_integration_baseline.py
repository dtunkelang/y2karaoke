"""Baseline/rollback helpers for Whisper integration."""

from __future__ import annotations

from typing import List, Tuple

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
    return duration < 0.5 and (duration / max(word_count, 1)) < 0.14


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
        if _is_implausibly_short_multiword_line(
            aligned
        ) and not _is_implausibly_short_multiword_line(original):
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


def _constrain_line_starts_to_baseline(
    mapped_lines: List[models.Line],
    baseline_lines: List[models.Line],
    *,
    min_gap: float = 0.01,
    max_shift_sec: float = 2.5,
) -> List[models.Line]:
    """Force mapped line starts to baseline (LRC) starts while preserving within-line shape."""
    constrained: List[models.Line] = []
    prev_output_start: float | None = None
    for idx, line in enumerate(mapped_lines):
        if idx >= len(baseline_lines) or not line.words:
            constrained.append(line)
            if line.words:
                prev_output_start = line.start_time
            continue

        baseline = baseline_lines[idx]
        if not baseline.words:
            constrained.append(line)
            if line.words:
                prev_output_start = line.start_time
            continue

        target_start = baseline.start_time
        shift = target_start - line.start_time
        if abs(shift) > max_shift_sec:
            constrained.append(line)
            prev_output_start = line.start_time
            continue
        if prev_output_start is not None and target_start < (
            prev_output_start + min_gap
        ):
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

        next_baseline_start = None
        for nxt in baseline_lines[idx + 1 :]:
            if nxt.words:
                next_baseline_start = nxt.start_time
                break

        if next_baseline_start is not None and shifted_line.end_time > (
            next_baseline_start - min_gap
        ):
            available = max(0.1, (next_baseline_start - min_gap) - target_start)
            current = max(0.1, shifted_line.end_time - target_start)
            scale = min(1.0, available / current)
            compressed_words = []
            for w in shifted_line.words:
                ws = target_start + (w.start_time - target_start) * scale
                we = target_start + (w.end_time - target_start) * scale
                if we < ws:
                    we = ws
                compressed_words.append(
                    models.Word(
                        text=w.text,
                        start_time=ws,
                        end_time=we,
                        singer=w.singer,
                    )
                )
            shifted_line = models.Line(words=compressed_words, singer=line.singer)

        constrained.append(shifted_line)
        prev_output_start = shifted_line.start_time

    return constrained
