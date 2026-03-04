"""Hybrid LRC/Whisper alignment helpers."""

from __future__ import annotations

from typing import Callable, List, Optional, Set, Tuple

from ...models import Line
from ..alignment.timing_models import TranscriptionSegment, TranscriptionWord


def calculate_drift_correction(
    recent_offsets: List[float], trust_threshold: float
) -> Optional[float]:
    if len(recent_offsets) < 2:
        return None
    recent_nonzero = [o for o in recent_offsets[-5:] if abs(o) > 0.5]
    if len(recent_nonzero) >= 2:
        avg_drift = sum(recent_nonzero) / len(recent_nonzero)
        if abs(avg_drift) > trust_threshold:
            return avg_drift
    return None


def interpolate_unmatched_lines(
    mapped_lines: List[Line],
    matched_lines: Set[int],
    *,
    line_duration_fn: Callable[[Line], float],
    shift_line_fn: Callable[[Line, float], Line],
) -> List[Line]:
    total = len(mapped_lines)
    prev_end = None
    idx = 0
    while idx < total:
        if idx in matched_lines:
            prev_end = mapped_lines[idx].end_time
            idx += 1
            continue
        run_start = idx
        while idx < total and idx not in matched_lines:
            idx += 1
        run_end = idx
        run_indices = list(range(run_start, run_end))
        if not run_indices:
            continue
        if run_end == total:
            tail_start = mapped_lines[run_start].start_time
            if prev_end is not None and tail_start < prev_end + 0.01:
                shift = (prev_end + 0.01) - tail_start
                for line_idx in run_indices:
                    line = mapped_lines[line_idx]
                    mapped_lines[line_idx] = shift_line_fn(line, shift)
                prev_end = mapped_lines[run_indices[-1]].end_time
            elif mapped_lines[run_indices[-1]].words:
                prev_end = mapped_lines[run_indices[-1]].end_time
            continue
        start_anchor = (
            prev_end if prev_end is not None else mapped_lines[run_start].start_time
        )
        next_anchor = (
            mapped_lines[run_end].start_time
            if run_end < total
            else start_anchor
            + sum(line_duration_fn(mapped_lines[i]) for i in run_indices)
            + 0.5
        )
        total_duration = sum(line_duration_fn(mapped_lines[i]) for i in run_indices)
        total_duration = max(total_duration, 0.5 * len(run_indices))
        available_gap = max(next_anchor - start_anchor, total_duration)
        duration_scale = available_gap / total_duration if total_duration > 0 else 1.0
        if len(run_indices) >= 3 and duration_scale > 1.2:
            duration_scale = 1.2
        elif duration_scale > 2.0:
            duration_scale = 2.0
        current = start_anchor
        for line_idx in run_indices:
            line = mapped_lines[line_idx]
            duration = line_duration_fn(line) * duration_scale
            line_shift = current - line.start_time
            mapped_lines[line_idx] = shift_line_fn(line, line_shift)
            current += duration + 0.01
        prev_end = current
    return mapped_lines


def refine_unmatched_lines_with_onsets(
    mapped_lines: List[Line],
    matched_lines: Set[int],
    vocals_path: str,
    *,
    refine_word_timing_fn: Callable[[List[Line], str], List[Line]],
    logger=None,
) -> List[Line]:
    unmatched = [
        i
        for i in range(len(mapped_lines))
        if i not in matched_lines and mapped_lines[i].words
    ]
    if not unmatched:
        return mapped_lines

    subset = [mapped_lines[i] for i in unmatched]
    try:
        refined = refine_word_timing_fn(subset, vocals_path)
    except Exception:
        return mapped_lines

    for idx, line_idx in enumerate(unmatched):
        mapped_lines[line_idx] = refined[idx]
    if logger is not None:
        logger.debug("Onset-refined %d unmatched line(s)", len(unmatched))
    return mapped_lines


def align_hybrid_lrc_whisper(
    lines: List[Line],
    segments: List[TranscriptionSegment],
    words: List[TranscriptionWord],
    *,
    language: str,
    trust_threshold: float,
    correct_threshold: float,
    min_similarity: float,
    get_ipa_fn: Callable[[str, str], str | None],
    find_best_whisper_segment_fn: Callable[
        [str, float, List[TranscriptionSegment], str, float],
        Tuple[Optional[TranscriptionSegment], float, float],
    ],
    apply_offset_to_line_fn: Callable[[Line, float], Line],
    calculate_drift_correction_fn: Callable[[List[float], float], Optional[float]],
) -> Tuple[List[Line], List[str]]:
    if not lines or not segments:
        return lines, []

    for w in words:
        get_ipa_fn(w.text, language)

    aligned_lines: List[Line] = []
    corrections: List[str] = []
    recent_offsets: List[float] = []
    sorted_segments = sorted(segments, key=lambda s: s.start)

    for line_idx, line in enumerate(lines):
        if not line.words:
            aligned_lines.append(line)
            continue

        line_text = " ".join(w.text for w in line.words)
        line_start = line.start_time

        best_segment, best_similarity, best_offset = find_best_whisper_segment_fn(
            line_text, line_start, sorted_segments, language, min_similarity
        )
        timing_error = abs(best_offset) if best_segment else float("inf")

        if best_segment and timing_error < trust_threshold:
            aligned_lines.append(line)
            recent_offsets.append(0.0)
            continue

        if (
            best_segment
            and timing_error >= correct_threshold
            and best_similarity >= 0.5
        ):
            aligned_lines.append(apply_offset_to_line_fn(line, best_offset))
            corrections.append(
                f'Line {line_idx} shifted {best_offset:+.1f}s (similarity: {best_similarity:.0%}): "{line_text[:35]}..."'
            )
            recent_offsets.append(best_offset)
            continue

        if timing_error >= trust_threshold and timing_error < correct_threshold:
            if len(recent_offsets) >= 2 and all(
                abs(o) > 0.5 for o in recent_offsets[-2:]
            ):
                avg_drift = sum(recent_offsets[-3:]) / len(recent_offsets[-3:])
                if abs(avg_drift) > trust_threshold:
                    aligned_lines.append(apply_offset_to_line_fn(line, avg_drift))
                    corrections.append(
                        f'Line {line_idx} drift-corrected {avg_drift:+.1f}s: "{line_text[:35]}..."'
                    )
                    recent_offsets.append(avg_drift)
                    continue

            aligned_lines.append(line)
            recent_offsets.append(0.0)
            continue

        drift = calculate_drift_correction_fn(recent_offsets, trust_threshold)
        if drift is not None:
            aligned_lines.append(apply_offset_to_line_fn(line, drift))
            corrections.append(
                f'Line {line_idx} drift-corrected {drift:+.1f}s (no match): "{line_text[:35]}..."'
            )
            recent_offsets.append(drift)
        else:
            aligned_lines.append(line)
            recent_offsets.append(0.0)

    return aligned_lines, corrections


def fix_ordering_violations(
    original_lines: List[Line],
    aligned_lines: List[Line],
    alignments: List[str],
    *,
    logger=None,
) -> Tuple[List[Line], List[str]]:
    if not aligned_lines:
        return aligned_lines, alignments

    fixed_lines: List[Line] = []
    prev_end_time = 0.0
    prev_start_time = 0.0
    reverted_count = 0

    for orig, aligned in zip(original_lines, aligned_lines):
        if not aligned.words:
            fixed_lines.append(aligned)
            continue

        aligned_start = aligned.start_time
        if (
            aligned_start < prev_start_time - 0.01
            or aligned_start < prev_end_time - 0.1
        ):
            fixed_lines.append(orig)
            if orig.words:
                prev_end_time = orig.end_time
                prev_start_time = orig.start_time
            reverted_count += 1
        else:
            fixed_lines.append(aligned)
            prev_end_time = aligned.end_time
            prev_start_time = aligned.start_time

    if reverted_count > 0:
        if logger is not None:
            logger.debug(
                "Reverted %d Whisper alignments due to ordering violations",
                reverted_count,
            )
        actual_corrections = len(alignments) - reverted_count
        fixed_alignments = (
            alignments[:actual_corrections] if actual_corrections > 0 else []
        )
        return fixed_lines, fixed_alignments

    return fixed_lines, alignments
