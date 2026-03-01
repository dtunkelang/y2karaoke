"""High-level refinement and hybrid alignment logic."""

import os
import logging
from contextlib import contextmanager
from typing import List, Optional, Tuple, Set
import numpy as np

from ...models import Line, Word
from ..alignment.timing_models import (
    TranscriptionSegment,
    TranscriptionWord,
    AudioFeatures,
)
from ...phonetic_utils import _get_ipa
from ... import audio_analysis
from .whisper_alignment_base import (
    _apply_offset_to_line,
    _line_duration,
    _shift_line,
)
from . import whisper_alignment_activity as _alignment_activity_helpers
from . import whisper_alignment_short_lines as _short_line_helpers
from .whisper_alignment_segments import _find_best_whisper_segment

_check_vocal_activity_in_range = audio_analysis._check_vocal_activity_in_range
_check_for_silence_in_range = audio_analysis._check_for_silence_in_range

logger = logging.getLogger(__name__)


@contextmanager
def use_alignment_refinement_hooks(
    *,
    find_best_whisper_segment_fn=None,
    get_ipa_fn=None,
    check_vocal_activity_in_range_fn=None,
    check_for_silence_in_range_fn=None,
):
    """Temporarily override hybrid-alignment collaborators for tests."""
    global _find_best_whisper_segment, _get_ipa
    global _check_vocal_activity_in_range, _check_for_silence_in_range

    prev_find_best = _find_best_whisper_segment
    prev_get_ipa = _get_ipa
    prev_check_activity = _check_vocal_activity_in_range
    prev_check_silence = _check_for_silence_in_range

    if find_best_whisper_segment_fn is not None:
        _find_best_whisper_segment = find_best_whisper_segment_fn
    if get_ipa_fn is not None:
        _get_ipa = get_ipa_fn
    if check_vocal_activity_in_range_fn is not None:
        _check_vocal_activity_in_range = check_vocal_activity_in_range_fn
    if check_for_silence_in_range_fn is not None:
        _check_for_silence_in_range = check_for_silence_in_range_fn

    try:
        yield
    finally:
        _find_best_whisper_segment = prev_find_best
        _get_ipa = prev_get_ipa
        _check_vocal_activity_in_range = prev_check_activity
        _check_for_silence_in_range = prev_check_silence


def _calculate_drift_correction(
    recent_offsets: List[float], trust_threshold: float
) -> Optional[float]:
    """Calculate drift correction from recent offsets if consistent drift exists."""
    if len(recent_offsets) < 2:
        return None

    # Check recent lines for consistent drift
    recent_nonzero = [o for o in recent_offsets[-5:] if abs(o) > 0.5]
    if len(recent_nonzero) >= 2:
        avg_drift = sum(recent_nonzero) / len(recent_nonzero)
        if abs(avg_drift) > trust_threshold:
            return avg_drift
    return None


def _interpolate_unmatched_lines(
    mapped_lines: List[Line], matched_lines: Set[int]
) -> List[Line]:
    """Spread lines without matches between their nearest matched neighbors."""
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
            # Trailing unmatched tail: preserve existing timing if already
            # monotonic, rather than stretching to synthetic anchors.
            tail_start = mapped_lines[run_start].start_time
            if prev_end is not None and tail_start < prev_end + 0.01:
                shift = (prev_end + 0.01) - tail_start
                for line_idx in run_indices:
                    line = mapped_lines[line_idx]
                    mapped_lines[line_idx] = _shift_line(line, shift)
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
            + sum(_line_duration(mapped_lines[i]) for i in run_indices)
            + 0.5
        )
        total_duration = sum(_line_duration(mapped_lines[i]) for i in run_indices)
        total_duration = max(total_duration, 0.5 * len(run_indices))
        available_gap = max(next_anchor - start_anchor, total_duration)
        duration_scale = available_gap / total_duration if total_duration > 0 else 1.0
        # Don't spread unmatched runs too aggressively; this can explode
        # timings in repeated/refrain-heavy songs when one late anchor is wrong.
        if len(run_indices) >= 3 and duration_scale > 1.2:
            duration_scale = 1.2
        elif duration_scale > 2.0:
            duration_scale = 2.0
        current = start_anchor
        for line_idx in run_indices:
            line = mapped_lines[line_idx]
            duration = _line_duration(line) * duration_scale
            line_shift = current - line.start_time
            mapped_lines[line_idx] = _shift_line(line, line_shift)
            current += duration + 0.01
        prev_end = current
    return mapped_lines


def _refine_unmatched_lines_with_onsets(
    mapped_lines: List[Line],
    matched_lines: Set[int],
    vocals_path: str,
) -> List[Line]:
    """Re-apply onset refinement to lines without Whisper word matches."""
    unmatched = [
        i
        for i in range(len(mapped_lines))
        if i not in matched_lines and mapped_lines[i].words
    ]
    if not unmatched:
        return mapped_lines

    from ...refine import refine_word_timing

    # Build a list of just the unmatched lines and refine them.
    subset = [mapped_lines[i] for i in unmatched]
    try:
        refined = refine_word_timing(subset, vocals_path)
    except Exception:
        return mapped_lines

    for idx, line_idx in enumerate(unmatched):
        mapped_lines[line_idx] = refined[idx]
    logger.debug(
        "Onset-refined %d unmatched line(s)",
        len(unmatched),
    )
    return mapped_lines


def align_hybrid_lrc_whisper(
    lines: List[Line],
    segments: List[TranscriptionSegment],
    words: List[TranscriptionWord],
    language: str = "fra-Latn",
    trust_threshold: float = 1.0,
    correct_threshold: float = 1.5,
    min_similarity: float = 0.4,
) -> Tuple[List[Line], List[str]]:
    """Hybrid alignment: preserve good LRC timing, use Whisper for broken sections."""
    if not lines or not segments:
        return lines, []

    logger.debug(f"Pre-computing IPA for {len(words)} Whisper words...")
    for w in words:
        _get_ipa(w.text, language)

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

        best_segment, best_similarity, best_offset = _find_best_whisper_segment(
            line_text, line_start, sorted_segments, language, min_similarity
        )

        timing_error = abs(best_offset) if best_segment else float("inf")

        # Case 1: LRC timing is good - keep it
        if best_segment and timing_error < trust_threshold:
            aligned_lines.append(line)
            recent_offsets.append(0.0)
            continue

        # Case 2: LRC timing is significantly off - use Whisper
        if (
            best_segment
            and timing_error >= correct_threshold
            and best_similarity >= 0.5
        ):
            aligned_lines.append(_apply_offset_to_line(line, best_offset))
            corrections.append(
                f'Line {line_idx} shifted {best_offset:+.1f}s (similarity: {best_similarity:.0%}): "{line_text[:35]}..."'
            )
            recent_offsets.append(best_offset)
            continue

        # Case 3: Intermediate timing error - check for consistent drift
        if timing_error >= trust_threshold and timing_error < correct_threshold:
            if len(recent_offsets) >= 2 and all(
                abs(o) > 0.5 for o in recent_offsets[-2:]
            ):
                avg_drift = sum(recent_offsets[-3:]) / len(recent_offsets[-3:])
                if abs(avg_drift) > trust_threshold:
                    aligned_lines.append(_apply_offset_to_line(line, avg_drift))
                    corrections.append(
                        f'Line {line_idx} drift-corrected {avg_drift:+.1f}s: "{line_text[:35]}..."'
                    )
                    recent_offsets.append(avg_drift)
                    continue

            aligned_lines.append(line)
            recent_offsets.append(0.0)
            continue

        # Case 4: No good match - apply drift correction if available
        drift = _calculate_drift_correction(recent_offsets, trust_threshold)
        if drift is not None:
            aligned_lines.append(_apply_offset_to_line(line, drift))
            corrections.append(
                f'Line {line_idx} drift-corrected {drift:+.1f}s (no match): "{line_text[:35]}..."'
            )
            recent_offsets.append(drift)
        else:
            aligned_lines.append(line)
            recent_offsets.append(0.0)

    return aligned_lines, corrections


def _fix_ordering_violations(
    original_lines: List[Line],
    aligned_lines: List[Line],
    alignments: List[str],
) -> Tuple[List[Line], List[str]]:
    """Fix lines that were moved out of order by Whisper alignment."""
    if not aligned_lines:
        return aligned_lines, alignments

    fixed_lines: List[Line] = []
    prev_end_time = 0.0
    prev_start_time = 0.0
    reverted_count = 0

    for i, (orig, aligned) in enumerate(zip(original_lines, aligned_lines)):
        if not aligned.words:
            fixed_lines.append(aligned)
            continue

        aligned_start = aligned.start_time

        # Check if this line would start before previous line (or end) with tolerance
        if (
            aligned_start < prev_start_time - 0.01
            or aligned_start < prev_end_time - 0.1
        ):
            # Revert to original timing
            fixed_lines.append(orig)
            if orig.words:
                prev_end_time = orig.end_time
                prev_start_time = orig.start_time
            reverted_count += 1
        else:
            # Keep the aligned timing
            fixed_lines.append(aligned)
            prev_end_time = aligned.end_time
            prev_start_time = aligned.start_time

    # Update alignments list (remove reverted ones)
    if reverted_count > 0:
        logger.debug(
            f"Reverted {reverted_count} Whisper alignments due to ordering violations"
        )
        actual_corrections = len(alignments) - reverted_count
        fixed_alignments = (
            alignments[:actual_corrections] if actual_corrections > 0 else []
        )
        return fixed_lines, fixed_alignments

    return fixed_lines, alignments


def _line_tokens(text: str) -> List[str]:
    tokens = []
    for raw in text.lower().split():
        token = "".join(ch for ch in raw if ch.isalpha())
        if token:
            tokens.append(token)
    return tokens


def _token_overlap(a: str, b: str) -> float:
    tokens_a = _line_tokens(a)
    tokens_b = _line_tokens(b)
    if not tokens_a or not tokens_b:
        return 0.0
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    return len(set_a & set_b) / max(len(set_a), len(set_b))


def _shift_line_words(line: Line, shift: float) -> Line:
    shifted_words = [
        Word(
            text=word.text,
            start_time=word.start_time + shift,
            end_time=word.end_time + shift,
            singer=word.singer,
        )
        for word in line.words
    ]
    return Line(words=shifted_words, singer=line.singer)


def _rebuild_line_with_target_end(line: Line, target_end: float) -> Optional[Line]:
    if not line.words:
        return None
    if target_end <= line.start_time + 0.2:
        return None
    spacing = (target_end - line.start_time) / len(line.words)
    compact_words = []
    for word_idx, word in enumerate(line.words):
        start = line.start_time + word_idx * spacing
        end = start + spacing * 0.9
        if word_idx == len(line.words) - 1:
            end = target_end
        compact_words.append(
            Word(
                text=word.text,
                start_time=start,
                end_time=end,
                singer=word.singer,
            )
        )
    return Line(words=compact_words, singer=line.singer)


def _compact_short_line_if_needed(
    line: Line,
    *,
    max_duration: float,
    next_start: Optional[float] = None,
) -> Line:
    if not line.words:
        return line
    duration = line.end_time - line.start_time
    if duration <= max_duration:
        return line
    target_end = line.start_time + max_duration
    if next_start is not None:
        target_end = min(target_end, next_start - 0.05)
    compacted = _rebuild_line_with_target_end(line, target_end)
    return compacted if compacted is not None else line


def _find_internal_silences(
    line_start: float,
    line_end: float,
    normalized_silences: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    return [
        (start, end)
        for start, end in normalized_silences
        if start >= line_start and end <= (line_end + 2.0)
    ]


def _has_near_start_silence(
    line_start: float, internal_silences: List[Tuple[float, float]]
) -> bool:
    return any(
        start <= line_start + 0.7 and end >= line_start - 0.1
        for start, end in internal_silences
    )


def _first_onset_after(onset_times, *, start: float, window: float) -> Optional[float]:
    candidate_onsets = onset_times[
        (onset_times >= start) & (onset_times <= start + window)
    ]
    if len(candidate_onsets) == 0:
        return None
    return float(candidate_onsets[0])


def _shift_lines_across_long_activity_gaps(
    lines: List[Line], audio_features: AudioFeatures, max_gap: float, onset_times
) -> int:
    fixes = 0
    for idx in range(1, len(lines)):
        prev_line = lines[idx - 1]
        line = lines[idx]
        if not prev_line.words or not line.words:
            continue
        gap = line.start_time - prev_line.end_time
        if gap <= max_gap:
            continue
        activity = _check_vocal_activity_in_range(
            prev_line.end_time, line.start_time, audio_features
        )
        has_silence = _check_for_silence_in_range(
            prev_line.end_time,
            line.start_time,
            audio_features,
            min_silence_duration=0.5,
        )
        if activity <= 0.6 or has_silence:
            continue
        candidate_onsets = onset_times[
            (onset_times >= prev_line.end_time) & (onset_times <= line.start_time)
        ]
        if len(candidate_onsets) == 0:
            continue
        new_start = max(float(candidate_onsets[0]), prev_line.end_time + 0.05)
        shift = new_start - line.start_time
        if shift > -0.3:
            continue
        lines[idx] = _shift_line_words(line, shift)
        fixes += 1
    return fixes


def _extend_line_ends_across_active_gaps(
    lines: List[Line],
    audio_features: AudioFeatures,
    *,
    min_gap: float = 1.25,
    min_extension: float = 0.25,
    max_extension: float = 2.5,
    activity_threshold: float = 0.65,
    silence_min_duration: float = 0.35,
) -> int:
    """Extend prior line ends when a gap has sustained vocal activity and no silence."""
    fixes = 0
    for idx in range(1, len(lines)):
        prev_line = lines[idx - 1]
        next_line = lines[idx]
        if not prev_line.words or not next_line.words:
            continue
        gap = next_line.start_time - prev_line.end_time
        if gap < min_gap:
            continue
        activity = _check_vocal_activity_in_range(
            prev_line.end_time, next_line.start_time, audio_features
        )
        if activity < activity_threshold:
            continue
        has_silence = _check_for_silence_in_range(
            prev_line.end_time,
            next_line.start_time,
            audio_features,
            min_silence_duration=silence_min_duration,
        )
        if has_silence:
            continue

        extension = min(gap - 0.05, max_extension)
        if extension < min_extension:
            continue
        target_end = prev_line.end_time + extension
        target_end = min(target_end, next_line.start_time - 0.05)
        stretched = _rebuild_line_with_target_end(prev_line, target_end)
        if stretched is None:
            continue
        if stretched.end_time <= prev_line.end_time + min_extension:
            continue
        lines[idx - 1] = stretched
        fixes += 1
    return fixes


def _is_uniform_word_timing(line: Line) -> bool:
    if not line.words or len(line.words) < 3:
        return False
    durations = np.array(
        [max(0.0, word.end_time - word.start_time) for word in line.words], dtype=float
    )
    mean_duration = float(np.mean(durations)) if len(durations) else 0.0
    if mean_duration <= 0.0:
        return False
    coeff_var = float(np.std(durations) / mean_duration)
    unique_rounded = len(set(round(float(d), 3) for d in durations))
    return coeff_var < 0.08 or unique_rounded <= 2


def _retime_line_words_to_onsets(
    line: Line,
    onset_times,
    *,
    min_word_duration: float = 0.08,
) -> Optional[Line]:
    if not line.words or len(line.words) < 3:
        return None
    line_start = line.start_time
    line_end = line.end_time
    if line_end <= line_start + min_word_duration * len(line.words):
        return None

    n_words = len(line.words)
    if n_words < 2:
        return None

    candidate_onsets = np.sort(
        onset_times[
            (onset_times >= line_start + 0.04) & (onset_times <= line_end - 0.01)
        ]
    )
    # Keep first word start anchored to the line start to avoid line-level shifts.
    target_word_count = n_words - 1
    if len(candidate_onsets) < target_word_count:
        return None

    original_duration = max(line_end - line_start, 1e-3)
    expected_starts = [
        line_start
        + max(0.0, min(1.0, (word.start_time - line_start) / original_duration))
        * original_duration
        for word in line.words[1:]
    ]

    chosen_indices = []
    prev_idx = -1
    for i in range(target_word_count):
        min_idx = prev_idx + 1
        max_idx = len(candidate_onsets) - (target_word_count - i)
        if max_idx < min_idx:
            return None
        target = expected_starts[i]
        best_idx = min(
            range(min_idx, max_idx + 1),
            key=lambda idx: abs(float(candidate_onsets[idx]) - target),
        )
        chosen_indices.append(best_idx)
        prev_idx = best_idx

    starts = [line_start] + [float(candidate_onsets[idx]) for idx in chosen_indices]
    adjusted_starts = [line_start]
    for start in starts[1:]:
        adjusted_starts.append(max(start, adjusted_starts[-1] + 0.02))

    if adjusted_starts[-1] >= line_end - 0.01:
        return None

    new_words: List[Word] = []
    for i, word in enumerate(line.words):
        start = adjusted_starts[i]
        if i + 1 < len(adjusted_starts):
            end = min(line_end, adjusted_starts[i + 1] - 0.02)
        else:
            end = line_end
        if end - start < min_word_duration:
            end = min(line_end, start + min_word_duration)
        if end <= start:
            return None
        new_words.append(
            Word(
                text=word.text,
                start_time=start,
                end_time=end,
                singer=word.singer,
            )
        )
    return Line(words=new_words, singer=line.singer)


def _retime_uniform_word_lines_with_onsets(
    lines: List[Line],
    onset_times,
) -> int:
    fixes = 0
    for idx, line in enumerate(lines):
        if not _is_uniform_word_timing(line):
            continue
        retimed = _retime_line_words_to_onsets(line, onset_times)
        if retimed is None:
            continue
        delta = sum(
            abs(retimed.words[i].start_time - line.words[i].start_time)
            for i in range(len(line.words))
        )
        if delta < 0.15:
            continue
        lines[idx] = retimed
        fixes += 1
    return fixes


def _short_line_silence_shift_candidate(
    line: Line,
    normalized_silences: List[Tuple[float, float]],
    onset_times,
) -> Optional[float]:
    return _short_line_helpers._short_line_silence_shift_candidate(
        line,
        normalized_silences,
        onset_times,
        find_internal_silences_fn=_find_internal_silences,
        has_near_start_silence_fn=_has_near_start_silence,
        first_onset_after_fn=_first_onset_after,
    )


def _short_line_run_end(lines: List[Line], start_idx: int) -> int:
    return _short_line_helpers._short_line_run_end(lines, start_idx)


def _shift_short_line_runs_after_silence(
    lines: List[Line], normalized_silences: List[Tuple[float, float]], onset_times
) -> int:
    return _short_line_helpers._shift_short_line_runs_after_silence(
        lines,
        normalized_silences,
        onset_times,
        shift_line_words_fn=_shift_line_words,
        compact_short_line_if_needed_fn=_compact_short_line_if_needed,
        find_internal_silences_fn=_find_internal_silences,
        has_near_start_silence_fn=_has_near_start_silence,
        first_onset_after_fn=_first_onset_after,
    )


def _shift_single_short_lines_after_silence(
    lines: List[Line], normalized_silences: List[Tuple[float, float]], onset_times
) -> int:
    return _short_line_helpers._shift_single_short_lines_after_silence(
        lines,
        normalized_silences,
        onset_times,
        shift_line_words_fn=_shift_line_words,
        compact_short_line_if_needed_fn=_compact_short_line_if_needed,
        find_internal_silences_fn=_find_internal_silences,
        has_near_start_silence_fn=_has_near_start_silence,
        first_onset_after_fn=_first_onset_after,
    )


def _compact_short_lines_near_silence(
    lines: List[Line], normalized_silences: List[Tuple[float, float]]
) -> int:
    return _short_line_helpers._compact_short_lines_near_silence(
        lines,
        normalized_silences,
        compact_short_line_if_needed_fn=_compact_short_line_if_needed,
        find_internal_silences_fn=_find_internal_silences,
        has_near_start_silence_fn=_has_near_start_silence,
    )


def _stretch_similar_adjacent_short_lines(
    lines: List[Line], normalized_silences: List[Tuple[float, float]]
) -> int:
    return _short_line_helpers._stretch_similar_adjacent_short_lines(
        lines,
        normalized_silences,
        token_overlap_fn=_token_overlap,
        rebuild_line_with_target_end_fn=_rebuild_line_with_target_end,
    )


def _cap_isolated_short_lines(lines: List[Line]) -> int:
    return _short_line_helpers._cap_isolated_short_lines(
        lines, rebuild_line_with_target_end_fn=_rebuild_line_with_target_end
    )


def _clone_lines(lines: List[Line]) -> List[Line]:
    return [
        Line(
            words=[
                Word(
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


def _long_gap_stats(lines: List[Line], threshold: float = 20.0) -> Tuple[int, float]:
    prev_end: Optional[float] = None
    long_count = 0
    max_gap = 0.0
    for line in lines:
        if not line.words:
            continue
        if prev_end is not None:
            gap = line.start_time - prev_end
            if gap > threshold:
                long_count += 1
            if gap > max_gap:
                max_gap = gap
        prev_end = line.end_time
    return long_count, max_gap


def _ordering_inversion_stats(
    lines: List[Line], tolerance: float = 0.01
) -> Tuple[int, float]:
    prev_start: Optional[float] = None
    inversions = 0
    max_drop = 0.0
    for line in lines:
        if not line.words:
            continue
        start = line.start_time
        if prev_start is not None and start < (prev_start - tolerance):
            inversions += 1
            max_drop = max(max_drop, prev_start - start)
        prev_start = start
    return inversions, max_drop


def _pull_lines_forward_for_continuous_vocals(
    lines: List[Line],
    audio_features: Optional[AudioFeatures],
    max_gap: float = 4.0,
    enable_silence_short_line_refinement: bool = True,
) -> Tuple[List[Line], int]:
    """Refine short-line placement from audio continuity/silence cues."""
    if not lines or audio_features is None:
        return lines, 0

    onset_times = audio_features.onset_times
    if onset_times is None or len(onset_times) == 0:
        return lines, 0

    original_lines = _clone_lines(lines)
    before_long_count, before_max_gap = _long_gap_stats(lines)
    before_inv_count, before_inv_drop = _ordering_inversion_stats(lines)

    fixes = _shift_lines_across_long_activity_gaps(
        lines, audio_features, max_gap, onset_times
    )
    fixes += _extend_line_ends_across_active_gaps(lines, audio_features)

    env_flag = os.getenv("Y2K_WHISPER_SILENCE_REFINEMENT", "1").strip().lower()
    env_enabled = env_flag not in {"0", "false", "off", "no"}
    if not (enable_silence_short_line_refinement and env_enabled):
        return lines, fixes

    silence_regions = getattr(audio_features, "silence_regions", None) or []
    normalized_silences = [
        (float(start), float(end))
        for start, end in silence_regions
        if float(end) - float(start) >= 0.8
    ]
    if normalized_silences:
        fixes += _shift_short_line_runs_after_silence(
            lines, normalized_silences, onset_times
        )
        fixes += _shift_single_short_lines_after_silence(
            lines, normalized_silences, onset_times
        )
        fixes += _compact_short_lines_near_silence(lines, normalized_silences)
        fixes += _stretch_similar_adjacent_short_lines(lines, normalized_silences)
        fixes += _cap_isolated_short_lines(lines)

    after_long_count, after_max_gap = _long_gap_stats(lines)
    after_inv_count, after_inv_drop = _ordering_inversion_stats(lines)
    if after_long_count > before_long_count or after_max_gap > max(
        before_max_gap + 8.0, 20.0
    ):
        logger.debug(
            "Reverting continuous-vocals refinement: long gaps worsened (%d→%d, %.2fs→%.2fs)",
            before_long_count,
            after_long_count,
            before_max_gap,
            after_max_gap,
        )
        return original_lines, 0
    if after_inv_count > before_inv_count and after_inv_drop > max(
        before_inv_drop + 0.75, 1.5
    ):
        logger.debug(
            "Reverting continuous-vocals refinement: ordering inversions worsened (%d→%d, %.2fs→%.2fs)",
            before_inv_count,
            after_inv_count,
            before_inv_drop,
            after_inv_drop,
        )
        return original_lines, 0
    return lines, fixes


def _fill_vocal_activity_gaps(
    whisper_words: List[TranscriptionWord],
    audio_features: AudioFeatures,
    threshold: float = 0.3,
    min_gap: float = 1.0,
    chunk_duration: float = 0.5,
    segments: Optional[List[TranscriptionSegment]] = None,
) -> Tuple[List[TranscriptionWord], Optional[List[TranscriptionSegment]]]:
    return _alignment_activity_helpers._fill_vocal_activity_gaps(
        whisper_words,
        audio_features,
        check_vocal_activity_in_range_fn=_check_vocal_activity_in_range,
        threshold=threshold,
        min_gap=min_gap,
        chunk_duration=chunk_duration,
        segments=segments,
    )


def _drop_duplicate_lines_by_timing(
    lines: List[Line],
    max_gap: float = 0.2,
) -> Tuple[List[Line], int]:
    return _alignment_activity_helpers._drop_duplicate_lines_by_timing(
        lines, max_gap=max_gap
    )
