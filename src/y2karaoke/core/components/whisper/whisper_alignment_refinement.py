"""High-level refinement and hybrid alignment logic."""

import os
import logging
from contextlib import contextmanager
from typing import List, Optional, Tuple, Set

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
        # Don't spread unmatched lines across a large instrumental gap;
        # keep them compact near the previous anchor.
        if duration_scale > 2.0:
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


def _pull_lines_forward_for_continuous_vocals(  # noqa: C901
    lines: List[Line],
    audio_features: Optional[AudioFeatures],
    max_gap: float = 4.0,
    enable_silence_short_line_refinement: bool = True,
) -> Tuple[List[Line], int]:
    """Refine short-line placement from audio continuity/silence cues."""

    def _line_tokens(text: str) -> List[str]:
        out = []
        for raw in text.lower().split():
            tok = "".join(ch for ch in raw if ch.isalpha())
            if tok:
                out.append(tok)
        return out

    def _token_overlap(a: str, b: str) -> float:
        ta = _line_tokens(a)
        tb = _line_tokens(b)
        if not ta or not tb:
            return 0.0
        sa = set(ta)
        sb = set(tb)
        return len(sa & sb) / max(len(sa), len(sb))

    fixes = 0
    if not lines or audio_features is None:
        return lines, fixes

    onset_times = audio_features.onset_times
    if onset_times is None or len(onset_times) == 0:
        return lines, fixes

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

        new_start = float(candidate_onsets[0])
        new_start = max(new_start, prev_line.end_time + 0.05)
        shift = new_start - line.start_time
        if shift > -0.3:
            continue

        new_words = [
            Word(
                text=w.text,
                start_time=w.start_time + shift,
                end_time=w.end_time + shift,
                singer=w.singer,
            )
            for w in line.words
        ]

        lines[idx] = Line(words=new_words, singer=line.singer)
        fixes += 1

    env_flag = os.getenv("Y2K_WHISPER_SILENCE_REFINEMENT", "1").strip().lower()
    env_enabled = env_flag not in {"0", "false", "off", "no"}
    if not (enable_silence_short_line_refinement and env_enabled):
        return lines, fixes

    silence_regions = getattr(audio_features, "silence_regions", None) or []
    normalized_silences = [
        (float(s), float(e)) for s, e in silence_regions if float(e) - float(s) >= 0.8
    ]
    if not normalized_silences:
        return lines, fixes

    shifted_indices = set()
    idx = 1
    while idx < len(lines) - 1:
        if idx in shifted_indices:
            idx += 1
            continue
        line = lines[idx]
        if not line.words or len(line.words) > 4:
            idx += 1
            continue

        line_start = line.start_time
        line_end = line.end_time
        if line_end <= line_start + 0.2:
            idx += 1
            continue

        internal_silences = [
            (s, e)
            for s, e in normalized_silences
            if s >= line_start and e <= (line_end + 2.0)
        ]
        if not internal_silences:
            idx += 1
            continue
        near_start_silence = any(
            s <= line_start + 0.7 and e >= line_start - 0.1
            for s, e in internal_silences
        )
        if not near_start_silence:
            idx += 1
            continue

        silence_start, silence_end = internal_silences[-1]
        candidate_onsets = onset_times[
            (onset_times >= silence_end) & (onset_times <= silence_end + 1.8)
        ]
        if len(candidate_onsets) == 0:
            idx += 1
            continue

        target_start = float(candidate_onsets[0])
        desired_shift = target_start - line_start
        if desired_shift < 0.8:
            idx += 1
            continue

        run_end = idx + 1
        while run_end < len(lines):
            prev_line = lines[run_end - 1]
            run_line = lines[run_end]
            if not prev_line.words or not run_line.words:
                break
            if len(run_line.words) > 4:
                break
            if run_line.start_time - prev_line.end_time > 2.2:
                break
            if run_end - idx >= 2:
                break
            run_end += 1

        available = desired_shift
        if run_end < len(lines) and lines[run_end].words:
            available = min(
                available,
                max(
                    0.0, lines[run_end].start_time - 0.05 - lines[run_end - 1].end_time
                ),
            )
        if available < 0.4:
            idx += 1
            continue

        for run_idx in range(idx, run_end):
            run_line = lines[run_idx]
            shifted_words = [
                Word(
                    text=w.text,
                    start_time=w.start_time + available,
                    end_time=w.end_time + available,
                    singer=w.singer,
                )
                for w in run_line.words
            ]
            shifted_line = Line(words=shifted_words, singer=run_line.singer)
            if (
                len(run_line.words) <= 4
                and shifted_line.end_time > shifted_line.start_time
            ):
                max_duration = 1.25 if len(run_line.words) <= 3 else 1.6
                duration = shifted_line.end_time - shifted_line.start_time
                if duration > max_duration:
                    next_start = None
                    if run_idx + 1 < run_end and lines[run_idx + 1].words:
                        next_start = lines[run_idx + 1].start_time + available
                    elif run_idx + 1 < len(lines) and lines[run_idx + 1].words:
                        next_start = lines[run_idx + 1].start_time
                    target_end = shifted_line.start_time + max_duration
                    if next_start is not None:
                        target_end = min(target_end, next_start - 0.05)
                    if target_end > shifted_line.start_time + 0.2:
                        spacing = (target_end - shifted_line.start_time) / len(
                            run_line.words
                        )
                        compact_words = []
                        for w_idx, w in enumerate(run_line.words):
                            ws = shifted_line.start_time + w_idx * spacing
                            we = ws + spacing * 0.9
                            if w_idx == len(run_line.words) - 1:
                                we = target_end
                            compact_words.append(
                                Word(
                                    text=w.text,
                                    start_time=ws,
                                    end_time=we,
                                    singer=w.singer,
                                )
                            )
                        shifted_line = Line(words=compact_words, singer=run_line.singer)
            lines[run_idx] = shifted_line
            shifted_indices.add(run_idx)
        fixes += 1
        idx += 1

    # Single-line follow-up pass: after block shifts, refine short lines
    # again when they still start before a long near-start silence.
    for idx in range(1, len(lines) - 1):
        line = lines[idx]
        if not line.words or len(line.words) > 4:
            continue
        next_line = lines[idx + 1]
        if not next_line.words:
            continue
        line_start = line.start_time
        line_end = line.end_time
        internal_silences = [
            (s, e)
            for s, e in normalized_silences
            if s >= line_start and e <= (line_end + 2.0)
        ]
        if not internal_silences:
            continue
        near_start_silence = any(
            s <= line_start + 0.7 and e >= line_start - 0.1
            for s, e in internal_silences
        )
        if not near_start_silence:
            continue
        _, silence_end = internal_silences[-1]
        candidate_onsets = onset_times[
            (onset_times >= silence_end) & (onset_times <= silence_end + 1.8)
        ]
        if len(candidate_onsets) == 0:
            continue
        desired_shift = float(candidate_onsets[0]) - line_start
        if desired_shift < 0.8:
            continue
        available = max(0.0, next_line.start_time - 0.05 - line_end)
        shift = min(desired_shift, available)
        if shift < 0.4:
            continue
        shifted_words = [
            Word(
                text=w.text,
                start_time=w.start_time + shift,
                end_time=w.end_time + shift,
                singer=w.singer,
            )
            for w in line.words
        ]
        shifted_line = Line(words=shifted_words, singer=line.singer)
        if len(line.words) <= 4 and shifted_line.end_time > shifted_line.start_time:
            max_duration = 1.25 if len(line.words) <= 3 else 1.6
            duration = shifted_line.end_time - shifted_line.start_time
            if duration > max_duration:
                target_end = min(
                    shifted_line.start_time + max_duration,
                    next_line.start_time - 0.05,
                )
                if target_end > shifted_line.start_time + 0.2:
                    spacing = (target_end - shifted_line.start_time) / len(line.words)
                    compact_words = []
                    for w_idx, w in enumerate(line.words):
                        ws = shifted_line.start_time + w_idx * spacing
                        we = ws + spacing * 0.9
                        if w_idx == len(line.words) - 1:
                            we = target_end
                        compact_words.append(
                            Word(
                                text=w.text,
                                start_time=ws,
                                end_time=we,
                                singer=w.singer,
                            )
                        )
                    shifted_line = Line(words=compact_words, singer=line.singer)
        lines[idx] = shifted_line
        fixes += 1

    # Even when no additional shift is needed, keep short lines concise after
    # near-start silence alignment to avoid overlong lingering highlights.
    for idx in range(1, len(lines) - 1):
        line = lines[idx]
        if not line.words or len(line.words) > 4:
            continue
        next_line = lines[idx + 1]
        if not next_line.words:
            continue
        line_start = line.start_time
        line_end = line.end_time
        internal_silences = [
            (s, e)
            for s, e in normalized_silences
            if s >= line_start and e <= (line_end + 2.0)
        ]
        if not internal_silences:
            continue
        near_start_silence = any(
            s <= line_start + 0.7 and e >= line_start - 0.1
            for s, e in internal_silences
        )
        if not near_start_silence:
            continue
        max_duration = 1.25 if len(line.words) <= 3 else 1.6
        duration = line_end - line_start
        if duration <= max_duration:
            continue
        target_end = min(line_start + max_duration, next_line.start_time - 0.05)
        if target_end <= line_start + 0.2:
            continue
        spacing = (target_end - line_start) / len(line.words)
        compact_words = []
        for w_idx, w in enumerate(line.words):
            ws = line_start + w_idx * spacing
            we = ws + spacing * 0.9
            if w_idx == len(line.words) - 1:
                we = target_end
            compact_words.append(
                Word(
                    text=w.text,
                    start_time=ws,
                    end_time=we,
                    singer=w.singer,
                )
            )
        lines[idx] = Line(words=compact_words, singer=line.singer)
        fixes += 1

    # When two adjacent short lines are lexically similar but separated by a
    # large gap, stretch the first line toward the next detected silence.
    for idx in range(1, len(lines)):
        prev_line = lines[idx - 1]
        line = lines[idx]
        if not prev_line.words or not line.words:
            continue
        if len(prev_line.words) > 5 or len(line.words) > 4:
            continue
        if _token_overlap(prev_line.text, line.text) < 0.25:
            continue
        gap = line.start_time - prev_line.end_time
        if gap <= 1.0:
            continue
        target_end = line.start_time - 0.05
        latest_silence_start = None
        for s, e in normalized_silences:
            if s >= prev_line.end_time and e <= line.start_time:
                latest_silence_start = s
        if latest_silence_start is not None:
            target_end = min(target_end, latest_silence_start - 0.05)
        if target_end <= prev_line.end_time + 0.6:
            continue
        span = target_end - prev_line.start_time
        if span <= 0.4:
            continue
        spacing = span / len(prev_line.words)
        stretched_words = []
        for w_idx, w in enumerate(prev_line.words):
            ws = prev_line.start_time + w_idx * spacing
            we = ws + spacing * 0.9
            if w_idx == len(prev_line.words) - 1:
                we = target_end
            stretched_words.append(
                Word(
                    text=w.text,
                    start_time=ws,
                    end_time=we,
                    singer=w.singer,
                )
            )
        lines[idx - 1] = Line(words=stretched_words, singer=prev_line.singer)
        fixes += 1

    # Cap isolated short lines that still linger too long.
    for idx in range(1, len(lines) - 1):
        line = lines[idx]
        prev_line = lines[idx - 1]
        next_line = lines[idx + 1]
        if not line.words or len(line.words) > 3:
            continue
        if not prev_line.words or not next_line.words:
            continue
        duration = line.end_time - line.start_time
        if duration <= 1.25:
            continue
        prev_gap = line.start_time - prev_line.end_time
        next_gap = next_line.start_time - line.end_time
        if prev_gap <= 0.5 or next_gap <= 1.0:
            continue
        target_end = min(line.start_time + 1.25, next_line.start_time - 0.05)
        if target_end <= line.start_time + 0.2:
            continue
        spacing = (target_end - line.start_time) / len(line.words)
        compact_words = []
        for w_idx, w in enumerate(line.words):
            ws = line.start_time + w_idx * spacing
            we = ws + spacing * 0.9
            if w_idx == len(line.words) - 1:
                we = target_end
            compact_words.append(
                Word(
                    text=w.text,
                    start_time=ws,
                    end_time=we,
                    singer=w.singer,
                )
            )
        lines[idx] = Line(words=compact_words, singer=line.singer)
        fixes += 1

    return lines, fixes


def _fill_vocal_activity_gaps(
    whisper_words: List[TranscriptionWord],
    audio_features: AudioFeatures,
    threshold: float = 0.3,
    min_gap: float = 1.0,
    chunk_duration: float = 0.5,
    segments: Optional[List[TranscriptionSegment]] = None,
) -> Tuple[List[TranscriptionWord], Optional[List[TranscriptionSegment]]]:
    """Inject pseudo-words and segments where vocal activity is high but transcription missing."""
    if not whisper_words:
        return whisper_words, segments

    filled_words = list(whisper_words)
    filled_words.sort(key=lambda w: w.start)

    filled_segments = list(segments) if segments is not None else None
    if filled_segments:
        filled_segments.sort(key=lambda s: s.start)

    new_words = []
    new_segments = []

    def add_vocal_block(start, end):
        curr = start
        seg_words = []
        while curr + chunk_duration <= end:
            w = TranscriptionWord(
                start=curr,
                end=curr + chunk_duration,
                text="[VOCAL]",
                probability=0.0,
            )
            new_words.append(w)
            seg_words.append(w)
            curr += chunk_duration

        if seg_words and filled_segments is not None:
            new_segments.append(
                TranscriptionSegment(
                    start=seg_words[0].start,
                    end=seg_words[-1].end,
                    text="[VOCAL]",
                    words=seg_words,
                )
            )

    # 1. Check gap before first word
    vocal_start = audio_features.vocal_start
    if filled_words[0].start - vocal_start >= min_gap:
        activity = _check_vocal_activity_in_range(
            vocal_start, filled_words[0].start, audio_features
        )
        if activity > threshold:
            add_vocal_block(vocal_start, filled_words[0].start)

    # 2. Check gaps between words
    for i in range(len(filled_words) - 1):
        gap_start = filled_words[i].end
        gap_end = filled_words[i + 1].start

        if gap_end - gap_start >= min_gap:
            activity = _check_vocal_activity_in_range(
                gap_start, gap_end, audio_features
            )
            if activity > threshold:
                add_vocal_block(gap_start, gap_end)

    # 3. Check gap after last word
    vocal_end = audio_features.vocal_end
    if vocal_end - filled_words[-1].end >= min_gap:
        activity = _check_vocal_activity_in_range(
            filled_words[-1].end, vocal_end, audio_features
        )
        if activity > threshold:
            add_vocal_block(filled_words[-1].end, vocal_end)

    if new_words:
        logger.info(f"Vocal gap filler: injected {len(new_words)} [VOCAL] pseudo-words")
        filled_words.extend(new_words)
        filled_words.sort(key=lambda w: w.start)

        if filled_segments is not None and new_segments:
            filled_segments.extend(new_segments)
            filled_segments.sort(key=lambda s: s.start)

    return filled_words, filled_segments


def _drop_duplicate_lines_by_timing(
    lines: List[Line],
    max_gap: float = 0.2,
) -> Tuple[List[Line], int]:
    """Drop adjacent duplicate lines that overlap or are nearly contiguous."""
    if not lines:
        return lines, 0

    deduped: List[Line] = []
    dropped = 0

    for line in lines:
        if not deduped:
            deduped.append(line)
            continue

        prev = deduped[-1]
        if not prev.words or not line.words:
            deduped.append(line)
            continue

        if prev.text.strip() != line.text.strip():
            deduped.append(line)
            continue

        gap = line.start_time - prev.end_time
        if gap <= max_gap:
            dropped += 1
            continue

        deduped.append(line)

    return deduped, dropped
