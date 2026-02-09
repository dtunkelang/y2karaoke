"""LRC to Whisper segment mapping logic."""

import logging
import re
from statistics import median
from typing import List, Optional, Tuple

from .models import Line, Word
from .timing_models import TranscriptionSegment, TranscriptionWord
from .phonetic_utils import _phonetic_similarity

logger = logging.getLogger(__name__)


def _norm_token(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w']+", "", text)
    return text


def _build_whisper_word_list(
    transcription: List[TranscriptionSegment],
) -> List[TranscriptionWord]:
    words: List[TranscriptionWord] = []
    for seg in sorted(transcription, key=lambda s: s.start):
        if not seg.words:
            continue
        for t_word in seg.words:
            if t_word.start is not None:
                words.append(t_word)
    words.sort(key=lambda w: w.start)
    return words


def _select_window_sequence(
    window_words: List[TranscriptionWord],
    line_text: str,
    language: str,
    target_len: int,
    first_token: Optional[str],
    phonetic_similarity,
) -> List[TranscriptionWord]:
    if not window_words:
        return []
    best_seq = window_words
    best_sim = -1.0
    best_score = -1.0
    lengths = [target_len - 1, target_len, target_len + 1]
    lengths = [length for length in lengths if length > 0]
    max_start = len(window_words)
    start_candidates = range(max_start)
    if first_token:
        anchor_idx = None
        anchor_sim = 0.0
        for idx, w in enumerate(window_words):
            sim = phonetic_similarity(first_token, w.text, language)
            if sim > anchor_sim:
                anchor_sim = sim
                anchor_idx = idx
        if anchor_idx is not None and anchor_sim >= 0.6:
            start_candidates = range(
                max(anchor_idx - 1, 0), min(anchor_idx + 1, max_start - 1) + 1
            )
    for start_idx in start_candidates:
        for length in lengths:
            end_idx = start_idx + length
            if end_idx > len(window_words):
                continue
            seq = window_words[start_idx:end_idx]
            seq_text = " ".join(w.text for w in seq)
            sim = phonetic_similarity(line_text, seq_text, language)
            length_penalty = 0.02 * abs(length - target_len)
            score = sim - length_penalty
            if score > best_score:
                best_score = score
                best_sim = sim
                best_seq = seq
    if best_sim < 0.2:
        return window_words
    return best_seq


def _slots_from_sequence(
    seq_words: List[TranscriptionWord], seq_end: Optional[float]
) -> Optional[Tuple[List[float], List[float]]]:
    if not seq_words:
        return None
    slots: List[float] = []
    spokens: List[float] = []
    for i, w in enumerate(seq_words):
        if w.start is None:
            return None
        if i + 1 < len(seq_words) and seq_words[i + 1].start is not None:
            slot = seq_words[i + 1].start - w.start
        elif w.end is not None and w.end > w.start:
            slot = w.end - w.start
        elif seq_end is not None and seq_end > w.start:
            slot = seq_end - w.start
        else:
            slot = 0.02
        slot = max(slot, 0.02)
        if w.end is not None and w.end > w.start:
            spoken = min(w.end - w.start, slot)
        else:
            spoken = slot * 0.85
        slots.append(slot)
        spokens.append(max(spoken, 0.02))
    return slots, spokens


def _resample_slots_to_line(
    slots: List[float],
    spokens: List[float],
    line_length: int,
) -> Tuple[List[float], List[float]]:
    if not slots or line_length <= 0:
        return [], []
    if line_length == 1:
        slot = median(slots)
        spoken = median(spokens) if spokens else slot * 0.85
        return [max(slot, 0.02)], [max(spoken, 0.02)]
    seg_count = len(slots)
    resampled: List[float] = []
    resampled_spoken: List[float] = []
    for i in range(line_length):
        pos = i * (seg_count - 1) / (line_length - 1)
        lo = int(pos)
        hi = min(lo + 1, seg_count - 1)
        frac = pos - lo
        slot = slots[lo] * (1 - frac) + slots[hi] * frac
        spoken = spokens[lo] * (1 - frac) + spokens[hi] * frac
        resampled.append(max(slot, 0.02))
        resampled_spoken.append(max(spoken, 0.02))
    return resampled, resampled_spoken


def _whisper_durations_for_line(  # noqa: C901
    line_words: List[Word],
    seg_words: Optional[List[TranscriptionWord]],
    seg_end: Optional[float],
    language: str,
    phonetic_similarity,
) -> Optional[Tuple[List[float], List[float]]]:
    if not seg_words:
        return None
    seg_tokens = [_norm_token(w.text) for w in seg_words]
    seg_starts = [w.start for w in seg_words if w.start is not None]
    if not seg_starts:
        return None
    seq_slots = _slots_from_sequence(seg_words, seg_end)
    if not seq_slots:
        return None
    seg_slots, seg_spoken = seq_slots
    default_slot = median(seg_slots) if seg_slots else 0.1
    default_spoken = median(seg_spoken) if seg_spoken else default_slot * 0.85
    slots: List[float] = []
    spokens: List[float] = []
    min_word_sim = 0.5

    def _token_similarity(token: str, seg_token: str) -> float:
        if not token or not seg_token:
            return 0.0
        if (
            seg_token == token
            or seg_token.startswith(token)
            or token.startswith(seg_token)
        ):
            return 1.0
        return phonetic_similarity(token, seg_token, language)

    # Monotonic DP alignment so repeated words cannot match out of order.
    line_tokens = [_norm_token(w.text) for w in line_words]
    n = len(line_tokens)
    m = len(seg_tokens)
    if n == 0 or m == 0:
        return None
    gap_line = 0.6
    gap_seg = 0.3
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] - gap_line
        back[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] - gap_seg
        back[0][j] = 2
    for i in range(1, n + 1):
        token = line_tokens[i - 1]
        for j in range(1, m + 1):
            seg_token = seg_tokens[j - 1]
            sim = _token_similarity(token, seg_token)
            match_cost = 2.0 if sim < min_word_sim else 1.0 - sim
            score_match = dp[i - 1][j - 1] - match_cost
            score_line_skip = dp[i - 1][j] - gap_line
            score_seg_skip = dp[i][j - 1] - gap_seg
            if score_match >= score_line_skip and score_match >= score_seg_skip:
                dp[i][j] = score_match
                back[i][j] = 0
            elif score_line_skip >= score_seg_skip:
                dp[i][j] = score_line_skip
                back[i][j] = 1
            else:
                dp[i][j] = score_seg_skip
                back[i][j] = 2
    mapping: List[Optional[int]] = [None] * n
    i = n
    j = m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and back[i][j] == 0:
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or back[i][j] == 1):
            i -= 1
        else:
            j -= 1

    matched = sum(1 for idx in mapping if idx is not None)
    match_ratio = matched / max(len(line_words), 1)
    if match_ratio < 0.4:
        return _resample_slots_to_line(seg_slots, seg_spoken, len(line_words))

    for idx in mapping:
        if idx is None:
            slots.append(default_slot)
            spokens.append(default_spoken)
        else:
            slots.append(max(seg_slots[idx], 0.02))
            spokens.append(max(seg_spoken[idx], 0.02))
    return slots, spokens


def _apply_weighted_slots_to_line(
    line_words: List[Word],
    desired_start: float,
    line_duration: float,
    slot_durations: Optional[List[float]],
    spoken_durations: Optional[List[float]],
) -> List[Word]:
    if not line_words:
        return []
    if not slot_durations or not spoken_durations:
        weights = [1.0 / len(line_words)] * len(line_words)
        slot_durations = [1.0] * len(line_words)
        spoken_durations = [0.85] * len(line_words)
    else:
        total = sum(slot_durations) or 1.0
        weights = [d / total for d in slot_durations]

    cursor = desired_start
    weighted_words = []
    for i, (word_obj, weight) in enumerate(zip(line_words, weights)):
        seg_dur = line_duration * weight
        slot = slot_durations[i]
        spoken = spoken_durations[i]
        spoken_ratio = min(max(spoken / max(slot, 0.02), 0.5), 1.0)
        start = cursor
        end = start + max(seg_dur * spoken_ratio, 0.02)
        weighted_words.append(
            Word(
                text=word_obj.text,
                start_time=start,
                end_time=end,
                singer=word_obj.singer,
            )
        )
        cursor += seg_dur
    return weighted_words


def _shift_words(words: List[Word], shift: float) -> List[Word]:
    return [
        Word(
            text=w.text,
            start_time=w.start_time + shift,
            end_time=w.end_time + shift,
            singer=w.singer,
        )
        for w in words
    ]


def _find_best_segment_for_line(
    line: Line,
    sorted_segments: List[TranscriptionSegment],
    seg_idx: int,
    lookahead: int,
    language: str,
    phonetic_similarity,
) -> Tuple[Optional[int], float, int]:
    best_idx = None
    best_sim = 0.0
    window_end = min(seg_idx + lookahead, len(sorted_segments))
    for idx in range(seg_idx, window_end):
        seg = sorted_segments[idx]
        sim = phonetic_similarity(line.text, seg.text, language)
        if sim > best_sim:
            best_sim = sim
            best_idx = idx
    return best_idx, best_sim, window_end


def _map_line_words_to_segment(line: Line, seg: TranscriptionSegment) -> List[Word]:
    duration = max(seg.end - seg.start, 0.2)
    orig_start = line.words[0].start_time
    orig_end = line.words[-1].end_time
    orig_duration = max(orig_end - orig_start, 0.2)
    mapped_duration = duration

    new_words: List[Word] = []
    if orig_duration > 0 and len(line.words) > 0:
        for word in line.words:
            rel_start = (word.start_time - orig_start) / orig_duration
            rel_end = (word.end_time - orig_start) / orig_duration
            start = seg.start + rel_start * mapped_duration
            end = seg.start + rel_end * mapped_duration
            new_words.append(
                Word(
                    text=word.text,
                    start_time=start,
                    end_time=max(end, start + 0.02),
                    singer=word.singer,
                )
            )
    else:
        spacing = mapped_duration / max(len(line.words), 1)
        for i, word in enumerate(line.words):
            start = seg.start + i * spacing
            end = start + spacing * 0.9
            new_words.append(
                Word(
                    text=word.text,
                    start_time=start,
                    end_time=end,
                    singer=word.singer,
                )
            )
    return new_words


def _select_window_words_for_line(
    all_words: List[TranscriptionWord],
    line: Line,
    desired_start: Optional[float],
    next_lrc_start: Optional[float],
    language: str,
    phonetic_similarity,
) -> Optional[List[TranscriptionWord]]:
    if desired_start is None or next_lrc_start is None or not all_words:
        return None
    window_start = desired_start - 1.0
    window_words = [w for w in all_words if window_start <= w.start < next_lrc_start]
    if not window_words:
        return None
    first_token = line.words[0].text if line.words else None
    min_count = max(3, len(line.words) // 2)
    min_prob = 0.5

    def _pick_sequence(candidates: List[TranscriptionWord]):
        seq = _select_window_sequence(
            candidates,
            line.text,
            language,
            len(line.words),
            first_token,
            phonetic_similarity,
        )
        sim = phonetic_similarity(line.text, " ".join(w.text for w in seq), language)
        return seq, sim

    def _preferred_sequence(candidates: List[TranscriptionWord]):
        filtered = [
            w for w in candidates if w.probability is None or w.probability >= min_prob
        ]
        seq_all, sim_all = _pick_sequence(candidates)
        if len(filtered) >= min_count:
            seq_f, sim_f = _pick_sequence(filtered)
            if sim_f >= 0.55 and sim_f >= sim_all - 0.02:
                return seq_f, sim_f
        return seq_all, sim_all

    initial_seq, initial_sim = _preferred_sequence(window_words)
    if len(window_words) >= min_count and initial_sim >= 0.55:
        return initial_seq

    # Expand window earlier to catch words that precede a late LRC start.
    expanded_start = max(desired_start - 4.0, 0.0)
    if expanded_start < window_start:
        expanded_words = [
            w for w in all_words if expanded_start <= w.start < next_lrc_start
        ]
        if expanded_words:
            expanded_seq, expanded_sim = _preferred_sequence(expanded_words)
            if expanded_sim > initial_sim + 0.05 or initial_sim < 0.45:
                return expanded_seq
    return initial_seq


def _line_duration_from_lrc(
    line: Line,
    desired_start: float,
    lrc_line_starts: Optional[List[float]],
    line_idx: int,
) -> float:
    if lrc_line_starts and line_idx + 1 < len(lrc_line_starts):
        return max(lrc_line_starts[line_idx + 1] - desired_start, 0.5)
    return max(line.words[-1].end_time - line.words[0].start_time, 0.5)


def _apply_lrc_weighted_timing(  # noqa: C901
    line: Line,
    desired_start: float,
    next_lrc_start: Optional[float],
    lrc_line_starts: Optional[List[float]],
    line_idx: int,
    all_words: List[TranscriptionWord],
    seg: Optional[TranscriptionSegment],
    language: str,
    phonetic_similarity,
    line_duration_override: Optional[float] = None,
) -> Tuple[List[Word], float]:
    line_duration = (
        line_duration_override
        if line_duration_override is not None
        else _line_duration_from_lrc(line, desired_start, lrc_line_starts, line_idx)
    )
    window_words = _select_window_words_for_line(
        all_words,
        line,
        desired_start,
        next_lrc_start,
        language,
        phonetic_similarity,
    )

    def _anchor_desired_start(
        desired_start: float,
        line_duration: float,
        offsets_scaled: List[float],
        align_by_index: bool = False,
    ) -> float:
        if not window_words or not offsets_scaled:
            return desired_start
        window_start = (
            window_words[0].start if window_words[0].start is not None else None
        )
        require_forward = (
            window_start is not None and desired_start + 0.2 < window_start
        )
        require_backward = (
            window_start is not None and desired_start - 0.2 > window_start
        )
        best_score: Optional[float] = None
        best_start: Optional[float] = None
        for i, word_obj in enumerate(line.words):
            token = _norm_token(word_obj.text)
            if not token or i >= len(offsets_scaled):
                continue
            if align_by_index and i < len(window_words):
                candidates = [window_words[i]]
            else:
                candidates = window_words
            for ww in candidates:
                if ww.start is None:
                    continue
                sim = phonetic_similarity(token, ww.text, language)
                if sim < 0.6:
                    continue
                expected = desired_start + offsets_scaled[i]
                delta = abs(expected - ww.start)
                score = sim - min(delta / max(line_duration, 0.01), 1.0) * 0.2
                candidate_start = ww.start - offsets_scaled[i]
                forward_shift = candidate_start - desired_start
                if forward_shift > 1.0:
                    allow_large_shift = (
                        i == 0
                        and offsets_scaled[i] <= line_duration * 0.05
                        and forward_shift <= 2.0
                    )
                    if not allow_large_shift:
                        continue
                if offsets_scaled[i] >= line_duration * 0.6:
                    if candidate_start - desired_start > 0.8:
                        continue
                allow_early_anchor = (
                    len(window_words) <= 3 and offsets_scaled[i] >= line_duration * 0.5
                )
                if not allow_early_anchor and i == len(line.words) - 1:
                    allow_early_anchor = len(window_words) <= 3
                if (
                    window_start is not None
                    and candidate_start < window_start - 0.05
                    and not allow_early_anchor
                ):
                    continue
                if (
                    allow_early_anchor
                    and window_start is not None
                    and candidate_start < window_start - 1.5
                ):
                    continue
                if require_forward and candidate_start < desired_start:
                    continue
                if require_backward and candidate_start > desired_start:
                    continue
                if best_score is None or score > best_score:
                    best_score = score
                    best_start = candidate_start
        if best_start is None:
            return desired_start
        shift = best_start - desired_start
        max_shift = 2.5
        if abs(shift) > max_shift:
            best_start = desired_start + max_shift * (1 if shift > 0 else -1)
        return max(best_start, 0.0)

    direct_offsets: Optional[List[float]] = None
    if (
        window_words is not None
        and len(window_words) == len(line.words)
        and next_lrc_start is not None
    ):
        first_start = window_words[0].start
        last_start = window_words[-1].start
        if (
            first_start is not None
            and last_start is not None
            and last_start > first_start
        ):
            offsets = [t_word.start - first_start for t_word in window_words]  # type: ignore[operator]
            if all(b > a for a, b in zip(offsets, offsets[1:])):
                direct_offsets = offsets

    if direct_offsets and next_lrc_start is not None:
        if line_duration_override is not None:
            line_duration = max(line_duration_override, 0.5)
        else:
            line_duration = max(next_lrc_start - desired_start, 0.5)
        if window_words and window_words[0].start is not None:
            whisper_span = None
            if window_words[-1].end is not None:
                whisper_span = window_words[-1].end - window_words[0].start
            elif window_words[-1].start is not None:
                whisper_span = window_words[-1].start - window_words[0].start
            if whisper_span is not None and whisper_span > 0.2:
                if next_lrc_start is not None and desired_start < window_words[0].start:
                    max_span = max(next_lrc_start - window_words[0].start - 0.1, 0.5)
                    whisper_span = min(whisper_span, max_span)
                if whisper_span < line_duration * 0.95:
                    line_duration = max(0.85 * whisper_span + 0.15 * line_duration, 0.5)
                lrc_start = None
                if lrc_line_starts and line_idx < len(lrc_line_starts):
                    lrc_start = lrc_line_starts[line_idx]
                elif line.words:
                    lrc_start = line.words[0].start_time
                forward_shift = None
                if lrc_start is not None and window_words[0].start is not None:
                    forward_shift = window_words[0].start - lrc_start
                if (
                    forward_shift is not None
                    and forward_shift >= 0.8
                    and whisper_span > 0.2
                ):
                    tightened = max(whisper_span * 1.05, 0.5)
                    if next_lrc_start is not None:
                        tightened = min(
                            tightened, max(next_lrc_start - desired_start, 0.5)
                        )
                    line_duration = min(line_duration, tightened)
        last_end = (
            window_words[-1].end
            if window_words and window_words[-1].end is not None
            else (window_words[-1].start if window_words else None)
        )
        if last_end is not None and first_start is not None:
            whisper_total = max(last_end - first_start, 0.01)
        else:
            whisper_total = max(direct_offsets[-1], 0.01)
        offsets_scaled = [
            (offset / whisper_total) * line_duration for offset in direct_offsets
        ]
        desired_start = _anchor_desired_start(
            desired_start,
            line_duration,
            offsets_scaled,
            align_by_index=True,
        )
        mapped_words: List[Word] = []
        for i, (word_obj, offset) in enumerate(zip(line.words, direct_offsets)):
            start = desired_start + (offset / whisper_total) * line_duration
            if i + 1 < len(direct_offsets):
                next_offset = direct_offsets[i + 1]
                slot = (next_offset - offset) / whisper_total * line_duration
            else:
                slot = line_duration - (offset / whisper_total) * line_duration
            spoken_ratio = 0.85
            if window_words is not None:
                ww = window_words[i]
                if ww.end is not None and ww.start is not None and ww.end > ww.start:
                    spoken_ratio = min(
                        max((ww.end - ww.start) / max(slot, 0.02), 0.5), 1.0
                    )
            end = start + max(slot * spoken_ratio, 0.02)
            mapped_words.append(
                Word(
                    text=word_obj.text,
                    start_time=start,
                    end_time=end,
                    singer=word_obj.singer,
                )
            )
        return mapped_words, desired_start

    if window_words and len(window_words) == len(line.words):
        whisper_timing = _slots_from_sequence(
            window_words, window_words[-1].end if window_words else None
        )
    else:
        whisper_timing = _whisper_durations_for_line(
            line.words,
            window_words if window_words else (seg.words if seg else None),
            (
                (window_words[-1].end if window_words else None)
                if window_words
                else (seg.end if seg else None)
            ),
            language,
            phonetic_similarity,
        )
    slot_durations = whisper_timing[0] if whisper_timing else None
    spoken_durations = whisper_timing[1] if whisper_timing else None
    if window_words and window_words[0].start is not None:
        whisper_span = None
        if window_words[-1].end is not None:
            whisper_span = window_words[-1].end - window_words[0].start
        elif window_words[-1].start is not None:
            whisper_span = window_words[-1].start - window_words[0].start
        if whisper_span is not None and whisper_span > 0.2:
            if next_lrc_start is not None and desired_start < window_words[0].start:
                max_span = max(next_lrc_start - window_words[0].start - 0.1, 0.5)
                whisper_span = min(whisper_span, max_span)
            if whisper_span < line_duration * 0.95:
                line_duration = max(0.85 * whisper_span + 0.15 * line_duration, 0.5)
            lrc_start = None
            if lrc_line_starts and line_idx < len(lrc_line_starts):
                lrc_start = lrc_line_starts[line_idx]
            elif line.words:
                lrc_start = line.words[0].start_time
            forward_shift = None
            if lrc_start is not None and window_words[0].start is not None:
                forward_shift = window_words[0].start - lrc_start
            if (
                forward_shift is not None
                and forward_shift >= 0.8
                and whisper_span > 0.2
            ):
                tightened = max(whisper_span * 1.05, 0.5)
                if next_lrc_start is not None:
                    tightened = min(tightened, max(next_lrc_start - desired_start, 0.5))
                line_duration = min(line_duration, tightened)
    if slot_durations:
        total = sum(slot_durations) or 1.0
        offsets_scaled = []
        cursor = 0.0
        for duration in slot_durations[: len(line.words)]:
            offsets_scaled.append((cursor / total) * line_duration)
            cursor += duration
        desired_start = _anchor_desired_start(
            desired_start,
            line_duration,
            offsets_scaled,
        )
    return (
        _apply_weighted_slots_to_line(
            line.words,
            desired_start,
            line_duration,
            slot_durations,
            spoken_durations,
        ),
        desired_start,
    )


def _clamp_line_end_to_next_start(
    line: Line,
    new_words: List[Word],
    lrc_line_starts: Optional[List[float]],
    line_idx: int,
) -> List[Word]:
    if not new_words:
        return new_words
    line_start = new_words[0].start_time
    orig_duration = max(line.words[-1].end_time - line.words[0].start_time, 0.5)
    max_end = line_start + max(1.5, orig_duration * 1.5)
    if lrc_line_starts and line_idx + 1 < len(lrc_line_starts):
        next_start = lrc_line_starts[line_idx + 1]
        max_end = min(max_end, next_start - 0.1)
    if new_words[-1].end_time > max_end and max_end > line_start:
        span = new_words[-1].end_time - line_start
        scale = (max_end - line_start) / span if span > 0 else 1.0
        new_words = [
            Word(
                text=w.text,
                start_time=line_start + (w.start_time - line_start) * scale,
                end_time=line_start + (w.end_time - line_start) * scale,
                singer=w.singer,
            )
            for w in new_words
        ]
    return new_words


def _create_lines_from_whisper(
    transcription: List[TranscriptionSegment],
) -> List[Line]:
    """Create Line objects directly from Whisper transcription."""
    from .models import Line

    lines: List[Line] = []
    for segment in transcription:
        if segment is None:
            continue
        words: List[Word] = []
        if segment.words:
            for w in segment.words:
                text = (w.text or "").strip()
                if not text:
                    continue
                words.append(
                    Word(
                        text=text,
                        start_time=float(w.start),
                        end_time=float(w.end),
                        singer="",
                    )
                )
        else:
            tokens = [t for t in segment.text.strip().split() if t]
            if tokens:
                duration = max(segment.end - segment.start, 0.2)
                spacing = duration / len(tokens)
                for i, token in enumerate(tokens):
                    start = segment.start + i * spacing
                    end = start + spacing * 0.9
                    words.append(
                        Word(text=token, start_time=start, end_time=end, singer="")
                    )
        if not words:
            continue
        lines.append(Line(words=words))
    return lines


def _map_lrc_lines_to_whisper_segments(  # noqa: C901
    lines: List[Line],
    transcription: List[TranscriptionSegment],
    language: str,
    lrc_line_starts: Optional[List[float]] = None,
    min_similarity: float = 0.35,
    min_similarity_fallback: float = 0.2,
    max_time_offset: float = 4.0,
    lookahead: int = 6,
) -> Tuple[List[Line], int, List[str]]:  # noqa: C901
    """Map LRC lines onto Whisper segment timing without reordering."""
    if not lines or not transcription:
        return lines, 0, []

    sorted_segments = sorted(transcription, key=lambda s: s.start)
    all_words = _build_whisper_word_list(transcription)
    adjusted: List[Line] = []
    fixes = 0
    issues: List[str] = []
    seg_idx = 0
    last_end = None
    min_gap = 0.01
    prev_text = None

    for line_idx, line in enumerate(lines):
        if not line.words:
            adjusted.append(line)
            continue

        best_idx, best_sim, window_end = _find_best_segment_for_line(
            line,
            sorted_segments,
            seg_idx,
            lookahead,
            language,
            _phonetic_similarity,
        )

        text_norm = line.text.strip().lower() if line.text else ""
        gap_required = 0.21 if prev_text and text_norm == prev_text else min_gap

        desired_start = (
            lrc_line_starts[line_idx]
            if lrc_line_starts and line_idx < len(lrc_line_starts)
            else None
        )
        next_lrc_start = (
            lrc_line_starts[line_idx + 1]
            if lrc_line_starts and line_idx + 1 < len(lrc_line_starts)
            else None
        )
        if (
            best_idx is None
            or best_sim < min_similarity_fallback
            or (
                desired_start is not None
                and best_idx is not None
                and abs(sorted_segments[best_idx].start - desired_start)
                > max_time_offset
            )
        ):
            new_words: List[Word]
            window_fallback = False
            if desired_start is not None and next_lrc_start is not None and all_words:
                window_words = _select_window_words_for_line(
                    all_words,
                    line,
                    desired_start,
                    next_lrc_start,
                    language,
                    _phonetic_similarity,
                )
                if window_words:
                    if window_words[0].start is not None:
                        whisper_start = window_words[0].start
                        start_delta = desired_start - whisper_start
                        if start_delta > 0.4:
                            desired_start = max(
                                whisper_start, desired_start - min(start_delta, 0.8)
                            )
                        elif start_delta < -0.4:
                            desired_start = min(
                                whisper_start, desired_start + min(-start_delta, 2.5)
                            )
                    whisper_duration = None
                    if (
                        window_words[-1].end is not None
                        and window_words[0].start is not None
                    ):
                        whisper_duration = max(
                            window_words[-1].end - window_words[0].start, 0.5
                        )
                    elif (
                        window_words[-1].start is not None
                        and window_words[0].start is not None
                    ):
                        whisper_duration = max(
                            window_words[-1].start - window_words[0].start, 0.5
                        )
                    window_text = " ".join(w.text for w in window_words)
                    window_sim = _phonetic_similarity(line.text, window_text, language)
                    if window_sim >= min_similarity_fallback:
                        new_words, desired_start = _apply_lrc_weighted_timing(
                            line,
                            desired_start,
                            next_lrc_start,
                            lrc_line_starts,
                            line_idx,
                            all_words,
                            None,
                            language,
                            _phonetic_similarity,
                            line_duration_override=whisper_duration,
                        )
                        window_fallback = True
                        issues.append(
                            f"Used window-only mapping for line '{line.text[:30]}...' "
                            f"(segment offset {sorted_segments[best_idx].start - desired_start:+.2f}s)"
                            if best_idx is not None and desired_start is not None
                            else f"Used window-only mapping for line '{line.text[:30]}...'"
                        )
            if not window_fallback:
                new_words = list(line.words)
            if desired_start is not None and new_words:
                shift = desired_start - new_words[0].start_time
                new_words = _shift_words(new_words, shift)
            new_words = _clamp_line_end_to_next_start(
                line, new_words, lrc_line_starts, line_idx
            )
            if last_end is not None and new_words:
                if new_words[0].start_time < last_end + gap_required:
                    shift = (last_end + gap_required) - new_words[0].start_time
                    new_words = _shift_words(new_words, shift)
            if new_words:
                adjusted.append(Line(words=new_words, singer=line.singer))
                last_end = new_words[-1].end_time
                if window_fallback:
                    fixes += 1
            else:
                adjusted.append(line)
                last_end = line.end_time
            if not window_fallback:
                if best_sim < min_similarity_fallback and best_idx is not None:
                    issues.append(
                        f"Skipped Whisper mapping for line '{line.text[:30]}...' "
                        f"(sim={best_sim:.2f})"
                    )
                if (
                    desired_start is not None
                    and best_idx is not None
                    and abs(sorted_segments[best_idx].start - desired_start)
                    > max_time_offset
                ):
                    issues.append(
                        f"Skipped Whisper mapping for line '{line.text[:30]}...' "
                        f"(segment offset {sorted_segments[best_idx].start - desired_start:+.2f}s)"
                    )
            prev_text = text_norm
            continue

        if best_idx is None:
            adjusted.append(line)
            prev_text = text_norm
            continue
        seg = sorted_segments[best_idx]
        forced_fallback = False
        if last_end is not None and seg.start < last_end + gap_required:
            next_idx: Optional[int] = None
            best_future_idx: Optional[int] = None
            best_future_sim: Optional[float] = None
            for idx in range(best_idx, window_end):
                seg_candidate = sorted_segments[idx]
                if seg_candidate.start < last_end + gap_required:
                    continue
                next_idx = idx
                sim_candidate = _phonetic_similarity(
                    line.text, seg_candidate.text, language
                )
                if best_future_sim is None or sim_candidate > best_future_sim:
                    best_future_sim = sim_candidate
                    best_future_idx = idx
            if best_future_idx is not None and best_future_sim is not None:
                if best_future_sim >= min_similarity_fallback:
                    best_idx = best_future_idx
                    seg = sorted_segments[best_idx]
                else:
                    if next_idx is None:
                        adjusted.append(line)
                        prev_text = text_norm
                        continue
                    best_idx = next_idx
                    seg = sorted_segments[best_idx]
                    forced_fallback = True
        new_words = _map_line_words_to_segment(line, seg)

        # If we have an LRC line start, redistribute timing within the line
        # using Whisper word durations as proportions, while preserving the
        # line's total duration.
        if desired_start is not None and line.words:
            new_words, desired_start = _apply_lrc_weighted_timing(
                line,
                desired_start,
                next_lrc_start,
                lrc_line_starts,
                line_idx,
                all_words,
                seg,
                language,
                _phonetic_similarity,
            )

        if desired_start is not None and new_words:
            shift = desired_start - new_words[0].start_time
            new_words = _shift_words(new_words, shift)

        # Clamp line duration so it doesn't explode far past the next line.
        new_words = _clamp_line_end_to_next_start(
            line, new_words, lrc_line_starts, line_idx
        )

        if (
            last_end is not None
            and new_words
            and new_words[0].start_time < last_end + gap_required
        ):
            shift = (last_end + gap_required) - new_words[0].start_time
            new_words = _shift_words(new_words, shift)
        adjusted.append(Line(words=new_words, singer=line.singer))
        fixes += 1
        if best_sim < min_similarity:
            issues.append(
                f"Low similarity mapping for line '{line.text[:30]}...' (sim={best_sim:.2f})"
            )
        if forced_fallback:
            issues.append(
                f"Forced forward mapping for line '{line.text[:30]}...' to avoid overlap"
            )
        last_end = new_words[-1].end_time if new_words else last_end
        seg_idx = best_idx + 1 if best_idx is not None else seg_idx
        prev_text = text_norm

    return adjusted, fixes, issues


