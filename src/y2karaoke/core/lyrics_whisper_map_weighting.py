"""Timing weighting logic for LRC-to-Whisper mapping."""

from statistics import median
from typing import Callable, List, Optional, Tuple

from .models import Line, Word
from .timing_models import TranscriptionSegment, TranscriptionWord
from .lyrics_whisper_map_helpers import (
    _line_duration_from_lrc,
    _norm_token,
    _resample_slots_to_line,
    _select_window_words_for_line,
    _slots_from_sequence,
)


def _whisper_durations_for_line(  # noqa: C901
    line_words: List[Word],
    seg_words: Optional[List[TranscriptionWord]],
    seg_end: Optional[float],
    language: str,
    phonetic_similarity: Callable[[str, str, str], float],
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


def _apply_lrc_weighted_timing(  # noqa: C901
    line: Line,
    desired_start: float,
    next_lrc_start: Optional[float],
    lrc_line_starts: Optional[List[float]],
    line_idx: int,
    all_words: List[TranscriptionWord],
    seg: Optional[TranscriptionSegment],
    language: str,
    phonetic_similarity: Callable[[str, str, str], float],
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
