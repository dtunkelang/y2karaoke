"""Helper utilities for mapping LRC lines onto Whisper timings."""

import re
from statistics import median
from typing import Callable, List, Optional, Tuple

from .models import Line, Word
from .timing_models import TranscriptionSegment, TranscriptionWord


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
    phonetic_similarity: Callable[[str, str, str], float],
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
    phonetic_similarity: Callable[[str, str, str], float],
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
    phonetic_similarity: Callable[[str, str, str], float],
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
