"""Utility functions for Whisper integration."""

from typing import List, Any, Tuple, Dict, Set, Optional
from .timing_models import TranscriptionWord
from . import models

_SPEECH_BLOCK_GAP = 5.0  # seconds - gap between Whisper words that defines a new block


def _normalize_word(w: str) -> str:
    """Lowercase and strip punctuation for bag-of-words comparison."""
    return w.strip(".,!?;:'\"()- ").lower()


def _normalize_words_expanded(w: str) -> List[str]:
    """Normalize and split hyphenated/compound words for overlap matching."""
    base = w.strip(".,!?;:'\"()- ").lower()
    if not base:
        return []
    parts = [p for p in base.split("-") if p]
    return parts if len(parts) > 1 else [base]


def _segment_start(segment: Any) -> float:
    """Helper to get start time from either Dict or TranscriptionSegment."""
    if hasattr(segment, "start"):
        return float(segment.start)
    if isinstance(segment, dict):
        return float(segment.get("start", 0.0))
    return 0.0


def _segment_end(segment: Any) -> float:
    """Helper to get end time from either Dict or TranscriptionSegment."""
    if hasattr(segment, "end"):
        return float(segment.end)
    if isinstance(segment, dict):
        return float(segment.get("end", 0.0))
    return 0.0


def _get_segment_text(segment: Any) -> str:
    """Helper to get text from either Dict or TranscriptionSegment."""
    if hasattr(segment, "text"):
        return str(segment.text)
    if isinstance(segment, dict):
        return str(segment.get("text", ""))
    return ""


def _compute_speech_blocks(
    all_words: List[TranscriptionWord],
    min_gap: float = _SPEECH_BLOCK_GAP,
) -> List[Tuple[int, int]]:
    """Group Whisper words into speech blocks separated by gaps >= *min_gap*."""
    if not all_words:
        return []
    blocks: List[Tuple[int, int]] = []
    block_start = 0
    for i in range(1, len(all_words)):
        gap = all_words[i].start - all_words[i - 1].end
        if gap >= min_gap:
            blocks.append((block_start, i - 1))
            block_start = i
    blocks.append((block_start, len(all_words) - 1))
    return blocks


def _word_idx_to_block(word_idx: int, speech_blocks: List[Tuple[int, int]]) -> int:
    """Return the speech-block index that contains *word_idx*, or -1."""
    for i, (start, end) in enumerate(speech_blocks):
        if start <= word_idx <= end:
            return i
    return -1


def _block_time_range(
    block_idx: int,
    speech_blocks: List[Tuple[int, int]],
    all_words: List[TranscriptionWord],
) -> Tuple[float, float]:
    """Return (start_time, end_time) for the given speech block."""
    first, last = speech_blocks[block_idx]
    return all_words[first].start, all_words[last].end


def _redistribute_word_timings_to_line(
    line: "models.Line",
    matches: List[Tuple[int, Tuple[float, float]]],
    target_duration: float,
    min_word_duration: float,
    line_start: Optional[float] = None,
    max_gap: float = 0.4,
) -> "models.Line":
    """Redistribute word timings based on Whisper durations within the line."""
    from .whisper_alignment import _scale_line_to_duration
    from . import models

    if not line.words:
        return line

    min_word_duration = max(min_word_duration, 0.0)
    match_map = {
        word_idx: (start, end)
        for word_idx, (start, end) in matches
        if start is not None and end is not None
    }

    max_line_duration = min(max(4.0, len(line.words) * 0.6), 8.0)
    target_duration = min(target_duration, max_line_duration)
    min_weight = max(min_word_duration, 0.01)
    weights: List[float] = []
    for word_idx in range(len(line.words)):
        interval = match_map.get(word_idx)
        if interval:
            start, end = interval
            weight = max(end - start, min_weight)
        else:
            weight = min_weight
        weights.append(weight)

    total_weight = sum(weights) or 1.0
    durations = [
        (weights[i] / total_weight) * target_duration for i in range(len(line.words))
    ]
    duration_sum = sum(durations)
    if durations:
        durations[-1] += target_duration - duration_sum

    max_word_duration = min(3.0, target_duration * 0.5)
    durations = _cap_word_durations(durations, target_duration, max_word_duration)

    current = line_start if line_start is not None else line.start_time
    new_words: List[models.Word] = []
    for idx, word in enumerate(line.words):
        duration = durations[idx]
        start_time = current
        end_time = start_time + duration
        if idx == len(line.words) - 1:
            end_time = line.start_time + target_duration
            if end_time <= start_time:
                end_time = start_time + max(min_word_duration, 0.01)
        new_words.append(
            models.Word(
                text=word.text,
                start_time=start_time,
                end_time=end_time,
                singer=word.singer,
            )
        )
        current = end_time

    adjusted_words = _clamp_word_gaps(new_words, max_gap)
    clamped_line = models.Line(words=adjusted_words, singer=line.singer)
    scaled_line = _scale_line_to_duration(
        clamped_line,
        target_duration=target_duration,
    )
    target_start = line_start if line_start is not None else scaled_line.start_time
    offset = target_start - scaled_line.start_time
    adjusted_scaled_words = [
        models.Word(
            text=w.text,
            start_time=w.start_time + offset,
            end_time=w.end_time + offset,
            singer=w.singer,
        )
        for w in scaled_line.words
    ]
    return models.Line(words=adjusted_scaled_words, singer=line.singer)


def _clamp_word_gaps(words: List["models.Word"], max_gap: float) -> List["models.Word"]:
    from . import models

    if not words or max_gap is None:
        return words
    adjusted: List[models.Word] = []
    total_shift = 0.0
    adjusted.append(
        models.Word(
            text=words[0].text,
            start_time=words[0].start_time,
            end_time=words[0].end_time,
            singer=words[0].singer,
        )
    )
    for current in words[1:]:
        prev = adjusted[-1]
        shifted_start = current.start_time - total_shift
        gap = shifted_start - prev.end_time
        shift = 0.0
        if gap > max_gap:
            shift = gap - max_gap
        total_shift += shift
        duration = current.end_time - current.start_time
        new_start = shifted_start - shift
        new_end = new_start + duration
        adjusted.append(
            models.Word(
                text=current.text,
                start_time=new_start,
                end_time=new_end,
                singer=current.singer,
            )
        )
    return adjusted


def _cap_word_durations(
    durations: List[float], total_duration: float, max_word_duration: float
) -> List[float]:
    if not durations:
        return durations
    capped = []
    remainder = total_duration
    for duration in durations:
        limit = min(max_word_duration, remainder - 0.01 * len(durations))
        capped_value = min(duration, limit)
        capped.append(capped_value)
        remainder -= capped_value
    if remainder > 0 and capped:
        capped[-1] += remainder
    return capped


def _build_word_assignments_from_syllable_path(
    path: List[Tuple[int, int]],
    lrc_syllables: List[Dict],
    whisper_syllables: List[Dict],
) -> Dict[int, List[int]]:
    assignments: Dict[int, Set[int]] = {}
    for lrc_idx, whisper_idx in path:
        lrc_token = lrc_syllables[lrc_idx]
        whisper_token = whisper_syllables[whisper_idx]
        for word_idx in lrc_token["word_idxs"]:
            assignments.setdefault(word_idx, set()).update(whisper_token["parent_idxs"])
    return {
        word_idx: sorted(list(indices)) for word_idx, indices in assignments.items()
    }
