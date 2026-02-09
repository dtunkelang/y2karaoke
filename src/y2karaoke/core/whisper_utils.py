"""Utility functions for Whisper integration."""

from typing import List, Any, Tuple, Dict, Set
from .timing_models import TranscriptionWord

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
