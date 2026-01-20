"""
Lyrics merging utility: align timed lines (LRC or YouTube) with Genius canonical text and singer info.
"""

from typing import List, Tuple, Optional, Set
from difflib import SequenceMatcher
import logging

from .models import Line, Word, SongMetadata, SingerID
from .romanization import romanize_line

logger = logging.getLogger(__name__)

# ----------------------
# Helper: fuzzy match a single LRC line to Genius lines
# ----------------------
def _fuzzy_match_line(
    lrc_text: str,
    genius_lines: List[str],
    used_indices: Set[int],
    min_score: float = 0.5
) -> Optional[Tuple[int, str]]:
    """
    Find the best matching Genius line for a given LRC line using fuzzy matching.
    
    Returns:
        Tuple of (index, genius_text) if match found, else None
    """
    best_index = None
    best_text = None
    best_score = 0.0

    lrc_norm = lrc_text.lower().strip()

    for i, genius_text in enumerate(genius_lines):
        if i in used_indices:
            continue
        genius_norm = genius_text.lower().strip()
        score = SequenceMatcher(None, lrc_norm, genius_norm).ratio()
        if score > best_score:
            best_score = score
            best_index = i
            best_text = genius_text

    if best_score >= min_score:
        return best_index, best_text
    return None

# ----------------------
# Helper: create Word objects evenly spaced in time
# ----------------------
def _create_words_from_line(
    text: str,
    start_time: float,
    end_time: float,
    singer: Optional[SingerID] = None,
) -> List[Word]:
    """
    Split a line into words, assign start/end times evenly, and assign singer if given.
    """
    word_texts = text.split()
    if not word_texts:
        return []

    line_duration = end_time - start_time
    word_duration = (line_duration * 0.95) / len(word_texts)

    words = []
    for j, word_text in enumerate(word_texts):
        word_start = start_time + j * (line_duration / len(word_texts))
        word_end = word_start + word_duration
        words.append(Word(
            text=word_text,
            start_time=word_start,
            end_time=word_end,
            singer=singer
        ))
    return words

# ----------------------
# Main merge function
# ----------------------
def merge_lyrics_with_singer_info(
    timed_lines: List[Tuple[float, str]],
    genius_lines: List[Tuple[str, Optional[str]]],
    metadata: Optional[SongMetadata],
    romanize: bool = True,
) -> List[Line]:
    """
    Merge timed lines (from LRC or YouTube) with Genius canonical lyrics and singer info.
    
    Args:
        timed_lines: List of (timestamp, text)
        genius_lines: List of (text, singer_name)
        metadata: SongMetadata with singer info
        romanize: Whether to romanize non-ASCII text

    Returns:
        List of Line objects with word timing and singer info
    """
    if not timed_lines:
        # fallback: create evenly spaced lines from Genius only
        lines = []
        for i, (text, singer_name) in enumerate(genius_lines):
            start = i * 3.0
            end = start + 3.0
            singer_id = metadata.get_singer_id(singer_name) if (metadata and singer_name) else None
            words = _create_words_from_line(text, start, end, singer_id)
            if romanize:
                for word in words:
                    if any(ord(c) > 127 for c in word.text):
                        word.text = romanize_line(word.text)
            lines.append(Line(words=words, singer=singer_id))
        return lines

    genius_texts = [text for text, _ in genius_lines]
    used_genius = set()
    result_lines: List[Line] = []

    for i, (start_time, lrc_text) in enumerate(timed_lines):
        # Determine end_time
        if i + 1 < len(timed_lines):
            end_time = timed_lines[i + 1][0]
            if end_time - start_time > 10.0:
                end_time = start_time + 5.0
        else:
            end_time = start_time + 3.0

        match = _fuzzy_match_line(lrc_text, genius_texts, used_genius)
        if match:
            genius_index, line_text = match
            used_genius.add(genius_index)
            singer_name = genius_lines[genius_index][1]
        else:
            line_text = lrc_text
            singer_name = None

        singer_id = metadata.get_singer_id(singer_name) if (metadata and singer_name) else None

        if romanize:
            line_text = " ".join([romanize_line(w) if any(ord(c) > 127 for c in w) else w for w in line_text.split()])

        words = _create_words_from_line(line_text, start_time, end_time, singer_id)

        # skip duplicate line text
        line_text_str = " ".join([w.text for w in words]).strip()
        if result_lines and " ".join([w.text for w in result_lines[-1].words]).strip() == line_text_str:
            continue

        result_lines.append(Line(words=words, singer=singer_id))

    return result_lines
