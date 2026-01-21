"""Lyrics utility helpers."""

from typing import List, Tuple
from difflib import SequenceMatcher
from .models import Line, Word

def _create_lines_from_lrc_timings(
    lrc_timings: List[Tuple[float, str]],
    genius_lines: List[str],
) -> List[Line]:
    """Create Line objects from LRC timings + Genius canonical text."""
    lines: List[Line] = []
    used_genius = set()

    for i, (start_time, lrc_text) in enumerate(lrc_timings):
        if i + 1 < len(lrc_timings):
            end_time = lrc_timings[i + 1][0]
            if end_time - start_time > 10.0:
                end_time = start_time + 5.0
        else:
            end_time = start_time + 3.0

        best_match = None
        best_score = 0.0
        lrc_normalized = lrc_text.lower().strip()
        for j, genius_text in enumerate(genius_lines):
            if j in used_genius:
                continue
            genius_normalized = genius_text.lower().strip()
            score = SequenceMatcher(None, lrc_normalized, genius_normalized).ratio()
            if score > best_score:
                best_score = score
                best_match = (j, genius_text)

        line_text = best_match[1] if best_match and best_score > 0.5 else lrc_text
        if best_match:
            used_genius.add(best_match[0])

        word_texts = line_text.split()
        if not word_texts:
            continue

        line_duration = end_time - start_time
        word_count = len(word_texts)
        word_duration = (line_duration * 0.95) / word_count

        words = []
        for j, word_text in enumerate(word_texts):
            word_start = start_time + j * (line_duration / word_count)
            word_end = word_start + word_duration
            words.append(Word(text=word_text, start_time=word_start, end_time=word_end))

        line_text_str = " ".join([w.text for w in words]).strip()
        if lines and " ".join([w.text for w in lines[-1].words]).strip() == line_text_str:
            continue

        lines.append(Line(words=words))

    return lines
