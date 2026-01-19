from typing import List, Tuple, Optional
from difflib import SequenceMatcher

from .models import Word, Line, SongMetadata
from .romanization import romanize_line
from .genius_utils import normalize_text

# ----------------------
# Lyrics merging
# ----------------------
def merge_lyrics_with_singer_info(
    timed_lines: List[Tuple[float, str]],
    genius_lines: List[Tuple[str, str]],
    metadata: SongMetadata,
    romanize: bool = True,
) -> List[Line]:
    if not timed_lines:
        return []

    genius_normalized = [(normalize_text(t), s) for t, s in genius_lines]
    used_genius_indices: set = set()
    lines: List[Line] = []

    for i, (start_time, text) in enumerate(timed_lines):
        line_text = romanize_line(text) if romanize else text
        end_time = timed_lines[i + 1][0] if i + 1 < len(timed_lines) else start_time + 3.0
        if i + 1 < len(timed_lines) and end_time - start_time > 10.0:
            end_time = start_time + 5.0

        # Fuzzy match to Genius line
        best_match_idx: Optional[int] = None
        best_score = 0.0
        text_norm = normalize_text(line_text)

        for j, (genius_norm, singer_name) in enumerate(genius_normalized):
            if j in used_genius_indices:
                continue
            score = SequenceMatcher(None, text_norm, genius_norm).ratio()
            if j not in used_genius_indices:
                score += 0.05
            if score > best_score and score > 0.5:
                best_score = score
                best_match_idx = j

        singer_id = ""
        if best_match_idx is not None:
            used_genius_indices.add(best_match_idx)
            _, singer_name = genius_lines[best_match_idx]
            singer_id = metadata.get_singer_id(singer_name)

        word_texts = line_text.split()
        if not word_texts:
            continue

        line_duration = end_time - start_time
        word_duration = max((line_duration * 0.95) / len(word_texts), 0.01)
        words: List[Word] = []

        for j, word_text in enumerate(word_texts):
            word_start = start_time + j * (line_duration / len(word_texts))
            word_end = word_start + word_duration
            words.append(Word(
                text=word_text,
                start_time=word_start,
                end_time=word_end,
                singer=singer_id,
            ))

        lines.append(Line(words=words, singer=singer_id))

    return lines
