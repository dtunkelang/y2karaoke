"""Word-level timing using audio analysis.

This module scores and fixes word timing, and provides fallback onset/energy-based estimation.
"""

from dataclasses import dataclass
from typing import List

from ..utils.logging import get_logger
from .models import Word, Line

logger = get_logger(__name__)


# ----------------------
# Quality scoring
# ----------------------
@dataclass
class TimingQuality:
    """Quality metrics for timing."""
    score: float  # 0.0 (bad) to 1.0 (good)
    short_words: int
    overlapping_words: int
    large_gaps: int
    out_of_order: int
    issues: List[str]


def score_timing_quality(
    lines: List[Line],
    min_word_duration: float = 0.08,
    max_gap: float = 1.0,
) -> TimingQuality:
    """Score the quality of word timing."""
    issues = []
    short_words = 0
    overlapping_words = 0
    large_gaps = 0
    out_of_order = 0
    total_words = 0
    prev_end = 0.0

    for line in lines:
        for word in line.words:
            total_words += 1
            duration = word.end_time - word.start_time

            if duration < min_word_duration:
                short_words += 1
                if short_words <= 3:
                    issues.append(f"Short word: '{word.text}' ({duration:.2f}s) at {word.start_time:.2f}s")

            if word.start_time < prev_end - 0.01:
                overlapping_words += 1
                if overlapping_words <= 3:
                    issues.append(f"Overlap: '{word.text}' starts at {word.start_time:.2f}s, prev ended at {prev_end:.2f}s")

            gap = word.start_time - prev_end
            if gap > max_gap and prev_end > 0:
                large_gaps += 1
                if large_gaps <= 3:
                    issues.append(f"Large gap: {gap:.2f}s before '{word.text}'")

            if word.start_time < prev_end - 0.5:
                out_of_order += 1

            prev_end = word.end_time

    if total_words == 0:
        return TimingQuality(0.0, 0, 0, 0, 0, ["No words found"])

    penalty = 0.0
    penalty += min(0.3, short_words / total_words)
    penalty += min(0.3, overlapping_words / total_words * 2)
    penalty += min(0.2, large_gaps / max(1, len(lines)) * 0.1)
    penalty += min(0.2, out_of_order / total_words * 5)
    score = max(0.0, 1.0 - penalty)

    return TimingQuality(score, short_words, overlapping_words, large_gaps, out_of_order, issues)


# ----------------------
# Fix timing issues
# ----------------------
def fix_timing_issues(
    lines: List[Line],
    min_word_duration: float = 0.08,
) -> List[Line]:
    """Fix short words, overlapping words, and ordering."""
    fixed_lines = []

    for line in lines:
        if not line.words:
            fixed_lines.append(line)
            continue

        fixed_words = []
        prev_end = line.start_time

        for i, word in enumerate(line.words):
            new_start = max(word.start_time, prev_end)
            new_end = word.end_time

            if new_end - new_start < min_word_duration:
                new_end = new_start + min_word_duration

            if i < len(line.words) - 1:
                new_end = min(new_end, line.end_time)

            fixed_words.append(Word(
                text=word.text,
                start_time=new_start,
                end_time=new_end,
                singer=word.singer,
            ))
            prev_end = new_end

        fixed_lines.append(Line(words=fixed_words, singer=line.singer))

    return fixed_lines


