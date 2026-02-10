"""Post-processing helpers for Whisper mapping output."""

from typing import Dict, List, Optional, Set, Tuple

from ... import models
from ..alignment import timing_models


def _build_word_assignments_from_phoneme_path(
    path: List[Tuple[int, int]],
    lrc_phonemes: List[Dict],
    whisper_phonemes: List[Dict],
) -> Dict[int, List[int]]:
    """Convert phoneme-level DTW matches back to word-level assignments."""
    assignments: Dict[int, Set[int]] = {}
    for lpc_idx, wpc_idx in path:
        lrc_token = lrc_phonemes[lpc_idx]
        whisper_token = whisper_phonemes[wpc_idx]
        word_idx = lrc_token["word_idx"]
        whisper_word_idx = whisper_token["parent_idx"]
        assignments.setdefault(word_idx, set()).add(whisper_word_idx)
    return {
        word_idx: sorted(list(indices)) for word_idx, indices in assignments.items()
    }


def _shift_repeated_lines_to_next_whisper(
    mapped_lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
) -> List[models.Line]:
    """Ensure repeated lines reserve later Whisper words when they reappear."""
    adjusted_lines: List[models.Line] = []
    last_idx_by_text: Dict[str, int] = {}
    last_end_time: Dict[str, float] = {}

    for line in mapped_lines:
        if not line.words:
            adjusted_lines.append(line)
            continue

        text_norm = line.text.strip().lower() if getattr(line, "text", "") else ""
        prev_idx = last_idx_by_text.get(text_norm)
        prev_end = last_end_time.get(text_norm)
        assigned_end_idx: Optional[int] = None

        if prev_idx is not None and prev_end is not None:
            required_time = max(prev_end + 0.4, line.start_time)
            start_idx = next(
                (
                    idx
                    for idx, ww in enumerate(all_words)
                    if idx > prev_idx and ww.start >= required_time
                ),
                None,
            )
            if start_idx is None:
                start_idx = next(
                    (idx for idx, _ww in enumerate(all_words) if idx > prev_idx),
                    None,
                )
            if start_idx is not None and (all_words[start_idx].start - prev_end > 10.0):
                start_idx = None
            if start_idx is not None:
                adjusted_words: List[models.Word] = []
                for word_idx, w in enumerate(line.words):
                    new_idx = min(start_idx + word_idx, len(all_words) - 1)
                    ww = all_words[new_idx]
                    adjusted_words.append(
                        models.Word(
                            text=w.text,
                            start_time=ww.start,
                            end_time=ww.end,
                            singer=w.singer,
                        )
                    )
                line = models.Line(words=adjusted_words, singer=line.singer)
                assigned_end_idx = min(
                    start_idx + len(line.words) - 1, len(all_words) - 1
                )

        adjusted_lines.append(line)
        if line.words:
            if assigned_end_idx is None:
                assigned_end_idx = next(
                    (
                        idx
                        for idx, ww in enumerate(all_words)
                        if abs(ww.start - line.words[-1].start_time) < 0.05
                    ),
                    len(all_words) - 1,
                )
            last_idx_by_text[text_norm] = assigned_end_idx
            last_end_time[text_norm] = line.end_time

    return adjusted_lines


def _enforce_monotonic_line_starts_whisper(
    mapped_lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
) -> List[models.Line]:
    """Ensure line starts are monotonic by shifting backwards lines forward."""
    prev_start = None
    prev_end = None
    monotonic_lines: List[models.Line] = []
    for line in mapped_lines:
        if not line.words:
            monotonic_lines.append(line)
            continue

        if prev_start is not None and line.start_time < prev_start:
            required_time = (prev_end or line.start_time) + 0.01
            start_idx = next(
                (idx for idx, ww in enumerate(all_words) if ww.start >= required_time),
                None,
            )
            if start_idx is not None and (
                all_words[start_idx].start - required_time <= 10.0
            ):
                adjusted_words_2: List[models.Word] = []
                for word_idx, w in enumerate(line.words):
                    new_idx = min(start_idx + word_idx, len(all_words) - 1)
                    ww = all_words[new_idx]
                    adjusted_words_2.append(
                        models.Word(
                            text=w.text,
                            start_time=ww.start,
                            end_time=ww.end,
                            singer=w.singer,
                        )
                    )
                line = models.Line(words=adjusted_words_2, singer=line.singer)
            else:
                shift = required_time - line.start_time
                shifted_words: List[models.Word] = [
                    models.Word(
                        text=w.text,
                        start_time=w.start_time + shift,
                        end_time=w.end_time + shift,
                        singer=w.singer,
                    )
                    for w in line.words
                ]
                line = models.Line(words=shifted_words, singer=line.singer)

        monotonic_lines.append(line)
        if line.words:
            prev_start = line.start_time
            prev_end = line.end_time

    return monotonic_lines


def _resolve_line_overlaps(lines: List[models.Line]) -> List[models.Line]:
    """Ensure consecutive lines never overlap in time."""
    from ...models import Line as LineModel

    resolved: List[models.Line] = list(lines)
    for i in range(len(resolved) - 1):
        cur = resolved[i]
        nxt = resolved[i + 1]
        if not cur.words or not nxt.words:
            continue
        if cur.end_time > nxt.start_time:
            gap_point = nxt.start_time
            new_words: List[models.Word] = []
            for w in cur.words:
                if w.start_time >= gap_point:
                    new_words.append(
                        models.Word(
                            text=w.text,
                            start_time=gap_point - 0.01,
                            end_time=gap_point,
                            singer=w.singer,
                        )
                    )
                elif w.end_time > gap_point:
                    new_words.append(
                        models.Word(
                            text=w.text,
                            start_time=w.start_time,
                            end_time=gap_point,
                            singer=w.singer,
                        )
                    )
                else:
                    new_words.append(w)
            resolved[i] = LineModel(words=new_words, singer=cur.singer)
    return resolved
