"""Trailing-line extension helpers for repetition-heavy Whisper mapping."""

from __future__ import annotations

from typing import Callable, List, Tuple

from ... import models
from ..alignment import timing_models


def _extend_line_to_trailing_whisper_matches(  # noqa: C901
    mapped_lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
    *,
    normalize_match_token_fn: Callable[[str], str],
    soft_token_match_fn: Callable[[str, str], bool],
    smooth_adjacent_duplicate_line_cadence_fn: Callable[
        [List[models.Line]], List[models.Line]
    ],
    rebalance_short_question_pairs_fn: Callable[[List[models.Line]], List[models.Line]],
    min_extension: float = 0.35,
    min_gap: float = 0.05,
) -> List[models.Line]:
    """Extend line tails when repeated-word ambiguity caused premature completion."""
    if not mapped_lines or not all_words:
        return mapped_lines

    adjusted = list(mapped_lines)
    for idx, line in enumerate(adjusted):
        if not line.words:
            continue

        token_pairs = [
            (word_idx, normalize_match_token_fn(w.text))
            for word_idx, w in enumerate(line.words)
        ]
        token_pairs = [(word_idx, tok) for word_idx, tok in token_pairs if tok]
        if len(token_pairs) < 2:
            continue

        next_start = (
            adjusted[idx + 1].start_time
            if idx + 1 < len(adjusted) and adjusted[idx + 1].words
            else float("inf")
        )
        window_end = (
            next_start + 3.0 if next_start != float("inf") else line.end_time + 3.0
        )
        candidates = [
            w for w in all_words if (line.start_time - 0.25) <= w.start <= window_end
        ]
        if not candidates:
            continue

        best_end = line.end_time
        best_match_count = 0
        best_pairs: List[Tuple[int, timing_models.TranscriptionWord]] = []
        for start_i in range(len(candidates)):
            wi = start_i
            matched = 0
            last_end = None
            matched_pairs: List[Tuple[int, timing_models.TranscriptionWord]] = []
            for word_idx, tok in token_pairs:
                found = False
                while wi < len(candidates):
                    ww_tok = normalize_match_token_fn(candidates[wi].text)
                    if soft_token_match_fn(tok, ww_tok):
                        matched += 1
                        last_end = candidates[wi].end
                        matched_pairs.append((word_idx, candidates[wi]))
                        wi += 1
                        found = True
                        break
                    wi += 1
                if wi >= len(candidates):
                    break
                if not found:
                    continue

            if last_end is None:
                continue
            min_required = max(2, int(len(token_pairs) * 0.66))
            if matched < min_required:
                continue
            if (matched > best_match_count) or (
                matched == best_match_count and last_end > best_end
            ):
                best_match_count = matched
                best_end = last_end
                best_pairs = matched_pairs

        target_end = best_end
        if next_start != float("inf"):
            target_end = min(target_end, next_start - min_gap)
        if target_end <= line.end_time + min_extension:
            continue

        new_words = list(line.words)
        if best_pairs:
            anchor_times = {word_idx: ww for word_idx, ww in best_pairs}
            first_anchor_idx = min(anchor_times)
            last_anchor_idx = max(anchor_times)
            for word_idx, ww in anchor_times.items():
                new_words[word_idx] = models.Word(
                    text=new_words[word_idx].text,
                    start_time=ww.start,
                    end_time=ww.end,
                    singer=new_words[word_idx].singer,
                )

            for word_idx in range(first_anchor_idx - 1, -1, -1):
                right = new_words[word_idx + 1].start_time
                duration = max(
                    0.12,
                    new_words[word_idx].end_time - new_words[word_idx].start_time,
                )
                end = max(line.start_time, right - 0.02)
                start = max(line.start_time, end - duration)
                new_words[word_idx] = models.Word(
                    text=new_words[word_idx].text,
                    start_time=start,
                    end_time=end,
                    singer=new_words[word_idx].singer,
                )

            for word_idx in range(last_anchor_idx + 1, len(new_words)):
                left = new_words[word_idx - 1].end_time
                end_cap = target_end if word_idx == len(new_words) - 1 else left + 0.28
                duration = max(0.12, min(0.25, end_cap - left - 0.02))
                start = max(left + 0.02, end_cap - duration)
                end = max(start + 0.08, end_cap)
                new_words[word_idx] = models.Word(
                    text=new_words[word_idx].text,
                    start_time=start,
                    end_time=end,
                    singer=new_words[word_idx].singer,
                )
        else:
            start = line.start_time
            duration = max(target_end - start, 0.2)
            spacing = duration / len(line.words)
            rebuilt_words: List[models.Word] = []
            for word_idx, w in enumerate(line.words):
                ws = start + word_idx * spacing
                we = ws + spacing * 0.9
                if word_idx == len(line.words) - 1:
                    we = target_end
                rebuilt_words.append(
                    models.Word(
                        text=w.text,
                        start_time=ws,
                        end_time=we,
                        singer=w.singer,
                    )
                )
            new_words = rebuilt_words

        if new_words[-1].end_time < target_end:
            last = new_words[-1]
            new_words[-1] = models.Word(
                text=last.text,
                start_time=min(last.start_time, target_end - 0.08),
                end_time=target_end,
                singer=last.singer,
            )

        for word_idx in range(1, len(new_words)):
            prev = new_words[word_idx - 1]
            cur = new_words[word_idx]
            if cur.start_time <= prev.end_time:
                shift = prev.end_time + 0.01 - cur.start_time
                new_words[word_idx] = models.Word(
                    text=cur.text,
                    start_time=cur.start_time + shift,
                    end_time=max(cur.end_time + shift, cur.start_time + shift + 0.06),
                    singer=cur.singer,
                )

        if next_start != float("inf") and new_words[-1].end_time >= next_start:
            last = new_words[-1]
            clipped_end = max(last.start_time + 0.06, next_start - min_gap)
            new_words[-1] = models.Word(
                text=last.text,
                start_time=last.start_time,
                end_time=clipped_end,
                singer=last.singer,
            )

        adjusted[idx] = models.Line(words=new_words, singer=line.singer)

    adjusted = smooth_adjacent_duplicate_line_cadence_fn(adjusted)
    return rebalance_short_question_pairs_fn(adjusted)
