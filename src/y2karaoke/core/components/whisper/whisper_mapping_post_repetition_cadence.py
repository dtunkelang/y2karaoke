"""Cadence and spacing postpasses for repetitive Whisper mapping lines."""

from typing import Callable, List

from ... import models


def pull_adjacent_similar_lines_across_long_gaps(
    lines: List[models.Line],
    *,
    normalize_match_token_fn: Callable[[str], str],
    soft_token_overlap_ratio_fn: Callable[[List[str], List[str]], float],
    min_gap: float = 0.05,
    long_gap_threshold: float = 8.0,
    min_similarity: float = 0.5,
) -> List[models.Line]:
    """Pull a line forward when a near-duplicate follows after a long gap."""
    if not lines:
        return lines
    adjusted = list(lines)
    for idx in range(len(adjusted) - 1):
        cur = adjusted[idx]
        nxt = adjusted[idx + 1]
        if not cur.words or not nxt.words:
            continue
        gap = nxt.start_time - cur.end_time
        if gap < long_gap_threshold:
            continue
        cur_tokens = [
            normalize_match_token_fn(w.text)
            for w in cur.words
            if normalize_match_token_fn(w.text)
        ]
        nxt_tokens = [
            normalize_match_token_fn(w.text)
            for w in nxt.words
            if normalize_match_token_fn(w.text)
        ]
        if not cur_tokens or not nxt_tokens:
            continue
        if soft_token_overlap_ratio_fn(cur_tokens, nxt_tokens) < min_similarity:
            continue
        if len(cur.words) > 8 or len(nxt.words) > 8:
            continue
        prev_end = (
            adjusted[idx - 1].end_time if idx > 0 and adjusted[idx - 1].words else None
        )
        duration = max(0.7, min(cur.end_time - cur.start_time, 2.5))
        new_end = nxt.start_time - min_gap
        new_start = new_end - duration
        if prev_end is not None:
            new_start = max(new_start, prev_end + min_gap)
            new_end = max(new_end, new_start + 0.4)
        if new_start <= cur.start_time + 0.5:
            continue
        shift = new_start - cur.start_time
        shifted_words = [
            models.Word(
                text=w.text,
                start_time=w.start_time + shift,
                end_time=w.end_time + shift,
                singer=w.singer,
            )
            for w in cur.words
        ]
        adjusted[idx] = models.Line(words=shifted_words, singer=cur.singer)
    return adjusted


def smooth_adjacent_duplicate_line_cadence(  # noqa: C901
    lines: List[models.Line],
    *,
    normalize_match_token_fn: Callable[[str], str],
    soft_token_overlap_ratio_fn: Callable[[List[str], List[str]], float],
    min_gap: float = 0.05,
    target_gap: float = 0.18,
    min_adjust_gap: float = 0.75,
    max_adjust_gap: float = 4.0,
    max_pull: float = 2.5,
) -> List[models.Line]:
    if not lines:
        return lines
    adjusted = list(lines)
    for idx in range(1, len(adjusted)):
        prev = adjusted[idx - 1]
        cur = adjusted[idx]
        if not prev.words or not cur.words:
            continue
        prev_text = prev.text.strip().lower()
        cur_text = cur.text.strip().lower()
        if not prev_text or not cur_text:
            continue
        prev_tokens = [
            normalize_match_token_fn(w.text)
            for w in prev.words
            if normalize_match_token_fn(w.text)
        ]
        cur_tokens = [
            normalize_match_token_fn(w.text)
            for w in cur.words
            if normalize_match_token_fn(w.text)
        ]
        overlap = soft_token_overlap_ratio_fn(prev_tokens, cur_tokens)
        exact_duplicate = prev_text == cur_text
        question_pair = (
            prev_text.endswith("?")
            and cur_text.endswith("?")
            and len(prev.words) <= 3
            and len(cur.words) <= 3
        )
        similar_refrain = overlap >= 0.5 and min(len(prev_tokens), len(cur_tokens)) >= 2
        short_similar_refrain = (
            similar_refrain
            and len(cur.words) <= 4
            and len(prev.words) <= 4
            and overlap >= 0.75
        )
        if not (exact_duplicate or similar_refrain or question_pair):
            continue
        gap = cur.start_time - prev.end_time
        if (
            exact_duplicate
            and len(cur.words) <= 3
            and cur_text.endswith("?")
            and gap >= 1.3
        ):
            continue
        if question_pair:
            min_gap_for_adjust = 0.0
        elif exact_duplicate:
            min_gap_for_adjust = min_adjust_gap
        elif short_similar_refrain:
            min_gap_for_adjust = min_adjust_gap
        elif len(cur.words) >= 5:
            min_gap_for_adjust = 0.15
        else:
            min_gap_for_adjust = 1.9
        if gap < min_gap_for_adjust or gap > max_adjust_gap:
            continue
        repetitive_tokens = len(set(cur_tokens)) <= max(3, len(cur_tokens) // 2)
        max_duration = 1.8 if exact_duplicate else 2.2
        duration = max(0.25, min(cur.end_time - cur.start_time, max_duration))
        next_start = (
            adjusted[idx + 1].start_time
            if idx + 1 < len(adjusted) and adjusted[idx + 1].words
            else None
        )
        desired_gap = (
            0.9
            if question_pair
            else (target_gap if (exact_duplicate or similar_refrain) else 1.2)
        )
        pull_limit = (
            3.2
            if question_pair
            else (max_pull if (exact_duplicate or similar_refrain) else 2.8)
        )
        target_start = prev.end_time + desired_gap
        target_start = max(target_start, cur.start_time - pull_limit)
        if next_start is not None:
            target_start = min(
                target_start,
                max(prev.end_time + min_gap, next_start - min_gap - duration),
            )
        if target_start >= cur.start_time - 0.2:
            continue
        if exact_duplicate and (cur.end_time - cur.start_time) > 1.8:
            target_duration = min(cur.end_time - target_start, 1.6)
            if next_start is not None:
                target_duration = min(
                    target_duration, next_start - min_gap - target_start
                )
            target_duration = max(0.8, target_duration)
            spacing = target_duration / len(cur.words)
            rebuilt_words_exact: List[models.Word] = []
            for word_idx, w in enumerate(cur.words):
                ws = target_start + word_idx * spacing
                we = ws + spacing * 0.9
                if word_idx == len(cur.words) - 1:
                    we = target_start + target_duration
                rebuilt_words_exact.append(
                    models.Word(
                        text=w.text,
                        start_time=ws,
                        end_time=we,
                        singer=w.singer,
                    )
                )
            adjusted[idx] = models.Line(words=rebuilt_words_exact, singer=cur.singer)
        elif (similar_refrain and not exact_duplicate) or question_pair:
            if repetitive_tokens and len(cur.words) >= 5:
                similar_max_duration = 1.7
            elif short_similar_refrain:
                similar_max_duration = 1.6
            else:
                similar_max_duration = 2.2
            target_duration = min(cur.end_time - target_start, similar_max_duration)
            if next_start is not None:
                target_duration = min(
                    target_duration, next_start - min_gap - target_start
                )
            target_duration = max(1.0, target_duration)
            spacing = target_duration / len(cur.words)
            rebuilt_words_refrain: List[models.Word] = []
            for word_idx, w in enumerate(cur.words):
                ws = target_start + word_idx * spacing
                we = ws + spacing * 0.9
                if word_idx == len(cur.words) - 1:
                    we = target_start + target_duration
                rebuilt_words_refrain.append(
                    models.Word(
                        text=w.text,
                        start_time=ws,
                        end_time=we,
                        singer=w.singer,
                    )
                )
            adjusted[idx] = models.Line(words=rebuilt_words_refrain, singer=cur.singer)
        else:
            shift = target_start - cur.start_time
            shifted_words = [
                models.Word(
                    text=w.text,
                    start_time=w.start_time + shift,
                    end_time=w.end_time + shift,
                    singer=w.singer,
                )
                for w in cur.words
            ]
            adjusted[idx] = models.Line(words=shifted_words, singer=cur.singer)
    return adjusted


def rebalance_short_question_pairs(
    lines: List[models.Line],
    *,
    normalize_match_token_fn: Callable[[str], str],
    soft_token_match_fn: Callable[[str, str], bool],
    min_gap: float = 0.05,
) -> List[models.Line]:
    """Rebalance collapsed adjacent short question lines into separate windows."""
    if not lines:
        return lines
    adjusted = list(lines)
    for idx in range(len(adjusted) - 1):
        prev = adjusted[idx]
        cur = adjusted[idx + 1]
        if not prev.words or not cur.words:
            continue
        if len(prev.words) > 4 or len(cur.words) > 4:
            continue
        prev_text = prev.text.strip()
        cur_text = cur.text.strip()
        if not prev_text.endswith("?") or not cur_text.endswith("?"):
            continue
        prev_first = normalize_match_token_fn(prev.words[0].text)
        cur_first = normalize_match_token_fn(cur.words[0].text)
        if not soft_token_match_fn(prev_first, cur_first):
            continue

        prev_dur = prev.end_time - prev.start_time
        cur_dur = cur.end_time - cur.start_time
        if prev_dur >= 0.9 or cur_dur <= 2.4:
            continue

        target_cur_duration = 2.0 if len(cur.words) <= 2 else 2.2
        new_cur_start = max(prev.start_time + 0.8, cur.end_time - target_cur_duration)
        if new_cur_start <= cur.start_time + 0.2:
            continue

        next_start = (
            adjusted[idx + 2].start_time
            if idx + 2 < len(adjusted) and adjusted[idx + 2].words
            else None
        )
        target_cur_duration = 2.0 if len(cur.words) <= 2 else 2.2
        if next_start is not None:
            target_cur_duration = min(
                target_cur_duration, next_start - min_gap - new_cur_start
            )
        target_cur_duration = max(0.8, target_cur_duration)
        spacing = target_cur_duration / len(cur.words)
        rebuilt_cur_words: List[models.Word] = []
        for w_idx, w in enumerate(cur.words):
            ws = new_cur_start + w_idx * spacing
            we = ws + spacing * 0.9
            if w_idx == len(cur.words) - 1:
                we = new_cur_start + target_cur_duration
            rebuilt_cur_words.append(
                models.Word(
                    text=w.text,
                    start_time=ws,
                    end_time=we,
                    singer=w.singer,
                )
            )
        adjusted[idx + 1] = models.Line(words=rebuilt_cur_words, singer=cur.singer)

        target_prev_end = min(
            adjusted[idx + 1].start_time - min_gap,
            prev.start_time + max(1.6, prev_dur),
        )
        if target_prev_end > prev.end_time + 0.2:
            duration = max(target_prev_end - prev.start_time, 0.3)
            spacing = duration / len(prev.words)
            stretched_prev_words: List[models.Word] = []
            for w_idx, w in enumerate(prev.words):
                ws = prev.start_time + w_idx * spacing
                we = ws + spacing * 0.9
                if w_idx == len(prev.words) - 1:
                    we = target_prev_end
                stretched_prev_words.append(
                    models.Word(
                        text=w.text,
                        start_time=ws,
                        end_time=we,
                        singer=w.singer,
                    )
                )
            adjusted[idx] = models.Line(words=stretched_prev_words, singer=prev.singer)

    return adjusted
