"""Post-processing helpers for repetitive Whisper mapping patterns."""

import re
from typing import List, Optional, Tuple

from ... import models
from ..alignment import timing_models


def _normalize_interjection_token(token: str) -> str:
    cleaned = "".join(ch for ch in token.lower() if ch.isalpha())
    if not cleaned:
        return ""
    return re.sub(r"(.)\1{2,}", r"\1\1", cleaned)


def _normalize_text_tokens(text: str) -> List[str]:
    tokens = []
    for raw in text.lower().split():
        tok = "".join(ch for ch in raw if ch.isalpha())
        if tok:
            tokens.append(re.sub(r"(.)\1{2,}", r"\1\1", tok))
    return tokens


def _light_text_similarity(a: str, b: str) -> float:
    a_tokens = _normalize_text_tokens(a)
    b_tokens = _normalize_text_tokens(b)
    if not a_tokens or not b_tokens:
        return 0.0
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union else 0.0


def _soft_text_similarity(a: str, b: str) -> float:
    a_tokens = _normalize_text_tokens(a)
    b_tokens = _normalize_text_tokens(b)
    if not a_tokens or not b_tokens:
        return 0.0
    used = [False] * len(b_tokens)
    matched = 0
    for at in a_tokens:
        best_idx = None
        for idx, bt in enumerate(b_tokens):
            if used[idx]:
                continue
            if _soft_token_match(at, bt):
                best_idx = idx
                if at == bt:
                    break
        if best_idx is not None:
            used[best_idx] = True
            matched += 1
    return matched / max(len(a_tokens), len(b_tokens))


def _normalize_match_token(token: str) -> str:
    base = _normalize_interjection_token(token)
    if not base:
        base = "".join(ch for ch in token.lower() if ch.isalpha())
    if base.endswith("s") and len(base) > 3:
        base = base[:-1]
    return base


def _soft_token_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    return a in b or b in a


def _overlap_suffix_prefix(
    left_tokens: List[str],
    right_tokens: List[str],
    max_overlap: int = 3,
) -> int:
    if not left_tokens or not right_tokens:
        return 0
    upper = min(max_overlap, len(left_tokens), len(right_tokens))
    for size in range(upper, 0, -1):
        ok = True
        for i in range(size):
            if not _soft_token_match(left_tokens[-size + i], right_tokens[i]):
                ok = False
                break
        if ok:
            return size
    return 0


def _soft_token_overlap_ratio(left_tokens: List[str], right_tokens: List[str]) -> float:
    if not left_tokens or not right_tokens:
        return 0.0
    matched = 0
    used = [False] * len(right_tokens)
    for lt in left_tokens:
        for idx, rt in enumerate(right_tokens):
            if used[idx]:
                continue
            if _soft_token_match(lt, rt):
                used[idx] = True
                matched += 1
                break
    return matched / max(len(left_tokens), len(right_tokens))


def _realign_repetitive_runs_to_matching_segments(  # noqa: C901
    lines: List[models.Line],
    segments: List[timing_models.TranscriptionSegment],
    *,
    min_gap: float = 0.05,
    min_run_len: int = 3,
    min_overlap: float = 0.4,
    min_seg_similarity: float = 0.15,
    min_shift: float = 0.8,
    max_shift: float = 3.0,
) -> List[models.Line]:
    """Shift early repetitive runs later to segment starts as a block."""
    if not lines or not segments:
        return lines

    adjusted = list(lines)
    sorted_segments = sorted(segments, key=lambda s: s.start)
    idx = 0
    while idx < len(adjusted):
        line = adjusted[idx]
        if not line.words:
            idx += 1
            continue

        run_end = idx + 1
        exact_duplicates = 0
        while run_end < len(adjusted):
            prev = adjusted[run_end - 1]
            cur = adjusted[run_end]
            if not prev.words or not cur.words:
                break
            prev_tokens = [
                _normalize_match_token(w.text)
                for w in prev.words
                if _normalize_match_token(w.text)
            ]
            cur_tokens = [
                _normalize_match_token(w.text)
                for w in cur.words
                if _normalize_match_token(w.text)
            ]
            if not prev_tokens or not cur_tokens:
                break
            if _soft_token_overlap_ratio(prev_tokens, cur_tokens) < min_overlap:
                break
            if prev.text.strip().lower() == cur.text.strip().lower():
                exact_duplicates += 1
            run_end += 1

        run_len = run_end - idx
        if run_len < min_run_len or exact_duplicates == 0:
            idx = run_end
            continue

        run_lines = adjusted[idx:run_end]
        run_window_start = run_lines[0].start_time - 1.0
        run_window_end = run_lines[-1].end_time + 8.0
        first_text = run_lines[0].text
        matching_segments: List[timing_models.TranscriptionSegment] = []
        for seg in sorted_segments:
            if seg.start < run_window_start or seg.start > run_window_end:
                continue
            if len(_normalize_text_tokens(seg.text)) < 2:
                continue
            sim = max(
                _light_text_similarity(first_text, seg.text),
                _soft_text_similarity(first_text, seg.text),
            )
            if sim < min_seg_similarity:
                continue
            matching_segments.append(seg)

        if len(matching_segments) < run_len:
            idx = run_end
            continue

        deltas = [
            run_lines[k].start_time - matching_segments[k].start for k in range(run_len)
        ]
        strongly_early = sum(1 for d in deltas if d <= -min_shift)
        if strongly_early < max(1, run_len // 3):
            idx = run_end
            continue

        median_delta = sorted(deltas)[run_len // 2]
        if median_delta >= -min_shift:
            idx = run_end
            continue

        shift = min(max_shift, -median_delta)
        shift_end = run_end
        next_start = (
            adjusted[shift_end].start_time
            if shift_end < len(adjusted) and adjusted[shift_end].words
            else None
        )
        if (
            next_start is not None
            and next_start - run_lines[-1].end_time < min_shift
            and shift_end + 1 < len(adjusted)
            and adjusted[shift_end].words
            and adjusted[shift_end + 1].words
        ):
            boundary_prev_tokens = [
                _normalize_match_token(w.text)
                for w in adjusted[shift_end - 1].words
                if _normalize_match_token(w.text)
            ]
            boundary_tokens = [
                _normalize_match_token(w.text)
                for w in adjusted[shift_end].words
                if _normalize_match_token(w.text)
            ]
            boundary_overlap = _soft_token_overlap_ratio(
                boundary_prev_tokens, boundary_tokens
            )
            trailing_gap = (
                adjusted[shift_end + 1].start_time - adjusted[shift_end].end_time
            )
            if boundary_overlap >= 0.3 and trailing_gap > 3.0:
                shift_end += 1
                next_start = adjusted[shift_end].start_time

        if next_start is not None:
            shift = min(
                shift,
                max(0.0, next_start - min_gap - adjusted[shift_end - 1].end_time),
            )
        if shift < 0.25:
            idx = run_end
            continue

        for run_idx in range(idx, shift_end):
            cur_line = adjusted[run_idx]
            shifted_words = [
                models.Word(
                    text=w.text,
                    start_time=w.start_time + shift,
                    end_time=w.end_time + shift,
                    singer=w.singer,
                )
                for w in cur_line.words
            ]
            adjusted[run_idx] = models.Line(words=shifted_words, singer=cur_line.singer)
        idx = shift_end

    return adjusted


def _retime_repetitive_question_runs_to_segment_windows(  # noqa: C901
    lines: List[models.Line],
    segments: List[timing_models.TranscriptionSegment],
    *,
    min_gap: float = 0.05,
    min_run_len: int = 3,
    min_overlap: float = 0.3,
    min_seg_similarity: float = 0.3,
) -> List[models.Line]:
    """Retarget repeated short question runs to successive segment windows."""
    if not lines or not segments:
        return lines

    adjusted = list(lines)
    sorted_segments = sorted(segments, key=lambda s: s.start)
    idx = 0
    while idx < len(adjusted):
        cur = adjusted[idx]
        if not cur.words or not cur.text.strip().endswith("?"):
            idx += 1
            continue

        run_end = idx + 1
        while run_end < len(adjusted):
            prev = adjusted[run_end - 1]
            nxt = adjusted[run_end]
            if not prev.words or not nxt.words or not nxt.text.strip().endswith("?"):
                break
            prev_tokens = [
                _normalize_match_token(w.text)
                for w in prev.words
                if _normalize_match_token(w.text)
            ]
            nxt_tokens = [
                _normalize_match_token(w.text)
                for w in nxt.words
                if _normalize_match_token(w.text)
            ]
            overlap = _soft_token_overlap_ratio(prev_tokens, nxt_tokens)
            shared_long_token = any(
                len(tok) >= 4 and tok in set(nxt_tokens) for tok in prev_tokens
            )
            if overlap < min_overlap or (overlap < 0.5 and not shared_long_token):
                break
            run_end += 1

        run_len = run_end - idx
        if run_len < min_run_len:
            idx = run_end
            continue

        run_lines = adjusted[idx:run_end]
        run_text = run_lines[0].text
        run_start = run_lines[0].start_time - 1.0
        run_stop = run_lines[-1].end_time + 10.0
        match_segments: List[timing_models.TranscriptionSegment] = []
        for seg in sorted_segments:
            if seg.start < run_start or seg.start > run_stop:
                continue
            if len(_normalize_text_tokens(seg.text)) < 2:
                continue
            sim = max(
                _light_text_similarity(run_text, seg.text),
                _soft_text_similarity(run_text, seg.text),
            )
            if sim < min_seg_similarity:
                continue
            match_segments.append(seg)

        if len(match_segments) < min_run_len:
            idx = run_end
            continue
        # Don't retime runs beyond Whisper segment coverage; this can pull
        # end-of-song refrains (after transcript tail) incorrectly early.
        if match_segments[-1].end < run_lines[-1].start_time - 0.8:
            idx = run_end
            continue

        assigned_seg_idx: List[int] = []
        seg_ptr = 0
        prev_seg_start: Optional[float] = None
        for run_line in run_lines:
            best_idx: Optional[int] = None
            best_cost = float("inf")
            for cand_idx in range(seg_ptr, len(match_segments)):
                seg = match_segments[cand_idx]
                if prev_seg_start is not None and seg.start < prev_seg_start + 0.6:
                    continue
                cost = abs(seg.start - run_line.start_time)
                if cost < best_cost:
                    best_cost = cost
                    best_idx = cand_idx
                if best_idx is not None and seg.start > run_line.start_time + 4.0:
                    break
            if best_idx is None:
                break
            assigned_seg_idx.append(best_idx)
            prev_seg_start = match_segments[best_idx].start
            seg_ptr = best_idx + 1

        apply_count = min(run_len, len(assigned_seg_idx))
        if apply_count < min_run_len:
            idx = run_end
            continue

        for rel_idx in range(apply_count):
            line = run_lines[rel_idx]
            seg = match_segments[assigned_seg_idx[rel_idx]]
            next_start = (
                match_segments[assigned_seg_idx[rel_idx + 1]].start
                if rel_idx + 1 < apply_count
                else (
                    adjusted[run_end].start_time
                    if run_end < len(adjusted) and adjusted[run_end].words
                    else None
                )
            )
            target_start = seg.start
            if rel_idx > 0:
                target_start = max(
                    target_start, adjusted[idx + rel_idx - 1].end_time + min_gap
                )
            target_start = max(target_start, line.start_time - 0.5)
            base_duration = max(0.9, min(1.8, 0.65 + 0.18 * len(line.words)))
            if len(line.words) >= 5:
                base_duration = min(2.2, base_duration + 0.25)
            target_end = target_start + base_duration
            if next_start is not None:
                target_end = min(target_end, next_start - min_gap)
            target_end = min(target_end, seg.end + 0.4)
            target_end = max(target_end, target_start + 0.7)
            if target_end <= target_start + 0.2:
                continue
            spacing = (target_end - target_start) / len(line.words)
            new_words: List[models.Word] = []
            for w_idx, w in enumerate(line.words):
                ws = target_start + w_idx * spacing
                we = ws + spacing * 0.9
                if w_idx == len(line.words) - 1:
                    we = target_end
                new_words.append(
                    models.Word(
                        text=w.text,
                        start_time=ws,
                        end_time=we,
                        singer=w.singer,
                    )
                )
            adjusted[idx + rel_idx] = models.Line(words=new_words, singer=line.singer)

        idx = run_end

    return adjusted


def _enforce_min_duration_for_short_question_lines(  # noqa: C901
    lines: List[models.Line],
    *,
    min_duration: float = 1.0,
    min_gap: float = 0.05,
) -> List[models.Line]:
    """Prevent short question lines from flashing too quickly."""
    if not lines:
        return lines
    adjusted = list(lines)
    for idx, line in enumerate(adjusted):
        if not line.words or not line.text.strip().endswith("?"):
            continue
        if len(line.words) > 6:
            continue
        duration = line.end_time - line.start_time
        if duration >= min_duration:
            continue
        next_start = (
            adjusted[idx + 1].start_time
            if idx + 1 < len(adjusted) and adjusted[idx + 1].words
            else None
        )
        target_end = line.start_time + min_duration
        if next_start is not None:
            target_end = min(target_end, next_start - min_gap)
            if target_end <= line.end_time + 0.05 and idx + 1 < len(adjusted):
                next_line = adjusted[idx + 1]
                if next_line.words and next_line.text.strip().endswith("?"):
                    need = min_duration - duration
                    if need > 0:
                        after_next_start = (
                            adjusted[idx + 2].start_time
                            if idx + 2 < len(adjusted) and adjusted[idx + 2].words
                            else None
                        )
                        available = (
                            max(0.0, after_next_start - next_line.end_time)
                            if after_next_start is not None
                            else 0.0
                        )
                        shift = min(need, available)
                        if shift > 0.05:
                            j = idx + 1
                            while j < len(adjusted):
                                run_line = adjusted[j]
                                if not run_line.words:
                                    break
                                if j > idx + 1 and not run_line.text.strip().endswith(
                                    "?"
                                ):
                                    break
                                shifted_words = [
                                    models.Word(
                                        text=w.text,
                                        start_time=w.start_time + shift,
                                        end_time=w.end_time + shift,
                                        singer=w.singer,
                                    )
                                    for w in run_line.words
                                ]
                                adjusted[j] = models.Line(
                                    words=shifted_words, singer=run_line.singer
                                )
                                j += 1
                            next_start = adjusted[idx + 1].start_time
                            target_end = min(
                                line.start_time + min_duration, next_start - min_gap
                            )
        if target_end <= line.end_time + 0.05:
            continue
        spacing = (target_end - line.start_time) / len(line.words)
        new_words: List[models.Word] = []
        for w_idx, w in enumerate(line.words):
            ws = line.start_time + w_idx * spacing
            we = ws + spacing * 0.9
            if w_idx == len(line.words) - 1:
                we = target_end
            new_words.append(
                models.Word(
                    text=w.text,
                    start_time=ws,
                    end_time=we,
                    singer=w.singer,
                )
            )
        adjusted[idx] = models.Line(words=new_words, singer=line.singer)
    return adjusted


def _pull_adjacent_similar_lines_across_long_gaps(
    lines: List[models.Line],
    *,
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
            _normalize_match_token(w.text)
            for w in cur.words
            if _normalize_match_token(w.text)
        ]
        nxt_tokens = [
            _normalize_match_token(w.text)
            for w in nxt.words
            if _normalize_match_token(w.text)
        ]
        if not cur_tokens or not nxt_tokens:
            continue
        if _soft_token_overlap_ratio(cur_tokens, nxt_tokens) < min_similarity:
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


def _smooth_adjacent_duplicate_line_cadence(  # noqa: C901
    lines: List[models.Line],
    *,
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
            _normalize_match_token(w.text)
            for w in prev.words
            if _normalize_match_token(w.text)
        ]
        cur_tokens = [
            _normalize_match_token(w.text)
            for w in cur.words
            if _normalize_match_token(w.text)
        ]
        overlap = _soft_token_overlap_ratio(prev_tokens, cur_tokens)
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
            # Keep naturally wider spacing for short duplicate refrain lines.
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


def _rebalance_short_question_pairs(
    lines: List[models.Line],
    *,
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
        prev_first = _normalize_match_token(prev.words[0].text)
        cur_first = _normalize_match_token(cur.words[0].text)
        if not _soft_token_match(prev_first, cur_first):
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


def _extend_line_to_trailing_whisper_matches(  # noqa: C901
    mapped_lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
    *,
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
            (word_idx, _normalize_match_token(w.text))
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
                    ww_tok = _normalize_match_token(candidates[wi].text)
                    if _soft_token_match(tok, ww_tok):
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

        # Prefer real Whisper word timings where available to preserve natural pauses.
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
                duration = max(
                    0.12,
                    min(0.25, end_cap - left - 0.02),
                )
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

    adjusted = _smooth_adjacent_duplicate_line_cadence(adjusted)
    return _rebalance_short_question_pairs(adjusted)
