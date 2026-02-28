"""Onset snapping helpers for Whisper mapping post-processing."""

from __future__ import annotations

from collections import Counter
from typing import Callable, List

from ... import models
from ..alignment import timing_models


def _is_uniform_word_timing(line: models.Line) -> bool:
    if not line.words or len(line.words) < 3:
        return False
    durations = [max(0.0, w.end_time - w.start_time) for w in line.words]
    mean_duration = sum(durations) / len(durations)
    if mean_duration <= 0.0:
        return False
    variance = sum((d - mean_duration) ** 2 for d in durations) / len(durations)
    coeff_var = (variance**0.5) / mean_duration
    unique_rounded = len(set(round(d, 3) for d in durations))
    return coeff_var < 0.12 or unique_rounded <= 2


def _retime_uniform_words_to_whisper_anchors(  # noqa: C901
    mapped_lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
    *,
    normalize_interjection_token_fn: Callable[[str], str],
    normalize_match_token_fn: Callable[[str], str],
) -> List[models.Line]:
    if not mapped_lines or not all_words:
        return mapped_lines
    adjusted = list(mapped_lines)
    for idx, line in enumerate(adjusted):
        has_front_delay = (
            len(line.words) >= 3 and (line.words[1].start_time - line.start_time) > 0.8
        )
        if not _is_uniform_word_timing(line) and not has_front_delay:
            continue
        n_words = len(line.words)
        if n_words < 3:
            continue
        line_start = line.start_time
        line_end = line.end_time
        next_start = (
            adjusted[idx + 1].start_time
            if idx + 1 < len(adjusted) and adjusted[idx + 1].words
            else line_end + 2.0
        )

        window_words = [
            w for w in all_words if (line_start - 0.6) <= w.start <= (next_start + 0.3)
        ]
        if not window_words:
            continue

        matches: dict[int, float] = {}
        match_ends: dict[int, float] = {}
        cursor = 0

        def _strict_token_match(a: str, b: str) -> bool:
            if not a or not b:
                return False
            if a == b:
                return True
            return a.startswith(b) or b.startswith(a)

        for word_idx, word in enumerate(line.words):
            if word_idx == 0:
                continue
            token = normalize_interjection_token_fn(
                word.text
            ) or normalize_match_token_fn(word.text)
            if not token:
                continue
            found_idx = None
            for j in range(cursor, len(window_words)):
                wt = normalize_interjection_token_fn(
                    window_words[j].text
                ) or normalize_match_token_fn(window_words[j].text)
                if not wt:
                    continue
                if _strict_token_match(token, wt):
                    # Allow aggressive pulls earlier, but resist pushing words much later.
                    if window_words[j].start > (word.start_time + 0.6):
                        continue
                    found_idx = j
                    break
            if found_idx is None:
                continue
            matches[word_idx] = float(window_words[found_idx].start)
            match_ends[word_idx] = float(window_words[found_idx].end)
            cursor = found_idx + 1

        min_matches = max(2, int(0.5 * n_words))
        if len(matches) < min_matches:
            continue

        starts = [w.start_time for w in line.words]
        original_starts = list(starts)
        starts[0] = line_start
        for word_idx, anchor in matches.items():
            if word_idx == 0:
                continue
            starts[word_idx] = anchor

        min_word_duration = 0.08
        min_inter_word_gap = 0.02
        for i in range(1, n_words):
            min_start = starts[i - 1] + min_word_duration + min_inter_word_gap
            tail_reserved = (n_words - i - 1) * (min_word_duration + min_inter_word_gap)
            max_start = line_end - min_word_duration - tail_reserved
            starts[i] = max(min_start, min(starts[i], max_start))
        changed = sum(abs(starts[i] - original_starts[i]) for i in range(1, n_words))
        if changed < 0.08:
            continue
        if starts[-1] >= line_end - min_word_duration:
            continue

        target_line_end = line_end
        last_idx = n_words - 1
        if last_idx in match_ends:
            # Whisper can provide strong evidence that the displayed lyric phrase
            # ends well before the line boundary inherited from baseline timing.
            candidate_end = match_ends[last_idx] + 0.25
            next_line_start = (
                adjusted[idx + 1].start_time
                if idx + 1 < len(adjusted) and adjusted[idx + 1].words
                else line_end + 3.0
            )
            candidate_end = min(candidate_end, next_line_start - 0.05, line_end)
            if candidate_end <= line_end - 0.5:
                target_line_end = max(
                    starts[last_idx] + min_word_duration, candidate_end
                )

        retimed_words: List[models.Word] = []
        for i, word in enumerate(line.words):
            start = starts[i]
            if i + 1 < n_words:
                end = min(target_line_end, starts[i + 1] - min_inter_word_gap)
            else:
                end = target_line_end
            if end - start < min_word_duration:
                end = min(target_line_end, start + min_word_duration)
            if end <= start:
                retimed_words = []
                break
            retimed_words.append(
                models.Word(
                    text=word.text,
                    start_time=start,
                    end_time=end,
                    singer=word.singer,
                )
            )
        if retimed_words and len(retimed_words) == n_words:
            adjusted[idx] = models.Line(words=retimed_words, singer=line.singer)
    return adjusted


def _snap_first_word_to_whisper_onset(  # noqa: C901
    mapped_lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
    *,
    normalize_interjection_token_fn: Callable[[str], str],
    normalize_match_token_fn: Callable[[str], str],
    soft_token_match_fn: Callable[[str, str], bool],
    soft_token_overlap_ratio_fn: Callable[[List[str], List[str]], float],
    early_threshold: float = 0.12,
    max_shift: float = 0.8,
    min_gap: float = 0.05,
) -> List[models.Line]:
    """Shift lines later when the first word starts clearly before matching Whisper."""
    if not mapped_lines or not all_words:
        return mapped_lines

    adjusted = list(mapped_lines)
    line_text_keys: Counter[str] = Counter()
    for src_line in mapped_lines:
        tokens = [
            normalize_match_token_fn(w.text)
            for w in src_line.words
            if normalize_match_token_fn(w.text)
        ]
        if tokens:
            line_text_keys[" ".join(tokens)] += 1
    for idx, line in enumerate(adjusted):
        if not line.words:
            continue
        line_tokens = [
            normalize_match_token_fn(w.text)
            for w in line.words
            if normalize_match_token_fn(w.text)
        ]
        line_key = " ".join(line_tokens)
        repeated_phrase_count = line_text_keys.get(line_key, 0) if line_key else 0
        repetition_guard = False
        if line_tokens:
            token_counts = Counter(line_tokens)
            max_token_share = max(token_counts.values()) / len(line_tokens)
            repetition_guard = (
                len(line_tokens) >= 5
                and max_token_share >= 0.33
                and len(token_counts) <= max(3, int(len(line_tokens) * 0.6))
            )
            if repeated_phrase_count >= 3 and len(line_tokens) >= 4:
                repetition_guard = True
        if line.text.strip().endswith("?"):
            cur_tokens = [
                normalize_match_token_fn(w.text)
                for w in line.words
                if normalize_match_token_fn(w.text)
            ]
            prev_overlap = 0.0
            if idx > 0 and adjusted[idx - 1].words:
                prev_tokens = [
                    normalize_match_token_fn(w.text)
                    for w in adjusted[idx - 1].words
                    if normalize_match_token_fn(w.text)
                ]
                prev_overlap = soft_token_overlap_ratio_fn(cur_tokens, prev_tokens)
            if prev_overlap >= 0.4 and len(line.words) <= 6:
                continue

        fw = line.words[0]
        fw_norm = normalize_interjection_token_fn(fw.text)
        if not fw_norm:
            fw_norm = "".join(ch for ch in fw.text.lower() if ch.isalpha())
        if not fw_norm:
            continue

        prev_end = (
            adjusted[idx - 1].end_time if idx > 0 and adjusted[idx - 1].words else None
        )
        next_start = (
            adjusted[idx + 1].start_time
            if idx + 1 < len(adjusted) and adjusted[idx + 1].words
            else None
        )

        search_after = max(2.0, max_shift + 0.4)
        candidates = [
            w
            for w in all_words
            if (fw.start_time - 1.0) <= w.start <= (fw.start_time + search_after)
        ]
        best = None
        repetitive_following = False
        if idx + 1 < len(adjusted) and adjusted[idx + 1].words:
            cur_tokens = [
                normalize_match_token_fn(w.text)
                for w in line.words
                if normalize_match_token_fn(w.text)
            ]
            next_tokens = [
                normalize_match_token_fn(w.text)
                for w in adjusted[idx + 1].words
                if normalize_match_token_fn(w.text)
            ]
            if cur_tokens and next_tokens:
                repetitive_following = (
                    soft_token_overlap_ratio_fn(cur_tokens, next_tokens) >= 0.4
                )
        later_match_starts: List[float] = []
        for word in candidates:
            wn = normalize_interjection_token_fn(word.text)
            if not wn:
                wn = "".join(ch for ch in word.text.lower() if ch.isalpha())
            if not wn:
                continue
            match_score = 0
            if wn == fw_norm:
                match_score = 3
            elif wn.startswith(fw_norm) or fw_norm.startswith(wn):
                match_score = 2
            elif fw_norm in wn or wn in fw_norm:
                match_score = 1
            if match_score == 0:
                continue
            if (
                repetitive_following
                and match_score >= 2
                and word.start >= fw.start_time + 0.8
                and (not repetition_guard or word.start <= fw.start_time + 1.2)
            ):
                later_match_starts.append(word.start)
            score = (match_score, -abs(word.start - fw.start_time))
            if best is None or score > best[0]:
                best = (score, word)
        if best is None:
            continue

        target_start = best[1].start
        if repetitive_following and later_match_starts:
            target_start = min(later_match_starts)
        delta = fw.start_time - target_start
        if delta >= -early_threshold:
            continue

        max_shift_effective = min(max_shift, 1.0) if repetition_guard else max_shift
        desired_shift = min(-delta, max_shift_effective)
        max_allowed = desired_shift
        if prev_end is not None:
            max_allowed = min(
                max_allowed, max(0.0, fw.start_time - (prev_end + min_gap))
            )
        if next_start is not None:
            max_allowed = min(
                max_allowed, max(0.0, (next_start - min_gap) - line.end_time)
            )

        run_end = idx + 1
        base_tokens = [
            normalize_match_token_fn(w.text)
            for w in line.words
            if normalize_match_token_fn(w.text)
        ]
        while run_end < len(adjusted):
            nxt = adjusted[run_end]
            if not nxt.words:
                break
            nxt_tokens = [
                normalize_match_token_fn(w.text)
                for w in nxt.words
                if normalize_match_token_fn(w.text)
            ]
            if not base_tokens or not nxt_tokens:
                break
            if soft_token_overlap_ratio_fn(base_tokens, nxt_tokens) < 0.4:
                break
            run_end += 1

        should_shift_run = (
            (not repetition_guard)
            and run_end > idx + 1
            and desired_shift - max_allowed >= 0.1
        )
        if should_shift_run:
            desired_run_shift = min(-delta, 2.5)
            run_lines = adjusted[idx:run_end]
            run_start = run_lines[0].start_time - 1.0
            run_end_time = run_lines[-1].end_time + 6.0
            onset_candidates = [
                ww.start
                for ww in all_words
                if run_start <= ww.start <= run_end_time
                and soft_token_match_fn(
                    fw_norm,
                    (
                        normalize_interjection_token_fn(ww.text)
                        or "".join(ch for ch in ww.text.lower() if ch.isalpha())
                    ),
                )
            ]
            onset_candidates.sort()
            if len(onset_candidates) >= len(run_lines):
                assigned_starts: List[float] = []
                cand_idx = 0
                for run_line in run_lines:
                    min_start = run_line.start_time + 0.15
                    if assigned_starts:
                        min_spacing = 1.0 if len(run_line.words) <= 4 else 0.35
                        min_start = max(min_start, assigned_starts[-1] + min_spacing)
                    while (
                        cand_idx < len(onset_candidates)
                        and onset_candidates[cand_idx] < min_start
                    ):
                        cand_idx += 1
                    if cand_idx >= len(onset_candidates):
                        assigned_starts = []
                        break
                    if onset_candidates[cand_idx] - run_line.start_time > 3.0:
                        assigned_starts = []
                        break
                    assigned_starts.append(onset_candidates[cand_idx])
                    cand_idx += 1
                if assigned_starts and len(assigned_starts) == len(run_lines):
                    shifted_any = False
                    for rel_idx, run_line in enumerate(run_lines):
                        target = assigned_starts[rel_idx]
                        delta_run = target - run_line.start_time
                        if delta_run <= 0.1:
                            continue
                        shifted_any = True
                        shifted_words = [
                            models.Word(
                                text=w.text,
                                start_time=w.start_time + delta_run,
                                end_time=w.end_time + delta_run,
                                singer=w.singer,
                            )
                            for w in run_line.words
                        ]
                        adjusted[idx + rel_idx] = models.Line(
                            words=shifted_words, singer=run_line.singer
                        )
                    if shifted_any:
                        continue

            after_run_start = (
                adjusted[run_end].start_time
                if run_end < len(adjusted) and adjusted[run_end].words
                else None
            )
            if after_run_start is not None:
                available = max(
                    0.0, after_run_start - min_gap - adjusted[run_end - 1].end_time
                )
                run_shift = min(desired_run_shift, available)
                if run_shift > 0:
                    for run_idx in range(idx, run_end):
                        run_line = adjusted[run_idx]
                        shifted_words = [
                            models.Word(
                                text=w.text,
                                start_time=w.start_time + run_shift,
                                end_time=w.end_time + run_shift,
                                singer=w.singer,
                            )
                            for w in run_line.words
                        ]
                        adjusted[run_idx] = models.Line(
                            words=shifted_words, singer=run_line.singer
                        )
                    continue

        if max_allowed <= 0:
            best_match_strength = best[0][0] if best is not None else 0
            # In tightly packed regions, whole-line shifting may be blocked by neighbor
            # constraints. If Whisper provides strong onset evidence, retime the first
            # word only so line-start can still move later without cascading overlaps.
            if (
                desired_shift >= 0.25
                and best_match_strength >= 2
                and len(line.words) >= 3
                and not line.text.strip().endswith("?")
                and not repetition_guard
            ):
                first = line.words[0]
                second = line.words[1]
                target_first_start = min(target_start, second.start_time - 0.1)
                if target_first_start - first.start_time >= 0.12:
                    target_first_end = min(second.start_time - 0.02, first.end_time)
                    if target_first_end - target_first_start < 0.06:
                        target_first_end = min(
                            second.start_time - 0.02, target_first_start + 0.06
                        )
                    if target_first_end - target_first_start >= 0.05:
                        shifted_words = [
                            models.Word(
                                text=first.text,
                                start_time=target_first_start,
                                end_time=target_first_end,
                                singer=first.singer,
                            ),
                            *line.words[1:],
                        ]
                        adjusted[idx] = models.Line(
                            words=shifted_words,
                            singer=line.singer,
                        )
                        continue

            tightly_packed = False
            if next_start is not None and (next_start - line.end_time) <= 0.12:
                packed_count = 0
                probe = idx
                while probe + 1 < len(adjusted) and packed_count < 5:
                    cur_probe = adjusted[probe]
                    nxt_probe = adjusted[probe + 1]
                    if not cur_probe.words or not nxt_probe.words:
                        break
                    if nxt_probe.start_time - cur_probe.end_time > 0.12:
                        break
                    packed_count += 1
                    probe += 1
                tightly_packed = packed_count >= 2
            if (
                desired_shift >= 0.8
                and tightly_packed
                and best_match_strength >= 2
                and idx >= len(adjusted) // 2
                and not line.text.strip().endswith("?")
                and not repetition_guard
            ):
                suffix_shift = min(desired_shift, max_shift)
                local_end = idx + 1
                while local_end < len(adjusted):
                    prev_local = adjusted[local_end - 1]
                    cur_local = adjusted[local_end]
                    if not prev_local.words or not cur_local.words:
                        break
                    if (cur_local.start_time - prev_local.end_time) > 0.2:
                        break
                    if local_end - idx >= 10:
                        break
                    local_end += 1
                for tail_idx in range(idx, local_end):
                    tail_line = adjusted[tail_idx]
                    if not tail_line.words:
                        break
                    shifted_words = [
                        models.Word(
                            text=w.text,
                            start_time=w.start_time + suffix_shift,
                            end_time=w.end_time + suffix_shift,
                            singer=w.singer,
                        )
                        for w in tail_line.words
                    ]
                    adjusted[tail_idx] = models.Line(
                        words=shifted_words, singer=tail_line.singer
                    )
                continue
            continue

        shifted_words = [
            models.Word(
                text=w.text,
                start_time=w.start_time + max_allowed,
                end_time=w.end_time + max_allowed,
                singer=w.singer,
            )
            for w in line.words
        ]
        adjusted[idx] = models.Line(words=shifted_words, singer=line.singer)

    adjusted = _retime_uniform_words_to_whisper_anchors(
        adjusted,
        all_words,
        normalize_interjection_token_fn=normalize_interjection_token_fn,
        normalize_match_token_fn=normalize_match_token_fn,
    )

    return adjusted
