"""Experimental follow-up helpers for Whisper alignment."""

from __future__ import annotations

import os
import re
from typing import Any, List, Optional

from ... import models, phonetic_utils
from ..alignment import timing_models


def enable_low_support_onset_reanchor() -> bool:
    return os.getenv("Y2K_WHISPER_ENABLE_LOW_SUPPORT_ONSET_REANCHOR", "0") == "1"


def enable_repeat_cadence_reanchor() -> bool:
    return os.getenv("Y2K_WHISPER_ENABLE_REPEAT_CADENCE_REANCHOR", "0") == "1"


def enable_restored_run_onset_shift() -> bool:
    return os.getenv("Y2K_WHISPER_ENABLE_RESTORED_RUN_ONSET_SHIFT", "0") == "1"


def count_non_vocal_words_near_time(
    words: List[timing_models.TranscriptionWord],
    center_time: float,
    *,
    window_sec: float = 1.0,
) -> int:
    lo = center_time - window_sec
    hi = center_time + window_sec
    count = 0
    for word in words:
        if word.text == "[VOCAL]":
            continue
        if lo <= word.start <= hi:
            count += 1
    return count


def normalized_prefix_tokens(line: models.Line, *, limit: int = 3) -> list[str]:
    return [
        re.sub(r"[^a-z]+", "", w.text.lower())
        for w in line.words[:limit]
        if re.sub(r"[^a-z]+", "", w.text.lower())
    ]


def normalized_tokens(line: models.Line) -> list[str]:
    return [
        re.sub(r"[^a-z]+", "", w.text.lower())
        for w in line.words
        if re.sub(r"[^a-z]+", "", w.text.lower())
    ]


_LIGHT_LEADING_TOKENS = {
    "the",
    "a",
    "an",
    "la",
    "el",
    "de",
    "del",
    "que",
    "lo",
    "me",
    "tu",
}


def rescale_line_to_new_start(line: models.Line, target_start: float) -> models.Line:
    old_duration = line.end_time - line.start_time
    new_duration = line.end_time - target_start
    span = old_duration if old_duration > 0 else 1.0
    reanchored_words: list[models.Word] = []
    for word in line.words:
        rel_start = (word.start_time - line.start_time) / span
        rel_end = (word.end_time - line.start_time) / span
        reanchored_words.append(
            models.Word(
                text=word.text,
                start_time=target_start + rel_start * new_duration,
                end_time=target_start + rel_end * new_duration,
                singer=word.singer,
            )
        )
    return models.Line(words=reanchored_words, singer=line.singer)


def _rescale_line_to_new_end(line: models.Line, target_end: float) -> models.Line:
    old_duration = line.end_time - line.start_time
    new_duration = target_end - line.start_time
    span = old_duration if old_duration > 0 else 1.0
    reanchored_words: list[models.Word] = []
    for word in line.words:
        rel_start = (word.start_time - line.start_time) / span
        rel_end = (word.end_time - line.start_time) / span
        reanchored_words.append(
            models.Word(
                text=word.text,
                start_time=line.start_time + rel_start * new_duration,
                end_time=line.start_time + rel_end * new_duration,
                singer=word.singer,
            )
        )
    return models.Line(words=reanchored_words, singer=line.singer)


def _rescale_line_to_new_bounds(
    line: models.Line, target_start: float, target_end: float
) -> models.Line:
    old_duration = line.end_time - line.start_time
    new_duration = target_end - target_start
    span = old_duration if old_duration > 0 else 1.0
    reanchored_words: list[models.Word] = []
    for word in line.words:
        rel_start = (word.start_time - line.start_time) / span
        rel_end = (word.end_time - line.start_time) / span
        reanchored_words.append(
            models.Word(
                text=word.text,
                start_time=target_start + rel_start * new_duration,
                end_time=target_start + rel_end * new_duration,
                singer=word.singer,
            )
        )
    return models.Line(words=reanchored_words, singer=line.singer)


def line_text_key(line: models.Line) -> str:
    return " ".join(normalized_tokens(line))


def _count_leading_light_tokens(tokens: list[str]) -> int:
    count = 0
    for token in tokens:
        if token in _LIGHT_LEADING_TOKENS:
            count += 1
            continue
        break
    return count


def _token_stem(token: str) -> str:
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def _tokens_match_with_light_stemming(
    a: str, b: str, *, min_shared_prefix: int = 4
) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    a_stem = _token_stem(a)
    b_stem = _token_stem(b)
    if a_stem == b_stem:
        return True
    shared_prefix = 0
    for a_ch, b_ch in zip(a_stem, b_stem):
        if a_ch != b_ch:
            break
        shared_prefix += 1
    return shared_prefix >= min_shared_prefix


def _tokens_match_with_short_fragment_stemming(a: str, b: str) -> bool:
    if _tokens_match_with_light_stemming(a, b):
        return True
    a_stem = _token_stem(a)
    b_stem = _token_stem(b)
    if min(len(a_stem), len(b_stem)) > 3:
        return False
    shared_prefix = 0
    for a_ch, b_ch in zip(a_stem, b_stem):
        if a_ch != b_ch:
            break
        shared_prefix += 1
    return shared_prefix >= 2


def line_token_overlap_ratio(a: models.Line, b: models.Line) -> float:
    a_tokens = set(normalized_tokens(a))
    b_tokens = set(normalized_tokens(b))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(len(a_tokens), len(b_tokens))


def local_lexical_overlap_ratio(
    line: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
    *,
    pad_before: float = 0.8,
    pad_after: float = 0.8,
) -> float:
    line_tokens = set(normalized_tokens(line))
    if not line_tokens:
        return 0.0
    nearby_tokens = {
        re.sub(r"[^a-z]+", "", word.text.lower())
        for word in whisper_words
        if line.start_time - pad_before <= word.start <= line.end_time + pad_after
        and word.text != "[VOCAL]"
    }
    nearby_tokens.discard("")
    if not nearby_tokens:
        return 0.0
    overlap = len(line_tokens & nearby_tokens)
    return overlap / max(len(line_tokens), len(nearby_tokens))


def _find_local_content_word_start(
    line: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
    *,
    lookback_sec: float = 0.35,
    lookahead_sec: float = 0.9,
) -> float | None:
    tokens = normalized_tokens(line)
    if len(tokens) < 2:
        return None
    leading_count = _count_leading_light_tokens(tokens)
    if leading_count != 1 or leading_count >= len(tokens):
        return None
    content_token = tokens[leading_count]
    for word in whisper_words:
        token = re.sub(r"[^a-z]+", "", word.text.lower())
        if not _tokens_match_with_light_stemming(content_token, token):
            continue
        if not (
            line.start_time - lookback_sec
            <= word.start
            <= line.end_time + lookahead_sec
        ):
            continue
        return float(word.start)
    return None


def reanchor_light_leading_lines_to_content_words(
    mapped_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord],
    *,
    min_shift_sec: float = 0.2,
    max_shift_sec: float = 0.75,
    min_word_count: int = 4,
    max_word_count: int = 10,
    min_overlap_ratio: float = 0.45,
    min_gap: float = 0.05,
) -> tuple[List[models.Line], int]:
    adjusted = list(mapped_lines)
    applied = 0
    for idx, line in enumerate(adjusted):
        if (
            not line.words
            or len(line.words) < min_word_count
            or len(line.words) > max_word_count
        ):
            continue
        if local_lexical_overlap_ratio(line, whisper_words) < min_overlap_ratio:
            continue
        target_start = _find_local_content_word_start(line, whisper_words)
        if target_start is None:
            continue
        shift = target_start - line.start_time
        if shift < min_shift_sec or shift > max_shift_sec:
            continue
        if (
            idx > 0
            and adjusted[idx - 1].words
            and target_start < adjusted[idx - 1].end_time + min_gap
        ):
            continue
        adjusted[idx] = rescale_line_to_new_start(line, target_start)
        applied += 1
    return adjusted, applied


def _match_stemmed_prefix_near_word(
    *,
    prefix: list[str],
    whisper_words: List[timing_models.TranscriptionWord],
    start_idx: int,
    support_window: float,
) -> int:
    match_count = 1
    cursor = start_idx + 1
    last_end = whisper_words[start_idx].end
    anchor_start = whisper_words[start_idx].start
    for token in prefix[1:]:
        while cursor < len(whisper_words):
            candidate = whisper_words[cursor]
            candidate_norm = re.sub(r"[^a-z]+", "", candidate.text.lower())
            if candidate.start > anchor_start + support_window:
                return match_count
            cursor += 1
            if candidate.text == "[VOCAL]" or not candidate_norm:
                continue
            if candidate.start + 1e-6 < last_end:
                continue
            if not _tokens_match_with_light_stemming(token, candidate_norm):
                continue
            match_count += 1
            last_end = candidate.end
            break
        else:
            return match_count
    return match_count


def _find_stemmed_prefix_anchor_start(
    line: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
    *,
    min_shift_sec: float,
    max_shift_sec: float,
    min_prefix_matches: int,
    support_window: float,
) -> float | None:
    prefix = normalized_prefix_tokens(line, limit=3)
    if len(prefix) < min_prefix_matches:
        return None
    search_start = line.start_time - max_shift_sec
    search_end = line.start_time - min_shift_sec
    for idx, word in enumerate(whisper_words):
        token = re.sub(r"[^a-z]+", "", word.text.lower())
        if word.text == "[VOCAL]" or not token:
            continue
        if word.start < search_start or word.start > search_end:
            continue
        if not _tokens_match_with_light_stemming(prefix[0], token):
            continue
        match_count = _match_stemmed_prefix_near_word(
            prefix=prefix,
            whisper_words=whisper_words,
            start_idx=idx,
            support_window=support_window,
        )
        if match_count >= min_prefix_matches:
            return float(word.start)
    return None


def _is_single_letter_followup(tokens: list[str]) -> bool:
    return bool(tokens) and len(tokens[0]) == 1 and len(tokens) >= 4


def _short_followup_prefix_settings(
    line: models.Line,
    *,
    default_max_shift_sec: float,
    default_min_prefix_matches: int,
    default_support_window: float,
) -> tuple[bool, float, int, float]:
    tokens = normalized_tokens(line)
    leading_light_count = _count_leading_light_tokens(tokens)
    single_letter_followup = leading_light_count == 0 and _is_single_letter_followup(
        tokens
    )
    if single_letter_followup:
        return True, 1.3, max(default_min_prefix_matches, 2), 1.35
    return (
        leading_light_count == 1,
        default_max_shift_sec,
        default_min_prefix_matches,
        default_support_window,
    )


def _latest_supported_whisper_end(
    line: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
    *,
    search_start: float,
    search_end: float,
) -> float | None:
    token_variants: set[str] = set()
    for line_word in line.words:
        normalized = re.sub(r"[^a-z]+", "", line_word.text.lower())
        if normalized:
            token_variants.add(normalized)
        for fragment in re.split(r"[^a-z]+", line_word.text.lower()):
            if fragment:
                token_variants.add(fragment)
    if not token_variants:
        return None
    latest_end: float | None = None
    for whisper_word in whisper_words:
        token = re.sub(r"[^a-z]+", "", whisper_word.text.lower())
        if whisper_word.text == "[VOCAL]" or not token:
            continue
        if whisper_word.start < search_start or whisper_word.end > search_end:
            continue
        if not any(
            _tokens_match_with_short_fragment_stemming(line_token, token)
            for line_token in token_variants
        ):
            continue
        latest_end = float(whisper_word.end)
    return latest_end


def rebalance_short_followup_boundaries_from_whisper(
    mapped_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord],
    *,
    min_shift_sec: float = 0.35,
    max_shift_sec: float = 0.95,
    min_prefix_matches: int = 2,
    support_window: float = 1.1,
    min_gap: float = 0.05,
    max_word_count: int = 6,
    min_prev_word_count: int = 8,
) -> tuple[List[models.Line], int]:
    updated = list(mapped_lines)
    applied = 0
    for idx in range(1, len(updated)):
        prev_line = updated[idx - 1]
        line = updated[idx]
        if (
            not prev_line.words
            or not line.words
            or len(line.words) > max_word_count
            or len(prev_line.words) < min_prev_word_count
        ):
            continue
        (
            allow_followup_shape,
            line_max_shift_sec,
            line_min_prefix_matches,
            line_support_window,
        ) = _short_followup_prefix_settings(
            line,
            default_max_shift_sec=max_shift_sec,
            default_min_prefix_matches=min_prefix_matches,
            default_support_window=support_window,
        )
        if not allow_followup_shape:
            continue
        target_start = _find_stemmed_prefix_anchor_start(
            line,
            whisper_words,
            min_shift_sec=min_shift_sec,
            max_shift_sec=line_max_shift_sec,
            min_prefix_matches=line_min_prefix_matches,
            support_window=line_support_window,
        )
        if target_start is None or target_start >= prev_line.end_time - 0.25:
            continue
        prev_support_end = _latest_supported_whisper_end(
            prev_line,
            whisper_words,
            search_start=max(prev_line.start_time - 0.6, target_start - 1.4),
            search_end=target_start,
        )
        current_support_end = _latest_supported_whisper_end(
            line,
            whisper_words,
            search_start=target_start,
            search_end=line.end_time + line_support_window,
        )
        if (
            prev_support_end is None
            or current_support_end is None
            or prev_support_end > target_start - min_gap
            or current_support_end <= target_start + 0.45
        ):
            continue
        new_prev_end = min(prev_line.end_time, target_start - min_gap)
        next_start = (
            updated[idx + 1].start_time
            if idx + 1 < len(updated) and updated[idx + 1].words
            else None
        )
        new_line_end = current_support_end
        if next_start is not None:
            new_line_end = min(new_line_end, next_start - min_gap)
        if (
            new_prev_end <= prev_line.start_time + 0.8
            or new_line_end <= target_start + 0.6
        ):
            continue
        updated[idx - 1] = _rescale_line_to_new_end(prev_line, new_prev_end)
        updated[idx] = _rescale_line_to_new_bounds(line, target_start, new_line_end)
        applied += 1
    return updated, applied


def _truncated_followup_tokens(line: models.Line) -> tuple[str, str] | None:
    if not line.words or len(line.words) > 3:
        return None
    tokens = normalized_tokens(line)
    leading_count = _count_leading_light_tokens(tokens)
    if leading_count != 1 or leading_count >= len(tokens):
        return None
    if not line.words[-1].text.endswith("..."):
        return None
    content_token = tokens[leading_count]
    tail_token = tokens[-1]
    if not content_token or not tail_token:
        return None
    return content_token, tail_token


def _phonetic_followup_candidate(
    *,
    word_idx: int,
    word: timing_models.TranscriptionWord,
    whisper_words: List[timing_models.TranscriptionWord],
    content_token: str,
    tail_token: str,
    line_start: float,
    prev_end: float | None,
    min_shift_sec: float,
    max_shift_sec: float,
    min_content_similarity: float,
    min_tail_similarity: float,
    min_gap: float,
) -> tuple[float, float] | None:
    token = re.sub(r"[^a-z]+", "", word.text.lower())
    if word.text == "[VOCAL]" or not token:
        return None
    shift = line_start - word.start
    if shift < min_shift_sec or shift > max_shift_sec:
        return None
    if prev_end is not None and word.start < prev_end + min_gap:
        return None
    similarity = phonetic_utils._phonetic_similarity(content_token, token, "es")
    if similarity < min_content_similarity:
        return None
    next_word = whisper_words[word_idx + 1]
    next_token = re.sub(r"[^a-z]+", "", next_word.text.lower())
    if next_word.text == "[VOCAL]" or not next_token:
        return None
    tail_similarity = phonetic_utils._phonetic_similarity(tail_token, next_token, "es")
    if tail_similarity < min_tail_similarity:
        return None
    return float(word.start), similarity + tail_similarity


def _best_truncated_followup_target(
    *,
    line: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
    prev_end: float | None,
    min_shift_sec: float,
    max_shift_sec: float,
    min_content_similarity: float,
    min_tail_similarity: float,
    min_gap: float,
) -> tuple[float, float] | None:
    tokens = _truncated_followup_tokens(line)
    if tokens is None:
        return None
    content_token, tail_token = tokens
    best_target: float | None = None
    best_score = 0.0
    for word_idx, word in enumerate(whisper_words[:-1]):
        candidate = _phonetic_followup_candidate(
            word_idx=word_idx,
            word=word,
            whisper_words=whisper_words,
            content_token=content_token,
            tail_token=tail_token,
            line_start=line.start_time,
            prev_end=prev_end,
            min_shift_sec=min_shift_sec,
            max_shift_sec=max_shift_sec,
            min_content_similarity=min_content_similarity,
            min_tail_similarity=min_tail_similarity,
            min_gap=min_gap,
        )
        if candidate is None:
            continue
        target, score = candidate
        if score > best_score:
            best_score = score
            best_target = target
    if best_target is None:
        return None
    return best_target, best_score


def _leading_light_token_slot_sec(line: models.Line) -> float:
    if len(line.words) < 2:
        return 0.0
    return max(line.words[1].start_time - line.start_time, 0.0)


def _truncated_followup_boundary_targets(
    *,
    prev_line: models.Line | None,
    line: models.Line,
    content_anchor_start: float,
    min_gap: float,
    min_existing_gap_sec: float,
    max_leading_slot_sec: float,
    max_prev_overlap_sec: float,
    min_prev_duration_sec: float,
    min_current_duration_sec: float,
) -> tuple[float | None, float]:
    if prev_line is None or not prev_line.words:
        return None, content_anchor_start
    if prev_line.end_time < content_anchor_start - max_prev_overlap_sec:
        return None, content_anchor_start
    if line.start_time - prev_line.end_time < min_existing_gap_sec:
        return None, content_anchor_start

    leading_slot = min(_leading_light_token_slot_sec(line), max_leading_slot_sec)
    if leading_slot <= 0.0:
        return None, content_anchor_start

    target_start = content_anchor_start - leading_slot
    target_prev_end = target_start - min_gap
    if target_prev_end <= prev_line.start_time + min_prev_duration_sec:
        return None, content_anchor_start
    if line.end_time <= target_start + min_current_duration_sec:
        return None, content_anchor_start
    return target_prev_end, target_start


def _allow_truncated_followup_start_only_reanchor(
    *,
    prev_line: models.Line | None,
    line: models.Line,
    content_anchor_start: float,
    candidate_score: float,
    min_gap: float,
    min_existing_gap_sec: float,
    min_candidate_score: float,
    min_prev_duration_sec: float,
    min_current_duration_sec: float,
) -> bool:
    if prev_line is None or not prev_line.words:
        return False
    if candidate_score < min_candidate_score:
        return False
    existing_gap = line.start_time - prev_line.end_time
    if existing_gap >= min_existing_gap_sec or existing_gap < min_gap:
        return False
    if content_anchor_start < prev_line.end_time + min_gap:
        return False
    if prev_line.end_time <= prev_line.start_time + min_prev_duration_sec:
        return False
    if line.end_time <= content_anchor_start + min_current_duration_sec:
        return False
    return True


def reanchor_truncated_followup_lines_from_phonetic_variants(
    mapped_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord],
    *,
    min_shift_sec: float = 0.18,
    max_shift_sec: float = 0.55,
    min_content_similarity: float = 0.55,
    min_tail_similarity: float = 0.6,
    min_gap: float = 0.05,
    min_existing_gap_sec: float = 0.4,
    max_leading_slot_sec: float = 0.24,
    max_prev_overlap_sec: float = 0.12,
    min_prev_duration_sec: float = 0.8,
    min_current_duration_sec: float = 0.9,
    min_start_only_candidate_score: float = 1.2,
) -> tuple[List[models.Line], int]:
    updated = list(mapped_lines)
    applied = 0
    for idx, line in enumerate(updated):
        prev_line = updated[idx - 1] if idx > 0 and updated[idx - 1].words else None
        prev_end = prev_line.end_time if prev_line is not None else None
        best_target = _best_truncated_followup_target(
            line=line,
            whisper_words=whisper_words,
            prev_end=prev_end,
            min_shift_sec=min_shift_sec,
            max_shift_sec=max_shift_sec,
            min_content_similarity=min_content_similarity,
            min_tail_similarity=min_tail_similarity,
            min_gap=min_gap,
        )
        if best_target is None:
            continue
        target_anchor_start, target_score = best_target
        target_prev_end, target_start = _truncated_followup_boundary_targets(
            prev_line=prev_line,
            line=line,
            content_anchor_start=target_anchor_start,
            min_gap=min_gap,
            min_existing_gap_sec=min_existing_gap_sec,
            max_leading_slot_sec=max_leading_slot_sec,
            max_prev_overlap_sec=max_prev_overlap_sec,
            min_prev_duration_sec=min_prev_duration_sec,
            min_current_duration_sec=min_current_duration_sec,
        )
        if prev_line is not None and target_prev_end is None:
            if not _allow_truncated_followup_start_only_reanchor(
                prev_line=prev_line,
                line=line,
                content_anchor_start=target_anchor_start,
                candidate_score=target_score,
                min_gap=min_gap,
                min_existing_gap_sec=min_existing_gap_sec,
                min_candidate_score=min_start_only_candidate_score,
                min_prev_duration_sec=min_prev_duration_sec,
                min_current_duration_sec=min_current_duration_sec,
            ):
                continue
            updated[idx] = rescale_line_to_new_start(line, target_anchor_start)
            applied += 1
            continue
        if target_prev_end is not None and prev_line is not None:
            updated[idx - 1] = _rescale_line_to_new_end(prev_line, target_prev_end)
        updated[idx] = rescale_line_to_new_start(line, target_start)
        applied += 1
    return updated, applied


def _is_late_compact_repetitive_tail_candidate(
    lines: List[models.Line],
    baseline_lines: List[models.Line],
    idx: int,
    *,
    baseline_anchor_tolerance: float,
    min_overlap_ratio: float = 0.66,
    min_words: int = 3,
    max_words: int = 6,
) -> bool:
    if idx <= 0 or idx >= len(lines) or idx >= len(baseline_lines):
        return False
    line = lines[idx]
    baseline = baseline_lines[idx]
    prev = lines[idx - 1]
    if (
        not line.words
        or not baseline.words
        or not prev.words
        or len(line.words) < min_words
        or len(line.words) > max_words
        or abs(line.start_time - baseline.start_time) > baseline_anchor_tolerance
    ):
        return False
    if line_token_overlap_ratio(prev, line) < min_overlap_ratio:
        return False
    if idx + 1 >= len(lines) or not lines[idx + 1].words:
        return False
    return line_token_overlap_ratio(line, lines[idx + 1]) >= min_overlap_ratio


def _match_prefix_tokens_near_word(
    *,
    prefix: list[str],
    whisper_words: List[timing_models.TranscriptionWord],
    normalized_whisper: list[str],
    start_idx: int,
    support_window: float,
) -> int:
    match_count = 1
    cursor = start_idx + 1
    last_end = whisper_words[start_idx].end
    anchor_start = whisper_words[start_idx].start
    for token in prefix[1:]:
        while cursor < len(whisper_words):
            candidate = whisper_words[cursor]
            candidate_norm = normalized_whisper[cursor]
            if candidate.start > anchor_start + support_window:
                return match_count
            cursor += 1
            if candidate.text == "[VOCAL]" or not candidate_norm:
                continue
            if candidate.start + 1e-6 < last_end:
                continue
            if candidate_norm != token:
                continue
            match_count += 1
            last_end = candidate.end
            break
        else:
            return match_count
    return match_count


def _choose_earlier_whisper_target(
    *,
    line: models.Line,
    prev_end: float | None,
    prefix: list[str],
    whisper_words: List[timing_models.TranscriptionWord],
    normalized_whisper: list[str],
    min_shift: float,
    max_shift: float,
    min_prefix_matches: int,
    min_gap: float,
    max_prev_overlap: float,
    support_window: float,
) -> float | None:
    search_start = line.start_time - max_shift
    search_end = line.start_time - min_shift
    best_target: float | None = None
    best_match_count = 0

    for word_idx, word in enumerate(whisper_words):
        if word.text == "[VOCAL]":
            continue
        if word.start < search_start or word.start > search_end:
            continue
        if normalized_whisper[word_idx] != prefix[0]:
            continue
        match_count = _match_prefix_tokens_near_word(
            prefix=prefix,
            whisper_words=whisper_words,
            normalized_whisper=normalized_whisper,
            start_idx=word_idx,
            support_window=support_window,
        )
        if match_count < min_prefix_matches:
            continue
        target_start = word.start
        if prev_end is not None:
            target_start = max(target_start, prev_end + min_gap)
            overlap = prev_end - word.start
            if overlap > max_prev_overlap:
                continue
        if target_start > line.start_time - min_shift:
            continue
        if match_count > best_match_count or (
            match_count == best_match_count
            and (best_target is None or target_start < best_target)
        ):
            best_target = target_start
            best_match_count = match_count

    return best_target


def reanchor_late_supported_lines_to_earlier_whisper(
    lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord],
    *,
    min_shift: float = 0.18,
    max_shift: float = 0.95,
    min_prefix_matches: int = 2,
    min_gap: float = 0.05,
    max_prev_overlap: float = 0.18,
    support_window: float = 1.2,
) -> tuple[List[models.Line], int]:
    adjusted = list(lines)
    applied = 0
    normalized_whisper = [
        re.sub(r"[^a-z]+", "", word.text.lower()) for word in whisper_words
    ]
    for idx, line in enumerate(adjusted):
        if len(line.words) < max(2, min_prefix_matches):
            continue
        prefix = normalized_prefix_tokens(line, limit=3)
        if len(prefix) < min_prefix_matches:
            continue

        prev_end = (
            adjusted[idx - 1].end_time if idx > 0 and adjusted[idx - 1].words else None
        )
        best_target = _choose_earlier_whisper_target(
            line=line,
            prev_end=prev_end,
            prefix=prefix,
            whisper_words=whisper_words,
            normalized_whisper=normalized_whisper,
            min_shift=min_shift,
            max_shift=max_shift,
            min_prefix_matches=min_prefix_matches,
            min_gap=min_gap,
            max_prev_overlap=max_prev_overlap,
            support_window=support_window,
        )
        if best_target is None:
            continue
        adjusted[idx] = rescale_line_to_new_start(line, best_target)
        applied += 1

    return adjusted, applied


def _choose_i_said_reanchor_start(
    line: models.Line,
    next_line: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
    onset_times: Any,
) -> Optional[float]:
    if not line.words or not next_line.words or len(line.words) < 6:
        return None
    if normalized_prefix_tokens(line)[:2] != ["i", "said"]:
        return None
    if count_non_vocal_words_near_time(whisper_words, line.start_time, window_sec=0.9):
        return None
    gap_after = next_line.start_time - line.end_time
    if gap_after < 0.0 or gap_after > 1.8:
        return None
    candidate_onsets = onset_times[
        (onset_times >= line.start_time + 0.35)
        & (onset_times <= min(line.start_time + 1.2, line.end_time - 3.5))
    ]
    if len(candidate_onsets) == 0:
        return None
    target_start = float(candidate_onsets[0])
    old_duration = line.end_time - line.start_time
    new_duration = line.end_time - target_start
    if new_duration < 3.5 or new_duration < old_duration * 0.72:
        return None
    return target_start


def choose_low_support_reanchor_start(  # noqa: C901
    line: models.Line,
    next_line: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
    onset_times: Any,
) -> Optional[float]:
    if not line.words or not next_line.words or len(line.words) < 4:
        return None
    tokens = normalized_tokens(line)
    if len(tokens) < 3:
        return None
    if normalized_prefix_tokens(line)[:2] == ["i", "said"]:
        return None
    if set(tokens) <= {"hey", "oh", "ooh", "ah", "yeah"}:
        return None

    gap_after = next_line.start_time - line.end_time
    if gap_after < 0.0 or gap_after > 2.5:
        return None

    local_density = count_non_vocal_words_near_time(
        whisper_words,
        line.start_time,
        window_sec=0.9,
    )
    if local_density > 1:
        return None
    overlap_ratio = local_lexical_overlap_ratio(
        line,
        whisper_words,
        pad_before=0.7,
        pad_after=0.7,
    )
    if overlap_ratio > 0.2:
        return None

    max_target_start = min(line.start_time + 1.4, line.end_time - 1.0)
    candidate_onsets = onset_times[
        (onset_times >= line.start_time + 0.25) & (onset_times <= max_target_start)
    ]
    if len(candidate_onsets) == 0:
        return None
    target_start = float(candidate_onsets[0])
    if target_start - line.start_time < 0.25:
        return None

    old_duration = line.end_time - line.start_time
    new_duration = line.end_time - target_start
    min_duration = max(1.2, 0.18 * len(line.words))
    if new_duration < min_duration or new_duration < old_duration * 0.68:
        return None
    return target_start


def reanchor_repeated_cadence_lines(  # noqa: C901
    mapped_lines: List[models.Line],
    *,
    min_word_count: int = 5,
    min_shift_sec: float = 0.25,
    max_shift_sec: float = 0.6,
    min_pair_gap_sec: float = 1.0,
) -> tuple[List[models.Line], int]:
    adjusted = list(mapped_lines)
    text_to_indices: dict[str, list[int]] = {}
    for idx, line in enumerate(adjusted):
        if not line.words:
            continue
        key = line_text_key(line)
        if key:
            text_to_indices.setdefault(key, []).append(idx)

    applied = 0
    for idx in range(1, len(adjusted)):
        line = adjusted[idx]
        prev_line = adjusted[idx - 1]
        if not line.words or not prev_line.words or len(line.words) < min_word_count:
            continue
        key = line_text_key(line)
        prev_key = line_text_key(prev_line)
        if not key or not prev_key:
            continue
        later_line_indices = [i for i in text_to_indices.get(key, []) if i > idx]
        later_prev_idx = next(
            (
                later_idx - 1
                for later_idx in later_line_indices
                if later_idx - 1 >= 0
                and line_text_key(adjusted[later_idx - 1]) == prev_key
            ),
            None,
        )
        if later_prev_idx is None:
            continue
        later_idx = later_prev_idx + 1

        current_gap = line.start_time - prev_line.start_time
        later_gap = adjusted[later_idx].start_time - adjusted[later_prev_idx].start_time
        desired_shift = later_gap - current_gap
        if desired_shift < min_shift_sec or desired_shift > max_shift_sec:
            continue
        if later_gap < min_pair_gap_sec or current_gap < min_pair_gap_sec:
            continue

        next_start = (
            adjusted[idx + 1].start_time
            if idx + 1 < len(adjusted) and adjusted[idx + 1].words
            else None
        )
        max_allowed = desired_shift
        if next_start is not None:
            max_allowed = min(
                max_allowed,
                max(0.0, (next_start - 0.05) - line.end_time),
            )
        if max_allowed < min_shift_sec:
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
        applied += 1
    return adjusted, applied


def shift_restored_low_support_runs_to_onset(  # noqa: C901
    mapped_lines: List[models.Line],
    baseline_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord],
    audio_features: Optional[timing_models.AudioFeatures],
    *,
    baseline_anchor_tolerance: float = 0.08,
    max_shift_sec: float = 0.65,
    min_shift_sec: float = 0.18,
    onset_window_sec: float = 0.8,
    max_run_size: int = 4,
    max_relative_index_ratio: float = 0.5,
    max_lexical_overlap_ratio: float = 0.12,
    moderate_support_skip_min: int = 2,
    moderate_support_skip_max: int = 12,
    moderate_support_long_line_word_count: int = 8,
) -> tuple[List[models.Line], int]:
    if audio_features is None or audio_features.onset_times is None:
        return mapped_lines, 0
    onset_times = audio_features.onset_times
    if len(onset_times) == 0:
        return mapped_lines, 0

    adjusted = list(mapped_lines)
    applied = 0
    idx = 0
    limit = min(len(adjusted), len(baseline_lines))
    while idx < limit:
        if idx / max(limit, 1) > max_relative_index_ratio and not (
            _is_late_compact_repetitive_tail_candidate(
                adjusted,
                baseline_lines,
                idx,
                baseline_anchor_tolerance=baseline_anchor_tolerance,
            )
        ):
            break
        line = adjusted[idx]
        base = baseline_lines[idx]
        if (
            not line.words
            or not base.words
            or len(line.words) < 4
            or abs(line.start_time - base.start_time) > baseline_anchor_tolerance
            or local_lexical_overlap_ratio(
                line, whisper_words, pad_before=0.7, pad_after=0.7
            )
            > max_lexical_overlap_ratio
        ):
            idx += 1
            continue
        support_count = count_non_vocal_words_near_time(
            whisper_words,
            line.start_time,
            window_sec=0.9,
        )
        if (
            moderate_support_skip_min <= support_count <= moderate_support_skip_max
            and len(line.words) >= moderate_support_long_line_word_count
        ):
            idx += 1
            continue

        run_end = idx + 1
        while run_end < limit and run_end - idx < max_run_size:
            cur = adjusted[run_end]
            cur_base = baseline_lines[run_end]
            prev = adjusted[run_end - 1]
            if (
                not cur.words
                or not cur_base.words
                or (
                    len(cur.words) < 4
                    and (
                        len(cur.words) < 3 or line_token_overlap_ratio(cur, line) < 0.66
                    )
                )
                or abs(cur.start_time - cur_base.start_time) > baseline_anchor_tolerance
                or cur.start_time - prev.end_time > 0.9
                or local_lexical_overlap_ratio(
                    cur, whisper_words, pad_before=0.7, pad_after=0.7
                )
                > 0.2
            ):
                break
            run_end += 1

        candidates = onset_times[
            (onset_times >= line.start_time + min_shift_sec)
            & (onset_times <= line.start_time + onset_window_sec)
        ]
        if len(candidates) == 0:
            idx = run_end
            continue
        desired_shift = min(float(candidates[0]) - line.start_time, max_shift_sec)
        if desired_shift < min_shift_sec:
            idx = run_end
            continue
        next_start = (
            adjusted[run_end].start_time
            if run_end < len(adjusted) and adjusted[run_end].words
            else None
        )
        max_allowed = desired_shift
        if next_start is not None:
            max_allowed = min(
                max_allowed,
                max(0.0, (next_start - 0.05) - adjusted[run_end - 1].end_time),
            )
        if max_allowed < min_shift_sec:
            idx = run_end
            continue

        for run_idx in range(idx, run_end):
            run_line = adjusted[run_idx]
            shifted_words = [
                models.Word(
                    text=w.text,
                    start_time=w.start_time + max_allowed,
                    end_time=w.end_time + max_allowed,
                    singer=w.singer,
                )
                for w in run_line.words
            ]
            adjusted[run_idx] = models.Line(words=shifted_words, singer=run_line.singer)
        applied += run_end - idx
        idx = run_end
    return adjusted, applied


def reanchor_unsupported_i_said_lines_to_later_onset(
    mapped_lines: List[models.Line],
    baseline_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord],
    audio_features: Optional[timing_models.AudioFeatures],
) -> tuple[List[models.Line], int]:
    if audio_features is None or audio_features.onset_times is None:
        return mapped_lines, 0
    onset_times = audio_features.onset_times
    if len(onset_times) == 0:
        return mapped_lines, 0

    updated = list(mapped_lines)
    applied = 0
    for idx in range(len(updated) - 1):
        line = updated[idx]
        next_line = updated[idx + 1]
        baseline_line = baseline_lines[idx] if idx < len(baseline_lines) else None
        if (
            baseline_line is not None
            and baseline_line.words
            and line.start_time - baseline_line.start_time > 0.75
        ):
            continue
        target_start = _choose_i_said_reanchor_start(
            line, next_line, whisper_words, onset_times
        )
        if target_start is None:
            continue
        updated[idx] = rescale_line_to_new_start(line, target_start)
        applied += 1
    return updated, applied


def reanchor_low_support_lines_to_later_onset(
    mapped_lines: List[models.Line],
    baseline_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord],
    audio_features: Optional[timing_models.AudioFeatures],
    *,
    baseline_anchor_tolerance: float = 0.08,
) -> tuple[List[models.Line], int]:
    if audio_features is None or audio_features.onset_times is None:
        return mapped_lines, 0
    onset_times = audio_features.onset_times
    if len(onset_times) == 0:
        return mapped_lines, 0

    updated = list(mapped_lines)
    applied = 0
    for idx in range(len(updated) - 1):
        line = updated[idx]
        next_line = updated[idx + 1]
        baseline_line = baseline_lines[idx] if idx < len(baseline_lines) else None
        if (
            baseline_line is not None
            and baseline_line.words
            and abs(line.start_time - baseline_line.start_time)
            > baseline_anchor_tolerance
        ):
            continue
        target_start = choose_low_support_reanchor_start(
            line,
            next_line,
            whisper_words,
            onset_times,
        )
        if target_start is None:
            continue
        updated[idx] = rescale_line_to_new_start(line, target_start)
        applied += 1
    return updated, applied


def _late_compact_tail_line_is_eligible(
    *,
    idx: int,
    limit: int,
    line: models.Line,
    baseline: models.Line,
    prev_line: models.Line,
    whisper_words: List[timing_models.TranscriptionWord],
    min_shift_from_baseline_sec: float,
    max_shift_from_baseline_sec: float,
    max_lexical_overlap_ratio: float,
    max_gap_from_prev_sec: float,
) -> bool:
    if not line.words or not baseline.words or not prev_line.words:
        return False
    if len(line.words) < 3 or len(line.words) > 6:
        return False
    shift_from_baseline = line.start_time - baseline.start_time
    if (
        shift_from_baseline < min_shift_from_baseline_sec
        or shift_from_baseline > max_shift_from_baseline_sec
    ):
        return False
    if line_token_overlap_ratio(prev_line, line) < 0.66:
        return False
    if (
        idx != limit - 1
        and line.start_time - prev_line.end_time > max_gap_from_prev_sec
    ):
        return False
    if (
        local_lexical_overlap_ratio(line, whisper_words, pad_before=0.7, pad_after=0.7)
        > max_lexical_overlap_ratio
    ):
        return False
    if count_non_vocal_words_near_time(whisper_words, line.start_time, window_sec=0.9):
        return False
    return True


def _late_compact_tail_target_start(
    *,
    line: models.Line,
    onset_times: Any,
    min_shift_sec: float,
    max_shift_sec: float,
) -> float | None:
    max_target_start = min(line.start_time + max_shift_sec, line.end_time - 0.35)
    candidates = onset_times[
        (onset_times >= line.start_time + min_shift_sec)
        & (onset_times <= max_target_start)
    ]
    if len(candidates) == 0:
        return None
    return float(candidates[0])


def _late_compact_tail_target_is_safe(
    *,
    line: models.Line,
    prev_line: models.Line,
    next_line: models.Line | None,
    target_start: float,
    min_gap: float,
    min_duration_sec: float,
    min_duration_per_word_sec: float,
) -> bool:
    if target_start < prev_line.end_time + min_gap:
        return False
    if (
        next_line is not None
        and next_line.words
        and target_start >= next_line.start_time
    ):
        return False
    new_duration = line.end_time - target_start
    min_duration = max(min_duration_sec, min_duration_per_word_sec * len(line.words))
    return new_duration >= min_duration


def reanchor_late_compact_repetitive_tail_lines_to_later_onsets(
    mapped_lines: List[models.Line],
    baseline_lines: List[models.Line],
    whisper_words: List[timing_models.TranscriptionWord],
    audio_features: Optional[timing_models.AudioFeatures],
    *,
    min_shift_from_baseline_sec: float = 0.3,
    max_shift_from_baseline_sec: float = 0.75,
    min_shift_sec: float = 0.25,
    max_shift_sec: float = 0.8,
    max_lexical_overlap_ratio: float = 0.05,
    max_gap_from_prev_sec: float = 0.12,
    min_gap: float = 0.05,
    min_duration_sec: float = 0.75,
    min_duration_per_word_sec: float = 0.15,
) -> tuple[List[models.Line], int]:
    if audio_features is None or audio_features.onset_times is None:
        return mapped_lines, 0
    onset_times = audio_features.onset_times
    if len(onset_times) == 0:
        return mapped_lines, 0

    updated = list(mapped_lines)
    applied = 0
    limit = min(len(updated), len(baseline_lines))
    for idx in range(1, limit):
        line = updated[idx]
        baseline = baseline_lines[idx]
        prev_line = updated[idx - 1]
        next_line = updated[idx + 1] if idx + 1 < len(updated) else None
        if not _late_compact_tail_line_is_eligible(
            idx=idx,
            limit=limit,
            line=line,
            baseline=baseline,
            prev_line=prev_line,
            whisper_words=whisper_words,
            min_shift_from_baseline_sec=min_shift_from_baseline_sec,
            max_shift_from_baseline_sec=max_shift_from_baseline_sec,
            max_lexical_overlap_ratio=max_lexical_overlap_ratio,
            max_gap_from_prev_sec=max_gap_from_prev_sec,
        ):
            continue
        target_start = _late_compact_tail_target_start(
            line=line,
            onset_times=onset_times,
            min_shift_sec=min_shift_sec,
            max_shift_sec=max_shift_sec,
        )
        if target_start is None:
            continue
        if not _late_compact_tail_target_is_safe(
            line=line,
            prev_line=prev_line,
            next_line=next_line,
            target_start=target_start,
            min_gap=min_gap,
            min_duration_sec=min_duration_sec,
            min_duration_per_word_sec=min_duration_per_word_sec,
        ):
            continue
        updated[idx] = rescale_line_to_new_start(line, target_start)
        applied += 1
    return updated, applied
