"""Shared text/token helpers for Whisper mapping post-processing."""

from __future__ import annotations

import re
from typing import List

_INTERJECTION_TOKENS = {
    "ooh",
    "oh",
    "ah",
    "aah",
    "mmm",
    "mm",
    "uh",
    "uhh",
    "la",
    "na",
}


def _normalize_interjection_token(token: str) -> str:
    cleaned = "".join(ch for ch in token.lower() if ch.isalpha())
    if not cleaned:
        return ""
    return re.sub(r"(.)\1{2,}", r"\1\1", cleaned)


def _is_interjection_line(text: str, max_tokens: int = 3) -> bool:
    tokens = [_normalize_interjection_token(t) for t in text.split()]
    tokens = [t for t in tokens if t]
    if not tokens or len(tokens) > max_tokens:
        return False
    return all(t in _INTERJECTION_TOKENS for t in tokens)


def _interjection_similarity(line_text: str, seg_text: str) -> float:
    line_tokens = [_normalize_interjection_token(t) for t in line_text.split()]
    seg_tokens = [_normalize_interjection_token(t) for t in seg_text.split()]
    line_tokens = [t for t in line_tokens if t]
    seg_tokens = [t for t in seg_tokens if t]
    if not line_tokens or not seg_tokens:
        return 0.0
    if len(line_tokens) == 1 and len(seg_tokens) == 1:
        if line_tokens[0] == seg_tokens[0]:
            return 1.0
        if line_tokens[0] in seg_tokens[0] or seg_tokens[0] in line_tokens[0]:
            return 0.9
    overlap = len(set(line_tokens) & set(seg_tokens))
    return overlap / max(len(set(line_tokens)), len(set(seg_tokens)))


def _normalize_text_tokens(text: str) -> List[str]:
    tokens = []
    for raw in text.lower().split():
        tok = "".join(ch for ch in raw if ch.isalpha())
        if tok:
            tokens.append(re.sub(r"(.)\1{2,}", r"\1\1", tok))
    return tokens


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


def _contains_token_sequence(
    needle_text: str,
    haystack_text: str,
    *,
    min_tokens: int = 3,
) -> bool:
    needle = _normalize_text_tokens(needle_text)
    haystack = _normalize_text_tokens(haystack_text)
    if len(needle) < min_tokens or len(haystack) < len(needle):
        return False
    for start in range(0, len(haystack) - len(needle) + 1):
        ok = True
        for offset, tok in enumerate(needle):
            if not _soft_token_match(tok, haystack[start + offset]):
                ok = False
                break
        if ok:
            return True
    return False


def _max_contiguous_soft_match_run(needle_text: str, haystack_text: str) -> int:
    needle = _normalize_text_tokens(needle_text)
    haystack = _normalize_text_tokens(haystack_text)
    if not needle or not haystack:
        return 0
    best = 0
    for ni in range(len(needle)):
        for hi in range(len(haystack)):
            run = 0
            while (
                ni + run < len(needle)
                and hi + run < len(haystack)
                and _soft_token_match(needle[ni + run], haystack[hi + run])
            ):
                run += 1
            if run > best:
                best = run
    return best


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
    for left in left_tokens:
        for idx, right in enumerate(right_tokens):
            if used[idx]:
                continue
            if _soft_token_match(left, right):
                used[idx] = True
                matched += 1
                break
    return matched / max(len(left_tokens), len(right_tokens))


def _is_placeholder_whisper_token(text: str) -> bool:
    cleaned = "".join(ch for ch in text.lower() if ch.isalpha())
    return cleaned in {"vocal", "silence", "gap"}
