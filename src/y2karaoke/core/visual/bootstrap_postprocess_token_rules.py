"""Token-level fallback and normalization rules for visual postprocessing."""

from __future__ import annotations

import re
from typing import Callable


def looks_fused_prefix_candidate(
    token: str, *, fused_fallback_prefix_anchors: tuple[str, ...]
) -> bool:
    low = token.strip().lower()
    if not low.isalpha() or len(low) < 5:
        return False
    for anchor in fused_fallback_prefix_anchors:
        if low.startswith(anchor) and len(low) > len(anchor) + 2:
            return True
    return False


def fallback_split_fused_token(  # noqa: C901
    token: str,
    *,
    fused_fallback_short_functions: tuple[str, ...],
    fused_fallback_prefix_anchors: tuple[str, ...],
    fused_fallback_suffix_anchors: tuple[str, ...],
    fallback_spell_validated_split_fn: Callable[[str], list[str] | None],
    repair_fallback_token_fn: Callable[[str], str],
) -> list[str] | None:
    if not token or " " in token:
        return None
    if not token.isalpha():
        return None
    lower = token.lower()
    if len(lower) < 5:
        for i in range(1, len(lower)):
            left = lower[:i]
            right = lower[i:]
            if left in fused_fallback_short_functions and right in (
                fused_fallback_short_functions
            ):
                return ["I" if left == "i" else left, "I" if right == "i" else right]
        return None

    for anchor in fused_fallback_prefix_anchors:
        if not lower.startswith(anchor):
            continue
        if len(lower) <= len(anchor):
            continue
        right = lower[len(anchor) :]
        min_right_len = 4 if len(anchor) == 1 else 3
        if len(right) < min_right_len:
            continue
        if sum(ch in "aeiouy" for ch in right) < 1:
            continue
        left_out = "I" if anchor == "i" else anchor
        return [left_out, right]

    if len(lower) >= 8:
        for anchor in ("a",):
            for idx in range(2, len(lower) - 2):
                if lower[idx] != anchor:
                    continue
                left = lower[:idx]
                right = lower[idx + 1 :]
                if len(left) < 4 or len(right) < 4:
                    continue
                if not right.endswith("in"):
                    continue
                if sum(ch in "aeiouy" for ch in left) < 1:
                    continue
                if sum(ch in "aeiouy" for ch in right) < 1:
                    continue
                right_parts = fallback_split_fused_token(
                    right,
                    fused_fallback_short_functions=fused_fallback_short_functions,
                    fused_fallback_prefix_anchors=fused_fallback_prefix_anchors,
                    fused_fallback_suffix_anchors=fused_fallback_suffix_anchors,
                    fallback_spell_validated_split_fn=fallback_spell_validated_split_fn,
                    repair_fallback_token_fn=repair_fallback_token_fn,
                )
                if right_parts:
                    return [left, anchor, *right_parts]
                return [left, anchor, repair_fallback_token_fn(right)]

    for anchor in fused_fallback_suffix_anchors:
        if not lower.endswith(anchor):
            continue
        left = lower[: -len(anchor)]
        if len(left) < 4:
            continue
        if sum(ch in "aeiouy" for ch in left) < 1:
            continue
        if anchor == "in" and not left[-1].isalpha():
            continue
        if anchor == "in" and left.endswith("ng"):
            return [repair_fallback_token_fn(lower)]
        if anchor == "in" and not left.endswith("r"):
            continue
        return [left, anchor]

    if lower.endswith("i") and len(lower) >= 6:
        left = lower[:-1]
        if sum(ch in "aeiouy" for ch in left) >= 1 and left.isalpha():
            return [left, "I"]

    spell_split = fallback_spell_validated_split_fn(lower)
    if spell_split:
        return spell_split
    return None


def repair_fallback_token(token: str) -> str:
    lower = token.lower()
    if len(lower) >= 6 and lower.endswith("in") and lower[:-2].endswith("ng"):
        return lower[:-2] + "ing"
    return token


def fallback_spell_validated_split(
    token: str,
    *,
    is_spelled_word_fn: Callable[[str], bool],
    fused_fallback_short_functions: tuple[str, ...],
) -> list[str] | None:
    if not token.isalpha() or len(token) < 7:
        return None
    if is_spelled_word_fn(token):
        return None

    best = _best_spell_validated_split(
        token,
        is_spelled_word_fn=is_spelled_word_fn,
        fused_fallback_short_functions=fused_fallback_short_functions,
    )

    if best is None:
        return None
    left, right = best[1]
    return ["I" if left == "i" else left, "I" if right == "i" else right]


def contextual_compound_split(
    token: str,
    next_token: str,
    confidence: float,
    *,
    normalize_text_basic_fn: Callable[[str], str],
    lyric_function_words: set[str],
    is_spelled_word_fn: Callable[[str], bool],
    case_like_fn: Callable[[str, str], str],
) -> list[str] | None:
    low = token.lower()
    next_norm = normalize_text_basic_fn(next_token or "")
    if confidence > 0.55:
        return None
    if not low.isalpha() or not (8 <= len(low) <= 12):
        return None
    if not next_norm or next_norm not in lyric_function_words:
        return None
    if not is_spelled_word_fn(low):
        return None

    best = _best_contextual_compound_split(low, is_spelled_word_fn=is_spelled_word_fn)

    if best is None:
        return None
    left, right = best[1]
    return [case_like_fn(token, left), right]


def maybe_expand_colloquial_token(
    text: str,
    confidence: float,
    *,
    colloquial_expansions: dict[str, tuple[str, str]],
    case_like_fn: Callable[[str, str], str],
) -> list[str] | None:
    token = text.strip()
    if not token:
        return None
    if confidence > 0.55:
        return None

    low = token.lower()
    exp = colloquial_expansions.get(low)
    if exp:
        first, second = exp
        return [case_like_fn(token, first), second]

    compact = low.replace("’", "'")
    if re.fullmatch(r"[a-z]{3,}in'", compact):
        return [case_like_fn(token, compact[:-1] + "g")]
    return None


def maybe_restore_contraction_token(
    text: str,
    confidence: float,
    *,
    contraction_restores: dict[str, str],
    case_like_fn: Callable[[str, str], str],
) -> str | None:
    token = text.strip()
    if confidence > 0.55:
        return None
    restored = contraction_restores.get(token.lower())
    if not restored:
        return None
    return case_like_fn(token, restored)


def maybe_restore_contextual_contraction_token(
    text: str,
    confidence: float,
    prev_token: str,
    next_token: str,
    *,
    normalize_text_basic_fn: Callable[[str], str],
    case_like_fn: Callable[[str, str], str],
) -> str | None:
    token = text.strip()
    if not token or confidence > 0.4:
        return None
    low = token.lower()
    if low != "i":
        return None

    prev_norm = normalize_text_basic_fn(prev_token or "")
    next_norm = normalize_text_basic_fn(next_token or "")
    if not next_norm:
        return None

    participle_like = (
        next_norm in {"been", "seen", "done", "gone", "known", "grown", "shown"}
        or next_norm.endswith("ed")
        or next_norm.endswith("en")
    )
    if not participle_like:
        return None
    if next_norm in {"feel", "know", "want", "need", "think", "am", "was"}:
        return None

    prev_support = bool(prev_norm) and prev_norm not in {
        "and",
        "but",
        "or",
        "if",
        "that",
        "because",
        "when",
        "while",
    }
    if not prev_support and next_norm not in {"been", "seen", "done", "gone"}:
        return None

    return case_like_fn(token, "i've")


def maybe_split_fused_contraction_token(token: str) -> list[str] | None:
    t = token.strip()
    if not t:
        return None
    low = t.lower().replace("’", "'")
    if low in {"i'min", "imin"}:
        return ["i'm", "in"]
    if low in {"andi'm", "andim"}:
        return ["and", "i'm"]
    return None


def maybe_contextual_inflection_token(
    text: str,
    confidence: float,
    prev_token: str,
    next_token: str,
    *,
    normalize_text_basic_fn: Callable[[str], str],
    is_spelled_word_fn: Callable[[str], bool],
    case_like_fn: Callable[[str, str], str],
) -> str | None:
    token = text.strip()
    low = token.lower()
    if confidence > 0.55 or not low.isalpha() or len(low) < 2:
        return None
    prev_norm = normalize_text_basic_fn(prev_token or "")
    next_norm = normalize_text_basic_fn(next_token or "")

    transformed = _contextual_inflection_basic_variants(
        low,
        prev_norm,
        next_norm,
        is_spelled_word_fn=is_spelled_word_fn,
    )
    if transformed is not None:
        return case_like_fn(token, transformed)
    if prev_norm == "" and low == "aw" and next_norm in {"who", "what", "why", "yeah"}:
        return case_like_fn(token, "oh")
    transformed = _contextual_inflection_prefix_variant(
        low,
        prev_norm,
        is_spelled_word_fn=is_spelled_word_fn,
    )
    if transformed is not None:
        return case_like_fn(token, transformed)

    return None


def _contextual_inflection_basic_variants(
    low: str,
    prev_norm: str,
    next_norm: str,
    *,
    is_spelled_word_fn: Callable[[str], bool],
) -> str | None:
    if next_norm in {"are", "were"} and is_spelled_word_fn(low + "s"):
        return low + "s"
    if (
        next_norm in {"is", "was"}
        and low.endswith("s")
        and is_spelled_word_fn(low[:-1])
    ):
        return low[:-1]
    if next_norm == "as" and is_spelled_word_fn(low + "ed"):
        return low + "ed"
    if (
        prev_norm in {"am", "is", "are", "was", "were", "be", "been"}
        and low.endswith("in")
        and is_spelled_word_fn(low + "g")
    ):
        return low + "g"
    if (
        next_norm in {"down", "up", "away"}
        and not is_spelled_word_fn(low)
        and is_spelled_word_fn(low + "w")
    ):
        return low + "w"
    return None


def _contextual_inflection_prefix_variant(
    low: str,
    prev_norm: str,
    *,
    is_spelled_word_fn: Callable[[str], bool],
) -> str | None:
    if prev_norm not in {"saint", "st"} or len(low) < 3 or is_spelled_word_fn(low):
        return None
    return _single_prefixed_candidate(low, is_spelled_word_fn=is_spelled_word_fn)


def _spell_split_score(
    left: str,
    right: str,
    *,
    fused_fallback_short_functions: tuple[str, ...],
) -> int:
    score = 0
    if left in fused_fallback_short_functions:
        score += 6
    if right in fused_fallback_short_functions:
        score += 6
    if len(left) <= 4:
        score += 2
    if len(right) <= 4:
        score += 2
    if abs(len(left) - len(right)) <= 2:
        score += 1
    return score


def _best_spell_validated_split(
    token: str,
    *,
    is_spelled_word_fn: Callable[[str], bool],
    fused_fallback_short_functions: tuple[str, ...],
) -> tuple[int, tuple[str, str]] | None:
    best: tuple[int, tuple[str, str]] | None = None
    for i in range(2, len(token) - 1):
        left = token[:i]
        right = token[i:]
        if len(left) < 2 or len(right) < 2:
            continue
        if not (is_spelled_word_fn(left) and is_spelled_word_fn(right)):
            continue
        score = _spell_split_score(
            left, right, fused_fallback_short_functions=fused_fallback_short_functions
        )
        if score < 4:
            continue
        if best is None or score > best[0]:
            best = (score, (left, right))
    return best


def _compound_split_score(left: str, right: str) -> int:
    score = 0
    if len(left) == len(right):
        score += 3
    if right.endswith(("ing", "ed", "s")):
        score -= 3
    if right.endswith(("ive", "all", "ight", "ove")):
        score += 1
    return score


def _best_contextual_compound_split(
    token: str,
    *,
    is_spelled_word_fn: Callable[[str], bool],
) -> tuple[int, tuple[str, str]] | None:
    best: tuple[int, tuple[str, str]] | None = None
    for i in range(4, len(token) - 3):
        left = token[:i]
        right = token[i:]
        if len(left) < 4 or len(right) < 4:
            continue
        if abs(len(left) - len(right)) > 1:
            continue
        if not (is_spelled_word_fn(left) and is_spelled_word_fn(right)):
            continue
        score = _compound_split_score(left, right)
        if score < 2:
            continue
        if best is None or score > best[0]:
            best = (score, (left, right))
    return best


def _single_prefixed_candidate(
    token: str,
    *,
    is_spelled_word_fn: Callable[[str], bool],
) -> str | None:
    candidates = [
        ch + token
        for ch in "abcdefghijklmnopqrstuvwxyz"
        if is_spelled_word_fn(ch + token)
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None
