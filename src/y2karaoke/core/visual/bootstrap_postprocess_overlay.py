"""Overlay/credit line detection for bootstrap postprocess outputs."""

from __future__ import annotations

from typing import Any, Callable


def overlay_line_signal_score(  # noqa: C901
    line: dict[str, Any],
    *,
    normalize_text_basic_fn: Callable[[str], str],
    overlay_platform_tokens: set[str],
    overlay_cta_tokens: set[str],
    overlay_legal_tokens: set[str],
    overlay_brand_tokens: set[str],
) -> int:
    text = str(line.get("text", "") or "")
    if not text:
        return 0
    words = line.get("words", [])
    n_words = len(words)
    norm = normalize_text_basic_fn(text)
    toks = [t for t in norm.split() if t]
    if not toks:
        return 0

    text_lower = text.lower()
    token_set = set(toks)
    score = 0

    if n_words >= 6:
        score += 1
    if n_words >= 10:
        score += 1

    platform_hits = len(token_set & overlay_platform_tokens)
    cta_hits = len(token_set & overlay_cta_tokens)
    legal_hits = len(token_set & overlay_legal_tokens)
    brand_hits = len(token_set & overlay_brand_tokens)

    if platform_hits:
        score += 3
    if platform_hits >= 2:
        score += 1
    if cta_hits and platform_hits:
        score += 3
    elif cta_hits >= 2 and n_words >= 8:
        score += 2
    if legal_hits >= 2:
        score += 3
    elif legal_hits and ("reserved" in token_set or "copyright" in token_set):
        score += 2
    if brand_hits and (platform_hits or cta_hits or legal_hits):
        score += 2

    urlish = (
        "www" in text_lower
        or ".com" in text_lower
        or ".co.uk" in text_lower
        or ".co" in text_lower
    )
    if urlish:
        score += 4

    if "all rights reserved" in text_lower:
        score += 4
    if "follow us" in text_lower or "like us" in text_lower:
        score += 3
    if "produced by" in text_lower:
        score += 2
    if "in association with" in text_lower:
        score += 3

    alnum_long_tokens = 0
    for raw in [str(w.get("text", "")) for w in words]:
        raw_low = raw.lower()
        if len(raw_low) >= 10 and any(ch.isdigit() for ch in raw_low):
            alnum_long_tokens += 1
        if any(ch in "./" for ch in raw_low) and len(raw_low) >= 6:
            alnum_long_tokens += 1
    if alnum_long_tokens:
        score += 2

    return score


def remove_overlay_credit_lines(
    lines_out: list[dict[str, Any]],
    *,
    overlay_line_signal_score_fn: Callable[[dict[str, Any]], int],
) -> None:
    if not lines_out:
        return

    kept: list[dict[str, Any]] = []
    for ln in lines_out:
        score = overlay_line_signal_score_fn(ln)
        if score >= 6:
            continue
        kept.append(ln)

    if len(kept) == len(lines_out):
        return
    lines_out[:] = kept
    for i, ln in enumerate(lines_out):
        ln["line_index"] = i + 1
