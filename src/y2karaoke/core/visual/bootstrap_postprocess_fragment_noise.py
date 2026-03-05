"""Fragment-noise suppression helpers for bootstrap postprocess outputs."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Callable


def tokens_contiguous_subphrase(needle: list[str], haystack: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    n = len(needle)
    for i in range(0, len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return True
    return False


def neighbor_supports_fragment_tokens(
    fragment_tokens: list[str],
    neighbor_tokens: list[str],
    *,
    tokens_contiguous_subphrase_fn: Callable[[list[str], list[str]], bool],
) -> bool:
    if tokens_contiguous_subphrase_fn(fragment_tokens, neighbor_tokens):
        return True

    return _supports_merged_fragment_token(
        fragment_tokens, neighbor_tokens
    ) or _supports_singular_plural_variant(fragment_tokens, neighbor_tokens)


def _supports_merged_fragment_token(
    fragment_tokens: list[str], neighbor_tokens: list[str]
) -> bool:
    if not (2 <= len(fragment_tokens) <= 3):
        return False
    if not all(1 <= len(token) <= 4 for token in fragment_tokens):
        return False
    merged = "".join(fragment_tokens)
    if len(merged) < 5:
        return False
    return any(
        token == merged
        or (
            len(token) >= len(merged)
            and SequenceMatcher(None, merged, token).ratio() >= 0.84
        )
        for token in neighbor_tokens
    )


def _supports_singular_plural_variant(
    fragment_tokens: list[str], neighbor_tokens: list[str]
) -> bool:
    if len(fragment_tokens) != 1:
        return False
    token = fragment_tokens[0]
    if len(token) < 4:
        return False
    singular = token[:-1] if token.endswith("s") else token
    plural = token if token.endswith("s") else f"{token}s"
    return any(
        neighbor == singular or neighbor == plural for neighbor in neighbor_tokens
    )


def remove_repeated_fragment_noise_lines(  # noqa: C901
    lines_out: list[dict[str, Any]],
    *,
    artist: str | None,
    title: str | None,
    normalize_text_basic_fn: Callable[[str], str],
    lyric_function_words: set[str],
    vocalization_noise_tokens: set[str],
    neighbor_supports_fragment_tokens_fn: Callable[[list[str], list[str]], bool],
) -> None:
    if len(lines_out) < 4:
        return

    artist_parts = set(normalize_text_basic_fn(artist or "").split())
    title_parts = set(normalize_text_basic_fn(title or "").split())
    protected = {t for t in (artist_parts | title_parts) if t}

    line_norms = [normalize_text_basic_fn(str(ln.get("text", ""))) for ln in lines_out]
    line_tokens = [[t for t in n.split() if t] for n in line_norms]

    frag_counts: dict[tuple[str, ...], int] = {}
    frag_conf_sums: dict[tuple[str, ...], float] = {}
    frag_indices: dict[tuple[str, ...], list[int]] = {}
    for idx, (ln, toks) in enumerate(zip(lines_out, line_tokens)):
        if not (1 <= len(toks) <= 3):
            continue
        if any(t in protected for t in toks):
            continue
        if all(
            t in lyric_function_words or t in vocalization_noise_tokens for t in toks
        ):
            continue
        key = tuple(toks)
        frag_counts[key] = frag_counts.get(key, 0) + 1
        frag_conf_sums[key] = frag_conf_sums.get(key, 0.0) + float(
            ln.get("confidence", 0.0) or 0.0
        )
        frag_indices.setdefault(key, []).append(idx)

    drops: set[int] = set()
    for key, count in frag_counts.items():
        avg_conf = frag_conf_sums[key] / max(count, 1)
        min_count = 3
        if count >= 2 and len(key) <= 2 and avg_conf <= 0.25:
            min_count = 2
        allow_high_conf_refrain_frag = count >= 5 and len(key) <= 3 and avg_conf <= 0.9
        if count < min_count:
            continue
        if avg_conf > 0.4 and not allow_high_conf_refrain_frag:
            continue

        for idx in frag_indices[key]:
            if idx in drops:
                continue
            ln = lines_out[idx]
            start = float(ln.get("start", 0.0) or 0.0)
            end = float(ln.get("end", start) or start)
            if end - start > 2.2:
                continue

            supported_by_neighbor = False
            for j in range(max(0, idx - 3), min(len(lines_out), idx + 4)):
                if j == idx:
                    continue
                neigh_toks = line_tokens[j]
                if len(neigh_toks) <= len(key):
                    continue
                n_conf = float(lines_out[j].get("confidence", 0.0) or 0.0)
                if n_conf + 0.1 < float(ln.get("confidence", 0.0) or 0.0):
                    continue
                if neighbor_supports_fragment_tokens_fn(list(key), neigh_toks):
                    supported_by_neighbor = True
                    break
            if not supported_by_neighbor:
                continue
            drops.add(idx)

    if not drops:
        return

    lines_out[:] = [ln for idx, ln in enumerate(lines_out) if idx not in drops]
    for i, ln in enumerate(lines_out):
        ln["line_index"] = i + 1
