"""Chant-noise detection and cleanup helpers for bootstrap postprocessing."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Callable

from ..text_utils import normalize_text_basic


def is_chant_noise_signature(
    tokens: list[str],
    *,
    vocalization_noise_tokens: set[str],
    common_lyric_chant_tokens: set[str],
) -> bool:
    if len(tokens) < 2:
        return False
    if len(tokens) > 8:
        return False
    if any(len(t) < 2 for t in tokens):
        return False
    if any(t in vocalization_noise_tokens for t in tokens):
        return False
    base = tokens[0]
    if base in common_lyric_chant_tokens:
        return False
    if len(base) > 5:
        return False
    for tok in tokens[1:]:
        if SequenceMatcher(None, base, tok).ratio() < 0.75:
            return False
        if tok in common_lyric_chant_tokens:
            return False
    return True


def remove_repeated_chant_noise_lines(  # noqa: C901
    lines_out: list[dict[str, Any]],
    *,
    is_chant_noise_signature_fn: Callable[[list[str]], bool],
) -> None:
    if len(lines_out) < 4:
        return

    norms = [normalize_text_basic(str(ln.get("text", ""))) for ln in lines_out]
    token_lists = [[t for t in n.split() if t] for n in norms]
    sig_counts: dict[tuple[str, ...], int] = {}
    sig_indices: dict[tuple[str, ...], list[int]] = {}
    sig_conf_sums: dict[tuple[str, ...], float] = {}

    for idx, (ln, toks) in enumerate(zip(lines_out, token_lists)):
        if not is_chant_noise_signature_fn(toks):
            continue
        base = min(toks, key=len)
        sig: tuple[str, ...] = (base[:4],)
        sig_counts[sig] = sig_counts.get(sig, 0) + 1
        sig_indices.setdefault(sig, []).append(idx)
        sig_conf_sums[sig] = sig_conf_sums.get(sig, 0.0) + float(
            ln.get("confidence", 0.0) or 0.0
        )

    drops: set[int] = set()
    for sig, count in sig_counts.items():
        if count < 3:
            continue
        avg_conf = sig_conf_sums[sig] / max(count, 1)
        if avg_conf > 0.4 and not (count >= 6 and avg_conf <= 0.55):
            continue
        for idx in sig_indices[sig]:
            ln = lines_out[idx]
            start = float(ln.get("start", 0.0) or 0.0)
            end = float(ln.get("end", start) or start)
            if end - start > 2.4:
                continue
            chant_token = sig[0]
            neighbor_mentions = False
            for j in range(max(0, idx - 2), min(len(lines_out), idx + 3)):
                if j == idx:
                    continue
                n_words = token_lists[j]
                if len(n_words) <= len(token_lists[idx]):
                    continue
                if is_chant_noise_signature_fn(n_words):
                    continue
                if chant_token in n_words:
                    neighbor_mentions = True
                    break
            if neighbor_mentions:
                continue
            drops.add(idx)

    if not drops:
        return
    lines_out[:] = [ln for idx, ln in enumerate(lines_out) if idx not in drops]
    for i, ln in enumerate(lines_out):
        ln["line_index"] = i + 1


def collect_chant_signature_stats(
    lines_out: list[dict[str, Any]],
    token_lists: list[list[str]],
    *,
    is_chant_noise_signature_fn: Callable[[list[str]], bool],
) -> tuple[
    dict[tuple[str, ...], list[int]],
    dict[tuple[str, ...], float],
    dict[tuple[str, ...], float],
]:
    sig_indices: dict[tuple[str, ...], list[int]] = {}
    sig_conf_sums: dict[tuple[str, ...], float] = {}
    sig_dur_sums: dict[tuple[str, ...], float] = {}
    for idx, (ln, toks) in enumerate(zip(lines_out, token_lists)):
        if not is_chant_noise_signature_fn(toks):
            continue
        base = min(toks, key=len)
        sig = (base[:4],)
        sig_indices.setdefault(sig, []).append(idx)
        sig_conf_sums[sig] = sig_conf_sums.get(sig, 0.0) + float(
            ln.get("confidence", 0.0) or 0.0
        )
        start = float(ln.get("start", 0.0) or 0.0)
        end = float(ln.get("end", start) or start)
        sig_dur_sums[sig] = sig_dur_sums.get(sig, 0.0) + max(0.0, end - start)
    return sig_indices, sig_conf_sums, sig_dur_sums


def line_has_neighbor_chant_token_support(
    idx: int,
    root: str,
    lines_out: list[dict[str, Any]],
    token_lists: list[list[str]],
    *,
    is_chant_noise_signature_fn: Callable[[list[str]], bool],
) -> bool:
    for j in range(max(0, idx - 2), min(len(lines_out), idx + 3)):
        if j == idx:
            continue
        n_words = token_lists[j]
        if len(n_words) <= len(token_lists[idx]):
            continue
        if is_chant_noise_signature_fn(n_words):
            continue
        if root in n_words:
            return True
    return False


def should_drop_high_repeat_chant_signature(
    sig: tuple[str, ...],
    idxs: list[int],
    sig_conf_sums: dict[tuple[str, ...], float],
    sig_dur_sums: dict[tuple[str, ...], float],
    *,
    is_spelled_word_fn: Callable[[str], bool],
) -> bool:
    count = len(idxs)
    if count < 5:
        return False
    root = sig[0]
    if is_spelled_word_fn(root) and count < 8:
        return False
    avg_conf = sig_conf_sums[sig] / max(count, 1)
    avg_dur = sig_dur_sums[sig] / max(count, 1)
    return avg_conf <= 0.95 and avg_dur <= 1.6


def remove_high_repeat_nonlexical_chant_noise_lines(
    lines_out: list[dict[str, Any]],
    *,
    collect_chant_signature_stats_fn: Callable[
        [list[dict[str, Any]], list[list[str]]],
        tuple[
            dict[tuple[str, ...], list[int]],
            dict[tuple[str, ...], float],
            dict[tuple[str, ...], float],
        ],
    ],
    should_drop_high_repeat_chant_signature_fn: Callable[
        [
            tuple[str, ...],
            list[int],
            dict[tuple[str, ...], float],
            dict[tuple[str, ...], float],
        ],
        bool,
    ],
    line_has_neighbor_chant_token_support_fn: Callable[
        [int, str, list[dict[str, Any]], list[list[str]]], bool
    ],
) -> None:
    if len(lines_out) < 5:
        return

    norms = [normalize_text_basic(str(ln.get("text", ""))) for ln in lines_out]
    token_lists = [[t for t in n.split() if t] for n in norms]
    sig_indices, sig_conf_sums, sig_dur_sums = collect_chant_signature_stats_fn(
        lines_out, token_lists
    )

    drops: set[int] = set()
    for sig, idxs in sig_indices.items():
        if not should_drop_high_repeat_chant_signature_fn(
            sig, idxs, sig_conf_sums, sig_dur_sums
        ):
            continue
        root = sig[0]

        for idx in idxs:
            ln = lines_out[idx]
            start = float(ln.get("start", 0.0) or 0.0)
            end = float(ln.get("end", start) or start)
            if end - start > 2.4:
                continue
            if line_has_neighbor_chant_token_support_fn(
                idx, root, lines_out, token_lists
            ):
                continue
            drops.add(idx)

    if not drops:
        return
    lines_out[:] = [ln for idx, ln in enumerate(lines_out) if idx not in drops]
    for i, ln in enumerate(lines_out):
        ln["line_index"] = i + 1
