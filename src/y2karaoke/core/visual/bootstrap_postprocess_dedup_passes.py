"""Near-duplicate suppression and repeated-line canonicalization passes."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from ..text_utils import normalize_text_basic
from .bootstrap_postprocess_repeat_cluster_helpers import (
    _repair_repeat_cluster_tokenization_variants,
)


def line_duplicate_quality_score(line: dict[str, Any]) -> float:
    conf = float(line.get("confidence", 0.0) or 0.0)
    meta = line.get("_reconstruction_meta", {})
    uncertainty = 0.0
    support = 1.0
    if isinstance(meta, dict):
        uncertainty = float(meta.get("uncertainty_score", 0.0) or 0.0)
        support = float(meta.get("selected_text_support_ratio", 1.0) or 0.0)
    return (0.8 * conf) + (0.4 * support) - (0.8 * uncertainty)


def line_uncertainty(line: dict[str, Any]) -> float:
    meta = line.get("_reconstruction_meta", {})
    if not isinstance(meta, dict):
        return 0.0
    return float(meta.get("uncertainty_score", 0.0) or 0.0)


def remove_weaker_near_duplicate_lines(  # noqa: C901
    lines_out: list[dict[str, Any]],
) -> None:
    if len(lines_out) < 2:
        return

    norms = [normalize_text_basic(str(ln.get("text", ""))) for ln in lines_out]
    token_counts = [len([t for t in n.split() if t]) for n in norms]
    drops: set[int] = set()

    for i, base in enumerate(lines_out):
        if i in drops:
            continue
        base_norm = norms[i]
        if not base_norm:
            continue
        base_count = token_counts[i]
        if base_count < 3:
            continue
        base_start = float(base.get("start", 0.0) or 0.0)
        base_block = None
        base_meta = base.get("_reconstruction_meta", {})
        if isinstance(base_meta, dict):
            bf = base_meta.get("block_first")
            if isinstance(bf, dict):
                base_block = bf.get("block_id")
        base_vs = base.get("_visibility_start")
        base_ve = base.get("_visibility_end")
        base_y = float(base.get("y", 0.0) or 0.0)

        for j in range(i + 1, min(i + 5, len(lines_out))):
            if j in drops:
                continue
            cand_norm = norms[j]
            if not cand_norm:
                continue
            cand_count = token_counts[j]
            if cand_count < 3:
                continue
            if abs(cand_count - base_count) > 4:
                continue
            cand_start = float(lines_out[j].get("start", 0.0) or 0.0)
            if cand_start - base_start > 12.0:
                break
            cand = lines_out[j]
            cand_meta = cand.get("_reconstruction_meta", {})
            cand_block = None
            if isinstance(cand_meta, dict):
                bf = cand_meta.get("block_first")
                if isinstance(bf, dict):
                    cand_block = bf.get("block_id")
            if (
                base_block is not None
                and cand_block is not None
                and base_block == cand_block
            ):
                continue
            cand_vs = cand.get("_visibility_start")
            cand_ve = cand.get("_visibility_end")
            cand_y = float(cand.get("y", 0.0) or 0.0)
            if (
                base_vs is not None
                and base_ve is not None
                and cand_vs is not None
                and cand_ve is not None
            ):
                overlap = min(float(base_ve), float(cand_ve)) - max(
                    float(base_vs), float(cand_vs)
                )
                if overlap > 0.5 and abs(base_y - cand_y) >= 24.0:
                    continue
            ratio = SequenceMatcher(None, base_norm, cand_norm).ratio()
            if ratio < 0.84:
                continue

            qi = line_duplicate_quality_score(base)
            qj = line_duplicate_quality_score(lines_out[j])
            diff = abs(qi - qj)
            if diff < 0.18:
                continue
            weak_idx = i if qi < qj else j
            weak_line = lines_out[weak_idx]
            weak_conf = float(weak_line.get("confidence", 0.0) or 0.0)
            weak_uncertainty = line_uncertainty(weak_line)
            if weak_conf > 0.55 and weak_uncertainty < 0.18:
                continue
            drops.add(weak_idx)

    if not drops:
        return

    lines_out[:] = [ln for idx, ln in enumerate(lines_out) if idx not in drops]
    for i, ln in enumerate(lines_out):
        ln["line_index"] = i + 1


def canonicalize_repeated_line_text_variants(  # noqa: C901
    lines_out: list[dict[str, Any]],
) -> None:
    if len(lines_out) < 3:
        return

    n = len(lines_out)
    norms = [normalize_text_basic(str(ln.get("text", ""))) for ln in lines_out]
    token_lists = [[t for t in nrm.split() if t] for nrm in norms]
    parents = list(range(n))

    def find(x: int) -> int:
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parents[rb] = ra

    for i in range(n):
        toks_i = token_lists[i]
        if len(toks_i) < 3:
            continue
        for j in range(i + 1, n):
            toks_j = token_lists[j]
            if len(toks_j) < 3:
                continue
            if abs(len(toks_i) - len(toks_j)) > 1:
                continue
            dt = abs(
                float(lines_out[j].get("start", 0.0))
                - float(lines_out[i].get("start", 0.0))
            )
            if dt < 8.0:
                continue
            ratio = SequenceMatcher(None, norms[i], norms[j]).ratio()
            if ratio < 0.82:
                continue
            shared = len(set(toks_i) & set(toks_j))
            if shared < min(2, len(set(toks_i))):
                continue
            union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    for group in groups.values():
        if len(group) < 2:
            continue
        _repair_repeat_cluster_tokenization_variants(lines_out, group)
        for idx in group:
            norms[idx] = normalize_text_basic(str(lines_out[idx].get("text", "")))
            token_lists[idx] = [t for t in norms[idx].split() if t]
        group_sorted = sorted(
            group,
            key=lambda idx: (
                line_duplicate_quality_score(lines_out[idx]),
                len(token_lists[idx]),
            ),
            reverse=True,
        )
        canon_idx = group_sorted[0]
        canon_words = lines_out[canon_idx].get("words", [])
        canon_tokens = [str(w.get("text", "")) for w in canon_words]
        canon_norm = norms[canon_idx]
        if len(canon_tokens) < 3:
            continue

        for idx in group_sorted[1:]:
            line = lines_out[idx]
            words = line.get("words", [])
            if len(words) != len(canon_tokens):
                continue
            if len(words) != len(token_lists[idx]):
                continue
            if (
                line_duplicate_quality_score(lines_out[canon_idx])
                - line_duplicate_quality_score(line)
                < 0.18
            ):
                continue
            line_conf = float(line.get("confidence", 0.0) or 0.0)
            if line_conf > 0.6 and line_uncertainty(line) < 0.2:
                continue
            ratio = SequenceMatcher(None, norms[idx], canon_norm).ratio()
            if ratio < 0.86:
                continue

            current_tokens = [str(w.get("text", "")) for w in words]
            if all(
                normalize_text_basic(a) == normalize_text_basic(b)
                for a, b in zip(current_tokens, canon_tokens)
            ):
                continue

            for w, replacement in zip(words, canon_tokens):
                w["text"] = replacement
            line["text"] = " ".join(str(w.get("text", "")) for w in words)
