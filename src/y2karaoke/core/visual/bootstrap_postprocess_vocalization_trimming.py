"""Vocalization-prefix trimming helpers for bootstrap postprocess."""

from __future__ import annotations

from typing import Any

from ..text_utils import normalize_text_basic


def _has_suffix_support_nearby(
    lines_out: list[dict[str, Any]],
    token_lines: list[list[str]],
    *,
    line_index: int,
    suffix: list[str],
    start: float,
    end: float,
) -> bool:
    for j in range(max(0, line_index - 10), min(len(lines_out), line_index + 11)):
        if j == line_index:
            continue
        other = lines_out[j]
        if token_lines[j] != suffix:
            continue
        other_start = float(other.get("start", 0.0) or 0.0)
        other_end = float(other.get("end", other_start) or other_start)
        overlaps = max(0.0, min(end, other_end) - max(start, other_start)) > 0.0
        near_time = abs(other_start - start) <= 20.0
        if overlaps or near_time:
            return True
    return False


def _apply_leading_trim(line: dict[str, Any], *, start: float) -> bool:
    kept_words = list(line.get("words", [])[1:])
    if len(kept_words) < 2:
        return False
    line["words"] = kept_words
    line["text"] = " ".join(str(w.get("text", "")) for w in kept_words if w.get("text"))
    line["start"] = float(kept_words[0].get("start", start) or start)
    for wi, w in enumerate(kept_words):
        w["word_index"] = wi + 1
    return True


def trim_leading_vocalization_prefixes(
    lines_out: list[dict[str, Any]],
    *,
    vocalization_noise_tokens_set: set[str],
) -> None:
    """Drop a leading vocalization token when the same phrase appears nearby without it."""
    if len(lines_out) < 2:
        return

    norm_lines = [normalize_text_basic(str(ln.get("text", ""))) for ln in lines_out]
    token_lines = [[t for t in n.split() if t] for n in norm_lines]

    for i, ln in enumerate(lines_out):
        words = ln.get("words", [])
        toks = token_lines[i]
        if len(words) != len(toks) or len(toks) < 3:
            continue
        lead = toks[0]
        if lead not in vocalization_noise_tokens_set:
            continue
        suffix = toks[1:]
        if len(suffix) < 2:
            continue
        conf = float(ln.get("confidence", 0.0) or 0.0)
        if conf > 0.4:
            continue
        start = float(ln.get("start", 0.0) or 0.0)
        end = float(ln.get("end", start) or start)
        if not _has_suffix_support_nearby(
            lines_out,
            token_lines,
            line_index=i,
            suffix=suffix,
            start=start,
            end=end,
        ):
            continue
        _apply_leading_trim(ln, start=start)


def _block_first_meta(line: dict[str, Any]) -> dict[str, Any] | None:
    meta = line.get("_reconstruction_meta", {})
    if not isinstance(meta, dict):
        return None
    bf = meta.get("block_first")
    return bf if isinstance(bf, dict) else None


def _block_first_cycle_index_groups(
    lines_out: list[dict[str, Any]],
) -> dict[object, list[int]]:
    by_block: dict[object, list[int]] = {}
    for idx, ln in enumerate(lines_out):
        bf = _block_first_meta(ln)
        if not bf:
            continue
        try:
            if int(bf.get("cycle_count", 1) or 1) <= 1:
                continue
        except Exception:
            continue
        by_block.setdefault(bf.get("block_id"), []).append(idx)
    return by_block


def _token_map_for_indices(
    lines_out: list[dict[str, Any]],
    by_block: dict[object, list[int]],
) -> dict[int, tuple[str, ...]]:
    return {
        idx: tuple(
            t
            for t in normalize_text_basic(str(lines_out[idx].get("text", ""))).split()
            if t
        )
        for idxs in by_block.values()
        for idx in idxs
    }


def _trim_line_leading_vocalization(
    line: dict[str, Any],
    *,
    token_list: list[str],
) -> None:
    words = list(line.get("words", []) or [])
    if len(words) != len(token_list) or len(words) < 3:
        return
    kept = words[1:]
    line["words"] = kept
    line["text"] = " ".join(str(w.get("text", "")) for w in kept if w.get("text"))
    line["start"] = float(
        kept[0].get("start", line.get("start", 0.0)) or line.get("start", 0.0)
    )
    for wi, w in enumerate(kept):
        w["word_index"] = wi + 1


def trim_leading_vocalization_in_block_first_cycle_rows(
    lines_out: list[dict[str, Any]],
    *,
    vocalization_noise_tokens_set: set[str],
) -> None:
    """Trim leading vocalization tokens on cycle rows when a sibling cycle has the suffix."""
    if len(lines_out) < 2:
        return

    by_block = _block_first_cycle_index_groups(lines_out)
    if not by_block:
        return

    token_map = _token_map_for_indices(lines_out, by_block)

    for idxs in by_block.values():
        tok_set = {token_map[i] for i in idxs if token_map.get(i)}
        for idx in idxs:
            toks_list = list(token_map.get(idx, ()))
            if len(toks_list) < 3:
                continue
            if toks_list[0] not in vocalization_noise_tokens_set:
                continue
            suffix = tuple(toks_list[1:])
            if suffix not in tok_set:
                continue
            ln = lines_out[idx]
            conf = float(ln.get("confidence", 0.0) or 0.0)
            if conf > 0.45:
                continue
            _trim_line_leading_vocalization(ln, token_list=toks_list)
