"""Dedupe helpers for block-first cycle rows."""

from __future__ import annotations

from typing import Any

from ..text_utils import normalize_text_basic


def _cycle_key_for_line(line: dict[str, Any]) -> tuple[object, int, int] | None:
    meta = line.get("_reconstruction_meta", {})
    if not isinstance(meta, dict):
        return None
    bf = meta.get("block_first")
    if not isinstance(bf, dict):
        return None
    cycle_count = int(bf.get("cycle_count", 1) or 1)
    if cycle_count <= 1:
        return None
    block_id = bf.get("block_id")
    cycle_idx = bf.get("cycle_index")
    row_order = bf.get("row_order")
    if cycle_idx is None or row_order is None:
        return None
    return (block_id, int(cycle_idx), int(row_order))


def _line_quality(line: dict[str, Any]) -> tuple[float, float, int, int]:
    conf = float(line.get("confidence", 0.0) or 0.0)
    start = float(line.get("start", 0.0) or 0.0)
    end = float(line.get("end", start) or start)
    dur = max(0.0, end - start)
    toks = [t for t in normalize_text_basic(str(line.get("text", ""))).split() if t]
    words = line.get("words", []) or []
    return (conf, dur, len(toks), len(words))


def _dedupe_drop_indexes(lines_out: list[dict[str, Any]]) -> set[int]:
    keep_by_key: dict[tuple[object, int, int], int] = {}
    drops: set[int] = set()
    for idx, ln in enumerate(lines_out):
        key = _cycle_key_for_line(ln)
        if key is None:
            continue
        prev_idx = keep_by_key.get(key)
        if prev_idx is None:
            keep_by_key[key] = idx
            continue
        if _line_quality(ln) > _line_quality(lines_out[prev_idx]):
            drops.add(prev_idx)
            keep_by_key[key] = idx
            continue
        drops.add(idx)
    return drops


def dedupe_block_first_cycle_rows(lines_out: list[dict[str, Any]]) -> None:
    """Keep one line per (block_id, cycle_index, row_order) for block-first cycle rows."""
    if not lines_out:
        return

    drops = _dedupe_drop_indexes(lines_out)

    if not drops:
        return

    lines_out[:] = [ln for idx, ln in enumerate(lines_out) if idx not in drops]
    for i, ln in enumerate(lines_out):
        ln["line_index"] = i + 1
