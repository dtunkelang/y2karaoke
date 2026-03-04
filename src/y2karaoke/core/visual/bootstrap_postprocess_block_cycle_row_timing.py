"""Row-timing normalization helpers for block-first postprocess output."""

from __future__ import annotations

from typing import Any, Callable, Optional


def normalize_block_first_row_timings(  # noqa: C901
    lines_out: list[dict[str, Any]],
    *,
    snap_fn: Callable[[float], float],
) -> None:
    """Normalize starts within block-first rows using row order."""
    if len(lines_out) < 2:
        return

    def _block_id(line: dict[str, Any]) -> object:
        meta = line.get("_reconstruction_meta", {})
        if not isinstance(meta, dict):
            return None
        bf = meta.get("block_first")
        if not isinstance(bf, dict):
            return None
        try:
            if int(bf.get("cycle_count", 1) or 1) > 1:
                return None
        except Exception:
            return None
        return bf.get("block_id")

    def _row_order(line: dict[str, Any]) -> int:
        meta = line.get("_reconstruction_meta", {})
        if not isinstance(meta, dict):
            return 10_000
        bf = meta.get("block_first")
        if not isinstance(bf, dict):
            return 10_000
        try:
            return int(bf.get("row_order", 10_000))
        except Exception:
            return 10_000

    def _shift_line(line: dict[str, Any], new_start: float) -> None:
        old_start = float(line.get("start", 0.0) or 0.0)
        old_end = float(line.get("end", old_start) or old_start)
        shift = new_start - old_start
        line["start"] = snap_fn(new_start)
        line["end"] = snap_fn(max(new_start + 0.1, old_end + shift))
        for w in line.get("words", []) or []:
            if "start" in w:
                w["start"] = snap_fn(float(w["start"]) + shift)
            if "end" in w:
                w["end"] = snap_fn(float(w["end"]) + shift)

    i = 0
    while i < len(lines_out):
        bid = _block_id(lines_out[i])
        if bid is None:
            i += 1
            continue
        j = i + 1
        while j < len(lines_out) and _block_id(lines_out[j]) == bid:
            j += 1

        block = lines_out[i:j]
        if 2 <= len(block) <= 6:
            starts = [float(ln.get("start", 0.0) or 0.0) for ln in block]
            by_row = sorted(
                block, key=lambda ln: (_row_order(ln), float(ln.get("y", 0.0) or 0.0))
            )
            row_starts = [float(ln.get("start", 0.0) or 0.0) for ln in by_row]
            has_inversion = any(
                row_starts[k] > row_starts[k + 1] + 0.35
                for k in range(len(row_starts) - 1)
            )
            if has_inversion:
                target_starts = sorted(starts)
                prev_end: Optional[float] = None
                for k, ln in enumerate(by_row):
                    new_start = target_starts[k]
                    if prev_end is not None:
                        new_start = max(new_start, prev_end + 0.05)
                    vis_start = ln.get("_visibility_start")
                    vis_end = ln.get("_visibility_end")
                    if vis_start is not None:
                        new_start = max(new_start, float(vis_start) - 0.15)
                    if vis_end is not None:
                        new_start = min(new_start, float(vis_end) - 0.12)
                    old_start = float(ln.get("start", 0.0) or 0.0)
                    if abs(new_start - old_start) > 0.15:
                        _shift_line(ln, new_start)
                    prev_end = float(ln.get("end", new_start) or new_start)
        i = j

    for idx, ln in enumerate(lines_out):
        ln["line_index"] = idx + 1
