"""Block-order normalization helpers for visual bootstrap postprocessing."""

from __future__ import annotations

from typing import Any, Optional

from .reconstruction import snap


def reorder_clean_visibility_blocks(  # noqa: C901
    lines_out: list[dict[str, Any]],
) -> None:
    """Sort simple lyric blocks by visibility window, then top-to-bottom.

    This handles the common karaoke layout where one screenful of lyrics is shown
    at a time and lines in a screen are sung in vertical order.
    """
    if len(lines_out) < 2:
        return

    for idx, ln in enumerate(lines_out):
        ln.setdefault("_orig_order", idx)

    enriched: list[tuple[int, float, float, float, float]] = []
    for idx, ln in enumerate(lines_out):
        vs = ln.get("_visibility_start")
        ve = ln.get("_visibility_end")
        if vs is None or ve is None:
            continue
        vsf = float(vs)
        vef = float(ve)
        if vef <= vsf:
            continue
        enriched.append(
            (
                idx,
                vsf,
                vef,
                float(ln.get("y", 0.0) or 0.0),
                float(ln.get("start", 0.0) or 0.0),
            )
        )
    if len(enriched) < 2:
        return

    enriched.sort(key=lambda t: (t[1], t[2], t[3], t[4], t[0]))

    blocks: list[list[tuple[int, float, float, float, float]]] = []
    current: list[tuple[int, float, float, float, float]] = []
    current_start = -1.0
    current_end = -1.0
    for rec in enriched:
        if not current:
            current = [rec]
            current_start = rec[1]
            current_end = rec[2]
            continue

        _, rec_vs, rec_ve, _, _ = rec
        overlap_with_block = min(current_end, rec_ve) - max(current_start, rec_vs)
        # For simple karaoke layouts, all lines in a screen usually become visible
        # together. Cluster primarily by visibility onset, with an overlap check to
        # avoid joining distant repeated blocks.
        if (rec_vs - current_start) <= 1.4 and overlap_with_block > 0.2:
            current.append(rec)
            current_start = min(current_start, rec_vs)
            current_end = max(current_end, rec[2])
        else:
            blocks.append(current)
            current = [rec]
            current_start = rec[1]
            current_end = rec[2]
    if current:
        blocks.append(current)

    # Build desired order only for blocks we explicitly choose to reorder.
    desired_positions: dict[int, int] = {}

    def _shift_line_timing(rec: dict[str, Any], new_start: float) -> None:
        old_start = float(rec.get("start", 0.0) or 0.0)
        old_end = float(rec.get("end", old_start) or old_start)
        shift = new_start - old_start
        rec["start"] = snap(new_start)
        rec["end"] = snap(max(new_start + 0.1, old_end + shift))
        for w in rec.get("words", []) or []:
            if "start" in w:
                w["start"] = snap(float(w["start"]) + shift)
            if "end" in w:
                w["end"] = snap(float(w["end"]) + shift)

    for block in blocks:
        if len(block) == 1:
            continue
        y_values = [rec[3] for rec in block]
        if max(y_values) - min(y_values) < 20.0:
            continue
        block_by_y = sorted(block, key=lambda t: (t[3], t[1], t[4], t[0]))
        block_vs = [rec[1] for rec in block]
        block_ve = [rec[2] for rec in block]
        block_indices = sorted(rec[0] for rec in block)
        is_local_contiguous = block_indices[-1] - block_indices[0] + 1 == len(
            block_indices
        )
        is_tight_screen_block = (
            (max(block_vs) - min(block_vs)) <= 0.35
            and (max(block_ve) - min(block_ve)) <= 10.0
            and 2 <= len(block_by_y) <= 5
            and is_local_contiguous
        )
        # Only reorder when the current starts are clearly inconsistent with
        # top-to-bottom order. This avoids disturbing blocks that are already
        # correctly sequenced by selection timing.
        y_order_starts = [rec[4] for rec in block_by_y]
        has_strong_inversion = any(
            y_order_starts[k] > y_order_starts[k + 1] + 0.75
            for k in range(len(y_order_starts) - 1)
        )
        if not has_strong_inversion:
            continue
        if not is_tight_screen_block:
            continue
        # For the simple base case, also remap line starts monotonically in y-order
        # using the block's observed start times. This fixes obvious screen-order
        # inversions without inventing new timings.
        if 2 <= len(block_by_y) <= 6:
            observed_starts = sorted(rec[4] for rec in block)
            prev_end: Optional[float] = None
            for k, rec in enumerate(block_by_y):
                idx = rec[0]
                line = lines_out[idx]
                target_start = observed_starts[k]
                if prev_end is not None:
                    target_start = max(target_start, prev_end + 0.05)
                vis_end = line.get("_visibility_end")
                if vis_end is not None:
                    target_start = min(
                        float(vis_end) - 0.15,
                        target_start,
                    )
                old_start = float(line.get("start", 0.0) or 0.0)
                if target_start > old_start + 0.15 or target_start < old_start - 0.5:
                    _shift_line_timing(line, target_start)
                prev_end = float(line.get("end", target_start) or target_start)
        # Within a clear inverted block, preserve vertical order.
        anchor = block_indices[0]
        for offset, rec in enumerate(block_by_y):
            desired_positions[rec[0]] = anchor + offset

    if not desired_positions:
        return

    # Stable reorder only lines within targeted blocks; all others keep original order.
    indexed = list(enumerate(lines_out))
    indexed.sort(
        key=lambda pair: (
            desired_positions.get(pair[0], int(pair[1].get("_orig_order", pair[0]))),
            int(pair[1].get("_orig_order", pair[0])),
        )
    )
    lines_out[:] = [ln for _, ln in indexed]
    for i, ln in enumerate(lines_out):
        ln["line_index"] = i + 1
