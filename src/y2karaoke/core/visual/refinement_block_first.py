from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..models import TargetLine


@dataclass
class BlockRowRef:
    line: TargetLine
    original_index: int
    start: float
    y: float


@dataclass
class VisibilityBlock:
    block_id: int
    visibility_start: float
    visibility_end: float
    rows: List[BlockRowRef]


def _line_start(ln: TargetLine) -> float:
    if ln.word_starts and ln.word_starts[0] is not None:
        return float(ln.word_starts[0])
    return float(ln.start)


def _visibility_bounds(ln: TargetLine) -> Optional[tuple[float, float]]:
    if ln.visibility_start is None or ln.visibility_end is None:
        return None
    vs = float(ln.visibility_start)
    ve = float(ln.visibility_end)
    if ve <= vs:
        return None
    return vs, ve


def build_visibility_blocks(
    target_lines: List[TargetLine],
    *,
    max_visibility_start_span: float = 1.4,
    min_overlap_sec: float = 0.75,
    max_block_size: int = 6,
) -> List[VisibilityBlock]:
    """Build explicit visibility blocks from target lines.

    This is intentionally conservative: it captures the common karaoke case where
    multiple rows share a similar visibility window (screenful) and should be
    ordered top-to-bottom.
    """

    candidates: list[tuple[int, TargetLine, float, float]] = []
    for i, ln in enumerate(target_lines):
        bounds = _visibility_bounds(ln)
        if bounds is None:
            continue
        vs, ve = bounds
        candidates.append((i, ln, vs, ve))

    if len(candidates) < 2:
        return []

    # Use chronology of visibility onset as the primary block sequence axis.
    candidates.sort(key=lambda rec: (rec[2], rec[3], float(rec[1].y), rec[0]))

    blocks: list[VisibilityBlock] = []
    curr: list[tuple[int, TargetLine, float, float]] = []
    curr_vs_min = 0.0
    curr_vs_max = 0.0
    curr_ve_min = 0.0
    curr_ve_max = 0.0

    def flush_current() -> None:
        nonlocal curr
        if len(curr) >= 2:
            rows = [
                BlockRowRef(
                    line=ln,
                    original_index=idx,
                    start=_line_start(ln),
                    y=float(ln.y),
                )
                for idx, ln, _vs, _ve in curr
            ]
            rows.sort(key=lambda r: (r.y, r.start, r.original_index))
            blocks.append(
                VisibilityBlock(
                    block_id=len(blocks),
                    visibility_start=min(vs for _idx, _ln, vs, _ve in curr),
                    visibility_end=max(ve for _idx, _ln, _vs, ve in curr),
                    rows=rows[:max_block_size],
                )
            )
        curr = []

    for rec in candidates:
        idx, ln, vs, ve = rec
        if not curr:
            curr = [rec]
            curr_vs_min = curr_vs_max = vs
            curr_ve_min = curr_ve_max = ve
            continue

        # Shared screen block if visibility onsets are close and windows overlap.
        prospective_vs_min = min(curr_vs_min, vs)
        prospective_vs_max = max(curr_vs_max, vs)
        overlap = min(curr_ve_max, ve) - max(curr_vs_min, vs)
        rec_dur = max(0.0, ve - vs)
        span_limit = max_visibility_start_span
        min_overlap = min_overlap_sec
        if rec_dur <= 0.8:
            # Short first-selected rows can appear later than the rest of the screen.
            span_limit = max(span_limit, 1.8)
            min_overlap = min(min_overlap, 0.15)
        too_wide = (prospective_vs_max - prospective_vs_min) > span_limit
        if too_wide or overlap < min_overlap or len(curr) >= max_block_size:
            flush_current()
            curr = [rec]
            curr_vs_min = curr_vs_max = vs
            curr_ve_min = curr_ve_max = ve
            continue

        curr.append(rec)
        curr_vs_min = prospective_vs_min
        curr_vs_max = prospective_vs_max
        curr_ve_min = min(curr_ve_min, ve)
        curr_ve_max = max(curr_ve_max, ve)

    flush_current()
    return blocks


def apply_block_first_prototype_ordering(target_lines: List[TargetLine]) -> bool:
    """Assign block/row order hints from explicit visibility blocks.

    Returns True if any hint was assigned. This is a prototype path that does not
    yet replace line reconstruction; it establishes a true block object layer for
    sequencing and output flattening.
    """

    blocks = build_visibility_blocks(target_lines)
    if not blocks:
        return False

    block_members = _build_block_membership_index(blocks)

    if not block_members:
        return False

    # Assign global hints by chronological merge of block rows and singleton lines.
    order_records = _build_block_order_records(target_lines, block_members)
    for hint, ln in enumerate(order_records):
        ln.block_order_hint = hint

    return True


def _build_block_membership_index(
    blocks: List[VisibilityBlock],
) -> dict[int, tuple[int, float, int, float]]:
    block_members: dict[int, tuple[int, float, int, float]] = {}
    # original_index -> (block_id, block_start, row_order, row_y)
    for block in blocks:
        for row_order, row in enumerate(block.rows):
            block_members[row.original_index] = (
                block.block_id,
                float(block.visibility_start),
                row_order,
                float(row.y),
            )
    return block_members


def _ensure_block_first_meta(
    line: TargetLine, *, block_id: int, row_y: float, row_order: int
) -> None:
    if line.reconstruction_meta is None:
        line.reconstruction_meta = {}
    meta = line.reconstruction_meta
    if not isinstance(meta, dict):
        return
    meta.setdefault("block_first", {})
    block_meta = meta.get("block_first")
    if not isinstance(block_meta, dict):
        return
    block_meta["block_id"] = block_id
    block_meta["row_y"] = row_y
    block_meta["row_order"] = row_order


def _singleton_sort_key(
    line: TargetLine, idx: int
) -> tuple[float, int, int, float, int]:
    visibility_start = (
        float(line.visibility_start)
        if line.visibility_start is not None
        else _line_start(line)
    )
    return (visibility_start, 1, 0, float(line.y), idx)


def _build_block_order_records(
    target_lines: List[TargetLine],
    block_members: dict[int, tuple[int, float, int, float]],
) -> list[TargetLine]:
    order_records: list[tuple[tuple[float, int, int, float, int], TargetLine]] = []
    for idx, line in enumerate(target_lines):
        membership = block_members.get(idx)
        if membership is None:
            order_records.append((_singleton_sort_key(line, idx), line))
            continue
        block_id, block_start, row_order, row_y = membership
        _ensure_block_first_meta(
            line, block_id=block_id, row_y=row_y, row_order=row_order
        )
        order_records.append(((block_start, 0, block_id, float(row_order), idx), line))
    order_records.sort(key=lambda record: record[0])
    return [line for _key, line in order_records]
