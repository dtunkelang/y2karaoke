from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

from ..models import TargetLine

LineStartFn = Callable[[TargetLine], Optional[float]]
LineEndFn = Callable[[TargetLine], Optional[float]]
AssignLineTimingFn = Callable[
    [TargetLine, Optional[float], Optional[float], float], None
]


def trace_clean_blocks(
    label: str,
    target_lines: List[TargetLine],
) -> None:
    if os.environ.get("Y2K_VISUAL_BLOCK_TRACE", "0") != "1":
        return
    blocks: list[dict[str, object]] = []
    i = 0
    while i < len(target_lines):
        ln = target_lines[i]
        if ln.visibility_start is None or ln.visibility_end is None:
            i += 1
            continue
        vs0 = float(ln.visibility_start)
        ve0 = float(ln.visibility_end)
        block = [ln]
        j = i + 1
        while j < len(target_lines):
            lnj = target_lines[j]
            if lnj.visibility_start is None or lnj.visibility_end is None:
                break
            vs = float(lnj.visibility_start)
            ve = float(lnj.visibility_end)
            overlap = min(ve0, ve) - max(vs0, vs)
            if abs(vs - vs0) > 0.6 or overlap < 0.5:
                break
            block.append(lnj)
            j += 1
        if 2 <= len(block) <= 6:
            blocks.append(
                {
                    "start_idx": i,
                    "size": len(block),
                    "vs_span": round(
                        max(float(x.visibility_start or 0.0) for x in block)
                        - min(float(x.visibility_start or 0.0) for x in block),
                        2,
                    ),
                    "texts": [" ".join(x.words[:6]) for x in block],
                }
            )
        i = max(i + 1, j)
    import logging

    logging.getLogger(__name__).info("BLOCK_TRACE %s blocks=%s", label, blocks[:20])


def retime_clean_screen_blocks_by_vertical_order(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_start_fn: LineStartFn,
    line_end_fn: LineEndFn,
    assign_line_level_word_timings_fn: AssignLineTimingFn,
) -> None:
    """Base-case karaoke block retiming: preserve top-to-bottom order in clean blocks."""
    lines = [ln for ln, _, _ in g_jobs]
    n = len(lines)
    if n < 2:
        return

    i = 0
    while i < n:
        ln0 = lines[i]
        s0 = line_start_fn(ln0)
        if (
            s0 is None
            or ln0.visibility_start is None
            or ln0.visibility_end is None
            or (float(ln0.visibility_end) - float(ln0.visibility_start)) < 1.0
        ):
            i += 1
            continue

        vs0 = float(ln0.visibility_start)
        ve0 = float(ln0.visibility_end)
        block = [lines[i]]
        j = i + 1
        while j < n:
            ln = lines[j]
            sj = line_start_fn(ln)
            if sj is None or ln.visibility_start is None or ln.visibility_end is None:
                break
            vs = float(ln.visibility_start)
            ve = float(ln.visibility_end)
            overlap = min(ve0, ve) - max(vs0, vs)
            if abs(vs - vs0) > 0.45 or abs(ve - ve0) > 1.75 or overlap < 0.75:
                break
            block.append(ln)
            j += 1

        if 2 <= len(block) <= 5:
            y_vals = [float(ln.y) for ln in block]
            if (max(y_vals) - min(y_vals)) >= 24.0:
                by_y = sorted(
                    block, key=lambda ln: (float(ln.y), line_start_fn(ln) or 0.0)
                )
                starts_by_y = [line_start_fn(ln) or 0.0 for ln in by_y]
                has_inversion = any(
                    starts_by_y[k] > starts_by_y[k + 1] + 0.35
                    for k in range(len(starts_by_y) - 1)
                )
                if has_inversion:
                    target_starts = sorted(starts_by_y)
                    prev_end: Optional[float] = None
                    for k, ln in enumerate(by_y):
                        old_start = line_start_fn(ln)
                        if old_start is None:
                            continue
                        old_end = line_end_fn(ln)
                        dur = (
                            max(0.25, float(old_end) - float(old_start))
                            if old_end is not None
                            else max(0.8, 0.22 * max(1, len(ln.words)))
                        )
                        new_start = target_starts[k]
                        if prev_end is not None:
                            new_start = max(new_start, prev_end + 0.05)
                        if ln.visibility_start is not None:
                            new_start = max(
                                new_start, float(ln.visibility_start) - 0.15
                            )
                        new_end = new_start + dur
                        if ln.visibility_end is not None:
                            new_end = min(new_end, float(ln.visibility_end) + 0.4)
                        if new_end <= new_start + 0.12:
                            new_end = new_start + 0.12

                        if abs(new_start - float(old_start)) > 0.2:
                            conf = 0.4
                            if ln.word_confidences:
                                vals = [c for c in ln.word_confidences if c is not None]
                                if vals:
                                    conf = float(sum(vals) / len(vals))
                            assign_line_level_word_timings_fn(
                                ln, new_start, new_end, conf
                            )
                        end_val = line_end_fn(ln)
                        prev_end = float(end_val) if end_val is not None else new_end

        i = max(j, i + 1)


def reorder_clean_screen_blocks_target_lines(  # noqa: C901
    target_lines: List[TargetLine],
    *,
    line_start_fn: LineStartFn,
) -> None:
    """Reorder simple shared-visibility blocks in target-line list by vertical order."""
    if len(target_lines) < 2:
        return

    indexed: list[tuple[int, TargetLine, float, float, float]] = []
    for idx, ln in enumerate(target_lines):
        if ln.visibility_start is None or ln.visibility_end is None:
            continue
        s = line_start_fn(ln)
        if s is None:
            continue
        vs = float(ln.visibility_start)
        ve = float(ln.visibility_end)
        if ve <= vs:
            continue
        indexed.append((idx, ln, vs, ve, float(s)))
    if len(indexed) < 2:
        return

    blocks: list[list[tuple[int, TargetLine, float, float, float]]] = []
    cur: list[tuple[int, TargetLine, float, float, float]] = []
    for rec in indexed:
        if not cur:
            cur = [rec]
            continue
        prev_idx = cur[-1][0]
        idx, _ln, vs, ve, _s = rec
        if idx != prev_idx + 1:
            blocks.append(cur)
            cur = [rec]
            continue
        block_vs = [r[2] for r in cur]
        block_ve = [r[3] for r in cur]
        c_start = min(block_vs)
        c_end = max(block_ve)
        overlap = min(c_end, ve) - max(c_start, vs)
        if (vs - c_start) <= 1.4 and overlap > 0.2:
            cur.append(rec)
        else:
            blocks.append(cur)
            cur = [rec]
    if cur:
        blocks.append(cur)

    reordered_any = False
    for block in blocks:
        if not (2 <= len(block) <= 5):
            continue
        ys = [float(rec[1].y) for rec in block]
        if (max(ys) - min(ys)) < 20.0:
            continue
        block_vs = [rec[2] for rec in block]
        block_ve = [rec[3] for rec in block]
        if (max(block_vs) - min(block_vs)) > 0.45:
            continue
        if (max(block_ve) - min(block_ve)) > 12.0:
            continue

        by_y = sorted(block, key=lambda r: (float(r[1].y), r[4], r[0]))
        starts_by_y = [r[4] for r in by_y]
        has_inversion = any(
            starts_by_y[k] > starts_by_y[k + 1] + 0.75
            for k in range(len(starts_by_y) - 1)
        )
        if not has_inversion:
            continue

        start_idx = block[0][0]
        end_idx = block[-1][0] + 1
        replacement = [rec[1] for rec in by_y]
        target_lines[start_idx:end_idx] = replacement
        reordered_any = True

    if reordered_any:
        for i, ln in enumerate(target_lines):
            ln.line_index = i + 1
            ln.block_order_hint = i


def assign_block_sequence_hints_from_visibility(  # noqa: C901
    target_lines: List[TargetLine],
    *,
    line_start_fn: LineStartFn,
) -> None:
    """Assign block/row ordering hints using visibility chronology."""
    if len(target_lines) < 2:
        return

    indexed: list[tuple[int, TargetLine]] = list(enumerate(target_lines))
    blocks: list[list[tuple[int, TargetLine]]] = []
    current: list[tuple[int, TargetLine]] = []

    def _line_vs(ln: TargetLine) -> Optional[float]:
        return float(ln.visibility_start) if ln.visibility_start is not None else None

    def _line_ve(ln: TargetLine) -> Optional[float]:
        return float(ln.visibility_end) if ln.visibility_end is not None else None

    for idx, ln in indexed:
        vs = _line_vs(ln)
        ve = _line_ve(ln)
        if vs is None or ve is None or ve <= vs:
            if current:
                blocks.append(current)
                current = []
            blocks.append([(idx, ln)])
            continue

        if not current:
            current = [(idx, ln)]
            continue

        c_vs = [
            float(c_ln.visibility_start or 0.0)
            for _, c_ln in current
            if c_ln.visibility_start is not None
        ]
        c_ve = [
            float(c_ln.visibility_end or 0.0)
            for _, c_ln in current
            if c_ln.visibility_end is not None
        ]
        if not c_vs or not c_ve:
            blocks.append(current)
            current = [(idx, ln)]
            continue
        c_start = min(c_vs)
        c_end = max(c_ve)
        overlap = min(c_end, ve) - max(c_start, vs)
        same_block = abs(vs - c_start) <= 0.55 and overlap >= 0.5 and len(current) < 6
        if same_block:
            current.append((idx, ln))
        else:
            blocks.append(current)
            current = [(idx, ln)]

    if current:
        blocks.append(current)

    block_records: list[tuple[float, int, list[tuple[int, TargetLine]]]] = []
    for block in blocks:
        vs_vals = [
            float(ln.visibility_start)
            for _, ln in block
            if ln.visibility_start is not None
        ]
        block_vs_key = min(vs_vals) if vs_vals else float(block[0][1].start)
        block_records.append((block_vs_key, block[0][0], block))

    block_records.sort(key=lambda rec: (rec[0], rec[1]))

    sequenced: list[TargetLine] = []
    for _block_vs_key, _first_idx, block in block_records:
        if len(block) == 1:
            sequenced.append(block[0][1])
            continue
        ys = [float(ln.y) for _, ln in block]
        if max(ys) - min(ys) < 20.0:
            sequenced.extend([ln for _, ln in block])
            continue
        ordered_block = sorted(
            block,
            key=lambda p: (
                float(p[1].y),
                (
                    line_start_fn(p[1])
                    if line_start_fn(p[1]) is not None
                    else float(p[1].start)
                ),
                p[0],
            ),
        )
        sequenced.extend([ln for _, ln in ordered_block])

    if len(sequenced) != len(target_lines):
        return
    for i, ln in enumerate(sequenced):
        ln.block_order_hint = i
