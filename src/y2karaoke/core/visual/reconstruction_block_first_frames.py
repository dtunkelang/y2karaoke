from __future__ import annotations

import collections
import os
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional

from ..models import TargetLine
from ..text_utils import normalize_text_basic
from .reconstruction_frame_accumulation import (
    _calculate_visibility_threshold,
    _group_words_into_lines,
    _process_line_in_frame,
)

FrameFilter = Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]
SnapFn = Callable[[float], float]


@dataclass
class _FrameRow:
    time: float
    y: float
    lane: int
    text: str
    words: list[str]
    w_rois: list[tuple[int, int, int, int]]
    is_visible: bool
    brightness: float


@dataclass
class _FrameState:
    time: float
    rows: list[_FrameRow]


@dataclass
class _BlockRow:
    y: float
    observations: list[_FrameRow]


def _frame_rows_from_raw_frames(
    raw_frames: list[dict[str, Any]],
    *,
    filter_static_overlay_words: FrameFilter,
) -> list[_FrameState]:
    raw_frames = filter_static_overlay_words(raw_frames)
    thresholds = _calculate_visibility_threshold(raw_frames)
    out: list[_FrameState] = []
    for frame in raw_frames:
        t = float(frame["time"])
        rows: list[_FrameRow] = []
        words = frame.get("words", []) or []
        for line_words in _group_words_into_lines(words):
            res = _process_line_in_frame(line_words, t, thresholds)
            if not res:
                continue
            ent, is_visible = res
            rows.append(
                _FrameRow(
                    time=t,
                    y=float(ent["y"]),
                    lane=int(ent.get("lane", 0)),
                    text=str(ent["text"]),
                    words=list(ent.get("words", [])),
                    w_rois=list(ent.get("w_rois", [])),
                    is_visible=bool(is_visible),
                    brightness=float(ent.get("brightness", 0.0) or 0.0),
                )
            )
        rows.sort(key=lambda r: (r.y, r.text))
        out.append(_FrameState(time=t, rows=rows))
    return out


def _row_similarity(a: _FrameRow, b: _FrameRow) -> float:
    if abs(a.y - b.y) > 28.0:
        return 0.0
    ta = normalize_text_basic(a.text)
    tb = normalize_text_basic(b.text)
    if not ta or not tb:
        return 0.0
    return SequenceMatcher(None, ta, tb).ratio()


def _frame_state_same_block(a: _FrameState, b: _FrameState) -> bool:
    if not a.rows or not b.rows:
        return False
    if abs(len(a.rows) - len(b.rows)) > 1:
        return False
    ay = [r.y for r in a.rows]
    by = [r.y for r in b.rows]
    if len(ay) == len(by):
        ydiffs = [abs(x - y) for x, y in zip(ay, by)]
        if ydiffs and max(ydiffs) > 35.0:
            return False
    # Match rows by nearest y and require some text continuity.
    matches = 0
    strong = 0
    for ra in a.rows:
        best = max((_row_similarity(ra, rb) for rb in b.rows), default=0.0)
        if best >= 0.45:
            matches += 1
        if best >= 0.7:
            strong += 1
    min_rows = min(len(a.rows), len(b.rows))
    return matches >= max(1, min_rows - 1) and (strong >= 1 or min_rows <= 2)


def _segment_blocks(frames: list[_FrameState]) -> list[list[_FrameState]]:
    blocks: list[list[_FrameState]] = []
    cur: list[_FrameState] = []
    for fr in frames:
        if not fr.rows:
            if cur:
                blocks.append(cur)
                cur = []
            continue
        if not cur:
            cur = [fr]
            continue
        if _frame_state_same_block(cur[-1], fr):
            cur.append(fr)
        else:
            blocks.append(cur)
            cur = [fr]
    if cur:
        blocks.append(cur)
    # Keep only substantial blocks; tiny transition blips are usually noise.
    substantial = [b for b in blocks if len(b) >= 2]
    split_blocks: list[list[_FrameState]] = []
    use_phase_split = os.environ.get("Y2K_VISUAL_BLOCK_PHASE_SPLIT", "0") == "1"
    for b in substantial:
        for sb in _split_block_on_row_cycle_resets(b):
            if use_phase_split:
                split_blocks.extend(_split_block_on_row_text_phase_changes(sb))
            else:
                split_blocks.append(sb)
    split_blocks = _prune_tiny_identical_blocks(split_blocks)
    return split_blocks


def _split_block_on_row_cycle_resets(  # noqa: C901
    block_frames: list[_FrameState],
) -> list[list[_FrameState]]:
    """Split long stable-layout blocks that contain multiple top-to-bottom cycles.

    Some karaoke screens keep the same visible rows while the highlight sweeps through
    them multiple times (e.g. repeated "take that money / watch it burn ..."). A
    single block with one row onset per row loses the repeated cycles entirely.
    """
    duration = float(block_frames[-1].time) - float(block_frames[0].time)
    if len(block_frames) < 20 or duration < 6.5:
        return [block_frames]

    row_counts = [len(fr.rows) for fr in block_frames if fr.rows]
    if not row_counts:
        return [block_frames]
    stable_count, stable_frames = collections.Counter(row_counts).most_common(1)[0]
    if (
        stable_count < 3
        or stable_count > 5
        or stable_frames < int(0.7 * len(row_counts))
    ):
        return [block_frames]

    # Normalize brightness per row within the block so a consistently bright row
    # (often the bottom row) does not mask actual highlight sweeps.
    stable_row_brightness: list[list[float]] = [[] for _ in range(stable_count)]
    stable_frames_with_rows: list[tuple[int, _FrameState]] = []
    for i, fr in enumerate(block_frames):
        if len(fr.rows) != stable_count:
            continue
        stable_frames_with_rows.append((i, fr))
        for ridx, row in enumerate(fr.rows):
            stable_row_brightness[ridx].append(float(row.brightness))

    row_minmax: list[tuple[float, float]] = []
    for vals in stable_row_brightness:
        if not vals:
            row_minmax.append((0.0, 1.0))
            continue
        mn = min(vals)
        mx = max(vals)
        row_minmax.append((mn, mx if mx > mn else mn + 1.0))

    seq: list[tuple[int, float, int, float, float]] = []
    # (frame_idx, time, selected_row_idx, max_norm, sum_norm)
    for i, fr in stable_frames_with_rows:
        b_vals = [float(r.brightness) for r in fr.rows]
        if not b_vals:
            continue
        norm_vals: list[float] = []
        for ridx, b in enumerate(b_vals):
            mn, mx = row_minmax[ridx]
            denom = max(1.0, mx - mn)
            norm_vals.append((float(b) - mn) / denom)
        max_norm = max(norm_vals)
        if max_norm < 0.35:
            continue
        best_idx = max(range(len(fr.rows)), key=lambda idx: (norm_vals[idx], -idx))
        seq.append((i, float(fr.time), best_idx, max_norm, sum(norm_vals)))

    if len(seq) < 12:
        return [block_frames]

    split_frame_idxs: list[int] = []
    seg_start_seq = 0
    seen_max_idx = -1
    prev_idx = seq[0][2]
    last_split_time = seq[0][1]
    seg_peak_sum = seq[0][4]

    def _restart_after(pos: int) -> bool:
        # Look ahead for a fresh sweep after a reset valley.
        # We expect either a strong top-row selection or at least a strong new
        # selected row after the valley.
        limit = min(len(seq), pos + 28)
        for k in range(pos + 1, limit):
            _fi, _t, idx_k, max_norm_k, _sum_k = seq[k]
            if max_norm_k < 0.55:
                continue
            if idx_k <= 1 or max_norm_k >= 0.8:
                return True
        return False

    short_block_mode = duration < 10.0
    if short_block_mode and not _has_strong_top_row_phase_change(block_frames):
        return [block_frames]
    min_events_before_split = 5 if duration < 8.5 else 8
    min_peak_for_reset = max(0.8, 0.25 * stable_count) if short_block_mode else 0.0

    for j in range(1, len(seq)):
        frame_idx, t, idx, max_norm, sum_norm = seq[j]
        seen_max_idx = max(seen_max_idx, idx)
        seg_peak_sum = max(seg_peak_sum, float(sum_norm))
        is_reset = (
            prev_idx >= (stable_count - 2)
            and idx <= 1
            and seen_max_idx >= (stable_count - 1)
            and (t - last_split_time) >= 2.0
            and (j - seg_start_seq) >= min_events_before_split
            and seg_peak_sum >= min_peak_for_reset
            and _restart_after(j)
        )
        # Same-layout repeat in place: rows stay fixed but selection resets from
        # mostly-selected to mostly-unselected before sweeping again.
        reset_valley = (
            (not short_block_mode)
            and seg_peak_sum >= max(1.8, 0.55 * stable_count)
            and float(sum_norm) <= max(0.95, 0.32 * stable_count)
            and float(max_norm) <= 0.62
            and (t - last_split_time) >= 2.0
            and (j - seg_start_seq) >= min_events_before_split
            and _restart_after(j)
        )
        if is_reset or reset_valley:
            split_frame_idxs.append(frame_idx)
            seg_start_seq = j
            seen_max_idx = idx
            last_split_time = t
            seg_peak_sum = float(sum_norm)
        prev_idx = idx

    if not split_frame_idxs:
        return [block_frames]

    out: list[list[_FrameState]] = []
    start = 0
    for cut in split_frame_idxs:
        seg = block_frames[start:cut]
        if len(seg) >= 2:
            out.append(seg)
        start = cut
    tail = block_frames[start:]
    if len(tail) >= 2:
        out.append(tail)
    out = out or [block_frames]
    return _merge_adjacent_identical_row_blocks(out)


def _has_strong_top_row_phase_change(block_frames: list[_FrameState]) -> bool:
    if len(block_frames) < 12:
        return False
    rows = _cluster_rows_within_block(block_frames)
    if not (2 <= len(rows) <= 5):
        return False
    top = rows[0]
    by_variant: dict[str, list[float]] = collections.defaultdict(list)
    for obs in top.observations:
        norm = normalize_text_basic(obs.text)
        if norm:
            by_variant[norm].append(float(obs.time))
    if len(by_variant) < 2:
        return False

    variants = sorted(by_variant.items(), key=lambda kv: len(kv[1]), reverse=True)
    (v1, ts1), (v2, ts2) = variants[:2]
    min_count = max(8, int(0.15 * len(block_frames)))
    if len(ts1) < min_count or len(ts2) < min_count:
        return False

    t1_min, t1_max = min(ts1), max(ts1)
    t2_min, t2_max = min(ts2), max(ts2)
    disjoint_phases = t1_max <= (t2_min + 0.2) or t2_max <= (t1_min + 0.2)
    if not disjoint_phases:
        return False

    toks1 = [t for t in v1.split() if t]
    toks2 = [t for t in v2.split() if t]
    if not toks1 or not toks2:
        return False

    def _is_suffix_variant(a: list[str], b: list[str]) -> bool:
        if len(a) == len(b):
            return False
        longer, shorter = (a, b) if len(a) > len(b) else (b, a)
        if len(longer) - len(shorter) > 2:
            return False
        return longer[-len(shorter) :] == shorter

    sim = SequenceMatcher(None, v1, v2).ratio()
    return _is_suffix_variant(toks1, toks2) or sim >= 0.8


def _merge_adjacent_identical_row_blocks(
    blocks: list[list[_FrameState]],
) -> list[list[_FrameState]]:
    """Merge micro-block splits that are actually a continuous single sweep.

    The cycle splitter intentionally over-splits long repeated screens; merge back
    adjacent blocks when they are very short and have the same rows with no real gap.
    """
    if len(blocks) < 2:
        return blocks

    merged: list[list[_FrameState]] = []
    i = 0
    while i < len(blocks):
        cur = blocks[i]
        if i + 1 >= len(blocks):
            merged.append(cur)
            break
        nxt = blocks[i + 1]
        cur_gap = float(nxt[0].time) - float(cur[-1].time)
        if cur_gap > 0.2:
            merged.append(cur)
            i += 1
            continue

        cur_rows = _cluster_rows_within_block(cur)
        nxt_rows = _cluster_rows_within_block(nxt)
        if len(cur_rows) != len(nxt_rows) or len(cur_rows) < 3:
            merged.append(cur)
            i += 1
            continue

        cur_texts = [
            normalize_text_basic(_choose_canonical_observation(r).text)
            for r in cur_rows
        ]
        nxt_texts = [
            normalize_text_basic(_choose_canonical_observation(r).text)
            for r in nxt_rows
        ]
        same_rows = cur_texts == nxt_texts
        cur_dur = float(cur[-1].time) - float(cur[0].time)
        nxt_dur = float(nxt[-1].time) - float(nxt[0].time)
        reset_boundary = _looks_like_cycle_reset_boundary(cur, nxt, len(cur_rows))
        # Only merge back very short adjacent chunks (usually over-split tails).
        if same_rows and max(cur_dur, nxt_dur) <= 4.5 and not reset_boundary:
            merged.append(cur + nxt)
            i += 2
            continue

        merged.append(cur)
        i += 1

    return merged


def _looks_like_cycle_reset_boundary(
    cur: list[_FrameState], nxt: list[_FrameState], row_count: int
) -> bool:
    if not (2 <= row_count <= 5):
        return False
    cur_seq = _block_normalized_brightness_seq(cur, row_count)
    nxt_seq = _block_normalized_brightness_seq(nxt, row_count)
    if len(cur_seq) < 4 or len(nxt_seq) < 4:
        return False

    tail = cur_seq[-min(6, len(cur_seq)) :]
    head = nxt_seq[: min(6, len(nxt_seq))]
    tail_sum = sum(s for _t, _m, s in tail) / len(tail)
    tail_max = sum(m for _t, m, _s in tail) / len(tail)
    head_sum = sum(s for _t, _m, s in head) / len(head)
    head_max = sum(m for _t, m, _s in head) / len(head)

    valley_like = tail_sum <= max(0.95, 0.38 * row_count) and tail_max <= 0.55
    restart_like = head_sum >= max(1.8, 0.55 * row_count) and head_max >= 0.75
    return valley_like and restart_like


def _block_normalized_brightness_seq(
    block_frames: list[_FrameState], row_count: int
) -> list[tuple[float, float, float]]:
    stable = [fr for fr in block_frames if len(fr.rows) == row_count]
    if len(stable) < 4:
        return []

    row_vals: list[list[float]] = [[] for _ in range(row_count)]
    for fr in stable:
        for ridx, row in enumerate(fr.rows):
            row_vals[ridx].append(float(row.brightness))
    row_minmax: list[tuple[float, float]] = []
    for vals in row_vals:
        if not vals:
            return []
        mn = min(vals)
        mx = max(vals)
        row_minmax.append((mn, mx if mx > mn else mn + 1.0))

    seq: list[tuple[float, float, float]] = []
    for fr in stable:
        norms: list[float] = []
        for ridx, row in enumerate(fr.rows):
            mn, mx = row_minmax[ridx]
            norms.append((float(row.brightness) - mn) / max(1.0, mx - mn))
        seq.append((float(fr.time), max(norms), sum(norms)))
    return seq


def _split_block_on_row_text_phase_changes(  # noqa: C901
    block_frames: list[_FrameState],
) -> list[list[_FrameState]]:
    """Split stable blocks when a row text variant changes phase over time.

    Example: a top row transitions from "ooh take that money" to "take that money"
    while the rest of the screen remains unchanged. Layout-only segmentation won't
    split this, but the text-phase change often marks repeated-cycle boundaries.
    """
    if len(block_frames) < 16:
        return [block_frames]
    duration = float(block_frames[-1].time) - float(block_frames[0].time)
    if duration < 6.0:
        return [block_frames]

    rows = _cluster_rows_within_block(block_frames)
    if not (2 <= len(rows) <= 5):
        return [block_frames]

    # Start with the top row; ad-lib/lead variants commonly appear there.
    top = rows[0]
    by_variant: dict[str, list[float]] = collections.defaultdict(list)
    for obs in top.observations:
        norm = normalize_text_basic(obs.text)
        if norm:
            by_variant[norm].append(float(obs.time))
    if len(by_variant) < 2:
        return [block_frames]

    variants = sorted(by_variant.items(), key=lambda kv: len(kv[1]), reverse=True)
    (v1, ts1), (v2, ts2) = variants[:2]
    min_count = max(10, int(0.18 * len(block_frames)))
    if len(ts1) < min_count or len(ts2) < min_count:
        return [block_frames]

    t1_min, t1_max = min(ts1), max(ts1)
    t2_min, t2_max = min(ts2), max(ts2)
    # Require distinct phases rather than interleaved OCR noise.
    if not (t1_max <= (t2_min + 0.2) or t2_max <= (t1_min + 0.2)):
        return [block_frames]

    toks1 = [t for t in v1.split() if t]
    toks2 = [t for t in v2.split() if t]
    if not toks1 or not toks2:
        return [block_frames]

    def _is_suffix_variant(a: list[str], b: list[str]) -> bool:
        if len(a) == len(b):
            return False
        longer, shorter = (a, b) if len(a) > len(b) else (b, a)
        if len(longer) - len(shorter) > 2:
            return False
        return longer[-len(shorter) :] == shorter

    # Require a likely ad-lib/prefix variant relation or strong text similarity.
    sim = SequenceMatcher(None, v1, v2).ratio()
    if not (_is_suffix_variant(toks1, toks2) or sim >= 0.8):
        return [block_frames]

    early_end = t1_max if t1_max <= t2_min else t2_max
    late_start = t2_min if t1_max <= t2_min else t1_min
    if (late_start - early_end) < -0.1:
        return [block_frames]
    split_time = late_start - 0.02
    if not _has_selection_reset_near_time(block_frames, len(rows), split_time):
        return [block_frames]

    cut_idx: Optional[int] = None
    for i, fr in enumerate(block_frames):
        if float(fr.time) >= split_time:
            cut_idx = i
            break
    if cut_idx is None or cut_idx < 2 or cut_idx > len(block_frames) - 2:
        return [block_frames]

    return [block_frames[:cut_idx], block_frames[cut_idx:]]


def _has_selection_reset_near_time(
    block_frames: list[_FrameState], row_count: int, split_time: float
) -> bool:
    if not (2 <= row_count <= 5):
        return False

    stable_frames = [fr for fr in block_frames if len(fr.rows) == row_count]
    if len(stable_frames) < 10:
        return False

    row_brightness: list[list[float]] = [[] for _ in range(row_count)]
    for fr in stable_frames:
        for ridx, row in enumerate(fr.rows):
            row_brightness[ridx].append(float(row.brightness))

    row_minmax: list[tuple[float, float]] = []
    for vals in row_brightness:
        if not vals:
            return False
        mn = min(vals)
        mx = max(vals)
        row_minmax.append((mn, mx if mx > mn else mn + 1.0))

    seq: list[tuple[float, float, float]] = []
    # (time, max_norm, sum_norm)
    for fr in stable_frames:
        norm_vals: list[float] = []
        for ridx, row in enumerate(fr.rows):
            mn, mx = row_minmax[ridx]
            norm_vals.append((float(row.brightness) - mn) / max(1.0, mx - mn))
        seq.append((float(fr.time), max(norm_vals), sum(norm_vals)))

    if len(seq) < 10:
        return False

    heavy_thresh = max(1.4, 0.48 * row_count)
    valley_sum_thresh = max(0.95, 0.32 * row_count)
    window = 1.2

    valley_idx: Optional[int] = None
    for i, (t, max_norm, sum_norm) in enumerate(seq):
        if abs(t - split_time) > window:
            continue
        if sum_norm <= valley_sum_thresh and max_norm <= 0.62:
            valley_idx = i
            break
    if valley_idx is None:
        return False

    valley_t = seq[valley_idx][0]
    saw_heavy_before = any(
        (valley_t - 2.0) <= t < valley_t and sum_norm >= heavy_thresh
        for t, _max_norm, sum_norm in seq
    )
    saw_restart_after = any(
        valley_t < t <= (valley_t + 2.0) and max_norm >= 0.55
        for t, max_norm, _sum_norm in seq
    )
    return saw_heavy_before and saw_restart_after


def _prune_tiny_identical_blocks(
    blocks: list[list[_FrameState]],
) -> list[list[_FrameState]]:
    """Drop tiny duplicate blocks sandwiched between identical row blocks."""
    if len(blocks) < 3:
        return blocks

    row_text_cache: dict[int, list[str]] = {}

    def _row_texts(idx: int) -> list[str]:
        if idx not in row_text_cache:
            rows = _cluster_rows_within_block(blocks[idx])
            row_text_cache[idx] = [
                normalize_text_basic(_choose_canonical_observation(r).text)
                for r in rows
            ]
        return row_text_cache[idx]

    keep: list[list[_FrameState]] = []
    for i, b in enumerate(blocks):
        if i == 0 or i == len(blocks) - 1:
            keep.append(b)
            continue
        dur = float(b[-1].time) - float(b[0].time)
        if dur > 0.6 or len(b) > 12:
            keep.append(b)
            continue
        prev_texts = _row_texts(i - 1)
        cur_texts = _row_texts(i)
        next_texts = _row_texts(i + 1)
        if (
            len(cur_texts) >= 3
            and cur_texts == prev_texts == next_texts
            and len(cur_texts) == len(prev_texts)
        ):
            # Tiny duplicate pulse between two identical repeated blocks.
            continue
        prev_dur = float(blocks[i - 1][-1].time) - float(blocks[i - 1][0].time)
        next_gap = float(blocks[i + 1][0].time) - float(blocks[i][-1].time)
        if (
            len(cur_texts) >= 3
            and cur_texts == prev_texts
            and prev_dur >= 3.0
            and next_gap <= 0.2
        ):
            # Tiny residual duplicate pulse at a transition boundary (often produced
            # when a repeated screen fades into a different block).
            continue
        keep.append(b)
    return keep


def _cluster_rows_within_block(block_frames: list[_FrameState]) -> list[_BlockRow]:
    row_clusters: list[_BlockRow] = []
    for fr in block_frames:
        for row in fr.rows:
            best_idx: Optional[int] = None
            best_dist = 1e9
            for i, cluster in enumerate(row_clusters):
                dist = abs(cluster.y - row.y)
                if dist < best_dist and dist <= 28.0:
                    best_idx = i
                    best_dist = dist
            if best_idx is None:
                row_clusters.append(_BlockRow(y=row.y, observations=[row]))
            else:
                c = row_clusters[best_idx]
                c.observations.append(row)
                c.y = (c.y * (len(c.observations) - 1) + row.y) / len(c.observations)
    row_clusters.sort(key=lambda c: c.y)
    return row_clusters


def _estimate_row_windows_from_block(  # noqa: C901
    block_frames: list[_FrameState],
    row_clusters: list[_BlockRow],
) -> Optional[list[tuple[float, float]]]:
    """Estimate row onset/offsets from selected-row progression inside a block.

    Uses per-row normalized brightness so baseline row brightness does not dominate.
    Returns windows in row-cluster order when a stable sequence is detected.
    """
    row_count = len(row_clusters)
    if row_count < 3 or row_count > 5:
        return None

    # Map frame rows to cluster rows by y proximity.
    frame_rows: list[tuple[float, list[_FrameRow | None]]] = []
    row_brightness: list[list[float]] = [[] for _ in range(row_count)]
    for fr in block_frames:
        mapped: list[Optional[_FrameRow]] = [None] * row_count
        for r in fr.rows:
            best_idx: Optional[int] = None
            best_dist = 1e9
            for idx, c in enumerate(row_clusters):
                d = abs(float(c.y) - float(r.y))
                if d <= 28.0 and d < best_dist:
                    best_idx = idx
                    best_dist = d
            if best_idx is None:
                continue
            prev = mapped[best_idx]
            # Prefer visible or brighter observation when multiple rows collide.
            if (
                prev is None
                or (r.is_visible and not prev.is_visible)
                or (r.is_visible == prev.is_visible and r.brightness > prev.brightness)
            ):
                mapped[best_idx] = r
        if sum(1 for m in mapped if m is not None) < row_count - 1:
            continue
        resolved: list[_FrameRow] = []
        for idx in range(row_count):
            mapped_row = mapped[idx]
            if mapped_row is None:
                # If exactly one row missing, use a placeholder-like baseline row.
                # This frame will contribute weakly and may still be useful for timing.
                continue
            resolved.append(mapped_row)
            row_brightness[idx].append(float(mapped_row.brightness))
        if len(resolved) >= row_count - 1:
            frame_rows.append((float(fr.time), mapped))  # type: ignore[arg-type]

    if len(frame_rows) < 10:
        return None

    row_minmax: list[tuple[float, float]] = []
    for vals in row_brightness:
        if not vals:
            return None
        mn = min(vals)
        mx = max(vals)
        row_minmax.append((mn, mx if mx > mn else mn + 1.0))

    # For each frame, pick the most likely selected row by normalized brightness.
    seq: list[tuple[float, int, list[float]]] = []
    for t, mapped in frame_rows:
        norm_vals: list[float] = []
        for idx in range(row_count):
            row = mapped[idx]
            if row is None:
                norm_vals.append(0.0)
                continue
            mn, mx = row_minmax[idx]
            norm_vals.append((float(row.brightness) - mn) / max(1.0, mx - mn))
        if max(norm_vals) < 0.28:
            continue
        selected_idx = max(range(row_count), key=lambda idx: (norm_vals[idx], -idx))
        seq.append((t, selected_idx, norm_vals))

    if len(seq) < 8:
        return None

    # Row window onset: first time a row becomes dominant after previous row.
    onsets: list[Optional[float]] = [None] * row_count
    offsets: list[Optional[float]] = [None] * row_count
    prev_idx = seq[0][1]
    onsets[prev_idx] = seq[0][0]
    for t, idx, _vals in seq[1:]:
        if idx != prev_idx:
            if offsets[prev_idx] is None:
                offsets[prev_idx] = t
            if onsets[idx] is None:
                onsets[idx] = t
            prev_idx = idx
    if offsets[prev_idx] is None:
        offsets[prev_idx] = seq[-1][0] + 0.2

    # Fill missing rows conservatively from observation windows.
    for idx, cluster in enumerate(row_clusters):
        if onsets[idx] is None:
            obs_times = sorted(r.time for r in cluster.observations if r.is_visible)
            if obs_times:
                onsets[idx] = float(obs_times[0])
        if offsets[idx] is None:
            obs_times = sorted(r.time for r in cluster.observations if r.is_visible)
            if obs_times:
                offsets[idx] = float(obs_times[-1]) + 0.2

    if any(v is None for v in onsets) or any(v is None for v in offsets):
        return None

    windows: list[tuple[float, float]] = []
    prev_t = float(block_frames[0].time) - 0.05
    for idx in range(row_count):
        on = onsets[idx]
        off = offsets[idx]
        if on is None or off is None:
            return None
        s = max(float(on), prev_t + 0.05)
        e = max(s + 0.2, float(off))
        windows.append((s, e))
        prev_t = s
    return windows


def _estimate_cycle_row_windows_from_block(  # noqa: C901
    block_frames: list[_FrameState],
    row_clusters: list[_BlockRow],
) -> Optional[list[tuple[float, float, list[tuple[float, float]]]]]:
    """Estimate repeated cycles inside a stable block.

    Returns cycle segments as (cycle_start, cycle_end, row_windows) when the block
    appears to contain repeated highlight sweeps over the same rows.
    """
    row_count = len(row_clusters)
    if row_count < 2 or row_count > 5:
        return None
    if len(block_frames) < 20:
        return None
    duration = float(block_frames[-1].time) - float(block_frames[0].time)
    if duration < 6.5 or duration > 12.0:
        return None
    if not _has_strong_top_row_phase_change(block_frames):
        return None

    # Build normalized selected-row sequence (same approach as row-window estimation).
    frame_rows: list[tuple[float, list[Optional[_FrameRow]]]] = []
    row_brightness: list[list[float]] = [[] for _ in range(row_count)]
    for fr in block_frames:
        mapped: list[Optional[_FrameRow]] = [None] * row_count
        for r in fr.rows:
            best_idx: Optional[int] = None
            best_dist = 1e9
            for idx, c in enumerate(row_clusters):
                d = abs(float(c.y) - float(r.y))
                if d <= 28.0 and d < best_dist:
                    best_idx = idx
                    best_dist = d
            if best_idx is None:
                continue
            prev = mapped[best_idx]
            if (
                prev is None
                or (r.is_visible and not prev.is_visible)
                or (r.is_visible == prev.is_visible and r.brightness > prev.brightness)
            ):
                mapped[best_idx] = r
        if sum(1 for m in mapped if m is not None) < row_count - 1:
            continue
        frame_rows.append((float(fr.time), mapped))
        for idx, row in enumerate(mapped):
            if row is not None:
                row_brightness[idx].append(float(row.brightness))

    if len(frame_rows) < 12:
        return None
    row_minmax: list[tuple[float, float]] = []
    for vals in row_brightness:
        if not vals:
            return None
        mn = min(vals)
        mx = max(vals)
        row_minmax.append((mn, mx if mx > mn else mn + 1.0))

    seq: list[tuple[float, int, float, float]] = []
    # (time, selected_idx, max_norm, sum_norm)
    for t, mapped in frame_rows:
        norm_vals: list[float] = []
        for idx in range(row_count):
            row = mapped[idx]
            if row is None:
                norm_vals.append(0.0)
                continue
            mn, mx = row_minmax[idx]
            norm_vals.append((float(row.brightness) - mn) / max(1.0, mx - mn))
        max_norm = max(norm_vals)
        if max_norm < 0.28:
            continue
        selected_idx = max(range(row_count), key=lambda idx: (norm_vals[idx], -idx))
        seq.append((t, selected_idx, max_norm, sum(norm_vals)))
    if len(seq) < 10:
        return None

    def _restart_after(pos: int) -> bool:
        limit = min(len(seq), pos + 22)
        for k in range(pos + 1, limit):
            _t, idx_k, max_norm_k, _sum_k = seq[k]
            if max_norm_k < 0.55:
                continue
            if idx_k <= 1 or max_norm_k >= 0.8:
                return True
        return False

    split_times: list[float] = []
    seg_start_seq = 0
    seen_max_idx = seq[0][1]
    prev_idx = seq[0][1]
    last_split_time = seq[0][0]
    for j in range(1, len(seq)):
        t, idx, max_norm, _sum_norm = seq[j]
        seen_max_idx = max(seen_max_idx, idx)
        is_reset = (
            prev_idx >= (row_count - 2)
            and idx <= 1
            and seen_max_idx >= (row_count - 1)
            and (t - last_split_time) >= 1.8
            and (j - seg_start_seq) >= 5
            and max_norm >= 0.6
            and _restart_after(j)
        )
        if is_reset:
            split_times.append(t)
            seg_start_seq = j
            seen_max_idx = idx
            last_split_time = t
        prev_idx = idx

    if not split_times:
        return None

    # Convert split times to cycle frame segments and estimate row windows per cycle.
    cycles: list[tuple[float, float, list[tuple[float, float]]]] = []
    start_idx = 0
    cut_times = split_times + [float(block_frames[-1].time) + 1e-6]
    for cut_t in cut_times:
        seg_frames: list[_FrameState] = []
        while start_idx < len(block_frames):
            fr_t = float(block_frames[start_idx].time)
            if seg_frames and fr_t >= cut_t:
                break
            seg_frames.append(block_frames[start_idx])
            start_idx += 1
        if len(seg_frames) < 6:
            continue
        seg_start = float(seg_frames[0].time)
        seg_end = float(seg_frames[-1].time)
        if (seg_end - seg_start) < 1.0:
            continue
        seg_windows = _estimate_row_windows_from_seq(
            seg_start, seg_end, seq, row_clusters, row_count
        )
        if not seg_windows:
            seg_windows = _estimate_row_windows_from_block(seg_frames, row_clusters)
        if not seg_windows:
            continue
        cycles.append((seg_start, seg_end, seg_windows))

    if len(cycles) < 2:
        return None
    return cycles


def _estimate_row_windows_from_seq(
    cycle_start: float,
    cycle_end: float,
    seq: list[tuple[float, int, float, float]],
    row_clusters: list[_BlockRow],
    row_count: int,
) -> Optional[list[tuple[float, float]]]:
    seg_seq = [
        (t, idx)
        for (t, idx, _max_norm, _sum_norm) in seq
        if (cycle_start - 0.05) <= t <= (cycle_end + 0.05)
    ]
    if len(seg_seq) < 4:
        return None

    onsets: list[Optional[float]] = [None] * row_count
    offsets: list[Optional[float]] = [None] * row_count
    prev_idx = seg_seq[0][1]
    onsets[prev_idx] = max(cycle_start, seg_seq[0][0])
    for t, idx in seg_seq[1:]:
        if idx == prev_idx:
            continue
        if offsets[prev_idx] is None:
            offsets[prev_idx] = min(cycle_end, t)
        if onsets[idx] is None:
            onsets[idx] = max(cycle_start, t)
        prev_idx = idx
    if offsets[prev_idx] is None:
        offsets[prev_idx] = min(cycle_end + 0.2, seg_seq[-1][0] + 0.2)

    for idx, cluster in enumerate(row_clusters):
        if onsets[idx] is None:
            obs_times = sorted(
                float(r.time)
                for r in cluster.observations
                if r.is_visible
                and (cycle_start - 0.15) <= float(r.time) <= (cycle_end + 0.15)
            )
            if obs_times:
                onsets[idx] = max(cycle_start, obs_times[0])
        if offsets[idx] is None:
            obs_times = sorted(
                float(r.time)
                for r in cluster.observations
                if r.is_visible
                and (cycle_start - 0.15) <= float(r.time) <= (cycle_end + 0.15)
            )
            if obs_times:
                offsets[idx] = min(cycle_end + 0.2, obs_times[-1] + 0.2)

    if any(v is None for v in onsets) or any(v is None for v in offsets):
        return None

    windows: list[tuple[float, float]] = []
    prev_s = cycle_start - 0.05
    for idx in range(row_count):
        on = onsets[idx]
        off = offsets[idx]
        if on is None or off is None:
            return None
        s = max(float(on), prev_s + 0.05)
        e = max(s + 0.2, float(off))
        e = min(cycle_end + 0.4, e)
        windows.append((s, e))
        prev_s = s
    return windows


def _slice_cluster_observations_for_cycle(
    cluster: _BlockRow, cycle_start: float, cycle_end: float
) -> _BlockRow:
    obs = [
        r
        for r in cluster.observations
        if (cycle_start - 0.15) <= float(r.time) <= (cycle_end + 0.15)
    ]
    if not obs:
        return cluster
    return _BlockRow(y=cluster.y, observations=obs)


def _choose_canonical_observation(cluster: _BlockRow) -> _FrameRow:
    # Prefer the most frequently observed visible text variant, then quality signals.
    by_text: dict[str, list[_FrameRow]] = collections.defaultdict(list)
    for obs in cluster.observations:
        by_text[obs.text].append(obs)

    def _score_text(
        item: tuple[str, list[_FrameRow]],
    ) -> tuple[float, int, int, float, str]:
        txt, rows = item
        visible_rows = [r for r in rows if r.is_visible]
        sample_rows = visible_rows or rows
        max_words = max((len(r.words) for r in sample_rows), default=0)
        avg_b = sum(float(r.brightness) for r in sample_rows) / max(len(sample_rows), 1)
        return (
            float(len(visible_rows)),
            len(rows),
            max_words,
            avg_b,
            txt,
        )

    best_text, best_rows = max(by_text.items(), key=_score_text)
    ranked = sorted(
        best_rows,
        key=lambda r: (
            r.is_visible,
            len(r.words),
            len(r.text),
            r.brightness,
            r.text == best_text,
        ),
        reverse=True,
    )
    chosen = ranked[0]
    cleaned = _clean_row_tokens(list(chosen.words))
    if cleaned and cleaned != list(chosen.words):
        return _FrameRow(
            time=chosen.time,
            y=chosen.y,
            lane=chosen.lane,
            text=" ".join(cleaned),
            words=cleaned,
            w_rois=chosen.w_rois[: len(cleaned)],
            is_visible=chosen.is_visible,
            brightness=chosen.brightness,
        )
    return chosen


def _clean_row_tokens(tokens: list[str]) -> list[str]:
    if not tokens:
        return tokens
    # Remove immediate duplicate tokens (e.g., "like like", "swing my Swing my")
    deduped: list[str] = []
    for tok in tokens:
        if deduped and normalize_text_basic(deduped[-1]) == normalize_text_basic(tok):
            continue
        deduped.append(tok)
    if len(deduped) >= 4:
        # Remove repeated 2-token phrase immediately duplicated.
        out: list[str] = []
        i = 0
        while i < len(deduped):
            if i + 3 < len(deduped):
                a1 = normalize_text_basic(deduped[i])
                a2 = normalize_text_basic(deduped[i + 1])
                b1 = normalize_text_basic(deduped[i + 2])
                b2 = normalize_text_basic(deduped[i + 3])
                if a1 == b1 and a2 == b2:
                    out.extend(deduped[i : i + 2])
                    i += 4
                    continue
            out.append(deduped[i])
            i += 1
        deduped = out
    return deduped


def _merge_suffix_fragment_rows(rows: list[_BlockRow]) -> list[_BlockRow]:
    if len(rows) < 2:
        return rows
    kept: list[_BlockRow] = []
    i = 0
    while i < len(rows):
        if i + 1 >= len(rows):
            kept.append(rows[i])
            break
        a = rows[i]
        b = rows[i + 1]
        ca = _choose_canonical_observation(a)
        cb = _choose_canonical_observation(b)
        ta = [t for t in normalize_text_basic(ca.text).split() if t]
        tb = [t for t in normalize_text_basic(cb.text).split() if t]
        can_merge = (
            ta
            and tb
            and len(tb) <= 3
            and len(tb) < len(ta)
            and ta[-len(tb) :] == tb
            and abs(ca.time - cb.time) <= 0.5
            and (b.y - a.y) >= 20.0
        )
        if not can_merge:
            kept.append(a)
            i += 1
            continue

        # Append b observations into a cluster so canonical row can become fuller.
        merged = _BlockRow(
            y=a.y, observations=list(a.observations) + list(b.observations)
        )
        kept.append(merged)
        i += 2
    return kept


def _row_onset_time(cluster: _BlockRow, block_start: float) -> float:
    visible_obs = sorted(
        [r for r in cluster.observations if r.is_visible], key=lambda r: r.time
    )
    if visible_obs:
        # Use brightness rise within the block as a proxy for selection onset.
        # In many karaoke videos, all rows are visible but the selected row becomes
        # brighter; first-visible alone collapses row ordering.
        b_vals = [float(r.brightness) for r in visible_obs]
        b_min = min(b_vals)
        b_max = max(b_vals)
        if len(visible_obs) >= 3 and (b_max - b_min) >= 12.0:
            threshold = b_min + 0.55 * (b_max - b_min)
            for r in visible_obs:
                if float(r.brightness) >= threshold:
                    return float(r.time)
        return float(visible_obs[0].time)
    times = sorted(r.time for r in cluster.observations)
    return float(times[0]) if times else block_start


def _row_offset_time(cluster: _BlockRow, block_end: float, onset: float) -> float:
    visible_obs = sorted(
        [r for r in cluster.observations if r.is_visible], key=lambda r: r.time
    )
    if visible_obs:
        last_vis = float(visible_obs[-1].time)
        b_vals = [float(r.brightness) for r in visible_obs]
        b_min = min(b_vals)
        b_max = max(b_vals)
        if len(visible_obs) >= 4 and (b_max - b_min) >= 12.0:
            tail_threshold = b_min + 0.35 * (b_max - b_min)
            for r in reversed(visible_obs):
                if float(r.brightness) >= tail_threshold:
                    last_vis = float(r.time)
                    break
        return max(onset + 0.2, min(block_end + 0.8, last_vis + 0.25))
    return max(onset + 0.2, min(block_end + 0.8, block_end))


def _stagger_equal_onsets(
    row_specs: list[tuple[_BlockRow, _FrameRow, float]],
) -> list[float]:
    if not row_specs:
        return []
    onsets = [float(onset) for (_c, _cand, onset) in row_specs]
    adjusted = list(onsets)
    i = 0
    while i < len(adjusted):
        j = i + 1
        while j < len(adjusted) and abs(adjusted[j] - adjusted[i]) <= 0.05:
            j += 1
        if j - i >= 2:
            # Spread tied onsets across a short window in row order.
            step = 0.28
            for k in range(i, j):
                adjusted[k] = adjusted[i] + (k - i) * step
        i = j
    return adjusted


def reconstruct_lyrics_from_visuals_block_first_frames(
    raw_frames: list[dict[str, Any]],
    *,
    filter_static_overlay_words: FrameFilter,
    snap_fn: SnapFn,
) -> list[TargetLine]:
    frames = _frame_rows_from_raw_frames(
        raw_frames, filter_static_overlay_words=filter_static_overlay_words
    )
    blocks = _segment_blocks(frames)
    if not blocks:
        return []

    out: list[TargetLine] = []
    line_index = 1
    for block_id, bframes in enumerate(blocks):
        if not bframes:
            continue
        block_start = float(bframes[0].time)
        block_end = float(bframes[-1].time)
        next_block_start = (
            float(blocks[block_id + 1][0].time) if block_id + 1 < len(blocks) else None
        )
        rows = _cluster_rows_within_block(bframes)
        rows = _merge_suffix_fragment_rows(rows)
        if not rows:
            continue
        cycle_windows = _estimate_cycle_row_windows_from_block(bframes, rows)
        cycle_specs = (
            cycle_windows
            if cycle_windows
            else [
                (
                    block_start,
                    block_end,
                    _estimate_row_windows_from_block(bframes, rows) or [],
                )
            ]
        )

        for cycle_idx, (cycle_start, cycle_end, row_windows) in enumerate(cycle_specs):
            row_specs: list[tuple[_BlockRow, _FrameRow, float]] = []
            for row_order, cluster in enumerate(rows):
                cycle_cluster = _slice_cluster_observations_for_cycle(
                    cluster, cycle_start, cycle_end
                )
                cand = _choose_canonical_observation(cycle_cluster)
                onset = (
                    row_windows[row_order][0]
                    if row_windows and row_order < len(row_windows)
                    else _row_onset_time(cycle_cluster, cycle_start)
                )
                row_specs.append((cycle_cluster, cand, onset))

            # Enforce monotonic row starts in top-to-bottom order.
            row_specs.sort(key=lambda item: item[0].y)
            staged_onsets = _stagger_equal_onsets(row_specs)
            adjusted_starts: list[float] = []
            prev = cycle_start - 0.05
            for idx2, (_cluster, _cand, onset) in enumerate(row_specs):
                onset = staged_onsets[idx2] if idx2 < len(staged_onsets) else onset
                s = max(onset, prev + 0.05)
                adjusted_starts.append(s)
                prev = s

            for row_order, ((cluster, cand, _onset), s) in enumerate(
                zip(row_specs, adjusted_starts)
            ):
                next_start = (
                    adjusted_starts[row_order + 1]
                    if row_order + 1 < len(adjusted_starts)
                    else (cycle_end + 0.8)
                )
                observed_end = _row_offset_time(cluster, cycle_end, s)
                if row_windows and row_order < len(row_windows):
                    observed_end = row_windows[row_order][1]
                e_cap = cycle_end + 0.8
                if cycle_idx == len(cycle_specs) - 1 and next_block_start is not None:
                    e_cap = min(e_cap, next_block_start - 0.02)
                e = max(s + 0.2, min(e_cap, min(observed_end, next_start)))
                meta = {
                    "block_first": {
                        "block_id": block_id,
                        "row_order": row_order,
                        "row_y": round(float(cluster.y), 1),
                        "frame_block_size": len(bframes),
                        "cycle_index": cycle_idx,
                        "cycle_count": len(cycle_specs),
                    }
                }
                out.append(
                    TargetLine(
                        line_index=line_index,
                        start=snap_fn(s),
                        end=snap_fn(e),
                        text=cand.text,
                        words=list(cand.words),
                        y=float(cluster.y),
                        word_rois=list(cand.w_rois),
                        visibility_start=float(block_start),
                        visibility_end=float(block_end),
                        reconstruction_meta=meta,
                    )
                )
                line_index += 1

    return out
