from __future__ import annotations

import collections
from difflib import SequenceMatcher
from typing import Any, Callable, Optional

from ..text_utils import normalize_text_basic


def row_similarity(a: Any, b: Any) -> float:
    if abs(float(a.y) - float(b.y)) > 28.0:
        return 0.0
    ta = normalize_text_basic(str(a.text))
    tb = normalize_text_basic(str(b.text))
    if not ta or not tb:
        return 0.0
    return SequenceMatcher(None, ta, tb).ratio()


def frame_state_same_block(a: Any, b: Any) -> bool:
    if not a.rows or not b.rows:
        return False
    if abs(len(a.rows) - len(b.rows)) > 1:
        return False
    ay = [float(r.y) for r in a.rows]
    by = [float(r.y) for r in b.rows]
    if len(ay) == len(by):
        ydiffs = [abs(x - y) for x, y in zip(ay, by)]
        if ydiffs and max(ydiffs) > 35.0:
            return False
    matches = 0
    strong = 0
    for ra in a.rows:
        best = max((row_similarity(ra, rb) for rb in b.rows), default=0.0)
        if best >= 0.45:
            matches += 1
        if best >= 0.7:
            strong += 1
    min_rows = min(len(a.rows), len(b.rows))
    return matches >= max(1, min_rows - 1) and (strong >= 1 or min_rows <= 2)


def split_block_on_row_cycle_resets(  # noqa: C901
    block_frames: list[Any],
    *,
    cluster_rows_within_block_fn: Callable[[list[Any]], list[Any]],
    choose_canonical_observation_fn: Callable[[Any], Any],
) -> list[list[Any]]:
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

    stable_row_brightness: list[list[float]] = [[] for _ in range(stable_count)]
    stable_frames_with_rows: list[tuple[int, Any]] = []
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
        limit = min(len(seq), pos + 28)
        for k in range(pos + 1, limit):
            _fi, _t, idx_k, max_norm_k, _sum_k = seq[k]
            if max_norm_k < 0.55:
                continue
            if idx_k <= 1 or max_norm_k >= 0.8:
                return True
        return False

    short_block_mode = duration < 10.0
    if short_block_mode and not has_strong_top_row_phase_change(
        block_frames,
        cluster_rows_within_block_fn=cluster_rows_within_block_fn,
    ):
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

    out: list[list[Any]] = []
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
    return _merge_adjacent_identical_row_blocks(
        out,
        cluster_rows_within_block_fn=cluster_rows_within_block_fn,
        choose_canonical_observation_fn=choose_canonical_observation_fn,
    )


def split_block_on_row_text_phase_changes(  # noqa: C901
    block_frames: list[Any],
    *,
    cluster_rows_within_block_fn: Callable[[list[Any]], list[Any]],
) -> list[list[Any]]:
    if len(block_frames) < 16:
        return [block_frames]
    duration = float(block_frames[-1].time) - float(block_frames[0].time)
    if duration < 6.0:
        return [block_frames]

    rows = cluster_rows_within_block_fn(block_frames)
    if not (2 <= len(rows) <= 5):
        return [block_frames]

    top = rows[0]
    by_variant: dict[str, list[float]] = collections.defaultdict(list)
    for obs in top.observations:
        norm = normalize_text_basic(str(obs.text))
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


def prune_tiny_identical_blocks(
    blocks: list[list[Any]],
    *,
    cluster_rows_within_block_fn: Callable[[list[Any]], list[Any]],
    choose_canonical_observation_fn: Callable[[Any], Any],
) -> list[list[Any]]:
    if len(blocks) < 3:
        return blocks

    row_text_cache: dict[int, list[str]] = {}

    def _row_texts(idx: int) -> list[str]:
        if idx not in row_text_cache:
            rows = cluster_rows_within_block_fn(blocks[idx])
            row_text_cache[idx] = [
                normalize_text_basic(str(choose_canonical_observation_fn(r).text))
                for r in rows
            ]
        return row_text_cache[idx]

    keep: list[list[Any]] = []
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
            continue
        prev_dur = float(blocks[i - 1][-1].time) - float(blocks[i - 1][0].time)
        next_gap = float(blocks[i + 1][0].time) - float(blocks[i][-1].time)
        if (
            len(cur_texts) >= 3
            and cur_texts == prev_texts
            and prev_dur >= 3.0
            and next_gap <= 0.2
        ):
            continue
        keep.append(b)
    return keep


def has_strong_top_row_phase_change(
    block_frames: list[Any],
    *,
    cluster_rows_within_block_fn: Callable[[list[Any]], list[Any]],
) -> bool:
    if len(block_frames) < 12:
        return False
    rows = cluster_rows_within_block_fn(block_frames)
    if not (2 <= len(rows) <= 5):
        return False
    top = rows[0]
    by_variant: dict[str, list[float]] = collections.defaultdict(list)
    for obs in top.observations:
        norm = normalize_text_basic(str(obs.text))
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
    blocks: list[list[Any]],
    *,
    cluster_rows_within_block_fn: Callable[[list[Any]], list[Any]],
    choose_canonical_observation_fn: Callable[[Any], Any],
) -> list[list[Any]]:
    if len(blocks) < 2:
        return blocks

    merged: list[list[Any]] = []
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

        cur_rows = cluster_rows_within_block_fn(cur)
        nxt_rows = cluster_rows_within_block_fn(nxt)
        if len(cur_rows) != len(nxt_rows) or len(cur_rows) < 3:
            merged.append(cur)
            i += 1
            continue

        cur_texts = [
            normalize_text_basic(str(choose_canonical_observation_fn(r).text))
            for r in cur_rows
        ]
        nxt_texts = [
            normalize_text_basic(str(choose_canonical_observation_fn(r).text))
            for r in nxt_rows
        ]
        same_rows = cur_texts == nxt_texts
        cur_dur = float(cur[-1].time) - float(cur[0].time)
        nxt_dur = float(nxt[-1].time) - float(nxt[0].time)
        reset_boundary = _looks_like_cycle_reset_boundary(cur, nxt, len(cur_rows))
        if same_rows and max(cur_dur, nxt_dur) <= 4.5 and not reset_boundary:
            merged.append(cur + nxt)
            i += 2
            continue

        merged.append(cur)
        i += 1

    return merged


def _looks_like_cycle_reset_boundary(
    cur: list[Any], nxt: list[Any], row_count: int
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
    block_frames: list[Any], row_count: int
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


def _has_selection_reset_near_time(
    block_frames: list[Any], row_count: int, split_time: float
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
