from __future__ import annotations

from typing import Any, Optional


def _segment_sequence(
    seq: list[tuple[float, int, float, float]],
    cycle_start: float,
    cycle_end: float,
) -> list[tuple[float, int]]:
    return [
        (t, idx)
        for (t, idx, _max_norm, _sum_norm) in seq
        if (cycle_start - 0.05) <= t <= (cycle_end + 0.05)
    ]


def _assign_transition_windows(
    seg_seq: list[tuple[float, int]],
    row_count: int,
    *,
    cycle_start: float,
    cycle_end: float,
) -> tuple[list[Optional[float]], list[Optional[float]]]:
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
    return onsets, offsets


def _visible_obs_times(
    cluster: Any,
    *,
    cycle_start: float,
    cycle_end: float,
) -> list[float]:
    return sorted(
        float(r.time)
        for r in cluster.observations
        if r.is_visible and (cycle_start - 0.15) <= float(r.time) <= (cycle_end + 0.15)
    )


def _fill_missing_from_observations(
    onsets: list[Optional[float]],
    offsets: list[Optional[float]],
    row_clusters: list[Any],
    *,
    cycle_start: float,
    cycle_end: float,
) -> None:
    for idx, cluster in enumerate(row_clusters):
        if onsets[idx] is None or offsets[idx] is None:
            obs_times = _visible_obs_times(
                cluster, cycle_start=cycle_start, cycle_end=cycle_end
            )
            if not obs_times:
                continue
            if onsets[idx] is None:
                onsets[idx] = max(cycle_start, obs_times[0])
            if offsets[idx] is None:
                offsets[idx] = min(cycle_end + 0.2, obs_times[-1] + 0.2)


def _assemble_windows(
    onsets: list[Optional[float]],
    offsets: list[Optional[float]],
    *,
    row_count: int,
    cycle_start: float,
    cycle_end: float,
) -> Optional[list[tuple[float, float]]]:
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


def estimate_row_windows_from_block(  # noqa: C901
    block_frames: list[Any],
    row_clusters: list[Any],
) -> Optional[list[tuple[float, float]]]:
    row_count = len(row_clusters)
    if row_count < 3 or row_count > 5:
        return None

    frame_rows: list[tuple[float, list[Any | None]]] = []
    row_brightness: list[list[float]] = [[] for _ in range(row_count)]
    for fr in block_frames:
        mapped: list[Optional[Any]] = [None] * row_count
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
        resolved: list[Any] = []
        for idx in range(row_count):
            mapped_row = mapped[idx]
            if mapped_row is None:
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


def estimate_cycle_row_windows_from_block(  # noqa: C901
    block_frames: list[Any],
    row_clusters: list[Any],
    *,
    has_strong_top_row_phase_change_fn,
    estimate_row_windows_from_seq_fn,
    estimate_row_windows_from_block_fn,
) -> Optional[list[tuple[float, float, list[tuple[float, float]]]]]:
    row_count = len(row_clusters)
    if row_count < 2 or row_count > 5:
        return None
    if len(block_frames) < 20:
        return None
    duration = float(block_frames[-1].time) - float(block_frames[0].time)
    if duration < 6.5 or duration > 12.0:
        return None
    if not has_strong_top_row_phase_change_fn(block_frames):
        return None

    frame_rows: list[tuple[float, list[Optional[Any]]]] = []
    row_brightness: list[list[float]] = [[] for _ in range(row_count)]
    for fr in block_frames:
        mapped: list[Optional[Any]] = [None] * row_count
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

    cycles: list[tuple[float, float, list[tuple[float, float]]]] = []
    start_idx = 0
    cut_times = split_times + [float(block_frames[-1].time) + 1e-6]
    for cut_t in cut_times:
        seg_frames: list[Any] = []
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
        seg_windows = estimate_row_windows_from_seq_fn(
            seg_start, seg_end, seq, row_clusters, row_count
        )
        if not seg_windows:
            seg_windows = estimate_row_windows_from_block_fn(seg_frames, row_clusters)
        if not seg_windows:
            continue
        cycles.append((seg_start, seg_end, seg_windows))

    if len(cycles) < 2:
        return None
    return cycles


def estimate_row_windows_from_seq(
    cycle_start: float,
    cycle_end: float,
    seq: list[tuple[float, int, float, float]],
    row_clusters: list[Any],
    row_count: int,
) -> Optional[list[tuple[float, float]]]:
    seg_seq = _segment_sequence(seq, cycle_start, cycle_end)
    if len(seg_seq) < 4:
        return None

    onsets, offsets = _assign_transition_windows(
        seg_seq, row_count, cycle_start=cycle_start, cycle_end=cycle_end
    )
    _fill_missing_from_observations(
        onsets, offsets, row_clusters, cycle_start=cycle_start, cycle_end=cycle_end
    )
    return _assemble_windows(
        onsets,
        offsets,
        row_count=row_count,
        cycle_start=cycle_start,
        cycle_end=cycle_end,
    )
