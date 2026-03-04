from __future__ import annotations

import collections
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..models import TargetLine
from ..text_utils import normalize_text_basic
from .reconstruction_block_first_segmentation import (
    frame_state_same_block as _frame_state_same_block_impl,
    has_strong_top_row_phase_change as _has_strong_top_row_phase_change_impl,
    prune_tiny_identical_blocks as _prune_tiny_identical_blocks_impl,
    row_similarity as _row_similarity_impl,
    split_block_on_row_cycle_resets as _split_block_on_row_cycle_resets_impl,
    split_block_on_row_text_phase_changes as _split_block_on_row_text_phase_changes_impl,
)
from .reconstruction_block_first_windows import (
    estimate_cycle_row_windows_from_block as _estimate_cycle_row_windows_from_block_impl,
    estimate_row_windows_from_block as _estimate_row_windows_from_block_impl,
    estimate_row_windows_from_seq as _estimate_row_windows_from_seq_impl,
)
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
    return _row_similarity_impl(a, b)


def _frame_state_same_block(a: _FrameState, b: _FrameState) -> bool:
    return _frame_state_same_block_impl(a, b)


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
    return _split_block_on_row_cycle_resets_impl(
        block_frames,
        cluster_rows_within_block_fn=_cluster_rows_within_block,
        choose_canonical_observation_fn=_choose_canonical_observation,
    )


def _prune_tiny_identical_blocks(
    blocks: list[list[_FrameState]],
) -> list[list[_FrameState]]:
    return _prune_tiny_identical_blocks_impl(
        blocks,
        cluster_rows_within_block_fn=_cluster_rows_within_block,
        choose_canonical_observation_fn=_choose_canonical_observation,
    )


def _split_block_on_row_text_phase_changes(
    block_frames: list[_FrameState],
) -> list[list[_FrameState]]:
    return _split_block_on_row_text_phase_changes_impl(
        block_frames,
        cluster_rows_within_block_fn=_cluster_rows_within_block,
    )


def _has_strong_top_row_phase_change(block_frames: list[_FrameState]) -> bool:
    return _has_strong_top_row_phase_change_impl(
        block_frames,
        cluster_rows_within_block_fn=_cluster_rows_within_block,
    )


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
    return _estimate_row_windows_from_block_impl(block_frames, row_clusters)


def _estimate_cycle_row_windows_from_block(  # noqa: C901
    block_frames: list[_FrameState],
    row_clusters: list[_BlockRow],
) -> Optional[list[tuple[float, float, list[tuple[float, float]]]]]:
    return _estimate_cycle_row_windows_from_block_impl(
        block_frames,
        row_clusters,
        has_strong_top_row_phase_change_fn=_has_strong_top_row_phase_change,
        estimate_row_windows_from_seq_fn=_estimate_row_windows_from_seq,
        estimate_row_windows_from_block_fn=_estimate_row_windows_from_block,
    )


def _estimate_row_windows_from_seq(
    cycle_start: float,
    cycle_end: float,
    seq: list[tuple[float, int, float, float]],
    row_clusters: list[_BlockRow],
    row_count: int,
) -> Optional[list[tuple[float, float]]]:
    return _estimate_row_windows_from_seq_impl(
        cycle_start, cycle_end, seq, row_clusters, row_count
    )


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


def _build_cycle_specs(
    bframes: list[_FrameState],
    rows: list[_BlockRow],
    *,
    block_start: float,
    block_end: float,
) -> list[tuple[float, float, list[tuple[float, float]]]]:
    cycle_windows = _estimate_cycle_row_windows_from_block(bframes, rows)
    if cycle_windows:
        return cycle_windows
    return [
        (
            block_start,
            block_end,
            _estimate_row_windows_from_block(bframes, rows) or [],
        )
    ]


def _build_row_specs_for_cycle(
    rows: list[_BlockRow],
    *,
    cycle_start: float,
    cycle_end: float,
    row_windows: list[tuple[float, float]],
) -> list[tuple[_BlockRow, _FrameRow, float]]:
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
    row_specs.sort(key=lambda item: item[0].y)
    return row_specs


def _compute_adjusted_row_starts(
    row_specs: list[tuple[_BlockRow, _FrameRow, float]],
    *,
    cycle_start: float,
) -> list[float]:
    staged_onsets = _stagger_equal_onsets(row_specs)
    adjusted_starts: list[float] = []
    prev = cycle_start - 0.05
    for idx, (_cluster, _cand, onset) in enumerate(row_specs):
        onset = staged_onsets[idx] if idx < len(staged_onsets) else onset
        s = max(onset, prev + 0.05)
        adjusted_starts.append(s)
        prev = s
    return adjusted_starts


def _append_cycle_target_lines(
    out: list[TargetLine],
    *,
    row_specs: list[tuple[_BlockRow, _FrameRow, float]],
    adjusted_starts: list[float],
    row_windows: list[tuple[float, float]],
    line_index_start: int,
    block_id: int,
    cycle_idx: int,
    cycle_specs_count: int,
    bframes: list[_FrameState],
    cycle_end: float,
    next_block_start: Optional[float],
    snap_fn: SnapFn,
) -> int:
    line_index = line_index_start
    for row_order, ((cluster, cand, _onset), s) in enumerate(
        zip(row_specs, adjusted_starts)
    ):
        e = _compute_row_line_end(
            cluster=cluster,
            row_windows=row_windows,
            row_order=row_order,
            adjusted_starts=adjusted_starts,
            cycle_end=cycle_end,
            start=s,
            cycle_idx=cycle_idx,
            cycle_specs_count=cycle_specs_count,
            next_block_start=next_block_start,
        )
        meta = _build_block_first_reconstruction_meta(
            block_id=block_id,
            row_order=row_order,
            row_y=cluster.y,
            frame_block_size=len(bframes),
            cycle_idx=cycle_idx,
            cycle_specs_count=cycle_specs_count,
        )
        out.append(
            _build_target_line_for_cycle_row(
                line_index=line_index,
                start=s,
                end=e,
                candidate=cand,
                row_y=cluster.y,
                bframes=bframes,
                reconstruction_meta=meta,
                snap_fn=snap_fn,
            )
        )
        line_index += 1
    return line_index


def _compute_row_line_end(
    *,
    cluster: _BlockRow,
    row_windows: list[tuple[float, float]],
    row_order: int,
    adjusted_starts: list[float],
    cycle_end: float,
    start: float,
    cycle_idx: int,
    cycle_specs_count: int,
    next_block_start: Optional[float],
) -> float:
    next_start = (
        adjusted_starts[row_order + 1]
        if row_order + 1 < len(adjusted_starts)
        else (cycle_end + 0.8)
    )
    observed_end = _row_offset_time(cluster, cycle_end, start)
    if row_windows and row_order < len(row_windows):
        observed_end = row_windows[row_order][1]
    e_cap = cycle_end + 0.8
    if cycle_idx == cycle_specs_count - 1 and next_block_start is not None:
        e_cap = min(e_cap, next_block_start - 0.02)
    return max(start + 0.2, min(e_cap, min(observed_end, next_start)))


def _build_block_first_reconstruction_meta(
    *,
    block_id: int,
    row_order: int,
    row_y: float,
    frame_block_size: int,
    cycle_idx: int,
    cycle_specs_count: int,
) -> dict[str, dict[str, float | int]]:
    return {
        "block_first": {
            "block_id": block_id,
            "row_order": row_order,
            "row_y": round(float(row_y), 1),
            "frame_block_size": frame_block_size,
            "cycle_index": cycle_idx,
            "cycle_count": cycle_specs_count,
        }
    }


def _build_target_line_for_cycle_row(
    *,
    line_index: int,
    start: float,
    end: float,
    candidate: _FrameRow,
    row_y: float,
    bframes: list[_FrameState],
    reconstruction_meta: dict[str, dict[str, float | int]],
    snap_fn: SnapFn,
) -> TargetLine:
    return TargetLine(
        line_index=line_index,
        start=snap_fn(start),
        end=snap_fn(end),
        text=candidate.text,
        words=list(candidate.words),
        y=float(row_y),
        word_rois=list(candidate.w_rois),
        visibility_start=float(bframes[0].time),
        visibility_end=float(bframes[-1].time),
        reconstruction_meta=reconstruction_meta,
    )


def _append_block_cycles_to_output(
    out: list[TargetLine],
    *,
    block_id: int,
    bframes: list[_FrameState],
    rows: list[_BlockRow],
    block_start: float,
    block_end: float,
    next_block_start: Optional[float],
    line_index_start: int,
    snap_fn: SnapFn,
) -> int:
    cycle_specs = _build_cycle_specs(
        bframes,
        rows,
        block_start=block_start,
        block_end=block_end,
    )

    line_index = line_index_start
    for cycle_idx, (cycle_start, cycle_end, row_windows) in enumerate(cycle_specs):
        row_specs = _build_row_specs_for_cycle(
            rows,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            row_windows=row_windows,
        )
        adjusted_starts = _compute_adjusted_row_starts(
            row_specs, cycle_start=cycle_start
        )
        line_index = _append_cycle_target_lines(
            out,
            row_specs=row_specs,
            adjusted_starts=adjusted_starts,
            row_windows=row_windows,
            line_index_start=line_index,
            block_id=block_id,
            cycle_idx=cycle_idx,
            cycle_specs_count=len(cycle_specs),
            bframes=bframes,
            cycle_end=cycle_end,
            next_block_start=next_block_start,
            snap_fn=snap_fn,
        )
    return line_index


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
        line_index = _append_block_cycles_to_output(
            out,
            block_id=block_id,
            bframes=bframes,
            rows=rows,
            block_start=block_start,
            block_end=block_end,
            next_block_start=next_block_start,
            line_index_start=line_index,
            snap_fn=snap_fn,
        )

    return out
