"""Track matching and lifecycle helpers for frame accumulation."""

from __future__ import annotations

from typing import Any

from ..text_utils import text_similarity

_TRACK_MATCH_BUCKET_PX = 20
_TRACK_MATCH_NEIGHBOR_BUCKETS = 2


def candidate_track_indices_for_entry(
    entry: dict[str, Any],
    *,
    active_track_count: int,
    track_buckets: dict[int, list[int]],
) -> list[int]:
    y_bucket = int(entry["y"]) // _TRACK_MATCH_BUCKET_PX
    candidate_indices: list[int] = []
    for delta in range(
        -_TRACK_MATCH_NEIGHBOR_BUCKETS, _TRACK_MATCH_NEIGHBOR_BUCKETS + 1
    ):
        candidate_indices.extend(track_buckets.get(y_bucket + delta, []))
    if not candidate_indices:
        candidate_indices = list(range(active_track_count))
    # Preserve deterministic iteration order comparable to enumerate(active_tracks)
    return sorted(set(candidate_indices))


def find_best_track_match(
    *,
    entry: dict[str, Any],
    active_track_count: int,
    matched_track_indices: set[int],
    track_y_cache: list[int],
    track_text_cache: list[str],
    track_stale_cache: list[bool],
    track_buckets: dict[int, list[int]],
) -> tuple[int, float]:
    best_match_idx = -1
    best_match_score = 0.0
    candidate_indices = candidate_track_indices_for_entry(
        entry,
        active_track_count=active_track_count,
        track_buckets=track_buckets,
    )
    for idx in candidate_indices:
        if idx in matched_track_indices:
            continue
        if track_stale_cache[idx]:
            continue
        dy = abs(track_y_cache[idx] - entry["y"])
        if dy > 25:
            continue
        sim = text_similarity(entry["text"], track_text_cache[idx])
        if sim > 0.6 or (dy < 10 and sim > 0.5):
            score = sim * 0.7 + (1.0 - min(dy, 30) / 30.0) * 0.3
            if score > best_match_score:
                best_match_score = score
                best_match_idx = idx
    return best_match_idx, best_match_score


def initialize_new_track_visibility(
    *,
    track: Any,
    curr_time: float,
    is_visible: bool,
) -> None:
    if is_visible:
        track.vis_count = 1
        track.visible_yet = True
        track.first_visible = curr_time
        track.last_visible_seen = curr_time
        return
    track.vis_count = 0
    track.visible_yet = False
    track.first_visible = 999999.0
    track.last_visible_seen = None


def commit_or_keep_unmatched_tracks(
    *,
    active_tracks: list[Any],
    curr_time: float,
    matched_track_indices: set[int],
    committed: list[dict[str, Any]],
) -> list[Any]:
    remaining_tracks: list[Any] = []
    for idx, track in enumerate(active_tracks):
        if idx in matched_track_indices:
            remaining_tracks.append(track)
            continue
        if curr_time - track.last_seen > 1.0:
            if track.first_visible == 999999.0:
                track.first_visible = track.last_seen
            committed.append(track.to_dict())
            continue
        remaining_tracks.append(track)
    return remaining_tracks
