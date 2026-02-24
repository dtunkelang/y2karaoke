"""Frame-by-frame OCR word accumulation and line persistence tracking."""

from __future__ import annotations

import logging
import collections
import statistics
from typing import Any, Callable, Dict, List

from ..text_utils import (
    normalize_ocr_line,
    normalize_ocr_tokens,
    text_similarity,
    LYRIC_FUNCTION_WORDS,
)
from .word_segmentation import segment_line_tokens_by_visual_gaps

logger = logging.getLogger(__name__)
FrameFilter = Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]
_TRACK_MATCH_BUCKET_PX = 20
_TRACK_MATCH_NEIGHBOR_BUCKETS = 2


class TrackedLine:
    def __init__(self, entry: Dict[str, Any], entry_key: str):
        self.id = entry_key
        self.lane = entry["lane"]
        self.y_history = [entry["y"]]
        self.brightness_history = [entry.get("brightness", 0.0)]
        self.text_counts = collections.Counter({entry["text"]: 1})
        self.entries = [entry]
        self.first_seen = entry["first"]
        self.last_seen = entry["last"]

        # Visibility tracking
        self.visible_yet = entry.get("visible_yet", False)
        self.vis_count = entry.get("vis_count", 0)
        self.first_visible = entry.get("first_visible", 999999.0)

    def update(self, entry: Dict[str, Any], curr_time: float, is_visible: bool):
        self.entries.append(entry)
        self.last_seen = curr_time
        self.y_history.append(entry["y"])
        self.brightness_history.append(entry.get("brightness", 0.0))
        self.text_counts[entry["text"]] += 1

        # Update visibility
        if is_visible:
            self.vis_count += 1
        else:
            self.vis_count = 0

        if not self.visible_yet and self.vis_count >= 3:
            # Backtrack start time slightly to capture the fade-in
            # Using a simplified heuristic (2 frames back)
            # entry["first"] is curr_time.
            self.first_visible = max(self.first_seen, curr_time - 0.2)
            self.visible_yet = True

    @property
    def current_y(self) -> int:
        # Simple moving average or just last
        if not self.y_history:
            return 0
        return int(sum(self.y_history[-5:]) / len(self.y_history[-5:]))

    @property
    def avg_brightness(self) -> float:
        if not self.brightness_history:
            return 0.0
        return sum(self.brightness_history) / len(self.brightness_history)

    @property
    def best_text(self) -> str:
        # Fast frequency-based matching for track persistence during accumulation.
        if not self.text_counts:
            return ""
        return self.text_counts.most_common(1)[0][0]

    def get_voted_text(self) -> str:
        # Perform word-by-word voting across all frames to filter out transient OCR noise.
        # This is expensive and should only be called once per track on finalization.
        if not self.entries:
            return ""

        all_line_tokens = [e["words"] for e in self.entries]
        if not all_line_tokens:
            return self.best_text

        # Use the median word count as the target structure
        word_counts = [len(tokens) for tokens in all_line_tokens]
        target_wc = int(statistics.median(word_counts))

        # Only vote among entries that match the target word count
        valid_entries = [
            tokens for tokens in all_line_tokens if len(tokens) == target_wc
        ]
        if not valid_entries:
            return self.best_text

        valid_entry_texts = [
            str(e.get("text", ""))
            for e in self.entries
            if len(e.get("words", [])) == target_wc
        ]
        text_freq = collections.Counter(valid_entry_texts)

        final_words = []
        position_vote_counts: list[collections.Counter[str]] = []
        for i in range(target_wc):
            word_votes = collections.Counter(tokens[i] for tokens in valid_entries)
            position_vote_counts.append(word_votes)

            # Weighted vote
            weighted_votes = {}
            for word, count in word_votes.items():
                score = float(count)
                if word.lower() in LYRIC_FUNCTION_WORDS:
                    score += 0.5
                weighted_votes[word] = score

            best_word = max(weighted_votes.items(), key=lambda x: x[1])[0]
            final_words.append(best_word)
        voted_text = " ".join(final_words)

        # When per-position consensus is weak, per-word voting can synthesize a
        # never-observed "Frankenstein" line. In that case prefer the strongest
        # observed candidate with the target word count.
        if voted_text not in text_freq:
            n_valid = max(1, len(valid_entries))
            supports = [
                position_vote_counts[i].get(final_words[i], 0) / float(n_valid)
                for i in range(target_wc)
            ]
            weak_positions = sum(1 for s in supports if s < 0.5)
            avg_support = sum(supports) / len(supports) if supports else 1.0
            if avg_support < 0.72 or weak_positions >= 2:
                best_tokens = max(
                    valid_entries,
                    key=lambda toks: (
                        # Prefer candidates matching high-support position votes.
                        sum(
                            position_vote_counts[i].get(tok, 0)
                            for i, tok in enumerate(toks)
                        ),
                        text_freq.get(" ".join(toks), 0),
                        # Deterministic tie-breakers
                        sum(len(tok) for tok in toks),
                        " ".join(toks),
                    ),
                )
                return " ".join(best_tokens)

        return voted_text

    def to_dict(self) -> Dict[str, Any]:
        # Construct the final entry dictionary
        best_txt = self.get_voted_text()

        # Find the entry that matches best_txt to get 'words' and 'w_rois'
        # Prefer the longest/most complete one if multiple match
        candidates = [e for e in self.entries if e["text"] == best_txt]
        if not candidates:
            # Fallback to last entry if best text not found (shouldn't happen)
            candidates = [self.entries[-1]]

        # Pick the candidate with most words
        best_entry = max(candidates, key=lambda e: len(e["words"]))

        return {
            "text": best_txt,
            "words": best_entry["words"],
            "first": self.first_seen,
            "first_visible": (
                self.first_visible if self.visible_yet else self.first_seen
            ),  # Fallback
            "last": self.last_seen,
            "y": self.current_y,
            "lane": self.lane,
            "visible_yet": self.visible_yet,
            "vis_count": self.vis_count,
            "w_rois": best_entry["w_rois"],
            "avg_brightness": self.avg_brightness,
        }


def accumulate_persistent_lines_from_frames(  # noqa: C901
    raw_frames: List[Dict[str, Any]],
    *,
    filter_static_overlay_words: FrameFilter,
    visual_fps: float = 3.0,
) -> List[Dict[str, Any]]:
    """
    Groups OCR words into lines per frame and tracks them over time to form persistent line objects.
    """
    raw_frames = filter_static_overlay_words(raw_frames)
    visibility_thresholds = _calculate_visibility_threshold(raw_frames)

    active_tracks: List[TrackedLine] = []
    committed: List[Dict[str, Any]] = []

    # Unique ID counter
    _track_id_counter = 0

    for frame in raw_frames:
        curr_time = frame["time"]
        words = frame.get("words", [])

        current_frame_lines = []
        if words:
            lines_in_frame = _group_words_into_lines(words)
            for ln_w in lines_in_frame:
                res = _process_line_in_frame(
                    ln_w,
                    curr_time,
                    visibility_thresholds,
                )
                if res:
                    current_frame_lines.append(res)

        # Match current frame lines to active tracks
        matched_track_indices = set()
        track_y_cache = [track.current_y for track in active_tracks]
        track_text_cache = [track.best_text for track in active_tracks]
        track_stale_cache = [
            (curr_time - track.last_seen) > 1.0 for track in active_tracks
        ]
        track_buckets: Dict[int, List[int]] = {}
        for idx, y_val in enumerate(track_y_cache):
            bucket = int(y_val) // _TRACK_MATCH_BUCKET_PX
            track_buckets.setdefault(bucket, []).append(idx)

        for frame_line_data in current_frame_lines:
            entry, is_visible = frame_line_data

            # Find best match in active_tracks
            best_match_idx = -1
            best_match_score = 0.0
            y_bucket = int(entry["y"]) // _TRACK_MATCH_BUCKET_PX
            candidate_indices: list[int] = []
            for delta in range(
                -_TRACK_MATCH_NEIGHBOR_BUCKETS, _TRACK_MATCH_NEIGHBOR_BUCKETS + 1
            ):
                candidate_indices.extend(track_buckets.get(y_bucket + delta, []))
            if not candidate_indices:
                candidate_indices = list(range(len(active_tracks)))

            # Preserve deterministic iteration order comparable to enumerate(active_tracks)
            for idx in sorted(set(candidate_indices)):
                if idx in matched_track_indices:
                    continue
                track = active_tracks[idx]

                # Don't match if the track has been gone for too long
                if track_stale_cache[idx]:
                    continue

                # Spatial check (Y)
                dy = abs(track_y_cache[idx] - entry["y"])
                if dy > 25:  # Lane proximity
                    continue

                # Text similarity check
                sim = text_similarity(entry["text"], track_text_cache[idx])

                # If very close spatially and reasonable similarity
                # OR if exact text match (fast path)
                if sim > 0.6 or (dy < 10 and sim > 0.5):
                    # Score combines similarity and proximity
                    score = sim * 0.7 + (1.0 - min(dy, 30) / 30.0) * 0.3
                    if score > best_match_score:
                        best_match_score = score
                        best_match_idx = idx

            if best_match_idx != -1 and best_match_score > 0.4:
                # Update existing track
                active_tracks[best_match_idx].update(entry, curr_time, is_visible)
                matched_track_indices.add(best_match_idx)
            else:
                # Create new track
                _track_id_counter += 1
                new_track = TrackedLine(entry, f"track_{_track_id_counter}")
                # Initial visibility update
                if is_visible:
                    new_track.vis_count = 1
                    new_track.visible_yet = (
                        True  # Assume visible on first frame if bright enough?
                    )
                    # Actually logic in original was: "Check if it hits visibility threshold on its first frame"
                    new_track.first_visible = curr_time
                else:
                    new_track.vis_count = 0
                    new_track.visible_yet = False
                    new_track.first_visible = 999999.0

                active_tracks.append(new_track)
                matched_track_indices.add(len(active_tracks) - 1)

        # Commit tracks that have disappeared
        remaining_tracks = []
        for idx, track in enumerate(active_tracks):
            if idx in matched_track_indices:
                remaining_tracks.append(track)
            else:
                # Track was NOT matched in this frame
                if curr_time - track.last_seen > 1.0:
                    # It's been gone for > 1.0s, commit it
                    if track.first_visible == 999999.0:
                        track.first_visible = track.last_seen
                    committed.append(track.to_dict())
                else:
                    # Keep it alive for now (gap tolerance)
                    remaining_tracks.append(track)

        active_tracks = remaining_tracks

    # Commit any remaining active tracks at end of video
    for track in active_tracks:
        if track.first_visible == 999999.0:
            track.first_visible = track.last_seen
        committed.append(track.to_dict())

    return committed


def _calculate_visibility_threshold(
    raw_frames: List[Dict[str, Any]],
) -> Dict[int, float]:
    """Estimate 'full bright' threshold per vertical lane."""
    lane_brightness: Dict[int, List[float]] = {}
    for frame in raw_frames:
        for w in frame.get("words", []):
            if w.get("brightness", 0) > 0:
                lane = w["y"] // 40
                lane_brightness.setdefault(lane, []).append(w["brightness"])

    def _p95(values: List[float]) -> float:
        if not values:
            return 0.0
        try:
            import numpy as np  # type: ignore

            return float(np.percentile(values, 95))
        except ImportError:
            ordered = sorted(float(v) for v in values)
            if len(ordered) == 1:
                return ordered[0]
            pos = (len(ordered) - 1) * 0.95
            lo = int(pos)
            hi = min(lo + 1, len(ordered) - 1)
            frac = pos - lo
            return ordered[lo] * (1.0 - frac) + ordered[hi] * frac

    thresholds: Dict[int, float] = {}
    for lane, vals in lane_brightness.items():
        if not vals:
            continue
        full_bright = _p95(vals)
        thresholds[lane] = full_bright * 0.70

    # Default for unknown lanes
    all_vals = [v for vals in lane_brightness.values() for v in vals]
    global_def = _p95(all_vals) * 0.70 if all_vals else 150.0
    thresholds[-1] = global_def

    return thresholds


def _group_words_into_lines(words: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group words in a single frame into lines based on Y proximity."""
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: w["y"])
    lines = []
    curr = [sorted_words[0]]
    for i in range(1, len(sorted_words)):
        if sorted_words[i]["y"] - curr[-1]["y"] < 30:
            curr.append(sorted_words[i])
        else:
            lines.append(curr)
            curr = [sorted_words[i]]
    lines.append(curr)
    return lines


def _process_line_in_frame(
    ln_w: List[Dict[str, Any]],
    curr_time: float,
    visibility_thresholds: Dict[int, float],
) -> tuple[Dict[str, Any], bool] | None:
    """Normalize and prepare a line entry from the current frame."""
    ln_w.sort(key=lambda w: w["x"])
    line_tokens = segment_line_tokens_by_visual_gaps(ln_w)
    line_tokens = normalize_ocr_tokens(line_tokens)
    if not line_tokens:
        return None
    txt = normalize_ocr_line(" ".join(line_tokens))
    if not txt:
        return None

    y_pos = int(sum(w["y"] for w in ln_w) / len(ln_w))
    # Lane height of 40px is sufficient to distinguish standard karaoke rows
    lane = y_pos // 40

    # Local threshold for this lane
    threshold = visibility_thresholds.get(lane, visibility_thresholds.get(-1, 150.0))

    # If any word lacks brightness data (e.g. mock data in unit tests),
    # bypass the gate and assume it's visible.
    all_words_have_brightness = all("brightness" in w for w in ln_w)
    if not all_words_have_brightness:
        is_visible = True
    else:
        avg_brightness = sum(w["brightness"] for w in ln_w if "brightness" in w) / len(
            ln_w
        )
        is_visible = avg_brightness >= threshold

    entry = {
        "text": txt,
        "words": line_tokens,
        "first": curr_time,  # Logical detection onset
        "last": curr_time,
        "y": y_pos,
        "lane": lane,
        "w_rois": [(w["x"], w["y"], w["w"], w["h"]) for w in ln_w],
        "brightness": (
            sum(w.get("brightness", 0.0) for w in ln_w) / len(ln_w) if ln_w else 0.0
        ),
    }

    return entry, is_visible
