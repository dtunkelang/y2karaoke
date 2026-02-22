"""Frame-by-frame OCR word accumulation and line persistence tracking."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from ..text_utils import (
    normalize_ocr_line,
    normalize_ocr_tokens,
    normalize_text_basic,
)
from .word_segmentation import segment_line_tokens_by_visual_gaps

logger = logging.getLogger(__name__)
FrameFilter = Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]


def accumulate_persistent_lines_from_frames(
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

    on_screen: Dict[str, Dict[str, Any]] = {}
    committed: List[Dict[str, Any]] = []
    prev_time = None

    for frame in raw_frames:
        curr_time = frame["time"]
        words = frame.get("words", [])
        current_norms = {}

        if words:
            lines_in_frame = _group_words_into_lines(words)
            for ln_w in lines_in_frame:
                res = _process_line_in_frame(
                    ln_w,
                    curr_time,
                    prev_time,
                    visibility_thresholds,
                    on_screen,
                    visual_fps,
                )
                if res:
                    norm, entry, old_entry = res
                    if old_entry:
                        committed.append(old_entry)
                    current_norms[norm] = entry

        # Commit lines that have disappeared for a while
        for nt in list(on_screen.keys()):
            if on_screen[nt].get("first_visible") == 999999.0:
                on_screen[nt]["first_visible"] = on_screen[nt]["last"]

            if nt not in current_norms and curr_time - on_screen[nt]["last"] > 1.0:
                committed.append(on_screen.pop(nt))

        prev_time = curr_time

    for ent in on_screen.values():
        if ent.get("first_visible") == 999999.0:
            ent["first_visible"] = ent["last"]
        committed.append(ent)

    return committed


def _calculate_visibility_threshold(raw_frames: List[Dict[str, Any]]) -> Dict[int, float]:
    """Estimate 'full bright' threshold per vertical lane."""
    lane_brightness: Dict[int, List[float]] = {}
    for frame in raw_frames:
        for w in frame.get("words", []):
            if w.get("brightness", 0) > 0:
                lane = w["y"] // 20
                lane_brightness.setdefault(lane, []).append(w["brightness"])

    import numpy as np
    thresholds: Dict[int, float] = {}
    for lane, vals in lane_brightness.items():
        full_bright = float(np.percentile(vals, 95))
        thresholds[lane] = full_bright * 0.70
    
    # Default for unknown lanes
    all_vals = [v for l in lane_brightness.values() for v in l]
    global_def = float(np.percentile(all_vals, 95)) * 0.70 if all_vals else 150.0
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
    prev_time: float | None,
    visibility_thresholds: Dict[int, float],
    on_screen: Dict[str, Dict[str, Any]],
    visual_fps: float,
) -> tuple[str, Dict[str, Any], Dict[str, Any] | None] | None:
    """Normalize and track a line in the current frame."""
    ln_w.sort(key=lambda w: w["x"])
    line_tokens = segment_line_tokens_by_visual_gaps(ln_w)
    line_tokens = normalize_ocr_tokens(line_tokens)
    if not line_tokens:
        return None
    txt = normalize_ocr_line(" ".join(line_tokens))
    if not txt:
        return None

    y_pos = int(sum(w["y"] for w in ln_w) / len(ln_w))
    # Lane height of 30px is sufficient to distinguish standard karaoke rows
    lane = y_pos // 30
    norm = f"lane{lane}_{normalize_text_basic(txt)}"

    # Local threshold for this lane
    threshold = visibility_thresholds.get(lane, visibility_thresholds.get(-1, 150.0))

    # If any word lacks brightness data (e.g. mock data in unit tests), 
    # bypass the gate and assume it's visible.
    all_words_have_brightness = all("brightness" in w for w in ln_w)
    if not all_words_have_brightness:
        is_visible = True
    else:
        avg_brightness = sum(w["brightness"] for w in ln_w if "brightness" in w) / len(ln_w)
        is_visible = avg_brightness >= threshold

    if norm in on_screen and (
        prev_time is not None and on_screen[norm]["last"] == prev_time
    ):
        on_screen[norm]["last"] = curr_time
        if is_visible:
            on_screen[norm]["vis_count"] = on_screen[norm].get("vis_count", 0) + 1
        else:
            on_screen[norm]["vis_count"] = 0

        # Set first_visible only once when sustained visibility is reached
        if not on_screen[norm].get("visible_yet") and on_screen[norm]["vis_count"] >= 3:
            on_screen[norm]["first_visible"] = curr_time - (2.0 / visual_fps)
            on_screen[norm]["visible_yet"] = True
        return norm, on_screen[norm], None
    else:
        # Check if it hits visibility threshold on its first frame
        if is_visible:
            v_start = curr_time
            v_yet = True
            v_count = 1
        else:
            v_start = 999999.0
            v_yet = False
            v_count = 0

        entry = {
            "text": txt,
            "words": line_tokens,
            "first": curr_time,  # Logical detection onset
            "first_visible": v_start, # Visibility onset
            "last": curr_time,
            "y": y_pos,
            "lane": lane,
            "visible_yet": v_yet,
            "vis_count": v_count,
            "w_rois": [(w["x"], w["y"], w["w"], w["h"]) for w in ln_w],
        }
        old_entry = on_screen.get(norm)
        on_screen[norm] = entry
        return norm, entry, old_entry
