"""Visual timing refinement logic."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Any, Dict

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore

from .models import TargetLine
from .text_utils import normalize_ocr_line, normalize_text_basic, text_similarity
from ..exceptions import VisualRefinementError

logger = logging.getLogger(__name__)


def _word_fill_mask(roi_bgr: np.ndarray, c_bg: np.ndarray) -> np.ndarray:
    """Create a mask for text pixels (foreground)."""
    if cv2 is None or np is None:
        raise ImportError("OpenCV and Numpy required.")

    dist_bg = np.linalg.norm(roi_bgr - c_bg, axis=2)
    mask = (dist_bg > 35).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    return mask


def refine_word_timings_at_high_fps(  # noqa: C901
    video_path: Path,
    target_lines: List[TargetLine],
    roi_rect: tuple[int, int, int, int],
) -> None:
    """Refine start/end times for words in target_lines using high-FPS analysis."""
    if cv2 is None or np is None:
        raise ImportError("OpenCV and Numpy required.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VisualRefinementError(f"Could not open video: {video_path}")

    rx, ry, rw, rh = roi_rect
    logger.info("Refining timings with Departure-Onset detection...")

    for i, ln in enumerate(target_lines):
        if not ln.word_rois:
            continue
        # Window: independent per line
        # Handle potential None end time by defaulting to start + 5s (safe upper bound)
        line_end = ln.end if ln.end is not None else ln.start + 5.0
        v_start, v_end = max(0.0, ln.start - 1.0), line_end + 1.0

        cap.set(cv2.CAP_PROP_POS_MSEC, v_start * 1000.0)
        line_frames = []
        while True:
            ok, frame = cap.read()
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not ok or t > v_end:
                break
            line_frames.append((t, frame[ry : ry + rh, rx : rx + rw]))

        if len(line_frames) < 20:
            continue

        # Estimate background color from first few frames (assumed unlit)
        c_bg_line = np.mean(
            [np.mean(f[1], axis=(0, 1)) for f in line_frames[:10]], axis=0
        )

        new_starts: List[Optional[float]] = []
        new_ends: List[Optional[float]] = []

        # We know word_rois is not None from check above
        assert ln.word_rois is not None

        for wi in range(len(ln.words)):
            wx, wy, ww, wh = ln.word_rois[wi]
            # 1. Identify TEXT-ONLY frames
            word_vals = []
            for t, roi in line_frames:
                if wy + wh <= roi.shape[0] and wx + ww <= roi.shape[1]:
                    word_roi = roi[wy : wy + wh, wx : wx + ww]
                    mask = _word_fill_mask(word_roi, c_bg_line)
                    if np.sum(mask > 0) > 30:  # Glyph is present
                        lab = cv2.cvtColor(word_roi, cv2.COLOR_BGR2LAB).astype(
                            np.float32
                        )
                        word_vals.append(
                            {
                                "t": t,
                                "mask": mask,
                                "lab": lab,
                                "avg": lab[mask.astype(bool)].mean(axis=0),
                            }
                        )

            s, e = None, None
            if len(word_vals) > 10:
                l_vals = np.array([v["avg"][0] for v in word_vals])
                # Smooth lightness curve
                kernel_size = min(10, len(l_vals))
                l_smooth = np.convolve(
                    l_vals,
                    np.ones(kernel_size) / kernel_size,
                    mode="same",
                )

                idx_peak = int(np.argmax(l_smooth))
                c_initial = word_vals[idx_peak]["avg"]

                # Find valley after peak
                idx_valley = idx_peak + int(np.argmin(l_smooth[idx_peak:]))
                c_final = word_vals[idx_valley]["avg"]

                if np.linalg.norm(c_final - c_initial) > 2.0:
                    times = []
                    dists_in = []
                    for v in word_vals:
                        times.append(v["t"])
                        dists_in.append(np.linalg.norm(v["avg"] - c_initial))

                    # 2. Departure search: find exact frame where color starts moving
                    # Calculate noise floor from stable period around peak
                    start_stable = max(0, idx_peak - 5)
                    end_stable = min(len(dists_in), idx_peak + 5)
                    stable_range = dists_in[start_stable:end_stable]

                    noise_floor = 1.0
                    if stable_range:
                        noise_floor = float(
                            np.mean(stable_range) + 2 * np.std(stable_range)
                        )

                    for j in range(idx_peak, len(times)):
                        # Start trigger: Consistent departure from noise floor
                        if s is None and dists_in[j] > noise_floor:
                            if j + 3 < len(times) and all(
                                dists_in[j + k] > dists_in[j + k - 1]
                                for k in range(1, 4)
                            ):
                                s = times[j]

                        # End trigger: Closer to final state
                        if s is not None and e is None:
                            curr_dist_final = np.linalg.norm(
                                word_vals[j]["avg"] - c_final
                            )
                            curr_dist_initial = np.linalg.norm(
                                word_vals[j]["avg"] - c_initial
                            )
                            if curr_dist_final < curr_dist_initial:
                                e = times[j]
                                break
            new_starts.append(s)
            new_ends.append(e)

        ln.word_starts = new_starts
        ln.word_ends = new_ends

    cap.release()


def _snap(value: float) -> float:
    # Assuming 0.05s snap from original tool
    return round(round(float(value) / 0.05) * 0.05, 3)


def reconstruct_lyrics_from_visuals(  # noqa: C901
    raw_frames: list[dict[str, Any]], visual_fps: float
) -> list[TargetLine]:
    """Group raw OCR words into logical lines and assign timing."""
    on_screen: Dict[str, Dict[str, Any]] = {}
    committed = []

    for frame in raw_frames:
        words = frame.get("words", [])
        current_norms = set()

        if words:
            # Sort by Y to process lines top-to-bottom
            words.sort(key=lambda w: w["y"])
            lines_in_frame = []

            # Group words into lines based on Y-proximity
            if words:
                curr = [words[0]]
                for i in range(1, len(words)):
                    if words[i]["y"] - curr[-1]["y"] < 20:
                        curr.append(words[i])
                    else:
                        lines_in_frame.append(curr)
                        curr = [words[i]]
                lines_in_frame.append(curr)

            for ln_w in lines_in_frame:
                ln_w.sort(key=lambda w: w["x"])
                txt = normalize_ocr_line(" ".join([w["text"] for w in ln_w]))
                if not txt:
                    continue

                y_pos = int(sum(w["y"] for w in ln_w) / len(ln_w))
                # Create a key based on Y-bin and text content to track unique lines
                norm = f"y{y_pos // 30}_{normalize_text_basic(txt)}"
                current_norms.add(norm)

                if norm in on_screen:
                    on_screen[norm]["last"] = frame["time"]
                else:
                    on_screen[norm] = {
                        "text": txt,
                        "words": [w["text"] for w in ln_w],
                        "first": frame["time"],
                        "last": frame["time"],
                        "y": y_pos,
                        "w_rois": [(w["x"], w["y"], w["w"], w["h"]) for w in ln_w],
                    }

        # Commit lines that have disappeared
        for nt in list(on_screen.keys()):
            # If line not seen in current frame and hasn't been seen for > 1.0s
            if nt not in current_norms and frame["time"] - on_screen[nt]["last"] > 1.0:
                committed.append(on_screen.pop(nt))

    # Commit remaining lines
    for ent in on_screen.values():
        committed.append(ent)

    # Deduplicate
    unique: list[dict[str, Any]] = []
    for ent in committed:
        is_dup = False
        for u in unique:
            # Text similarity check
            if text_similarity(ent["text"], u["text"]) > 0.9:
                # Spatial and Temporal proximity check
                if abs(ent["y"] - u["y"]) < 20 and abs(ent["first"] - u["first"]) < 2.0:
                    is_dup = True
                    break
        if not is_dup:
            unique.append(ent)

    # Sort: Primary by time (2.0s bins), Secondary by Y (top-to-bottom)
    # This keeps multi-line blocks together
    unique.sort(key=lambda x: (round(float(x["first"]) / 2.0) * 2.0, x["y"]))

    out: list[TargetLine] = []
    for i, ent in enumerate(unique):
        s = _snap(float(ent["first"]))
        # Determine end time based on next line start or duration
        if i + 1 < len(unique):
            nxt_s = _snap(float(unique[i + 1]["first"]))
            # If next line starts soon (<3s), snap to it
            e = nxt_s if (nxt_s - s < 3.0) else _snap(float(ent["last"]) + 2.0)
        else:
            e = _snap(float(ent["last"]) + 2.0)

        out.append(
            TargetLine(
                line_index=i + 1,
                start=s,
                end=e,
                text=ent["text"],
                words=ent["words"],
                y=ent["y"],
                word_starts=None,
                word_ends=None,
                word_rois=ent["w_rois"],
                char_rois=None,
            )
        )
    return out
