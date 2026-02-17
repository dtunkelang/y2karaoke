from __future__ import annotations

from typing import Any, Dict

from ..models import TargetLine
from ..text_utils import normalize_ocr_line, normalize_text_basic, text_similarity


def snap(value: float) -> float:
    # Assuming 0.05s snap from original tool
    return round(round(float(value) / 0.05) * 0.05, 3)


def _filter_static_overlay_words(
    raw_frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not raw_frames:
        return raw_frames

    total_frames = len(raw_frames)
    if total_frames < 20:
        return raw_frames
    all_y, stats = _collect_overlay_stats(raw_frames)

    if not all_y:
        return raw_frames
    y_min = min(all_y)
    y_max = max(all_y)
    if (y_max - y_min) < 60.0:
        return raw_frames
    y_top_cut = y_min + 0.35 * (y_max - y_min)

    static_keys = _identify_static_overlay_keys(stats, total_frames, y_top_cut)

    if not static_keys:
        return raw_frames

    out: list[dict[str, Any]] = []
    for frame in raw_frames:
        new_words = []
        for w in frame.get("words", []):
            tok = normalize_text_basic(str(w.get("text", ""))).strip()
            key = (
                tok,
                int(round(float(w.get("x", 0.0)) / 16.0)),
                int(round(float(w.get("y", 0.0)) / 16.0)),
            )
            if key in static_keys:
                continue
            new_words.append(w)
        out.append({**frame, "words": new_words})
    return out


def _collect_overlay_stats(
    raw_frames: list[dict[str, Any]],
) -> tuple[list[float], dict[tuple[str, int, int], dict[str, float]]]:
    all_y: list[float] = []
    stats: dict[tuple[str, int, int], dict[str, float]] = {}
    for frame in raw_frames:
        seen: set[tuple[str, int, int]] = set()
        for w in frame.get("words", []):
            try:
                x = float(w["x"])
                y = float(w["y"])
            except Exception:
                continue
            all_y.append(y)
            tok = normalize_text_basic(str(w.get("text", ""))).strip()
            if len(tok) < 4:
                continue
            key = (tok, int(round(x / 16.0)), int(round(y / 16.0)))
            if key in seen:
                continue
            seen.add(key)
            rec = stats.setdefault(
                key,
                {
                    "count": 0.0,
                    "sum_x": 0.0,
                    "sum_y": 0.0,
                    "sum_x2": 0.0,
                    "sum_y2": 0.0,
                },
            )
            rec["count"] += 1.0
            rec["sum_x"] += x
            rec["sum_y"] += y
            rec["sum_x2"] += x * x
            rec["sum_y2"] += y * y
    return all_y, stats


def _identify_static_overlay_keys(
    stats: dict[tuple[str, int, int], dict[str, float]],
    total_frames: int,
    y_top_cut: float,
) -> set[tuple[str, int, int]]:
    static_keys: set[tuple[str, int, int]] = set()
    for key, rec in stats.items():
        n = max(rec["count"], 1.0)
        freq = rec["count"] / max(float(total_frames), 1.0)
        mean_x = rec["sum_x"] / n
        mean_y = rec["sum_y"] / n
        var_x = max(rec["sum_x2"] / n - mean_x * mean_x, 0.0)
        var_y = max(rec["sum_y2"] / n - mean_y * mean_y, 0.0)
        if (
            freq >= 0.45
            and (var_x**0.5) <= 8.0
            and (var_y**0.5) <= 8.0
            and mean_y <= y_top_cut
        ):
            static_keys.add(key)
    return static_keys


def reconstruct_lyrics_from_visuals(  # noqa: C901
    raw_frames: list[dict[str, Any]], visual_fps: float
) -> list[TargetLine]:
    """Group raw OCR words into logical lines and assign timing."""
    raw_frames = _filter_static_overlay_words(raw_frames)
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
        s = snap(float(ent["first"]))
        # Determine end time based on next line start or duration
        if i + 1 < len(unique):
            nxt_s = snap(float(unique[i + 1]["first"]))
            # If next line starts soon (<3s), snap to it
            e = nxt_s if (nxt_s - s < 3.0) else snap(float(ent["last"]) + 2.0)
        else:
            e = snap(float(ent["last"]) + 2.0)

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
