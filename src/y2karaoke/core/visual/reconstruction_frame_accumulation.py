"""Frame-by-frame OCR word accumulation and line persistence tracking."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from ..text_utils import (
    normalize_ocr_line,
    normalize_ocr_tokens,
    normalize_text_basic,
)
from .word_segmentation import segment_line_tokens_by_visual_gaps

FrameFilter = Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]


def accumulate_persistent_lines_from_frames(
    raw_frames: List[Dict[str, Any]],
    *,
    filter_static_overlay_words: FrameFilter,
) -> List[Dict[str, Any]]:
    """
    Groups OCR words into lines per frame and tracks them over time to form persistent line objects.
    """
    raw_frames = filter_static_overlay_words(raw_frames)
    on_screen: Dict[str, Dict[str, Any]] = {}
    committed: List[Dict[str, Any]] = []
    prev_time = None

    for frame in raw_frames:
        curr_time = frame["time"]
        words = frame.get("words", [])
        current_norms = {}

        if words:
            words.sort(key=lambda w: w["y"])
            lines_in_frame = []

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
                line_tokens = segment_line_tokens_by_visual_gaps(ln_w)
                line_tokens = normalize_ocr_tokens(line_tokens)
                if not line_tokens:
                    continue
                txt = normalize_ocr_line(" ".join(line_tokens))
                if not txt:
                    continue

                y_pos = int(sum(w["y"] for w in ln_w) / len(ln_w))
                lane = y_pos // 30
                norm = f"y{lane}_{normalize_text_basic(txt)}"

                if norm in on_screen and (
                    prev_time is not None and on_screen[norm]["last"] == prev_time
                ):
                    on_screen[norm]["last"] = curr_time
                    current_norms[norm] = on_screen[norm]
                else:
                    entry = {
                        "text": txt,
                        "words": line_tokens,
                        "first": curr_time,
                        "last": curr_time,
                        "y": y_pos,
                        "lane": lane,
                        "w_rois": [(w["x"], w["y"], w["w"], w["h"]) for w in ln_w],
                    }
                    if norm in on_screen:
                        committed.append(on_screen.pop(norm))
                    on_screen[norm] = entry
                    current_norms[norm] = entry

        for nt in list(on_screen.keys()):
            if nt not in current_norms and curr_time - on_screen[nt]["last"] > 1.0:
                committed.append(on_screen.pop(nt))

        prev_time = curr_time

    for ent in on_screen.values():
        committed.append(ent)

    return committed
