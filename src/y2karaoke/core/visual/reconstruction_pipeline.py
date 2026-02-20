from __future__ import annotations

from typing import Any, Callable, Dict

from ..models import TargetLine
from ..text_utils import (
    normalize_ocr_line,
    normalize_ocr_tokens,
    normalize_text_basic,
    text_similarity,
)
from .word_segmentation import segment_line_tokens_by_visual_gaps

EntriesPass = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
FrameFilter = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
SnapFn = Callable[[float], float]


def reconstruct_lyrics_from_visuals(  # noqa: C901
    raw_frames: list[dict[str, Any]],
    *,
    filter_static_overlay_words: FrameFilter,
    merge_overlapping_same_lane_duplicates: EntriesPass,
    merge_short_same_lane_reentries: EntriesPass,
    expand_overlapped_same_text_repetitions: EntriesPass,
    extrapolate_mirrored_lane_cycles: EntriesPass,
    split_persistent_line_epochs_from_context_transitions: EntriesPass,
    suppress_short_duplicate_reentries: EntriesPass,
    collapse_short_refrain_noise: EntriesPass,
    filter_intro_non_lyrics: EntriesPass,
    suppress_bottom_fragment_families: EntriesPass,
    snap_fn: SnapFn,
) -> list[TargetLine]:
    raw_frames = filter_static_overlay_words(raw_frames)
    on_screen: Dict[str, Dict[str, Any]] = {}
    committed = []

    for frame in raw_frames:
        words = frame.get("words", [])
        current_norms = set()

        if words:
            words.sort(key=lambda w: w["y"])
            lines_in_frame = []

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
                line_tokens = segment_line_tokens_by_visual_gaps(ln_w)
                line_tokens = normalize_ocr_tokens(line_tokens)
                if not line_tokens:
                    continue
                txt = normalize_ocr_line(" ".join(line_tokens))
                if not txt:
                    continue

                y_pos = int(sum(w["y"] for w in ln_w) / len(ln_w))
                norm = f"y{y_pos // 30}_{normalize_text_basic(txt)}"
                current_norms.add(norm)

                if norm in on_screen:
                    on_screen[norm]["last"] = frame["time"]
                else:
                    lane = y_pos // 30
                    on_screen[norm] = {
                        "text": txt,
                        "words": line_tokens,
                        "first": frame["time"],
                        "last": frame["time"],
                        "y": y_pos,
                        "lane": lane,
                        "w_rois": [(w["x"], w["y"], w["w"], w["h"]) for w in ln_w],
                    }

        for nt in list(on_screen.keys()):
            if nt not in current_norms and frame["time"] - on_screen[nt]["last"] > 1.0:
                committed.append(on_screen.pop(nt))

    for ent in on_screen.values():
        committed.append(ent)

    unique: list[dict[str, Any]] = []
    for ent in committed:
        is_dup = False
        for u in unique:
            if text_similarity(ent["text"], u["text"]) > 0.9:
                if abs(ent["y"] - u["y"]) < 20 and abs(ent["first"] - u["first"]) < 2.0:
                    is_dup = True
                    break
        if not is_dup:
            unique.append(ent)

    unique.sort(key=lambda x: (round(float(x["first"]) / 2.0) * 2.0, x["y"]))
    unique = merge_overlapping_same_lane_duplicates(unique)
    unique = merge_short_same_lane_reentries(unique)
    unique = expand_overlapped_same_text_repetitions(unique)
    unique = extrapolate_mirrored_lane_cycles(unique)
    unique = split_persistent_line_epochs_from_context_transitions(unique)
    unique = suppress_short_duplicate_reentries(unique)
    unique = collapse_short_refrain_noise(unique)
    unique = filter_intro_non_lyrics(unique)
    unique = suppress_bottom_fragment_families(unique)

    out: list[TargetLine] = []
    for i, ent in enumerate(unique):
        s = snap_fn(float(ent["first"]))
        if i + 1 < len(unique):
            nxt_s = snap_fn(float(unique[i + 1]["first"]))
            e = nxt_s if (nxt_s - s < 3.0) else snap_fn(float(ent["last"]) + 2.0)
        else:
            e = snap_fn(float(ent["last"]) + 2.0)

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
                visibility_start=float(ent["first"]),
                visibility_end=float(ent["last"]),
            )
        )
    return out
