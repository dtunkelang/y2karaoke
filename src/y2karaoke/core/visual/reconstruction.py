from __future__ import annotations

from typing import Any, Dict

from ..models import TargetLine
from ..text_utils import (
    normalize_ocr_line,
    normalize_ocr_tokens,
    normalize_text_basic,
    text_similarity,
)
from .reconstruction_mirrored_cycles import (
    extrapolate_mirrored_lane_cycles as _extrapolate_mirrored_lane_cycles_impl,
)
from .reconstruction_mirrored_cycles import (
    is_candidate_for_mirrored_cycle as _is_candidate_for_mirrored_cycle_impl,
)
from .reconstruction_mirrored_cycles import (
    mirrored_cycle_candidate as _mirrored_cycle_candidate_impl,
)
from .reconstruction_intro_filters import (
    filter_intro_non_lyrics as _filter_intro_non_lyrics_impl,
)
from .reconstruction_intro_filters import (
    is_intro_artifact as _is_intro_artifact_impl,
)
from .reconstruction_intro_filters import (
    suppress_bottom_fragment_families as _suppress_bottom_fragment_families_impl,
)
from .reconstruction_context_transitions import (
    split_persistent_line_epochs_from_context_transitions as _split_persistent_line_epochs_from_context_transitions_impl,
)
from .reconstruction_lane_merge import (
    is_same_lane as _is_same_lane_impl,
)
from .reconstruction_lane_merge import (
    merge_overlapping_same_lane_duplicates as _merge_overlapping_same_lane_duplicates_impl,
)
from .reconstruction_overlap_repetitions import (
    expand_overlapped_same_text_repetitions as _expand_overlapped_same_text_repetitions_impl,
)
from .reconstruction_reentry import (
    merge_short_same_lane_reentries as _merge_short_same_lane_reentries_impl,
)
from .reconstruction_reentry import (
    suppress_short_duplicate_reentries as _suppress_short_duplicate_reentries_impl,
)
from .reconstruction_refrain import (
    collapse_short_refrain_noise as _collapse_short_refrain_noise_impl,
)
from .reconstruction_refrain import (
    is_short_refrain_entry as _is_short_refrain_entry_impl,
)
from .reconstruction_overlay import _filter_static_overlay_words
from .word_segmentation import segment_line_tokens_by_visual_gaps

_LANE_PROXIMITY_PX = 18.0


def snap(value: float) -> float:
    # Assuming 0.05s snap from original tool
    return round(round(float(value) / 0.05) * 0.05, 3)


def _suppress_short_duplicate_reentries(  # noqa: C901
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _suppress_short_duplicate_reentries_impl(
        entries,
        is_same_lane=_is_same_lane,
    )


def _merge_short_same_lane_reentries(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _merge_short_same_lane_reentries_impl(
        entries,
        is_same_lane=_is_same_lane,
        is_short_refrain_entry=_is_short_refrain_entry,
    )


def _is_same_lane(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return _is_same_lane_impl(a, b, lane_proximity_px=_LANE_PROXIMITY_PX)


def _merge_overlapping_same_lane_duplicates(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _merge_overlapping_same_lane_duplicates_impl(
        entries,
        is_short_refrain_entry=_is_short_refrain_entry,
        is_same_lane=_is_same_lane,
    )


def _is_short_refrain_entry(entry: dict[str, Any]) -> bool:
    return _is_short_refrain_entry_impl(entry)


def _collapse_short_refrain_noise(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _collapse_short_refrain_noise_impl(
        entries,
        is_short_refrain_entry=_is_short_refrain_entry,
        is_same_lane=_is_same_lane,
    )


def _is_intro_artifact(entry: dict[str, Any]) -> bool:
    return _is_intro_artifact_impl(entry)


def _filter_intro_non_lyrics(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _filter_intro_non_lyrics_impl(entries)


def _suppress_bottom_fragment_families(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _suppress_bottom_fragment_families_impl(entries)


def _expand_overlapped_same_text_repetitions(  # noqa: C901
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _expand_overlapped_same_text_repetitions_impl(
        entries,
        lane_proximity_px=_LANE_PROXIMITY_PX,
        is_short_refrain_entry=_is_short_refrain_entry,
        is_same_lane=_is_same_lane,
    )


def _extrapolate_mirrored_lane_cycles(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _extrapolate_mirrored_lane_cycles_impl(
        entries,
        lane_proximity_px=_LANE_PROXIMITY_PX,
        is_candidate_for_mirrored_cycle=_is_candidate_for_mirrored_cycle,
        mirrored_cycle_candidate=_mirrored_cycle_candidate,
    )


def _is_candidate_for_mirrored_cycle(entry: dict[str, Any]) -> bool:
    return _is_candidate_for_mirrored_cycle_impl(
        entry,
        is_short_refrain_entry=_is_short_refrain_entry,
    )


def _mirrored_cycle_candidate(
    a: dict[str, Any], b: dict[str, Any]
) -> tuple[float, dict[str, Any]] | None:
    return _mirrored_cycle_candidate_impl(
        a,
        b,
        is_candidate_for_mirrored_cycle=_is_candidate_for_mirrored_cycle,
        is_same_lane=_is_same_lane,
    )


def _split_persistent_line_epochs_from_context_transitions(  # noqa: C901
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _split_persistent_line_epochs_from_context_transitions_impl(
        entries,
        lane_proximity_px=_LANE_PROXIMITY_PX,
        is_short_refrain_entry=_is_short_refrain_entry,
    )


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
                line_tokens = segment_line_tokens_by_visual_gaps(ln_w)
                line_tokens = normalize_ocr_tokens(line_tokens)
                if not line_tokens:
                    continue
                txt = normalize_ocr_line(" ".join(line_tokens))
                if not txt:
                    continue

                y_pos = int(sum(w["y"] for w in ln_w) / len(ln_w))
                # Create a key based on Y-bin and text content to track unique lines
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
    unique = _merge_overlapping_same_lane_duplicates(unique)
    unique = _merge_short_same_lane_reentries(unique)
    unique = _expand_overlapped_same_text_repetitions(unique)
    unique = _extrapolate_mirrored_lane_cycles(unique)
    unique = _split_persistent_line_epochs_from_context_transitions(unique)
    unique = _suppress_short_duplicate_reentries(unique)
    unique = _collapse_short_refrain_noise(unique)
    unique = _filter_intro_non_lyrics(unique)
    unique = _suppress_bottom_fragment_families(unique)

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
                visibility_start=float(ent["first"]),
                visibility_end=float(ent["last"]),
            )
        )
    return out
