"""Fragment suppression helpers for visual reconstruction."""

from __future__ import annotations

from typing import Any, Callable

from ..text_utils import normalize_text_basic


def _group_lane_items(
    lines: list[dict[str, Any]],
) -> dict[Any, list[tuple[int, dict[str, Any]]]]:
    by_lane: dict[Any, list[tuple[int, dict[str, Any]]]] = {}
    for idx, line in enumerate(lines):
        by_lane.setdefault(line.get("lane"), []).append((idx, line))
    for lane_items in by_lane.values():
        lane_items.sort(key=lambda item: float(item[1].get("first", 0.0)))
    return by_lane


def _is_transient_short_fragment(line: dict[str, Any]) -> bool:
    dur = float(line["last"]) - float(line["first"])
    wc = len(line["words"])
    return dur < 1.0 and wc < 4


def _has_stronger_lane_neighbor(
    line: dict[str, Any],
    *,
    lane_items: list[tuple[int, dict[str, Any]]],
    line_idx: int,
) -> bool:
    dur = float(line["last"]) - float(line["first"])
    wc = len(line["words"])
    line_first = float(line["first"])
    for j, other in lane_items:
        if line_idx == j:
            continue
        other_first = float(other["first"])
        if other_first < line_first - 3.0:
            continue
        if other_first > line_first + 3.0:
            break
        other_dur = float(other["last"]) - float(other["first"])
        other_wc = len(other["words"])
        if other_dur > dur * 2 and other_wc >= wc + 2:
            return True
    return False


def suppress_short_lane_fragments(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Identify and remove transient OCR noise overshadowed by stable lines."""
    if not lines:
        return []

    by_lane = _group_lane_items(lines)

    suppressed_indices: set[int] = set()
    for i, line in enumerate(lines):
        if not _is_transient_short_fragment(line):
            continue
        lane_items = by_lane.get(line.get("lane"), [])
        if _has_stronger_lane_neighbor(line, lane_items=lane_items, line_idx=i):
            suppressed_indices.add(i)

    return [l for idx, l in enumerate(lines) if idx not in suppressed_indices]


def suppress_repeated_short_fragment_clusters(  # noqa: C901
    lines: list[dict[str, Any]],
    *,
    is_band_fragment_subphrase_fn: Callable[[list[str], list[str]], bool],
) -> list[dict[str, Any]]:
    """Remove repeated short fragment shards backed by nearby fuller lyric lines."""
    if len(lines) < 4:
        return lines

    token_lists = [
        [t for t in normalize_text_basic(str(line.get("text", ""))).split() if t]
        for line in lines
    ]
    groups: dict[tuple[str, ...], list[int]] = {}
    for idx, (line, toks) in enumerate(zip(lines, token_lists)):
        if not (1 <= len(toks) <= 3):
            continue
        dur = float(line["last"]) - float(line["first"])
        if dur > 1.6:
            continue
        groups.setdefault(tuple(toks), []).append(idx)

    suppressed: set[int] = set()
    for key, idxs in groups.items():
        if len(idxs) < 2:
            continue
        avg_dur = sum(
            max(0.0, float(lines[i]["last"]) - float(lines[i]["first"])) for i in idxs
        ) / max(len(idxs), 1)
        min_count = 3
        if len(key) <= 2 and avg_dur <= 1.0:
            min_count = 2
        if len(idxs) < min_count:
            continue

        supported: list[int] = []
        for idx in idxs:
            line = lines[idx]
            start = float(line["first"])
            end = float(line["last"])
            for j, other in enumerate(lines):
                if j == idx:
                    continue
                other_toks = token_lists[j]
                if len(other_toks) <= len(key):
                    continue
                other_start = float(other["first"])
                other_end = float(other["last"])
                if other_end < start - 3.0 or other_start > end + 3.0:
                    continue
                if is_band_fragment_subphrase_fn(list(key), other_toks):
                    supported.append(idx)
                    break

        if len(supported) < min_count:
            continue
        suppressed.update(supported)

    if not suppressed:
        return lines
    return [line for idx, line in enumerate(lines) if idx not in suppressed]


def suppress_never_visible_ghost_reentries(
    lines: list[dict[str, Any]],
    *,
    is_same_lane_fn: Callable[[dict[str, Any], dict[str, Any]], bool],
    text_similarity_fn: Callable[[str, str], float],
) -> list[dict[str, Any]]:
    """Drop later never-visible reentries that mirror an earlier visible line."""
    if len(lines) < 2:
        return lines

    kept: list[dict[str, Any]] = []
    for ent in lines:
        if bool(ent.get("visible_yet", False)):
            kept.append(ent)
            continue

        suppress = False
        ent_first = float(ent.get("first", 0.0))
        ent_last = float(ent.get("last", ent_first))
        ent_dur = max(0.0, ent_last - ent_first)
        if ent_dur >= 1.0:
            suppress = _has_matching_visible_ancestor(
                ent,
                kept=kept,
                ent_first=ent_first,
                is_same_lane_fn=is_same_lane_fn,
                text_similarity_fn=text_similarity_fn,
            )

        if not suppress:
            kept.append(ent)

    return kept


def _has_matching_visible_ancestor(
    ent: dict[str, Any],
    *,
    kept: list[dict[str, Any]],
    ent_first: float,
    is_same_lane_fn: Callable[[dict[str, Any], dict[str, Any]], bool],
    text_similarity_fn: Callable[[str, str], float],
) -> bool:
    for prev in reversed(kept[-16:]):
        if not bool(prev.get("visible_yet", False)):
            continue
        if not is_same_lane_fn(prev, ent):
            continue
        if (
            text_similarity_fn(str(prev.get("text", "")), str(ent.get("text", "")))
            < 0.9
        ):
            continue
        prev_last = float(prev.get("last", prev.get("first", 0.0)))
        if ent_first >= prev_last + 0.8:
            return True
    return False
