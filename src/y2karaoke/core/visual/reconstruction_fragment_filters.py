"""Fragment suppression helpers for visual reconstruction."""

from __future__ import annotations

from typing import Any


def suppress_short_lane_fragments(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Identify and remove transient OCR noise overshadowed by stable lines."""
    if not lines:
        return []

    by_lane: dict[Any, list[tuple[int, dict[str, Any]]]] = {}
    for idx, line in enumerate(lines):
        by_lane.setdefault(line.get("lane"), []).append((idx, line))
    for lane_items in by_lane.values():
        lane_items.sort(key=lambda item: float(item[1].get("first", 0.0)))

    suppressed_indices: set[int] = set()
    for i, line in enumerate(lines):
        dur = line["last"] - line["first"]
        wc = len(line["words"])

        if dur < 1.0 and wc < 4:
            lane_items = by_lane.get(line.get("lane"), [])
            line_first = float(line["first"])
            for j, other in lane_items:
                if i == j:
                    continue
                other_first = float(other["first"])
                if other_first < line_first - 3.0:
                    continue
                if other_first > line_first + 3.0:
                    break

                other_dur = other["last"] - other["first"]
                other_wc = len(other["words"])
                if other_dur > dur * 2 and other_wc >= wc + 2:
                    suppressed_indices.add(i)
                    break

    return [l for idx, l in enumerate(lines) if idx not in suppressed_indices]
