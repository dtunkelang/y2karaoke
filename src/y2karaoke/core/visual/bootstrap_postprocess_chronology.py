"""Chronology repair passes for bootstrap postprocess outputs."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Callable

from ..text_utils import normalize_text_basic


def _block_id(line: dict[str, Any]) -> object:
    meta = line.get("_reconstruction_meta", {})
    if not isinstance(meta, dict):
        return None
    bf = meta.get("block_first")
    if not isinstance(bf, dict):
        return None
    return bf.get("block_id")


def _tokens(line: dict[str, Any]) -> list[str]:
    return [t for t in normalize_text_basic(str(line.get("text", ""))).split() if t]


def _is_comparable_short_pair(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    ratio: float,
    ta: list[str],
    tb: list[str],
) -> bool:
    comparable = ratio >= 0.82 or ta == tb
    short_fragment_case = (
        len(ta) <= 3
        and len(tb) <= 4
        and (
            float(a.get("confidence", 0.0) or 0.0) < 0.45
            or float(b.get("confidence", 0.0) or 0.0) < 0.45
        )
    )
    return comparable or short_fragment_case


def _large_inversion_swap_allowed(
    a: dict[str, Any], b: dict[str, Any], *, sa: float, sb: float
) -> bool:
    inversion = sa - sb
    if inversion < 0.5:
        return False
    ea = float(a.get("end", sa) or sa)
    eb = float(b.get("end", sb) or sb)
    overlap = max(0.0, min(ea, eb) - max(sa, sb))
    dur_a = max(0.1, ea - sa)
    dur_b = max(0.1, eb - sb)
    conf_b = float(b.get("confidence", 0.0) or 0.0)

    strong_no_overlap = overlap <= 0.05 and eb <= sa - 0.2
    short_fragment_inversion = (
        overlap <= 0.3
        and dur_b <= 1.4
        and (conf_b <= 0.35 or dur_a >= dur_b * 2.0)
        and sb + 0.4 < sa
    )
    if overlap > 0.8 and not (strong_no_overlap or short_fragment_inversion):
        return False
    if inversion < 0.9 and not short_fragment_inversion:
        return False
    return strong_no_overlap or short_fragment_inversion


def _reindex(lines_out: list[dict[str, Any]]) -> None:
    for i, ln in enumerate(lines_out):
        ln["line_index"] = i + 1


def repair_strong_local_chronology_inversions(
    lines_out: list[dict[str, Any]],
    *,
    line_duplicate_quality_score_fn: Callable[[dict[str, Any]], float],
) -> None:
    if len(lines_out) < 2:
        return

    max_passes = min(4, len(lines_out))
    for _ in range(max_passes):
        changed = False
        for i in range(len(lines_out) - 1):
            a = lines_out[i]
            b = lines_out[i + 1]
            if _block_id(a) is not None and _block_id(a) == _block_id(b):
                continue
            sa = float(a.get("start", 0.0) or 0.0)
            sb = float(b.get("start", 0.0) or 0.0)
            if sa <= sb + 0.75:
                continue

            ta = _tokens(a)
            tb = _tokens(b)
            if not ta or not tb:
                continue
            ratio = SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()
            if not _is_comparable_short_pair(a, b, ratio=ratio, ta=ta, tb=tb):
                continue

            qa = line_duplicate_quality_score_fn(a)
            qb = line_duplicate_quality_score_fn(b)
            if abs(qa - qb) < 0.05 and ratio < 0.9:
                continue

            lines_out[i], lines_out[i + 1] = lines_out[i + 1], lines_out[i]
            changed = True
        if not changed:
            break

    _reindex(lines_out)


def repair_large_adjacent_time_inversions(lines_out: list[dict[str, Any]]) -> None:
    if len(lines_out) < 2:
        return

    max_passes = min(6, len(lines_out))
    for _ in range(max_passes):
        changed = False
        for i in range(len(lines_out) - 1):
            a = lines_out[i]
            b = lines_out[i + 1]
            if _block_id(a) is not None and _block_id(a) == _block_id(b):
                continue
            sa = float(a.get("start", 0.0) or 0.0)
            sb = float(b.get("start", 0.0) or 0.0)
            if not _large_inversion_swap_allowed(a, b, sa=sa, sb=sb):
                continue
            lines_out[i], lines_out[i + 1] = lines_out[i + 1], lines_out[i]
            changed = True
        if not changed:
            break
    _reindex(lines_out)
