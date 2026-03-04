"""Fragment-focused refinement passes extracted from refinement_postpasses."""

from __future__ import annotations

from typing import List, Optional

from ..models import TargetLine
from ..text_utils import normalize_text_basic


def token_list(ln: TargetLine) -> list[str]:
    return [t for t in normalize_text_basic(" ".join(ln.words)).split() if t]


def is_fragment_of_line(a: TargetLine, b: TargetLine) -> bool:
    ta = token_list(a)
    tb = token_list(b)
    if not ta or not tb:
        return False
    if len(ta) >= len(tb):
        return False
    if len(ta) > 3:
        return False
    if any(len(t) <= 1 for t in ta):
        return False
    return ta == tb[: len(ta)] or ta == tb[-len(ta) :]


def _line_start(ln: TargetLine) -> Optional[float]:
    if ln.word_starts:
        ws0 = ln.word_starts[0]
        if ws0 is not None:
            return float(ws0)
    return float(ln.start) if ln.start is not None else 0.0


def _collect_clean_visibility_block(
    target_lines: List[TargetLine], i: int
) -> tuple[int, int, list[TargetLine]]:
    ln0 = target_lines[i]
    if ln0.visibility_start is None or ln0.visibility_end is None:
        return i, i + 1, []
    vs0 = float(ln0.visibility_start)
    ve0 = float(ln0.visibility_end)
    block = [target_lines[i]]
    j = i + 1
    while j < len(target_lines):
        ln = target_lines[j]
        if ln.visibility_start is None or ln.visibility_end is None:
            break
        vs = float(ln.visibility_start)
        ve = float(ln.visibility_end)
        overlap = min(ve0, ve) - max(vs0, vs)
        if abs(vs - vs0) > 0.45 or abs(ve - ve0) > 3.0 or overlap < 0.75:
            break
        block.append(ln)
        j += 1
    return i, j, block


def demote_fragment_lines_within_clean_blocks(target_lines: List[TargetLine]) -> None:
    if len(target_lines) < 2:
        return

    i = 0
    while i < len(target_lines):
        i0, j, block = _collect_clean_visibility_block(target_lines, i)
        if not block:
            i += 1
            continue

        if 2 <= len(block) <= 5:
            by_y = sorted(block, key=lambda ln: float(ln.y))
            fragment_flags: dict[int, bool] = {}
            for ln in by_y:
                fragment_flags[id(ln)] = any(
                    other is not ln and is_fragment_of_line(ln, other) for other in by_y
                )
            if any(fragment_flags.values()):
                reordered = sorted(
                    block,
                    key=lambda ln: (
                        1 if fragment_flags.get(id(ln), False) else 0,
                        float(ln.y),
                        _line_start(ln) or float(ln.start),
                    ),
                )
                if reordered != block:
                    target_lines[i0:j] = reordered
                    for k, ln in enumerate(target_lines):
                        ln.line_index = k + 1
                        ln.block_order_hint = k
        i = max(i + 1, j)


def merge_prefix_fragment_rows_in_clean_blocks(  # noqa: C901
    target_lines: List[TargetLine],
) -> None:
    if len(target_lines) < 2:
        return

    i = 0
    while i < len(target_lines):
        _i0, j, block = _collect_clean_visibility_block(target_lines, i)
        if not block:
            i += 1
            continue

        if 2 <= len(block) <= 5:
            by_y = sorted(block, key=lambda ln: float(ln.y))
            for upper, lower in zip(by_y, by_y[1:]):
                upper_tokens = token_list(upper)
                lower_tokens = token_list(lower)
                if len(lower_tokens) != 2:
                    continue
                if len(upper_tokens) < 3 or len(upper_tokens) > 6:
                    continue
                if upper_tokens[-2:] != lower_tokens:
                    continue
                if (
                    upper.visibility_end is not None
                    and upper.visibility_start is not None
                ):
                    if (
                        float(upper.visibility_end) - float(upper.visibility_start)
                    ) > 1.0:
                        continue
                merged_text = " ".join(upper.words + lower.words)
                found_support = False
                merged_norm = normalize_text_basic(merged_text)
                for other in target_lines:
                    if other is upper or other is lower:
                        continue
                    other_norm = normalize_text_basic(" ".join(other.words))
                    if not other_norm:
                        continue
                    if (
                        merged_norm == other_norm
                        or merged_norm in other_norm
                        or other_norm in merged_norm
                    ):
                        if len(other_norm.split()) >= len(merged_norm.split()):
                            found_support = True
                            break
                if not found_support:
                    continue

                upper.words = upper.words + lower.words
                if upper.word_rois and lower.word_rois:
                    upper.word_rois = list(upper.word_rois) + list(lower.word_rois)
                if upper.word_starts and lower.word_starts:
                    upper.word_starts = list(upper.word_starts) + list(
                        lower.word_starts
                    )
                if upper.word_ends and lower.word_ends:
                    upper.word_ends = list(upper.word_ends) + list(lower.word_ends)
                if upper.word_confidences and lower.word_confidences:
                    upper.word_confidences = list(upper.word_confidences) + list(
                        lower.word_confidences
                    )
                upper.text = " ".join(upper.words)

                lower.words = []
                lower.text = ""
                lower.word_starts = []
                lower.word_ends = []
                lower.word_confidences = []
                lower.word_rois = []
                lower.end = lower.start
                break

        i = max(i + 1, j)
