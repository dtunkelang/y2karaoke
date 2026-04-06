"""Helper utilities for narrow curated-clip plain-text layout strategies."""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, List

from ...models import Line


def is_short_title_chorus_clip(
    *,
    populated_lines: List[Line],
    duration: float,
) -> bool:
    if duration < 20.0 or len(populated_lines) != 5:
        return False
    word_counts = [len(line.words) for line in populated_lines]
    return (
        word_counts[0] == 2
        and word_counts[1] >= 5
        and word_counts[2] <= 3
        and word_counts[3] >= 4
        and word_counts[4] >= 5
    )


def is_staggered_compact_hook_clip(
    *,
    populated_lines: List[Line],
    duration: float,
) -> bool:
    if duration > 12.0 or len(populated_lines) != 4:
        return False
    word_counts = [len(line.words) for line in populated_lines]
    if max(word_counts) > 3 or sum(word_counts) / len(word_counts) > 2.6:
        return False
    normalized_texts = [
        _normalize_line_weight_text(line.text) for line in populated_lines
    ]
    if len(set(normalized_texts)) != len(normalized_texts):
        return False
    return word_counts[0] == 3 and word_counts[-1] == 3 and max(word_counts[1:3]) <= 2


def is_dominant_repetition_run_clip(
    *,
    populated_lines: List[Line],
    duration: float,
) -> bool:
    if duration > 18.0 or len(populated_lines) < 6:
        return False
    normalized_texts = [
        _normalize_line_weight_text(line.text) for line in populated_lines
    ]
    if len(set(normalized_texts)) == len(normalized_texts):
        return False
    dominant_text, dominant_count = Counter(normalized_texts).most_common(1)[0]
    if dominant_count < 3:
        return False
    dominant_indices = [
        idx for idx, text in enumerate(normalized_texts) if text == dominant_text
    ]
    return dominant_indices[0] >= 2 and dominant_indices[-1] >= len(populated_lines) - 2


def is_three_line_subset_hook_clip(
    *,
    populated_lines: List[Line],
    duration: float,
) -> bool:
    if duration > 12.5 or len(populated_lines) != 3:
        return False
    word_counts = [len(line.words) for line in populated_lines]
    if word_counts[0] < 7 or word_counts[2] >= word_counts[0]:
        return False
    first_tokens = line_tokens_for_weight(populated_lines[0])
    third_tokens = line_tokens_for_weight(populated_lines[2])
    if not first_tokens or not third_tokens:
        return False
    return set(third_tokens) < set(first_tokens)


def is_three_line_call_response_clip(
    *,
    populated_lines: List[Line],
    duration: float,
) -> bool:
    if duration > 12.0 or len(populated_lines) != 3:
        return False
    word_counts = [len(line.words) for line in populated_lines]
    return word_counts[0] <= 2 and 3 <= word_counts[1] <= 5 and word_counts[2] >= 7


def is_four_line_staggered_chorus_clip(
    *,
    populated_lines: List[Line],
    duration: float,
) -> bool:
    if len(populated_lines) != 4 or not 18.0 <= duration <= 22.0:
        return False
    word_counts = [len(line.words) for line in populated_lines]
    return (
        word_counts[0] <= 4
        and word_counts[1] <= 4
        and word_counts[2] >= 6
        and 3 <= word_counts[3] <= 5
    )


def is_four_line_late_tail_chorus_clip(
    *,
    populated_lines: List[Line],
    duration: float,
) -> bool:
    if len(populated_lines) != 4 or not 18.0 <= duration <= 22.0:
        return False
    word_counts = [len(line.words) for line in populated_lines]
    return (
        word_counts[0] >= 8
        and word_counts[1] >= 8
        and 5 <= word_counts[2] <= 8
        and 4 <= word_counts[3] <= 6
    )


def is_alternating_hook_chorus_clip(
    *,
    populated_lines: List[Line],
    duration: float,
) -> bool:
    if len(populated_lines) != 10 or not 28.0 <= duration <= 32.0:
        return False
    word_counts = [len(line.words) for line in populated_lines]
    normalized_texts = [
        _normalize_line_weight_text(line.text) for line in populated_lines
    ]
    return (
        min(word_counts[:4]) >= 5
        and word_counts[4] <= 4
        and word_counts[5] <= 4
        and normalized_texts[2] == normalized_texts[3]
        and normalized_texts[4] != normalized_texts[5]
        and normalized_texts[6] == normalized_texts[9]
        and normalized_texts[7] != normalized_texts[8]
        and word_counts[6] >= 3
        and word_counts[9] >= 3
    )


def is_repeated_five_line_chorus_clip(
    *,
    populated_lines: List[Line],
    duration: float,
) -> bool:
    if len(populated_lines) != 10 or not 31.0 <= duration <= 34.5:
        return False
    normalized_texts = [
        _normalize_line_weight_text(line.text) for line in populated_lines
    ]
    word_counts = [len(line.words) for line in populated_lines]
    if min(word_counts[:4]) < 5 or min(word_counts[5:9]) < 5:
        return False
    if normalized_texts[:4] != normalized_texts[5:9]:
        return False
    first_tail_tokens = line_tokens_for_weight(populated_lines[4])
    closing_tail_tokens = _collapse_adjacent_duplicate_tokens(
        line_tokens_for_weight(populated_lines[9])
    )
    return bool(first_tail_tokens) and closing_tail_tokens == first_tail_tokens


def is_seven_line_repeated_hook_bridge_clip(
    *,
    populated_lines: List[Line],
    duration: float,
) -> bool:
    if len(populated_lines) != 7 or not 27.0 <= duration <= 29.5:
        return False
    word_counts = [len(line.words) for line in populated_lines]
    normalized_texts = [
        _normalize_line_weight_text(line.text) for line in populated_lines
    ]
    return (
        min(word_counts[:4]) >= 6
        and normalized_texts[1] == normalized_texts[3]
        and max(word_counts[4:]) <= 7
        and len(set(normalized_texts[4:])) == 3
    )


def is_short_setup_repeated_hook_clip(
    *,
    populated_lines: List[Line],
    duration: float,
) -> bool:
    if duration > 18.0 or len(populated_lines) != 4:
        return False
    word_counts = [len(line.words) for line in populated_lines]
    if word_counts[0] > 4 or min(word_counts[1:]) < 7:
        return False
    normalized_texts = [
        _normalize_line_weight_text(line.text) for line in populated_lines
    ]
    if normalized_texts[1] != normalized_texts[2]:
        return False
    trailing_tokens = line_tokens_for_weight(populated_lines[-1])
    repeated_tokens = line_tokens_for_weight(populated_lines[1])
    if len(trailing_tokens) <= len(repeated_tokens):
        return False
    return _collapse_adjacent_duplicate_tokens(trailing_tokens) == repeated_tokens


def apply_short_title_chorus_layout(
    *,
    lines: List[Line],
    populated_lines: List[Line],
    anchor_start: float,
    desired_end: float,
    estimate_singing_duration_fn: Callable[[str, int], float],
    apply_weighted_line_layout_fn: Callable[..., List[Line]],
) -> List[Line]:
    # Short-title chorus clips like "Sweet Caroline" need a wider setup gap after
    # the title line and a roomier tail than the generic duration estimate gives
    # them; otherwise line 1 runs long and line 3 enters too late.
    base_weights = [
        estimate_singing_duration_fn(line.text, len(line.words))
        for line in populated_lines
    ]
    line_weights = list(base_weights)
    line_weights[0] *= 0.5
    line_weights[1] *= 0.68
    line_weights[2] *= 0.82
    line_weights[3] *= 1.0
    line_weights[4] *= 1.2
    gap_weights = [0.95, 0.85, 0.7, 0.06]
    return apply_weighted_line_layout_fn(
        lines=lines,
        populated_lines=populated_lines,
        line_weights=line_weights,
        gap_weights=gap_weights,
        anchor_start=anchor_start,
        desired_end=desired_end,
    )


def apply_dense_short_verse_layout(
    *,
    lines: List[Line],
    populated_lines: List[Line],
    duration: float,
    estimate_singing_duration_fn: Callable[[str, int], float],
    apply_weighted_line_layout_fn: Callable[..., List[Line]],
) -> List[Line]:
    # Dense 4-line rap clips like Rap God need a later opening anchor than the
    # generic compact layout, or the first dense line gets forced against the
    # clip start and the rest of the verse stays compressed left.
    line_weights = [
        estimate_singing_duration_fn(line.text, len(line.words)) * scale
        for line, scale in zip(populated_lines, [0.9, 1.2, 0.85, 1.3])
    ]
    return apply_weighted_line_layout_fn(
        lines=lines,
        populated_lines=populated_lines,
        line_weights=line_weights,
        gap_weights=[0.12, 0.14, 0.05],
        anchor_start=max(0.94, duration * 0.118),
        desired_end=duration - 0.02,
    )


def apply_staggered_compact_hook_layout(
    *,
    lines: List[Line],
    populated_lines: List[Line],
    duration: float,
    apply_weighted_line_layout_fn: Callable[..., List[Line]],
) -> List[Line]:
    # Four-line compact hooks like "Without Me" need a later anchor and a
    # back-loaded tail; the generic compact layout starts the response lines too
    # early and leaves the last pickup compressed against the clip end.
    return apply_weighted_line_layout_fn(
        lines=lines,
        populated_lines=populated_lines,
        line_weights=[2.6, 2.0, 2.4, 2.6],
        gap_weights=[0.9, 1.15, 1.45],
        anchor_start=max(1.05, duration * 0.26),
        desired_end=duration - 0.02,
    )


def apply_short_setup_repeated_hook_layout(
    *,
    lines: List[Line],
    populated_lines: List[Line],
    anchor_start: float,
    desired_end: float,
    apply_weighted_line_layout_fn: Callable[..., List[Line]],
) -> List[Line]:
    # Four-line hooks like "I Gotta Feeling" need a short setup line and a
    # wider first gap; otherwise the generic dense layout overstates line 1 and
    # keeps the repeated hook entrances too early.
    return apply_weighted_line_layout_fn(
        lines=lines,
        populated_lines=populated_lines,
        line_weights=[1.4, 3.2, 3.2, 3.45],
        gap_weights=[2.0, 0.05, 0.05],
        anchor_start=anchor_start,
        desired_end=desired_end,
    )


def apply_three_line_subset_hook_layout(
    *,
    lines: List[Line],
    populated_lines: List[Line],
    anchor_start: float,
    desired_end: float,
    apply_weighted_line_layout_fn: Callable[..., List[Line]],
) -> List[Line]:
    # Three-line hooks like "Hotline Bling" need a roomier first pickup gap and
    # a later callback entrance than the generic dense layout gives them.
    return apply_weighted_line_layout_fn(
        lines=lines,
        populated_lines=populated_lines,
        line_weights=[8.1, 6.2, 5.7],
        gap_weights=[2.5, 1.7],
        anchor_start=anchor_start,
        desired_end=desired_end,
    )


def apply_three_line_call_response_layout(
    *,
    lines: List[Line],
    populated_lines: List[Line],
    anchor_start: float,
    desired_end: float,
    apply_weighted_line_layout_fn: Callable[..., List[Line]],
) -> List[Line]:
    # Three-line openings like "Shout" need a longer first call and wider gaps
    # before the response and closing line than the generic dense layout allows.
    return apply_weighted_line_layout_fn(
        lines=lines,
        populated_lines=populated_lines,
        line_weights=[3.4, 3.0, 7.4],
        gap_weights=[2.0, 1.35],
        anchor_start=anchor_start,
        desired_end=desired_end,
    )


def apply_four_line_staggered_chorus_layout(
    *,
    lines: List[Line],
    populated_lines: List[Line],
    duration: float,
    apply_weighted_line_layout_fn: Callable[..., List[Line]],
) -> List[Line]:
    # Four-line choruses like "Creep" need wider gaps before the second, third,
    # and fourth lines so the long third line can hold more span.
    return apply_weighted_line_layout_fn(
        lines=lines,
        populated_lines=populated_lines,
        line_weights=[2.1, 3.2, 6.6, 4.1],
        gap_weights=[3.2, 2.9, 3.0],
        anchor_start=max(1.25, duration * 0.06),
        desired_end=duration - 0.02,
    )


def apply_four_line_late_tail_chorus_layout(
    *,
    lines: List[Line],
    populated_lines: List[Line],
    duration: float,
    apply_weighted_line_layout_fn: Callable[..., List[Line]],
) -> List[Line]:
    # Four-line choruses like "Taste" need a longer closing line and slightly
    # later first-two-line entrances than the generic dense layout provides.
    return apply_weighted_line_layout_fn(
        lines=lines,
        populated_lines=populated_lines,
        line_weights=[6.7, 6.3, 4.7, 8.7],
        gap_weights=[0.55, 0.35, 0.18],
        anchor_start=max(1.2, duration * 0.065),
        desired_end=duration,
    )


def apply_alternating_hook_chorus_layout(
    *,
    lines: List[Line],
    populated_lines: List[Line],
    duration: float,
    apply_weighted_line_layout_fn: Callable[..., List[Line]],
) -> List[Line]:
    # Choruses like "Radioactive" alternate long repeated hook lines with short
    # "Whoa-oh" responses; the generic dense layout leaves the long repeated
    # lines too late and the short responses too compressed around them.
    return apply_weighted_line_layout_fn(
        lines=lines,
        populated_lines=populated_lines,
        line_weights=[7.2, 5.8, 5.2, 5.2, 2.1, 1.8, 4.7, 2.0, 1.8, 4.6],
        gap_weights=[0.22, 0.18, 0.14, 0.85, 0.18, 0.9, 0.18, 0.16, 0.6],
        anchor_start=max(0.32, duration * 0.012),
        desired_end=duration - 0.04,
    )


def apply_repeated_five_line_chorus_layout(
    *,
    lines: List[Line],
    populated_lines: List[Line],
    duration: float,
    apply_weighted_line_layout_fn: Callable[..., List[Line]],
) -> List[Line]:
    # Repeated five-line choruses like "Counting Stars" need a later opening
    # anchor and larger handoff gaps before the short closing line in each half.
    return apply_weighted_line_layout_fn(
        lines=lines,
        populated_lines=populated_lines,
        line_weights=[6.9, 6.2, 7.2, 4.2, 3.0, 6.6, 6.0, 6.9, 4.1, 4.2],
        gap_weights=[0.95, 0.88, 1.75, 0.12, 0.82, 0.82, 0.9, 1.68, 0.08],
        anchor_start=max(1.35, duration * 0.04),
        desired_end=duration - 0.02,
    )


def apply_seven_line_repeated_hook_bridge_layout(
    *,
    lines: List[Line],
    populated_lines: List[Line],
    duration: float,
    apply_weighted_line_layout_fn: Callable[..., List[Line]],
) -> List[Line]:
    # Choruses like "Call Me Maybe" need a large handoff gap after the repeated
    # fourth line before the closing three-line tail; the generic dense layout
    # keeps the back half several seconds too early.
    return apply_weighted_line_layout_fn(
        lines=lines,
        populated_lines=populated_lines,
        line_weights=[6.7, 5.8, 6.3, 5.8, 3.9, 3.8, 3.7],
        gap_weights=[0.42, 0.34, 0.42, 3.6, 0.22, 0.18],
        anchor_start=max(2.15, duration * 0.08),
        desired_end=duration,
    )


def apply_special_plain_text_layout(
    *,
    lines: List[Line],
    populated_lines: List[Line],
    duration: float,
    short_title_chorus_clip: bool,
    four_line_staggered_chorus_clip: bool,
    four_line_late_tail_chorus_clip: bool,
    alternating_hook_chorus_clip: bool,
    repeated_five_line_chorus_clip: bool,
    seven_line_repeated_hook_bridge_clip: bool,
    staggered_compact_hook_clip: bool,
    three_line_call_response_clip: bool,
    three_line_subset_hook_clip: bool,
    short_setup_repeated_hook_clip: bool,
    dense_short_verse_clip: bool,
    estimate_singing_duration_fn: Callable[[str, int], float],
    apply_weighted_line_layout_fn: Callable[..., List[Line]],
) -> List[Line] | None:
    layout_options: List[tuple[bool, Callable[[], List[Line]]]] = [
        (
            short_title_chorus_clip,
            lambda: apply_short_title_chorus_layout(
                lines=lines,
                populated_lines=populated_lines,
                anchor_start=max(0.95, duration * 0.038),
                desired_end=duration,
                estimate_singing_duration_fn=estimate_singing_duration_fn,
                apply_weighted_line_layout_fn=apply_weighted_line_layout_fn,
            ),
        ),
        (
            four_line_staggered_chorus_clip,
            lambda: apply_four_line_staggered_chorus_layout(
                lines=lines,
                populated_lines=populated_lines,
                duration=duration,
                apply_weighted_line_layout_fn=apply_weighted_line_layout_fn,
            ),
        ),
        (
            four_line_late_tail_chorus_clip,
            lambda: apply_four_line_late_tail_chorus_layout(
                lines=lines,
                populated_lines=populated_lines,
                duration=duration,
                apply_weighted_line_layout_fn=apply_weighted_line_layout_fn,
            ),
        ),
        (
            alternating_hook_chorus_clip,
            lambda: apply_alternating_hook_chorus_layout(
                lines=lines,
                populated_lines=populated_lines,
                duration=duration,
                apply_weighted_line_layout_fn=apply_weighted_line_layout_fn,
            ),
        ),
        (
            repeated_five_line_chorus_clip,
            lambda: apply_repeated_five_line_chorus_layout(
                lines=lines,
                populated_lines=populated_lines,
                duration=duration,
                apply_weighted_line_layout_fn=apply_weighted_line_layout_fn,
            ),
        ),
        (
            seven_line_repeated_hook_bridge_clip,
            lambda: apply_seven_line_repeated_hook_bridge_layout(
                lines=lines,
                populated_lines=populated_lines,
                duration=duration,
                apply_weighted_line_layout_fn=apply_weighted_line_layout_fn,
            ),
        ),
        (
            staggered_compact_hook_clip,
            lambda: apply_staggered_compact_hook_layout(
                lines=lines,
                populated_lines=populated_lines,
                duration=duration,
                apply_weighted_line_layout_fn=apply_weighted_line_layout_fn,
            ),
        ),
        (
            three_line_call_response_clip,
            lambda: apply_three_line_call_response_layout(
                lines=lines,
                populated_lines=populated_lines,
                anchor_start=max(0.7, duration * 0.07),
                desired_end=duration - 0.05,
                apply_weighted_line_layout_fn=apply_weighted_line_layout_fn,
            ),
        ),
        (
            three_line_subset_hook_clip,
            lambda: apply_three_line_subset_hook_layout(
                lines=lines,
                populated_lines=populated_lines,
                anchor_start=max(0.75, duration * 0.07),
                desired_end=duration - 0.05,
                apply_weighted_line_layout_fn=apply_weighted_line_layout_fn,
            ),
        ),
        (
            short_setup_repeated_hook_clip,
            lambda: apply_short_setup_repeated_hook_layout(
                lines=lines,
                populated_lines=populated_lines,
                anchor_start=max(0.95, duration * 0.07),
                desired_end=duration - 0.02,
                apply_weighted_line_layout_fn=apply_weighted_line_layout_fn,
            ),
        ),
        (
            dense_short_verse_clip,
            lambda: apply_dense_short_verse_layout(
                lines=lines,
                populated_lines=populated_lines,
                duration=duration,
                estimate_singing_duration_fn=estimate_singing_duration_fn,
                apply_weighted_line_layout_fn=apply_weighted_line_layout_fn,
            ),
        ),
    ]
    for enabled, apply_layout in layout_options:
        if enabled:
            return apply_layout()
    return None


def line_tokens_for_weight(line: Line) -> List[str]:
    words = [re.sub(r"[^a-z0-9]", "", word.text.lower()) for word in line.words]
    return [word for word in words if word]


def _normalize_line_weight_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\\s]", "", text.lower()).strip()


def _collapse_adjacent_duplicate_tokens(tokens: List[str]) -> List[str]:
    collapsed: List[str] = []
    for token in tokens:
        if collapsed and collapsed[-1] == token:
            continue
        collapsed.append(token)
    return collapsed


def adjust_repetitive_compact_layout(
    lines: List[Line],
    line_weights: List[float],
    *,
    normalize_text_fn: Callable[[str], str],
) -> tuple[List[float], List[float]]:
    gap_weights = [1.0] * max(len(lines) - 1, 0)
    if not lines:
        return line_weights, gap_weights
    if len(lines) == 2:
        return _adjust_two_line_repetitive_layout(lines, line_weights, gap_weights)
    if len(lines) == 3:
        return _adjust_three_line_repeated_short_intro_layout(
            lines, line_weights, gap_weights, normalize_text_fn=normalize_text_fn
        )
    if len(lines) < 4:
        return line_weights, gap_weights
    return _adjust_dominant_repetition_layout(
        lines, line_weights, gap_weights, normalize_text_fn=normalize_text_fn
    )


def _adjust_two_line_repetitive_layout(
    lines: List[Line],
    line_weights: List[float],
    gap_weights: List[float],
) -> tuple[List[float], List[float]]:
    first_tokens = line_tokens_for_weight(lines[0])
    second_tokens = line_tokens_for_weight(lines[1])
    if (
        first_tokens
        and second_tokens
        and set(second_tokens) < set(first_tokens)
        and len(lines[1].words) < len(lines[0].words)
    ):
        line_weights = list(line_weights)
        line_weights[0] = max(2.0, min(line_weights[0], line_weights[1] * 0.53))
        line_weights[1] = max(line_weights[1], line_weights[0] * 1.85)
        gap_weights[0] = min(gap_weights[0], 0.1)
    return line_weights, gap_weights


def _adjust_dominant_repetition_layout(
    lines: List[Line],
    line_weights: List[float],
    gap_weights: List[float],
    *,
    normalize_text_fn: Callable[[str], str],
) -> tuple[List[float], List[float]]:
    normalized_texts = [normalize_text_fn(line.text) for line in lines]
    dominant_text, dominant_count = Counter(normalized_texts).most_common(1)[0]
    if dominant_count < 3:
        return line_weights, gap_weights
    dominant_indices = [
        i for i, text in enumerate(normalized_texts) if text == dominant_text
    ]
    first_dominant_idx = dominant_indices[0]
    last_dominant_idx = dominant_indices[-1]
    if first_dominant_idx <= 0:
        return line_weights, gap_weights
    for gap_idx in range(first_dominant_idx):
        remaining_prefix_gaps = first_dominant_idx - gap_idx - 1
        gap_weights[gap_idx] = max(
            gap_weights[gap_idx],
            2.95 + remaining_prefix_gaps * 0.35,
        )
    for gap_idx in range(first_dominant_idx, last_dominant_idx):
        gap_weights[gap_idx] = min(gap_weights[gap_idx], 0.05)
    if last_dominant_idx + 1 == len(lines) - 1:
        dominant_tokens = set(line_tokens_for_weight(lines[last_dominant_idx]))
        tail_tokens = set(line_tokens_for_weight(lines[-1]))
        if (
            tail_tokens
            and tail_tokens < dominant_tokens
            and len(lines[-1].words) < len(lines[last_dominant_idx].words)
        ):
            line_weights[-1] = max(1.0, line_weights[-1] * 0.34)
            gap_weights[last_dominant_idx] = min(gap_weights[last_dominant_idx], 0.15)
    return line_weights, gap_weights


def _adjust_three_line_repeated_short_intro_layout(
    lines: List[Line],
    line_weights: List[float],
    gap_weights: List[float],
    *,
    normalize_text_fn: Callable[[str], str],
) -> tuple[List[float], List[float]]:
    first_text = normalize_text_fn(lines[0].text)
    second_text = normalize_text_fn(lines[1].text)
    third_text = normalize_text_fn(lines[2].text)
    if not first_text or first_text != second_text or third_text == first_text:
        return line_weights, gap_weights
    if len(lines[0].words) > 2 or len(lines[1].words) > 2:
        return line_weights, gap_weights
    if len(lines[2].words) < 4:
        return line_weights, gap_weights

    adjusted_line_weights = list(line_weights)
    adjusted_gap_weights = list(gap_weights)
    adjusted_line_weights[0] = max(2.4, adjusted_line_weights[0])
    adjusted_line_weights[1] = max(2.2, adjusted_line_weights[1])
    adjusted_line_weights[2] = min(adjusted_line_weights[2], 1.1)
    adjusted_gap_weights[0] = min(adjusted_gap_weights[0], 0.7)
    adjusted_gap_weights[1] = max(adjusted_gap_weights[1], 6.1)
    return adjusted_line_weights, adjusted_gap_weights
