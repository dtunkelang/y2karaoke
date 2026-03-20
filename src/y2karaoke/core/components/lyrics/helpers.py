"""Helper functions for lyrics processing."""

import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ...models import Word, Line, SongMetadata
from ...romanization import romanize_line
from .lyrics_clip_layout_helpers import (
    adjust_repetitive_compact_layout as _adjust_repetitive_compact_layout,
    apply_special_plain_text_layout as _apply_special_plain_text_layout,
    is_short_title_chorus_clip as _is_short_title_chorus_clip,
    line_tokens_for_weight as _line_tokens_for_weight,
)
from .lrc import (
    parse_lrc_with_timing,
    create_lines_from_lrc,
    create_lines_from_lrc_timings,
    uncensor_lyrics_text,
)

logger = logging.getLogger(__name__)

__all__ = [
    "create_lines_from_lrc",
    "create_lines_from_lrc_timings",
]


def _whisper_metrics_quality_score(metrics: Dict[str, float]) -> float:
    """Compute a bounded quality score from Whisper alignment metrics."""
    if not metrics:
        return 0.0

    matched_ratio = float(metrics.get("matched_ratio", 0.0))
    avg_similarity = float(metrics.get("avg_similarity", 0.0))
    line_coverage = float(metrics.get("line_coverage", 0.0))
    phonetic_similarity_coverage = float(
        metrics.get("phonetic_similarity_coverage", matched_ratio * avg_similarity)
    )
    exact_match_ratio = float(metrics.get("exact_match_ratio", 0.0))
    high_similarity_ratio = float(metrics.get("high_similarity_ratio", avg_similarity))

    score = (
        matched_ratio * 0.34
        + avg_similarity * 0.23
        + line_coverage * 0.23
        + phonetic_similarity_coverage * 0.12
        + exact_match_ratio * 0.04
        + high_similarity_ratio * 0.04
    )
    return max(0.0, min(1.0, score))


def _should_try_whisper_map_fallback(metrics: Dict[str, float]) -> bool:
    """Determine whether hybrid alignment quality is low enough to try map fallback."""
    if not metrics:
        return False
    return _whisper_metrics_quality_score(metrics) < 0.72


def _choose_whisper_map_fallback(
    baseline_metrics: Dict[str, float],
    map_metrics: Dict[str, float],
    *,
    min_gain: float = 0.05,
    allow_coverage_promotion: bool = True,
) -> Dict[str, float]:
    """Choose whether to use map fallback based on stable internal quality signals."""
    baseline_score = _whisper_metrics_quality_score(baseline_metrics)
    map_score = _whisper_metrics_quality_score(map_metrics)
    score_gain = map_score - baseline_score
    baseline_coverage = float(baseline_metrics.get("line_coverage", 0.0))
    map_coverage = float(map_metrics.get("line_coverage", 0.0))

    selected = 0.0
    decision_code = 2.0  # rejected_insufficient_score_gain

    if map_score > baseline_score + min_gain:
        selected = 1.0
        decision_code = 1.0  # selected_score_gain
    elif (
        allow_coverage_promotion
        and map_score > baseline_score
        and score_gain > (min_gain * 0.4)
        and map_coverage > baseline_coverage + 0.08
    ):
        # Optional secondary path: allow smaller gains when line coverage
        # meaningfully improves and map score is still better than baseline.
        selected = 1.0
        decision_code = 4.0  # selected_coverage_promotion
    return {
        "selected": selected,
        "decision_code": decision_code,
        "baseline_score": round(baseline_score, 4),
        "candidate_score": round(map_score, 4),
        "score_gain": round(score_gain, 4),
        "min_gain_required": float(min_gain),
    }


def _coverage_promotion_enabled() -> bool:
    value = str(
        os.getenv("Y2KARAOKE_ENABLE_FALLBACK_MAP_COVERAGE_PROMOTION", "1")
    ).strip()
    return value not in {"0", "false", "False", "no", "off"}


def _clone_lines_for_fallback(lines: List[Line]) -> List[Line]:
    """Fast explicit clone of line/word timing structures for fallback mapping."""
    cloned: List[Line] = []
    for line in lines:
        cloned.append(
            Line(
                words=[
                    Word(
                        text=word.text,
                        start_time=word.start_time,
                        end_time=word.end_time,
                        singer=word.singer,
                    )
                    for word in line.words
                ],
                singer=line.singer,
            )
        )
    return cloned


def _estimate_singing_duration(text: str, word_count: int) -> float:
    """
    Estimate how long it takes to sing a line based on text content.

    Uses character count as primary heuristic since longer words take
    longer to sing. Assumes roughly 12-15 characters per second for
    typical singing tempo.

    Args:
        text: The line text
        word_count: Number of words in the line

    Returns:
        Estimated duration in seconds
    """
    char_count = len(text.replace(" ", ""))

    # Base estimate: ~0.07 seconds per character (roughly 14 chars/sec)
    char_based = char_count * 0.07

    # Minimum based on word count (~0.25 sec per word for fast singing)
    word_based = word_count * 0.25

    # Use the larger of the two estimates
    duration = max(char_based, word_based)

    # Clamp to reasonable range
    return max(0.5, min(duration, 8.0))


def _duration_cap_multiplier_for_line(
    line_text: str,
    word_count: int,
    gap_to_next: float,
    estimated_duration: float,
) -> float:
    """Allow extra room for long pause-heavy sung lines without broad gap fill."""
    tokens = [re.sub(r"[^a-z]+", "", tok.lower()) for tok in line_text.split()]
    tokens = [tok for tok in tokens if tok]
    if word_count < 6:
        return 1.3
    if gap_to_next < max(4.5, estimated_duration * 1.8):
        return 1.3
    punctuation_pauses = (
        line_text.count(",") + line_text.count(";") + line_text.count(":")
    )
    lower_text = line_text.lower()
    has_interjection = bool(re.search(r"\b(oh|ooh|ah|hey|yeah)\b", lower_text))
    has_phrase_break = punctuation_pauses > 0 or "(" in line_text or "'" in line_text
    leading_filler_phrase = (
        len(tokens) >= 2
        and tokens[0] in {"oh", "ooh", "ah", "hey", "yeah"}
        and tokens[1] not in {"oh", "ooh", "ah", "hey", "yeah"}
    )
    if punctuation_pauses > 0 and has_interjection and not leading_filler_phrase:
        if word_count <= 6 and re.search(r"\((oh|ooh|ah|hey|yeah)\)", lower_text):
            return 1.3
        return 2.7
    if (
        8 <= word_count <= 10
        and has_phrase_break
        and gap_to_next >= max(4.2, estimated_duration * 1.8)
        and not leading_filler_phrase
    ):
        return 1.7
    if punctuation_pauses == 0:
        return 1.3
    if not has_interjection:
        return 1.3
    if leading_filler_phrase:
        return 1.3
    return 1.3


def _extract_text_lines_from_lrc(lrc_text: str) -> List[str]:
    timed = parse_lrc_with_timing(lrc_text, "", "", filter_promos=False)
    if timed:
        return [text for _t, text in timed if text.strip()]
    lines: List[str] = []
    for raw in lrc_text.splitlines():
        line = uncensor_lyrics_text(re.sub(r"\[[0-9:.]+\]", "", raw).strip())
        if line:
            lines.append(line)
    return lines


def _is_non_substantive_offset_anchor(text: str) -> bool:
    tokens = [re.sub(r"[^a-z]+", "", tok.lower()) for tok in text.split()]
    tokens = [tok for tok in tokens if tok]
    if not tokens or len(tokens) > 2:
        return False
    filler_tokens = {"oh", "ooh", "ah", "hey", "yeah"}
    return all(tok in filler_tokens for tok in tokens)


def _select_offset_anchor_timing(
    line_timings: List[Tuple[float, str]],
) -> Tuple[float, str, bool]:
    if not line_timings:
        raise ValueError("line_timings must not be empty")
    first_time, first_text = line_timings[0]
    if len(line_timings) < 2 or not _is_non_substantive_offset_anchor(first_text):
        return first_time, first_text, False
    second_time, second_text = line_timings[1]
    if float(second_time) - float(first_time) < 8.0:
        return first_time, first_text, False
    return second_time, second_text, True


def _create_lines_from_plain_text(text_lines: List[str]) -> List[Line]:
    if not text_lines:
        return []

    lines: List[Line] = []
    current_time = 0.0
    for text in text_lines:
        text = uncensor_lyrics_text(text)
        word_texts = text.split()
        if not word_texts:
            continue
        duration = max(2.0, _estimate_singing_duration(text, len(word_texts)))
        start_time = current_time
        end_time = start_time + duration
        word_duration = (duration * 0.95) / len(word_texts)
        words: List[Word] = []
        for j, word_text in enumerate(word_texts):
            word_start = start_time + j * (duration / len(word_texts))
            word_end = word_start + word_duration
            words.append(Word(text=word_text, start_time=word_start, end_time=word_end))
        lines.append(Line(words=words))
        current_time = end_time + 0.2

    return lines


def _spread_lines_across_target_duration(
    lines: List[Line],
    target_duration: Optional[float],
    *,
    edge_padding_sec: float = 1.0,
    min_expansion_ratio: float = 1.15,
) -> List[Line]:
    """Stretch untimed plain-text lines across the known audio window.

    Plain-text lyrics are initially packed using local duration heuristics, which can
    compress a short clip's entire lyric block into the first few seconds. When the
    clip duration is known, spreading that sketch across the full window produces much
    saner seed timings for downstream Whisper alignment.
    """
    if not lines or not target_duration or float(target_duration) <= 0.0:
        return lines

    populated_lines = [line for line in lines if line.words]
    if not populated_lines:
        return lines

    current_start = min(line.start_time for line in populated_lines)
    current_end = max(line.end_time for line in populated_lines)
    current_span = current_end - current_start
    if current_span <= 0.0:
        return lines

    duration = float(target_duration)
    padding = min(float(edge_padding_sec), max(duration * 0.08, 0.0))
    usable_span = max(duration - (padding * 2.0), current_span)
    desired_start = padding
    desired_end = padding + usable_span
    needs_fit = current_end > desired_end + 0.01 or current_span > usable_span + 0.01
    if not needs_fit and usable_span <= current_span * float(min_expansion_ratio):
        return lines

    scale = usable_span / current_span
    for line in populated_lines:
        for word in line.words:
            word.start_time = desired_start + (word.start_time - current_start) * scale
            word.end_time = desired_start + (word.end_time - current_start) * scale
    return lines


def _anchor_plain_text_lines_to_audio_window(
    lines: List[Line],
    target_duration: Optional[float],
    vocals_path: Optional[str],
    *,
    min_detectable_start_sec: float = 0.35,
    fallback_anchor_ratio: float = 0.06,
    max_anchor_ratio: float = 0.35,
    trailing_padding_sec: float = 0.8,
    min_expansion_ratio: float = 1.05,
    compact_line_word_threshold: float = 5.0,
    compact_short_clip_word_threshold: float = 3.0,
    compact_short_clip_line_count: int = 4,
    compact_short_clip_duration_sec: float = 12.0,
    compact_short_clip_anchor_ratio: float = 0.24,
    compact_short_clip_trailing_padding_sec: float = 0.15,
    repetitive_compact_clip_trailing_padding_sec: float = 0.15,
    dense_short_verse_min_avg_words_per_line: float = 6.5,
    dense_short_verse_max_lines: int = 4,
    dense_short_verse_max_duration_sec: float = 10.0,
    sparse_sustained_clip_min_duration_sec: float = 18.0,
    sparse_sustained_clip_max_lines: int = 5,
    sparse_sustained_clip_max_words_per_line: int = 5,
    sparse_sustained_clip_min_avg_words_per_line: float = 2.5,
    sparse_sustained_clip_max_avg_words_per_line: float = 5.2,
    sparse_sustained_clip_anchor_ratio: float = 0.045,
    sparse_sustained_clip_trailing_padding_sec: float = 0.5,
    short_phrase_sustained_clip_min_duration_sec: float = 20.0,
    short_phrase_sustained_clip_max_lines: int = 4,
    short_phrase_sustained_clip_max_avg_words_per_line: float = 4.2,
    mixed_density_chorus_clip_min_duration_sec: float = 24.0,
    mixed_density_chorus_clip_min_lines: int = 8,
    mixed_density_chorus_clip_anchor_ratio: float = 0.026,
    mixed_density_chorus_clip_trailing_padding_sec: float = 0.02,
) -> List[Line]:
    """Anchor untimed plain-text lines to the detected vocal onset when available.

    Plain-text clip lyrics otherwise start at 0.0s, which can bias repeated-hook
    clips toward artificially early alignment. When vocals enter later, shift the
    seed lines forward to the detected onset and, if useful, stretch the remaining
    sketch across the rest of the clip window.
    """
    if not lines or not target_duration or float(target_duration) <= 0.0:
        return lines

    populated_lines = [line for line in lines if line.words]
    if not populated_lines:
        return lines

    duration = float(target_duration)
    (
        average_words_per_line,
        repetitive_compact_clip,
        dense_short_verse_clip,
        sparse_sustained_clip,
        compact_short_clip,
        short_phrase_sustained_clip,
        two_line_subset_refrain_clip,
        mixed_density_chorus_clip,
    ) = _classify_plain_text_clip_layout(
        populated_lines=populated_lines,
        duration=duration,
        compact_line_word_threshold=compact_line_word_threshold,
        compact_short_clip_word_threshold=compact_short_clip_word_threshold,
        compact_short_clip_line_count=compact_short_clip_line_count,
        compact_short_clip_duration_sec=compact_short_clip_duration_sec,
        dense_short_verse_min_avg_words_per_line=dense_short_verse_min_avg_words_per_line,
        dense_short_verse_max_lines=dense_short_verse_max_lines,
        dense_short_verse_max_duration_sec=dense_short_verse_max_duration_sec,
        sparse_sustained_clip_min_duration_sec=sparse_sustained_clip_min_duration_sec,
        sparse_sustained_clip_max_lines=sparse_sustained_clip_max_lines,
        sparse_sustained_clip_max_words_per_line=sparse_sustained_clip_max_words_per_line,
        sparse_sustained_clip_min_avg_words_per_line=sparse_sustained_clip_min_avg_words_per_line,
        sparse_sustained_clip_max_avg_words_per_line=sparse_sustained_clip_max_avg_words_per_line,
        short_phrase_sustained_clip_min_duration_sec=short_phrase_sustained_clip_min_duration_sec,
        short_phrase_sustained_clip_max_lines=short_phrase_sustained_clip_max_lines,
        short_phrase_sustained_clip_max_avg_words_per_line=short_phrase_sustained_clip_max_avg_words_per_line,
        mixed_density_chorus_clip_min_duration_sec=mixed_density_chorus_clip_min_duration_sec,
        mixed_density_chorus_clip_min_lines=mixed_density_chorus_clip_min_lines,
    )
    short_title_chorus_clip = _is_short_title_chorus_clip(
        populated_lines=populated_lines,
        duration=duration,
    )
    if not vocals_path:
        special_layout = _apply_special_plain_text_layout(
            lines=lines,
            populated_lines=populated_lines,
            duration=duration,
            short_title_chorus_clip=short_title_chorus_clip,
            dense_short_verse_clip=dense_short_verse_clip,
            estimate_singing_duration_fn=_estimate_singing_duration,
            apply_weighted_line_layout_fn=_apply_weighted_line_layout,
        )
        if special_layout is not None:
            return special_layout
        return _spread_lines_across_target_duration(lines, target_duration)

    from ..alignment.alignment import detect_song_start

    detected_start = float(detect_song_start(vocals_path))
    if detected_start < float(min_detectable_start_sec):
        special_layout = _apply_special_plain_text_layout(
            lines=lines,
            populated_lines=populated_lines,
            duration=duration,
            short_title_chorus_clip=short_title_chorus_clip,
            dense_short_verse_clip=dense_short_verse_clip,
            estimate_singing_duration_fn=_estimate_singing_duration,
            apply_weighted_line_layout_fn=_apply_weighted_line_layout,
        )
        if special_layout is not None:
            return special_layout
        if not two_line_subset_refrain_clip:
            return _spread_lines_across_target_duration(lines, target_duration)
        detected_start = duration * float(fallback_anchor_ratio)

    anchor_start, trailing_padding = _resolve_plain_text_clip_anchor(
        detected_start=detected_start,
        duration=duration,
        max_anchor_ratio=max_anchor_ratio,
        compact_short_clip=compact_short_clip,
        compact_short_clip_anchor_ratio=compact_short_clip_anchor_ratio,
        sparse_sustained_clip=sparse_sustained_clip,
        sparse_sustained_clip_anchor_ratio=sparse_sustained_clip_anchor_ratio,
        trailing_padding_sec=trailing_padding_sec,
        repetitive_compact_clip=repetitive_compact_clip,
        repetitive_compact_clip_trailing_padding_sec=repetitive_compact_clip_trailing_padding_sec,
        compact_short_clip_trailing_padding_sec=compact_short_clip_trailing_padding_sec,
        sparse_sustained_clip_trailing_padding_sec=sparse_sustained_clip_trailing_padding_sec,
        mixed_density_chorus_clip=mixed_density_chorus_clip,
        mixed_density_chorus_clip_anchor_ratio=mixed_density_chorus_clip_anchor_ratio,
        mixed_density_chorus_clip_trailing_padding_sec=mixed_density_chorus_clip_trailing_padding_sec,
    )
    desired_end = max(anchor_start, duration - trailing_padding)
    available_span = desired_end - anchor_start
    if available_span <= 0.0:
        return lines

    if (
        average_words_per_line > compact_line_word_threshold
        and not repetitive_compact_clip
        and not dense_short_verse_clip
        and not two_line_subset_refrain_clip
        and not mixed_density_chorus_clip
        and not short_title_chorus_clip
    ):
        return _scale_dense_plain_text_lines(
            lines=lines,
            populated_lines=populated_lines,
            anchor_start=anchor_start,
            available_span=available_span,
            min_expansion_ratio=min_expansion_ratio,
        )

    line_weights, gap_weights = _build_plain_text_clip_layout(
        populated_lines=populated_lines,
        sparse_sustained_clip=sparse_sustained_clip,
        short_phrase_sustained_clip=short_phrase_sustained_clip,
        two_line_subset_refrain_clip=two_line_subset_refrain_clip,
        repetitive_compact_clip=repetitive_compact_clip,
        mixed_density_chorus_clip=mixed_density_chorus_clip,
    )
    return _apply_weighted_line_layout(
        lines=lines,
        populated_lines=populated_lines,
        line_weights=line_weights,
        gap_weights=gap_weights,
        anchor_start=anchor_start,
        desired_end=desired_end,
    )


def _normalize_line_weight_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\\s]", "", text.lower()).strip()


def _classify_plain_text_clip_layout(
    *,
    populated_lines: List[Line],
    duration: float,
    compact_line_word_threshold: float,
    compact_short_clip_word_threshold: float,
    compact_short_clip_line_count: int,
    compact_short_clip_duration_sec: float,
    dense_short_verse_min_avg_words_per_line: float,
    dense_short_verse_max_lines: int,
    dense_short_verse_max_duration_sec: float,
    sparse_sustained_clip_min_duration_sec: float,
    sparse_sustained_clip_max_lines: int,
    sparse_sustained_clip_max_words_per_line: int,
    sparse_sustained_clip_min_avg_words_per_line: float,
    sparse_sustained_clip_max_avg_words_per_line: float,
    short_phrase_sustained_clip_min_duration_sec: float,
    short_phrase_sustained_clip_max_lines: int,
    short_phrase_sustained_clip_max_avg_words_per_line: float,
    mixed_density_chorus_clip_min_duration_sec: float,
    mixed_density_chorus_clip_min_lines: int,
) -> Tuple[float, bool, bool, bool, bool, bool, bool, bool]:
    average_words_per_line = sum(len(line.words) for line in populated_lines) / max(
        len(populated_lines), 1
    )
    max_words_per_line = max(len(line.words) for line in populated_lines)
    normalized_line_texts = [
        _normalize_line_weight_text(line.text) for line in populated_lines
    ]
    two_line_subset_refrain_clip = _is_two_line_subset_refrain_clip(populated_lines)
    mixed_density_chorus_clip = _is_mixed_density_chorus_clip(
        lines=populated_lines,
        duration=duration,
        min_duration_sec=mixed_density_chorus_clip_min_duration_sec,
        min_lines=mixed_density_chorus_clip_min_lines,
    )
    repetitive_compact_clip = len(set(normalized_line_texts)) < len(
        normalized_line_texts
    ) and max_words_per_line <= max(compact_line_word_threshold, 6.0)
    dense_short_verse_clip = (
        not repetitive_compact_clip
        and len(populated_lines) <= dense_short_verse_max_lines
        and duration <= dense_short_verse_max_duration_sec
        and average_words_per_line >= dense_short_verse_min_avg_words_per_line
    )
    sparse_sustained_clip = (
        not repetitive_compact_clip
        and not dense_short_verse_clip
        and duration >= sparse_sustained_clip_min_duration_sec
        and len(populated_lines) <= sparse_sustained_clip_max_lines
        and max_words_per_line <= sparse_sustained_clip_max_words_per_line
        and sparse_sustained_clip_min_avg_words_per_line
        <= average_words_per_line
        <= sparse_sustained_clip_max_avg_words_per_line
    )
    compact_short_clip = (
        average_words_per_line <= compact_short_clip_word_threshold
        and len(populated_lines) <= compact_short_clip_line_count
        and duration <= compact_short_clip_duration_sec
    )
    short_phrase_sustained_clip = (
        sparse_sustained_clip
        and duration >= short_phrase_sustained_clip_min_duration_sec
        and len(populated_lines) <= short_phrase_sustained_clip_max_lines
        and average_words_per_line <= short_phrase_sustained_clip_max_avg_words_per_line
    )
    return (
        average_words_per_line,
        repetitive_compact_clip,
        dense_short_verse_clip,
        sparse_sustained_clip,
        compact_short_clip,
        short_phrase_sustained_clip,
        two_line_subset_refrain_clip,
        mixed_density_chorus_clip,
    )


def _resolve_plain_text_clip_anchor(
    *,
    detected_start: float,
    duration: float,
    max_anchor_ratio: float,
    compact_short_clip: bool,
    compact_short_clip_anchor_ratio: float,
    sparse_sustained_clip: bool,
    sparse_sustained_clip_anchor_ratio: float,
    trailing_padding_sec: float,
    repetitive_compact_clip: bool,
    repetitive_compact_clip_trailing_padding_sec: float,
    compact_short_clip_trailing_padding_sec: float,
    sparse_sustained_clip_trailing_padding_sec: float,
    mixed_density_chorus_clip: bool,
    mixed_density_chorus_clip_anchor_ratio: float,
    mixed_density_chorus_clip_trailing_padding_sec: float,
) -> Tuple[float, float]:
    anchor_start = min(max(detected_start, 0.0), duration * float(max_anchor_ratio))
    if compact_short_clip:
        anchor_start = max(
            anchor_start, duration * float(compact_short_clip_anchor_ratio)
        )
    if sparse_sustained_clip:
        anchor_start = max(
            anchor_start, duration * float(sparse_sustained_clip_anchor_ratio)
        )
    if mixed_density_chorus_clip:
        anchor_start = max(
            anchor_start, duration * float(mixed_density_chorus_clip_anchor_ratio)
        )

    trailing_padding = min(float(trailing_padding_sec), 1.2)
    if repetitive_compact_clip:
        trailing_padding = min(
            trailing_padding,
            float(repetitive_compact_clip_trailing_padding_sec),
        )
    elif compact_short_clip:
        trailing_padding = min(
            trailing_padding,
            float(compact_short_clip_trailing_padding_sec),
        )
    elif sparse_sustained_clip:
        trailing_padding = min(
            trailing_padding,
            float(sparse_sustained_clip_trailing_padding_sec),
        )
    elif mixed_density_chorus_clip:
        trailing_padding = min(
            trailing_padding,
            float(mixed_density_chorus_clip_trailing_padding_sec),
        )
    return anchor_start, trailing_padding


def _scale_dense_plain_text_lines(
    *,
    lines: List[Line],
    populated_lines: List[Line],
    anchor_start: float,
    available_span: float,
    min_expansion_ratio: float,
) -> List[Line]:
    current_start = min(line.start_time for line in populated_lines)
    current_end = max(line.end_time for line in populated_lines)
    current_span = current_end - current_start
    if current_span <= 0.0:
        return lines
    if available_span <= current_span * min_expansion_ratio:
        shift = anchor_start - current_start
        if abs(shift) > 0.01:
            for line in populated_lines:
                for word in line.words:
                    word.start_time += shift
                    word.end_time += shift
        return lines

    scale = available_span / current_span
    for line in populated_lines:
        for word in line.words:
            rel_start = word.start_time - current_start
            rel_end = word.end_time - current_start
            word.start_time = anchor_start + rel_start * scale
            word.end_time = anchor_start + rel_end * scale
    return lines


def _build_plain_text_clip_layout(
    *,
    populated_lines: List[Line],
    sparse_sustained_clip: bool,
    short_phrase_sustained_clip: bool,
    two_line_subset_refrain_clip: bool,
    repetitive_compact_clip: bool,
    mixed_density_chorus_clip: bool,
) -> Tuple[List[float], List[float]]:
    if sparse_sustained_clip:
        if short_phrase_sustained_clip:
            line_weights = [
                _short_phrase_sustained_line_weight(line) for line in populated_lines
            ]
        else:
            line_weights = [
                _sparse_sustained_line_weight(line) for line in populated_lines
            ]
        return line_weights, [1.0] * max(len(populated_lines) - 1, 0)

    if two_line_subset_refrain_clip:
        return [3.4, 6.4], [0.12]

    if mixed_density_chorus_clip:
        return _build_mixed_density_chorus_layout(populated_lines)

    line_weights = [
        _compact_line_weight(line, prefer_unique_tokens=repetitive_compact_clip)
        for line in populated_lines
    ]
    if repetitive_compact_clip:
        return _adjust_repetitive_compact_layout(
            populated_lines,
            line_weights,
            normalize_text_fn=_normalize_line_weight_text,
        )
    return line_weights, [1.0] * max(len(populated_lines) - 1, 0)


def _apply_weighted_line_layout(
    *,
    lines: List[Line],
    populated_lines: List[Line],
    line_weights: List[float],
    gap_weights: List[float],
    anchor_start: float,
    desired_end: float,
) -> List[Line]:
    total_units = sum(line_weights) + sum(gap_weights)
    if total_units <= 0.0:
        return lines

    unit = (desired_end - anchor_start) / total_units
    cursor = anchor_start
    for line_idx, (line, weight) in enumerate(zip(populated_lines, line_weights)):
        line_span = max(unit * weight, 0.2)
        word_count = max(len(line.words), 1)
        word_step = line_span / word_count
        word_span = word_step * 0.9
        for word_idx, word in enumerate(line.words):
            word.start_time = cursor + word_idx * word_step
            word.end_time = min(word.start_time + word_span, desired_end)
        cursor += line_span
        if line_idx < len(gap_weights):
            cursor += unit * gap_weights[line_idx]
    return lines


def _compact_line_weight(line: Line, *, prefer_unique_tokens: bool) -> float:
    words = [re.sub(r"[^a-z0-9]", "", word.text.lower()) for word in line.words]
    words = [word for word in words if word]
    if not words:
        return 2.0
    if not prefer_unique_tokens:
        return max(float(len(words)), 2.0)
    unique_count = len(dict.fromkeys(words))
    return max(float(unique_count), 2.0)


def _is_two_line_subset_refrain_clip(lines: List[Line]) -> bool:
    if len(lines) != 2:
        return False
    first_tokens = _line_tokens_for_weight(lines[0])
    second_tokens = _line_tokens_for_weight(lines[1])
    if not first_tokens or not second_tokens:
        return False
    first_counts = Counter(first_tokens)
    second_counts = Counter(second_tokens)
    return (
        all(first_counts[token] >= count for token, count in second_counts.items())
        and any(first_counts[token] > count for token, count in second_counts.items())
        and len(lines[1].words) < len(lines[0].words)
    )


def _is_mixed_density_chorus_clip(
    *,
    lines: List[Line],
    duration: float,
    min_duration_sec: float,
    min_lines: int,
) -> bool:
    if duration < min_duration_sec or len(lines) < min_lines:
        return False
    normalized_texts = [_normalize_line_weight_text(line.text) for line in lines]
    counts = Counter(normalized_texts)
    repeated_exact_lines = sum(1 for count in counts.values() if count >= 2)
    word_counts = [len(line.words) for line in lines]
    if repeated_exact_lines < 2:
        return False
    if min(word_counts) > 4 or max(word_counts) < 8:
        return False
    parenthetical_lines = sum(
        1 for line in lines if "(" in line.text and ")" in line.text
    )
    return parenthetical_lines >= 1


def _build_mixed_density_chorus_layout(
    lines: List[Line],
) -> Tuple[List[float], List[float]]:
    normalized_texts = [_normalize_line_weight_text(line.text) for line in lines]
    counts = Counter(normalized_texts)
    line_weights = [
        _mixed_density_chorus_line_weight(line, repeated_count=counts[text])
        for line, text in zip(lines, normalized_texts)
    ]
    gap_weights = [
        _mixed_density_chorus_gap_weight(current=line, next_line=lines[idx + 1])
        for idx, line in enumerate(lines[:-1])
    ]
    return line_weights, gap_weights


def _mixed_density_chorus_line_weight(line: Line, *, repeated_count: int) -> float:
    words = len(line.words)
    weight = _estimate_singing_duration(line.text, words)
    if repeated_count >= 2 and words >= 8:
        weight *= 1.28
    elif repeated_count >= 2 and words <= 4:
        weight *= 1.02
    if "(" in line.text and ")" in line.text and words <= 7:
        weight *= 0.72
    return max(weight, 1.0)


def _mixed_density_chorus_gap_weight(current: Line, next_line: Line) -> float:
    current_words = len(current.words)
    next_words = len(next_line.words)
    if current_words <= 4 and next_words >= current_words + 5:
        return 0.8
    if "(" in current.text and ")" in current.text and current_words <= 7:
        return 0.35
    if next_words >= 10 and current_words <= 6:
        return 0.25
    if current_words >= 8 and next_words <= 4:
        return 0.11
    return 0.08


def _sparse_sustained_line_weight(line: Line) -> float:
    words = [re.sub(r"[^a-z0-9]", "", word.text.lower()) for word in line.words]
    words = [word for word in words if word]
    if not words:
        return 1.5
    unique_count = len(dict.fromkeys(words))
    return min(1.9, max(1.35, 1.1 + unique_count * 0.14))


def _short_phrase_sustained_line_weight(line: Line) -> float:
    words = [re.sub(r"[^a-z0-9]", "", word.text.lower()) for word in line.words]
    words = [word for word in words if word]
    if not words:
        return 2.0
    unique_count = len(dict.fromkeys(words))
    return min(2.35, max(2.0, 1.75 + unique_count * 0.12))


def _clean_text_lines(lines: List[str]) -> List[str]:
    cleaned = []
    for line in lines:
        line = uncensor_lyrics_text(re.sub(r"\s+", " ", line).strip())
        if not line:
            continue
        if len(line) < 2:
            continue
        cleaned.append(line)
    return cleaned


def _load_lyrics_file(
    lyrics_file: Path,
    filter_promos: bool,
) -> Tuple[Optional[str], Optional[List[Tuple[float, str]]], List[str]]:
    """Load lyrics from a local text or LRC file.

    Returns (lrc_text, line_timings, text_lines).
    """
    try:
        raw = lyrics_file.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"Failed to read lyrics file {lyrics_file}: {e}")
        return None, None, []

    if re.search(r"\[[0-9]{1,2}:[0-9]{2}(?:\.[0-9]{1,3})?\]", raw):
        line_timings = parse_lrc_with_timing(raw, "", "", filter_promos)
        if line_timings:
            return raw, line_timings, _extract_text_lines_from_lrc(raw)

    text_lines = _clean_text_lines(raw.splitlines())
    return None, None, text_lines


def _create_no_lyrics_placeholder(
    title: str, artist: str
) -> Tuple[List[Line], SongMetadata]:
    """Create placeholder content when no lyrics are found."""
    placeholder_word = Word(text="Lyrics not available", start_time=0.0, end_time=3.0)
    return [Line(words=[placeholder_word])], SongMetadata(
        singers=[], is_duet=False, title=title, artist=artist
    )


def _detect_and_apply_offset(
    vocals_path: str,
    line_timings: List[Tuple[float, str]],
    lyrics_offset: Optional[float],
) -> Tuple[List[Tuple[float, str]], float]:
    """Detect vocal offset and apply to line timings.

    Returns updated line_timings and the offset that was applied.
    """
    from ..alignment.alignment import detect_song_start

    detected_vocal_start = detect_song_start(vocals_path)
    anchor_time, _anchor_text, used_alternate_anchor = _select_offset_anchor_timing(
        line_timings
    )
    delta = detected_vocal_start - anchor_time

    logger.info(
        f"Vocal timing: audio_start={detected_vocal_start:.2f}s, "
        f"LRC_start={anchor_time:.2f}s, delta={delta:+.2f}s"
    )

    AUTO_OFFSET_MAX_ABS_SEC = 5.0

    offset = 0.0
    if lyrics_offset is not None:
        offset = lyrics_offset
    elif abs(delta) > 2.5 and abs(delta) <= AUTO_OFFSET_MAX_ABS_SEC:
        logger.warning(
            f"Detected vocal offset ({delta:+.2f}s) matches suspicious range (2.5-5.0s) - NOT auto-applying."
        )
    elif abs(delta) > 0.3 and abs(delta) <= 2.5:
        scale = 0.6 if used_alternate_anchor else 1.0
        offset = delta * scale
        logger.info(f"Auto-applying vocal offset: {offset:+.2f}s")
    elif abs(delta) > AUTO_OFFSET_MAX_ABS_SEC:
        logger.warning(
            "Large timing delta (%+.2fs) exceeds auto-offset clamp (%.1fs) - "
            "not auto-applying.",
            delta,
            AUTO_OFFSET_MAX_ABS_SEC,
        )

    if offset != 0.0:
        line_timings = [(ts + offset, text) for ts, text in line_timings]

    return line_timings, offset


def _distribute_word_timing_in_line(
    line: Line, line_start: float, next_line_start: float
) -> None:
    """Distribute word timing evenly within a line based on estimated duration."""
    word_count = len(line.words)
    if word_count == 0:
        return

    line_text = " ".join(w.text for w in line.words)
    estimated_duration = _estimate_singing_duration(line_text, word_count)

    gap_to_next = next_line_start - line_start
    max_duration = estimated_duration * _duration_cap_multiplier_for_line(
        line_text,
        word_count,
        gap_to_next,
        estimated_duration,
    )
    line_duration = min(gap_to_next, max_duration)
    line_duration = max(line_duration, word_count * 0.15)

    word_duration = (line_duration * 0.95) / word_count
    for j, word in enumerate(line.words):
        word.start_time = line_start + j * (line_duration / word_count)
        word.end_time = word.start_time + word_duration
        if j == word_count - 1:
            word.end_time = min(word.end_time, next_line_start - 0.05)


def _apply_timing_to_lines(
    lines: List[Line], line_timings: List[Tuple[float, str]]
) -> None:
    """Apply timing from line_timings to lines in place."""
    for i, line in enumerate(lines):
        if i < len(line_timings):
            line_start = line_timings[i][0]
            next_line_start = (
                line_timings[i + 1][0]
                if i + 1 < len(line_timings)
                else line_start + 5.0
            )
            _distribute_word_timing_in_line(line, line_start, next_line_start)


def _refine_timing_with_audio(
    lines: List[Line],
    vocals_path: str,
    line_timings: List[Tuple[float, str]],
    lrc_text: str,
    target_duration: Optional[int],
) -> List[Line]:
    """Refine word timing using audio onset detection and handle duration mismatch."""
    from ...refine import refine_word_timing
    from ..alignment.alignment import (
        _apply_adjustments_to_lines,
        adjust_timing_for_duration_mismatch,
    )
    from .sync import get_lrc_duration
    from ...audio_analysis import (
        extract_audio_features,
        _check_for_silence_in_range,
        _check_vocal_activity_in_range,
    )
    from ..whisper.whisper_alignment_refinement import (
        _pull_lines_forward_for_continuous_vocals,
    )
    from ..alignment.timing_evaluator_correction import (
        correct_line_timestamps,
        fix_spurious_gaps,
    )

    lines = refine_word_timing(lines, vocals_path)
    logger.debug("Word-level timing refined using vocals")

    lines = _apply_duration_mismatch_adjustment(
        lines,
        line_timings,
        vocals_path,
        lrc_text=lrc_text,
        target_duration=target_duration,
        get_lrc_duration=get_lrc_duration,
        adjust_timing_for_duration_mismatch=adjust_timing_for_duration_mismatch,
    )

    audio_features = extract_audio_features(vocals_path)
    if audio_features is None:
        logger.warning("Audio feature extraction failed; skipping onset-based fixes")
        return lines

    lines, spurious_gap_fixes = _compress_spurious_lrc_gaps(
        lines,
        line_timings,
        audio_features,
        _apply_adjustments_to_lines,
        _check_vocal_activity_in_range,
        _check_for_silence_in_range,
    )
    if spurious_gap_fixes:
        logger.info(
            f"Compressed {spurious_gap_fixes} large LRC gap(s) with vocals present"
        )

    needs_aggressive_correction = _has_large_continuous_vocal_gap(
        lines,
        audio_features,
        check_vocal_activity=_check_vocal_activity_in_range,
        check_for_silence=_check_for_silence_in_range,
    )
    max_correction = 15.0 if needs_aggressive_correction else 3.0
    lines, corrections = correct_line_timestamps(
        lines, audio_features, max_correction=max_correction
    )
    if corrections:
        logger.info(
            f"Adjusted {len(corrections)} line start(s) using audio onsets "
            f"(max_correction={max_correction:.1f}s)"
        )

    lines, pull_fixes = _pull_lines_forward_for_continuous_vocals(
        lines,
        audio_features,
    )
    if pull_fixes:
        logger.info(
            f"Pulled {pull_fixes} line(s) forward due to continuous vocals in gap"
        )

    lines, gap_fixes = fix_spurious_gaps(lines, audio_features)
    if gap_fixes:
        logger.info(f"Merged {len(gap_fixes)} spurious gap(s) based on vocals")

    return lines


def _apply_duration_mismatch_adjustment(
    lines: List[Line],
    line_timings: List[Tuple[float, str]],
    vocals_path: str,
    *,
    lrc_text: str,
    target_duration: Optional[int],
    get_lrc_duration,
    adjust_timing_for_duration_mismatch,
) -> List[Line]:
    lrc_duration = get_lrc_duration(lrc_text)
    if not target_duration or not lrc_duration:
        return lines
    if abs(target_duration - lrc_duration) <= 8:
        return lines
    logger.info(
        f"Duration mismatch: LRC={lrc_duration}s, "
        f"audio={target_duration}s (diff={target_duration - lrc_duration:+}s)"
    )
    return adjust_timing_for_duration_mismatch(
        lines,
        line_timings,
        vocals_path,
        lrc_duration=lrc_duration,
        audio_duration=target_duration,
    )


def _has_large_continuous_vocal_gap(
    lines: List[Line],
    audio_features,
    *,
    check_vocal_activity,
    check_for_silence,
) -> bool:
    for prev_line, next_line in zip(lines, lines[1:]):
        if not prev_line.words or not next_line.words:
            continue
        gap = next_line.start_time - prev_line.end_time
        if gap <= 4.0:
            continue
        activity = check_vocal_activity(
            prev_line.end_time, next_line.start_time, audio_features
        )
        has_silence = check_for_silence(
            prev_line.end_time,
            next_line.start_time,
            audio_features,
            min_silence_duration=0.5,
        )
        if activity > 0.6 and not has_silence:
            return True
    return False


def _compress_spurious_lrc_gaps(
    lines: List[Line],
    line_timings: List[Tuple[float, str]],
    audio_features,
    apply_adjustments,
    check_activity,
    check_silence,
) -> Tuple[List[Line], int]:
    """Compress large LRC gaps that contain continuous vocals."""
    if len(line_timings) < 2:
        return lines, 0

    adjustments = []
    cumulative_adj = 0.0
    fixes = 0

    for (start, _), (next_start, _) in zip(line_timings, line_timings[1:]):
        gap_start = start + cumulative_adj
        gap_end = next_start + cumulative_adj
        gap_duration = gap_end - gap_start
        if gap_duration < 8.0:
            continue

        activity = check_activity(gap_start, gap_end, audio_features)
        has_silence = check_silence(
            gap_start, gap_end, audio_features, min_silence_duration=0.5
        )
        if activity <= 0.6 or has_silence:
            continue

        target_gap = 0.5
        shift = gap_duration - target_gap
        if shift <= 0.5:
            continue

        cumulative_adj -= shift
        adjustments.append((next_start, cumulative_adj))
        fixes += 1

    if not adjustments:
        return lines, 0

    return apply_adjustments(lines, adjustments), fixes


def _apply_whisper_alignment(
    lines: List[Line],
    vocals_path: str,
    whisper_language: Optional[str],
    whisper_model: Optional[str],
    whisper_force_dtw: bool,
    whisper_aggressive: bool = False,
    whisper_temperature: float = 0.0,
    prefer_whisper_timing_map: bool = False,
    lenient_vocal_activity_threshold: float = 0.3,
    lenient_activity_bonus: float = 0.4,
    low_word_confidence_threshold: float = 0.5,
) -> Tuple[List[Line], List[str], Dict[str, float]]:
    """Apply Whisper alignment to lines. Returns (lines, fixes_list)."""
    from ...audio_analysis import extract_audio_features
    from ..whisper.whisper_integration import (
        align_lrc_text_to_whisper_timings,
        correct_timing_with_whisper,
        transcribe_vocals,
        use_whisper_integration_hooks,
    )

    audio_features = extract_audio_features(vocals_path)
    transcribe_cache: Dict[
        Tuple[str, Optional[str], str, bool, float],
        Tuple[object, object, str, str],
    ] = {}
    transcribe_cache_hits = 0
    transcribe_cache_misses = 0

    # Quality-first default: use Whisper large for alignment paths and rely on
    # cache reuse to keep iterative benchmark runs practical.
    default_model = "large"
    model_size = whisper_model or default_model

    def _memoized_transcribe_vocals(
        _vocals_path: str,
        language: Optional[str] = None,
        model_size: str = "base",
        aggressive: bool = False,
        temperature: float = 0.0,
    ) -> Tuple[object, object, str, str]:
        nonlocal transcribe_cache_hits, transcribe_cache_misses
        key = (_vocals_path, language, model_size, aggressive, float(temperature))
        cached = transcribe_cache.get(key)
        if cached is None:
            transcribe_cache_misses += 1
            miss_result = transcribe_vocals(
                _vocals_path,
                language=language,
                model_size=model_size,
                aggressive=aggressive,
                temperature=temperature,
            )
            transcribe_cache[key] = miss_result
            return miss_result
        else:
            transcribe_cache_hits += 1
        return cached

    with use_whisper_integration_hooks(
        transcribe_vocals_fn=_memoized_transcribe_vocals
    ):
        if prefer_whisper_timing_map:
            lines, whisper_fixes, whisper_metrics = align_lrc_text_to_whisper_timings(
                lines,
                vocals_path,
                language=whisper_language,
                model_size=model_size,
                aggressive=whisper_aggressive,
                temperature=whisper_temperature,
                audio_features=audio_features,
                lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
                lenient_activity_bonus=lenient_activity_bonus,
                low_word_confidence_threshold=low_word_confidence_threshold,
            )
            whisper_metrics = dict(whisper_metrics or {})
            whisper_metrics["fallback_map_attempted"] = 0.0
            whisper_metrics["fallback_map_selected"] = 0.0
            whisper_metrics["fallback_map_rejected"] = 0.0
            whisper_metrics["fallback_map_decision_code"] = 3.0
        else:
            baseline_lines = _clone_lines_for_fallback(lines)
            lines, whisper_fixes, whisper_metrics = correct_timing_with_whisper(
                lines,
                vocals_path,
                language=whisper_language,
                model_size=model_size,
                aggressive=whisper_aggressive,
                temperature=whisper_temperature,
                force_dtw=whisper_force_dtw,
                audio_features=audio_features,
                lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
                lenient_activity_bonus=lenient_activity_bonus,
                low_word_confidence_threshold=low_word_confidence_threshold,
            )
            whisper_metrics = dict(whisper_metrics or {})
            whisper_metrics["fallback_map_attempted"] = 0.0
            whisper_metrics["fallback_map_selected"] = 0.0
            whisper_metrics["fallback_map_rejected"] = 0.0
            whisper_metrics["fallback_map_decision_code"] = 0.0
            if _should_try_whisper_map_fallback(whisper_metrics):
                map_lines, map_fixes, map_metrics = align_lrc_text_to_whisper_timings(
                    baseline_lines,
                    vocals_path,
                    language=whisper_language,
                    model_size=model_size,
                    aggressive=whisper_aggressive,
                    temperature=whisper_temperature,
                    audio_features=audio_features,
                    lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
                    lenient_activity_bonus=lenient_activity_bonus,
                    low_word_confidence_threshold=low_word_confidence_threshold,
                )
                min_gain = 0.05
                decision = _choose_whisper_map_fallback(
                    whisper_metrics,
                    map_metrics,
                    min_gain=min_gain,
                    allow_coverage_promotion=_coverage_promotion_enabled(),
                )
                selected = bool(float(decision.get("selected", 0.0) or 0.0))
                if selected:
                    lines = map_lines
                    whisper_fixes = map_fixes
                    whisper_metrics = dict(map_metrics)
                    whisper_metrics["fallback_map_attempted"] = 1.0
                    whisper_metrics["fallback_map_selected"] = 1.0
                    whisper_metrics["fallback_map_rejected"] = 0.0
                    whisper_metrics["fallback_map_decision_code"] = float(
                        decision.get("decision_code", 1.0)
                    )
                    whisper_metrics["fallback_map_baseline_score"] = float(
                        decision.get("baseline_score", 0.0)
                    )
                    whisper_metrics["fallback_map_candidate_score"] = float(
                        decision.get("candidate_score", 0.0)
                    )
                    whisper_metrics["fallback_map_min_gain_required"] = float(
                        decision.get("min_gain_required", min_gain)
                    )
                    whisper_metrics["fallback_map_score_gain"] = float(
                        decision.get("score_gain", 0.0)
                    )
                else:
                    whisper_metrics["fallback_map_attempted"] = 1.0
                    whisper_metrics["fallback_map_selected"] = 0.0
                    whisper_metrics["fallback_map_rejected"] = 1.0
                    whisper_metrics["fallback_map_decision_code"] = float(
                        decision.get("decision_code", 2.0)
                    )
                    whisper_metrics["fallback_map_baseline_score"] = float(
                        decision.get("baseline_score", 0.0)
                    )
                    whisper_metrics["fallback_map_candidate_score"] = float(
                        decision.get("candidate_score", 0.0)
                    )
                    whisper_metrics["fallback_map_min_gain_required"] = float(
                        decision.get("min_gain_required", min_gain)
                    )
                    whisper_metrics["fallback_map_score_gain"] = float(
                        decision.get("score_gain", 0.0)
                    )
    whisper_metrics["local_transcribe_cache_hits"] = float(transcribe_cache_hits)
    whisper_metrics["local_transcribe_cache_misses"] = float(transcribe_cache_misses)
    if whisper_fixes:
        logger.info(f"Whisper aligned {len(whisper_fixes)} line(s)")
        for fix in whisper_fixes:
            logger.debug(f"  {fix}")
    if whisper_metrics:
        logger.info(
            "Whisper DTW metrics: "
            f"matched_ratio={whisper_metrics.get('matched_ratio', 0.0):.2f}, "
            f"avg_similarity={whisper_metrics.get('avg_similarity', 0.0):.2f}, "
            f"line_coverage={whisper_metrics.get('line_coverage', 0.0):.2f}"
        )
    return lines, whisper_fixes, whisper_metrics


def _romanize_lines(lines: List[Line]) -> None:
    """Apply romanization to non-Latin characters in lines."""
    for line in lines:
        for word in line.words:
            if any(ord(c) > 127 for c in word.text):
                word.text = romanize_line(word.text)
