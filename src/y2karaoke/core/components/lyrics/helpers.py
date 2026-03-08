"""Helper functions for lyrics processing."""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ...models import Word, Line, SongMetadata
from ...romanization import romanize_line
from .lrc import (
    parse_lrc_with_timing,
    create_lines_from_lrc,
    create_lines_from_lrc_timings,
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
    if word_count < 6:
        return 1.3
    if gap_to_next < max(4.5, estimated_duration * 1.8):
        return 1.3
    punctuation_pauses = (
        line_text.count(",") + line_text.count(";") + line_text.count(":")
    )
    if punctuation_pauses == 0:
        return 1.3
    lower_text = line_text.lower()
    if not re.search(r"\b(oh|ooh|ah|hey|yeah)\b", lower_text):
        return 1.3
    if re.search(r"\((oh|ooh|ah|hey|yeah)\)", lower_text):
        return 1.3
    return 2.7


def _extract_text_lines_from_lrc(lrc_text: str) -> List[str]:
    timed = parse_lrc_with_timing(lrc_text, "", "", filter_promos=False)
    if timed:
        return [text for _t, text in timed if text.strip()]
    lines: List[str] = []
    for raw in lrc_text.splitlines():
        line = re.sub(r"\[[0-9:.]+\]", "", raw).strip()
        if line:
            lines.append(line)
    return lines


def _create_lines_from_plain_text(text_lines: List[str]) -> List[Line]:
    if not text_lines:
        return []

    lines: List[Line] = []
    current_time = 0.0
    for text in text_lines:
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


def _clean_text_lines(lines: List[str]) -> List[str]:
    cleaned = []
    for line in lines:
        line = re.sub(r"\s+", " ", line).strip()
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
    first_lrc_time = line_timings[0][0]
    delta = detected_vocal_start - first_lrc_time

    logger.info(
        f"Vocal timing: audio_start={detected_vocal_start:.2f}s, "
        f"LRC_start={first_lrc_time:.2f}s, delta={delta:+.2f}s"
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
        offset = delta
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
