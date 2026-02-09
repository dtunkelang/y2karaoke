"""Helper functions for lyrics processing."""

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from .models import Word, Line, SongMetadata
from .romanization import romanize_line
from .lrc import (
    parse_lrc_with_timing,
)

logger = logging.getLogger(__name__)


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
    from .alignment import detect_song_start

    detected_vocal_start = detect_song_start(vocals_path)
    first_lrc_time = line_timings[0][0]
    delta = detected_vocal_start - first_lrc_time

    logger.info(
        f"Vocal timing: audio_start={detected_vocal_start:.2f}s, "
        f"LRC_start={first_lrc_time:.2f}s, delta={delta:+.2f}s"
    )

    offset = 0.0
    if lyrics_offset is not None:
        offset = lyrics_offset
    elif abs(delta) > 0.3 and abs(delta) <= 30.0:
        if abs(delta) > 10.0:
            logger.warning(
                f"Large vocal offset ({delta:+.2f}s) - audio may have intro/speech not in LRC"
            )
        offset = delta
        logger.info(f"Auto-applying vocal offset: {offset:+.2f}s")
    elif abs(delta) > 30.0:
        logger.warning(
            f"Large timing delta ({delta:+.2f}s) - not auto-applying. "
            "Use --lyrics-offset to adjust manually."
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
    max_duration = estimated_duration * 1.3  # 30% buffer
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
    from .refine import refine_word_timing
    from .alignment import (
        _apply_adjustments_to_lines,
        adjust_timing_for_duration_mismatch,
    )
    from .sync import get_lrc_duration
    from .audio_analysis import extract_audio_features
    from .timing_evaluator import (
        _check_for_silence_in_range,
        _check_vocal_activity_in_range,
        _pull_lines_forward_for_continuous_vocals,
        correct_line_timestamps,
        fix_spurious_gaps,
    )

    lines = refine_word_timing(lines, vocals_path)
    logger.debug("Word-level timing refined using vocals")

    lrc_duration = get_lrc_duration(lrc_text)
    if target_duration and lrc_duration and abs(target_duration - lrc_duration) > 8:
        logger.info(
            f"Duration mismatch: LRC={lrc_duration}s, "
            f"audio={target_duration}s (diff={target_duration - lrc_duration:+}s)"
        )
        lines = adjust_timing_for_duration_mismatch(
            lines,
            line_timings,
            vocals_path,
            lrc_duration=lrc_duration,
            audio_duration=target_duration,
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

    needs_aggressive_correction = False
    for prev_line, next_line in zip(lines, lines[1:]):
        if not prev_line.words or not next_line.words:
            continue
        gap = next_line.start_time - prev_line.end_time
        if gap <= 4.0:
            continue
        activity = _check_vocal_activity_in_range(
            prev_line.end_time, next_line.start_time, audio_features
        )
        has_silence = _check_for_silence_in_range(
            prev_line.end_time,
            next_line.start_time,
            audio_features,
            min_silence_duration=0.5,
        )
        if activity > 0.6 and not has_silence:
            needs_aggressive_correction = True
            break

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
    from .audio_analysis import extract_audio_features
    from .timing_evaluator import (
        align_lrc_text_to_whisper_timings,
        correct_timing_with_whisper,
    )

    audio_features = extract_audio_features(vocals_path)

    # When we have no LRC timings, we lean heavily on Whisper for all timing
    # so using "large" is worth the extra compute (and results get cached).
    default_model = "large" if prefer_whisper_timing_map else "base"
    model_size = whisper_model or default_model
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
    else:
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
