"""Lyrics public API.

This module provides the main interface for lyrics fetching and processing:
- Fetches lyrics from Genius (canonical text + singer info)
- Gets LRC timing from syncedlyrics
- Aligns text to audio for word-level timing
- Applies romanization for non-Latin scripts
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import Word, Line, SongMetadata
from .romanization import romanize_line
from .lrc import (
    parse_lrc_timestamp,
    parse_lrc_with_timing,
    create_lines_from_lrc,
    split_long_lines,
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


__all__ = [
    # Models
    "Word",
    "Line",
    "SongMetadata",
    # Utilities
    "split_long_lines",
    "parse_lrc_timestamp",
    "parse_lrc_with_timing",
    "romanize_line",
    # API
    "LyricsProcessor",
    "get_lyrics",
    "get_lyrics_simple",
]


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
    from .timing_evaluator import (
        _check_for_silence_in_range,
        _check_vocal_activity_in_range,
        correct_line_timestamps,
        extract_audio_features,
        fix_spurious_gaps,
    )

    lines = refine_word_timing(lines, vocals_path)
    logger.debug("Word-level timing refined using vocals")

    lrc_duration = get_lrc_duration(lrc_text)
    if target_duration and lrc_duration and abs(target_duration - lrc_duration) > 10:
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
        _check_vocal_activity_in_range,
        _check_for_silence_in_range,
    )
    if pull_fixes:
        logger.info(
            f"Pulled {pull_fixes} line(s) forward due to continuous vocals in gap"
        )

    lines, gap_fixes = fix_spurious_gaps(lines, audio_features)
    if gap_fixes:
        logger.info(f"Merged {len(gap_fixes)} spurious gap(s) based on vocals")

    return lines


def _pull_lines_forward_for_continuous_vocals(
    lines: List[Line], audio_features, check_activity, check_silence
) -> Tuple[List[Line], int]:
    """Pull lines earlier when a long gap contains continuous vocal activity."""
    fixes = 0
    if len(lines) < 2:
        return lines, fixes

    onset_times = audio_features.onset_times
    if onset_times is None or len(onset_times) == 0:
        return lines, fixes

    for idx in range(1, len(lines)):
        prev_line = lines[idx - 1]
        line = lines[idx]
        if not prev_line.words or not line.words:
            continue

        gap = line.start_time - prev_line.end_time
        if gap <= 4.0:
            continue

        activity = check_activity(prev_line.end_time, line.start_time, audio_features)
        has_silence = check_silence(
            prev_line.end_time,
            line.start_time,
            audio_features,
            min_silence_duration=0.5,
        )
        if activity <= 0.6 or has_silence:
            continue

        candidate_onsets = onset_times[
            (onset_times >= prev_line.end_time) & (onset_times <= line.start_time)
        ]
        if len(candidate_onsets) == 0:
            continue

        new_start = float(candidate_onsets[0])
        new_start = max(new_start, prev_line.end_time + 0.05)
        shift = new_start - line.start_time
        if shift > -0.3:
            continue

        new_words = [
            Word(
                text=w.text,
                start_time=w.start_time + shift,
                end_time=w.end_time + shift,
                singer=w.singer,
            )
            for w in line.words
        ]
        lines[idx] = Line(words=new_words, singer=line.singer)
        fixes += 1

    return lines, fixes


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
    whisper_model: str,
    whisper_force_dtw: bool,
) -> Tuple[List[Line], List[str], Dict[str, float]]:
    """Apply Whisper alignment to lines. Returns (lines, fixes_list)."""
    from .timing_evaluator import correct_timing_with_whisper

    lines, whisper_fixes, whisper_metrics = correct_timing_with_whisper(
        lines,
        vocals_path,
        language=whisper_language,
        model_size=whisper_model,
        force_dtw=whisper_force_dtw,
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


def _create_lines_from_whisper(
    transcription: List["TranscriptionSegment"],
) -> List[Line]:
    """Create Line objects directly from Whisper transcription."""
    from .models import Line, Word

    lines: List[Line] = []
    for segment in transcription:
        if segment is None:
            continue
        words: List[Word] = []
        if segment.words:
            for w in segment.words:
                text = (w.text or "").strip()
                if not text:
                    continue
                words.append(
                    Word(
                        text=text,
                        start_time=float(w.start),
                        end_time=float(w.end),
                        singer="",
                    )
                )
        else:
            tokens = [t for t in segment.text.strip().split() if t]
            if tokens:
                duration = max(segment.end - segment.start, 0.2)
                spacing = duration / len(tokens)
                for i, token in enumerate(tokens):
                    start = segment.start + i * spacing
                    end = start + spacing * 0.9
                    words.append(
                        Word(text=token, start_time=start, end_time=end, singer="")
                    )
        if not words:
            continue
        lines.append(Line(words=words))
    return lines


def _map_lrc_lines_to_whisper_segments(
    lines: List[Line],
    transcription: List["TranscriptionSegment"],
    language: str,
    min_similarity: float = 0.35,
    lookahead: int = 6,
) -> Tuple[List[Line], int, List[str]]:
    """Map LRC lines onto Whisper segment timing without reordering."""
    from .models import Line, Word
    from .timing_evaluator import _phonetic_similarity

    if not lines or not transcription:
        return lines, 0, []

    sorted_segments = sorted(transcription, key=lambda s: s.start)
    adjusted: List[Line] = []
    fixes = 0
    issues: List[str] = []
    seg_idx = 0
    last_end = None
    min_gap = 0.01
    prev_text = None

    for line in lines:
        if not line.words:
            adjusted.append(line)
            continue

        best_idx = None
        best_sim = 0.0
        window_end = min(seg_idx + lookahead, len(sorted_segments))
        for idx in range(seg_idx, window_end):
            seg = sorted_segments[idx]
            sim = _phonetic_similarity(line.text, seg.text, language)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        text_norm = line.text.strip().lower() if line.text else ""
        gap_required = 0.21 if prev_text and text_norm == prev_text else min_gap

        if best_idx is None:
            if last_end is not None and line.start_time < last_end + gap_required:
                duration = max(line.end_time - line.start_time, 0.2)
                offset = (last_end + gap_required) - line.start_time
                new_words = [
                    Word(
                        text=w.text,
                        start_time=w.start_time + offset,
                        end_time=w.end_time + offset,
                        singer=w.singer,
                    )
                    for w in line.words
                ]
                adjusted.append(Line(words=new_words, singer=line.singer))
                last_end = new_words[-1].end_time if new_words else last_end
            else:
                adjusted.append(line)
                last_end = line.end_time
            prev_text = text_norm
            continue

        seg = sorted_segments[best_idx]
        if last_end is not None and seg.start < last_end + gap_required:
            next_idx = None
            for idx in range(best_idx, window_end):
                if sorted_segments[idx].start >= last_end + gap_required:
                    next_idx = idx
                    break
            if next_idx is not None:
                best_idx = next_idx
                seg = sorted_segments[best_idx]

        duration = max(seg.end - seg.start, 0.2)
        spacing = duration / max(len(line.words), 1)
        new_words = []
        for i, word in enumerate(line.words):
            start = seg.start + i * spacing
            end = start + spacing * 0.9
            new_words.append(
                Word(
                    text=word.text,
                    start_time=start,
                    end_time=end,
                    singer=word.singer,
                )
            )
        if (
            last_end is not None
            and new_words
            and new_words[0].start_time < last_end + gap_required
        ):
            shift = (last_end + gap_required) - new_words[0].start_time
            new_words = [
                Word(
                    text=w.text,
                    start_time=w.start_time + shift,
                    end_time=w.end_time + shift,
                    singer=w.singer,
                )
                for w in new_words
            ]
        adjusted.append(Line(words=new_words, singer=line.singer))
        fixes += 1
        if best_sim < min_similarity:
            issues.append(
                f"Low similarity mapping for line '{line.text[:30]}...' (sim={best_sim:.2f})"
            )
        last_end = new_words[-1].end_time if new_words else last_end
        seg_idx = best_idx + 1
        prev_text = text_norm

    return adjusted, fixes, issues


def _apply_singer_info(
    lines: List[Line],
    genius_lines: List[Tuple[str, str]],
    metadata: SongMetadata,
) -> None:
    """Apply singer info from Genius to lines for duets."""
    for i, line in enumerate(lines):
        if i < len(genius_lines):
            _, singer_name = genius_lines[i]
            singer_id = metadata.get_singer_id(singer_name)
            line.singer = singer_id
            for word in line.words:
                word.singer = singer_id


def _detect_offset_with_issues(
    vocals_path: str,
    line_timings: List[Tuple[float, str]],
    lyrics_offset: Optional[float],
    issues: List[str],
) -> Tuple[List[Tuple[float, str]], float]:
    """Detect vocal offset, track issues for quality report.

    Returns (updated_line_timings, offset_applied).
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
            logger.warning(f"Large vocal offset ({delta:+.2f}s) - audio may have intro")
            issues.append(f"Large vocal offset ({delta:+.2f}s)")
        offset = delta
        logger.info(f"Auto-applying vocal offset: {offset:+.2f}s")
    elif abs(delta) > 30.0:
        logger.warning(f"Large timing delta ({delta:+.2f}s) - not auto-applying.")
        issues.append(f"Large timing delta ({delta:+.2f}s) not applied")

    if offset != 0.0:
        line_timings = [(ts + offset, text) for ts, text in line_timings]

    return line_timings, offset


def _refine_timing_with_quality(
    lines: List[Line],
    vocals_path: str,
    line_timings: List[Tuple[float, str]],
    lrc_text: str,
    target_duration: Optional[int],
    issues: List[str],
) -> Tuple[List[Line], str]:
    """Refine timing and track issues. Returns (lines, alignment_method)."""
    from .refine import refine_word_timing
    from .alignment import adjust_timing_for_duration_mismatch
    from .sync import get_lrc_duration

    lines = refine_word_timing(lines, vocals_path)
    alignment_method = "onset_refined"
    logger.debug("Word-level timing refined using vocals")

    lrc_duration = get_lrc_duration(lrc_text)
    if target_duration and lrc_duration and abs(target_duration - lrc_duration) > 10:
        logger.info(f"Duration mismatch: LRC={lrc_duration}s, audio={target_duration}s")
        issues.append(
            f"Duration mismatch: LRC={lrc_duration}s vs audio={target_duration}s"
        )
        lines = adjust_timing_for_duration_mismatch(
            lines,
            line_timings,
            vocals_path,
            lrc_duration=lrc_duration,
            audio_duration=target_duration,
        )

    return lines, alignment_method


def _calculate_quality_score(quality_report: dict) -> float:
    """Calculate overall quality score from report components."""
    # Base score on lyrics quality if available
    if quality_report["lyrics_quality"]:
        base_score = quality_report["lyrics_quality"].get("quality_score", 50.0)
    else:
        base_score = 30.0  # Genius fallback

    # Adjust for alignment method
    method_bonus = {
        "whisper_hybrid": 10,
        "onset_refined": 5,
        "lrc_only": 0,
        "genius_fallback": -20,
        "none": -50,
    }
    base_score += method_bonus.get(quality_report["alignment_method"], 0)

    # Adjust for issues
    base_score -= len(quality_report["issues"]) * 5

    return max(0.0, min(100.0, base_score))


def _fetch_lrc_text_and_timings(
    title: str,
    artist: str,
    target_duration: Optional[int] = None,
    vocals_path: Optional[str] = None,
    evaluate_sources: bool = False,
    filter_promos: bool = True,
) -> Tuple[Optional[str], Optional[List[Tuple[float, str]]], str]:
    """Fetch raw LRC text and parsed timings from available sources.

    Args:
        title: Song title
        artist: Artist name
        target_duration: Expected track duration in seconds (for validation)
        vocals_path: Path to vocals audio (for timing evaluation)
        evaluate_sources: If True, compare all sources and select best based on timing

    Returns:
        Tuple of (lrc_text, parsed_timings, source_name)
    """
    try:
        # If evaluation is requested and we have vocals, compare all sources
        if evaluate_sources and vocals_path:
            from .timing_evaluator import select_best_source

            lrc_text, source, report = select_best_source(
                title, artist, vocals_path, target_duration
            )
            if lrc_text and source:
                lines = parse_lrc_with_timing(
                    lrc_text, title, artist, filter_promos=filter_promos
                )
                score_str = f" (score: {report.overall_score:.1f})" if report else ""
                logger.info(f"Selected best source: {source}{score_str}")
                return lrc_text, lines, source
            # Fall through to standard fetch if evaluation fails

        if target_duration:
            # Use duration-aware fetch to find LRC matching target
            from .sync import fetch_lyrics_for_duration

            lrc_text, is_synced, source, lrc_duration = fetch_lyrics_for_duration(
                title, artist, target_duration, tolerance=20
            )
            if lrc_text and is_synced:
                lines = parse_lrc_with_timing(
                    lrc_text, title, artist, filter_promos=filter_promos
                )
                logger.debug(
                    f"Got {len(lines)} LRC lines from {source} (duration: {lrc_duration}s)"
                )
                return lrc_text, lines, source
            else:
                logger.debug("No duration-matched LRC available")
                return None, None, ""
        else:
            # Fallback to standard fetch without duration validation
            from .sync import fetch_lyrics_multi_source

            lrc_text, is_synced, source = fetch_lyrics_multi_source(title, artist)
            if lrc_text and is_synced:
                lines = parse_lrc_with_timing(
                    lrc_text, title, artist, filter_promos=filter_promos
                )
                logger.debug(f"Got {len(lines)} LRC lines from {source}")
                return lrc_text, lines, source
            else:
                logger.debug(f"No synced LRC available from {source}")
                return None, None, ""
    except Exception as e:
        logger.warning(f"LRC fetch failed: {e}")
        return None, None, ""


def get_lyrics_simple(
    title: str,
    artist: str,
    vocals_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    lyrics_offset: Optional[float] = None,
    romanize: bool = True,
    filter_promos: bool = True,
    target_duration: Optional[int] = None,
    evaluate_sources: bool = False,
    use_whisper: bool = False,
    whisper_only: bool = False,
    whisper_map_lrc: bool = False,
    whisper_language: Optional[str] = None,
    whisper_model: str = "base",
    whisper_force_dtw: bool = False,
) -> Tuple[List[Line], Optional[SongMetadata]]:
    """Simplified lyrics pipeline favoring LRC over Genius.

    Pipeline:
    1. Try to fetch LRC lyrics with timing (preferred source)
    2. If no LRC, fall back to Genius lyrics
    3. Detect vocal offset and align timing
    4. Create Line objects with word-level timing
    5. Refine timing using audio onset detection
    6. Optionally align to Whisper transcription for severely broken LRC
    7. Apply romanization if needed

    Args:
        title: Song title
        artist: Artist name
        vocals_path: Path to vocals audio (for timing refinement)
        cache_dir: Cache directory (unused, for API compatibility)
        lyrics_offset: Manual timing offset in seconds (auto-detected if None)
        romanize: Whether to romanize non-Latin scripts
        target_duration: Expected track duration in seconds (for LRC validation)
        evaluate_sources: If True, compare all lyrics sources and select best
                         based on timing alignment with audio
        use_whisper: If True, use Whisper transcription to align lyrics timing
        whisper_only: If True, generate lines directly from Whisper (no LRC/Genius)
        whisper_map_lrc: If True, map LRC text onto Whisper timing without shifting segments
        whisper_language: Language code for Whisper (auto-detected if None)
        whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')

    Returns:
        Tuple of (lines, metadata)
    """
    from .genius import fetch_genius_lyrics_with_singers

    # Whisper-only mode: generate lines directly from transcription.
    if whisper_only:
        if not vocals_path:
            logger.warning(
                "Whisper-only mode requires vocals; using placeholder lyrics"
            )
            return _create_no_lyrics_placeholder(title, artist)
        from .timing_evaluator import transcribe_vocals

        transcription, _, detected_lang = transcribe_vocals(
            vocals_path, whisper_language, whisper_model
        )
        if not transcription:
            logger.warning(
                "No Whisper transcription available; using placeholder lyrics"
            )
            return _create_no_lyrics_placeholder(title, artist)
        lines = _create_lines_from_whisper(transcription)
        metadata = SongMetadata(
            singers=[],
            is_duet=False,
            title=title,
            artist=artist,
        )
        if romanize:
            _romanize_lines(lines)
        logger.debug(f"Returning {len(lines)} lines from Whisper-only mode")
        return lines, metadata

    # 1. Try LRC first (preferred source), with duration validation if provided
    logger.debug(
        f"Fetching LRC lyrics... (target_duration={target_duration}, "
        f"evaluate={evaluate_sources})"
    )
    lrc_text, line_timings, source = _fetch_lrc_text_and_timings(
        title, artist, target_duration, vocals_path, evaluate_sources, filter_promos
    )

    # 2. Fetch Genius as fallback or for singer info
    genius_lines, metadata = None, None
    if not line_timings:
        logger.debug("No LRC found, fetching lyrics from Genius...")
        genius_lines, metadata = fetch_genius_lyrics_with_singers(title, artist)
        if not genius_lines:
            logger.warning("No lyrics found from any source, using placeholder")
            return _create_no_lyrics_placeholder(title, artist)
    else:
        # Still fetch Genius for singer/duet metadata only
        genius_lines, metadata = fetch_genius_lyrics_with_singers(title, artist)

    # 3. Apply vocal offset if available
    if vocals_path and line_timings:
        line_timings, _ = _detect_and_apply_offset(
            vocals_path, line_timings, lyrics_offset
        )

    # 4. Create Line objects
    if line_timings and lrc_text:
        lines = create_lines_from_lrc(
            lrc_text,
            romanize=False,
            title=title,
            artist=artist,
            filter_promos=filter_promos,
        )
        _apply_timing_to_lines(lines, line_timings)

        # 5. Refine word timing using audio
        if vocals_path and len(line_timings) > 1:
            lines = _refine_timing_with_audio(
                lines, vocals_path, line_timings, lrc_text, target_duration
            )

        # 5b. Optionally use Whisper for more accurate alignment
        if vocals_path and use_whisper:
            try:
                lines, _, _ = _apply_whisper_alignment(
                    lines,
                    vocals_path,
                    whisper_language,
                    whisper_model,
                    whisper_force_dtw,
                )
            except Exception as e:
                logger.warning(f"Whisper alignment failed: {e}")
        elif vocals_path and whisper_map_lrc:
            try:
                from .timing_evaluator import (
                    _whisper_lang_to_epitran,
                    transcribe_vocals,
                )

                transcription, _, detected_lang = transcribe_vocals(
                    vocals_path, whisper_language, whisper_model
                )
                if transcription:
                    lang = _whisper_lang_to_epitran(detected_lang)
                    lines, mapped, issues = _map_lrc_lines_to_whisper_segments(
                        lines, transcription, lang
                    )
                    if mapped:
                        logger.info(f"Mapped {mapped} LRC line(s) onto Whisper timing")
                    for issue in issues:
                        logger.debug(issue)
            except Exception as e:
                logger.warning(f"Whisper LRC mapping failed: {e}")
    else:
        # Fallback: use Genius text with evenly spaced lines
        if genius_lines:
            text_lines = [text for text, _ in genius_lines if text.strip()]
            lrc_text = "\n".join(text_lines)
        else:
            lrc_text = ""
        lines = create_lines_from_lrc(
            lrc_text,
            romanize=romanize,
            title=title,
            artist=artist,
            filter_promos=filter_promos,
        )

    # 6. Romanize
    if romanize:
        _romanize_lines(lines)

    # Apply singer info for duets (from Genius metadata)
    if metadata and metadata.is_duet and genius_lines:
        _apply_singer_info(lines, genius_lines, metadata)

    logger.debug(f"Returning {len(lines)} lines")
    return lines, metadata


def get_lyrics_with_quality(
    title: str,
    artist: str,
    vocals_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    lyrics_offset: Optional[float] = None,
    romanize: bool = True,
    filter_promos: bool = True,
    target_duration: Optional[int] = None,
    evaluate_sources: bool = False,
    use_whisper: bool = False,
    whisper_only: bool = False,
    whisper_map_lrc: bool = False,
    whisper_language: Optional[str] = None,
    whisper_model: str = "base",
    whisper_force_dtw: bool = False,
) -> Tuple[List[Line], Optional[SongMetadata], dict]:
    """Get lyrics with quality report.

    Same as get_lyrics_simple but also returns a quality report dict.

    Returns:
        Tuple of (lines, metadata, quality_report)
        quality_report contains:
        - lyrics_quality: dict from get_lyrics_quality_report
        - alignment_method: str describing how timing was aligned
        - whisper_used: bool
        - whisper_corrections: int (if whisper used)
        - total_lines: int
        - overall_score: float 0-100
        - issues: list of str
    """
    from .sync import get_lyrics_quality_report

    quality_report = {
        "lyrics_quality": {},
        "alignment_method": "none",
        "whisper_used": False,
        "whisper_corrections": 0,
        "whisper_requested": use_whisper or whisper_only or whisper_map_lrc,
        "whisper_force_dtw": whisper_force_dtw,
        "total_lines": 0,
        "overall_score": 0.0,
        "issues": [],
        "source": "",
    }

    if whisper_only:
        lines, metadata = get_lyrics_simple(
            title=title,
            artist=artist,
            vocals_path=vocals_path,
            cache_dir=cache_dir,
            lyrics_offset=lyrics_offset,
            romanize=romanize,
            filter_promos=filter_promos,
            target_duration=target_duration,
            evaluate_sources=evaluate_sources,
            use_whisper=use_whisper,
            whisper_only=True,
            whisper_map_lrc=whisper_map_lrc,
            whisper_language=whisper_language,
            whisper_model=whisper_model,
            whisper_force_dtw=whisper_force_dtw,
        )
        quality_report["alignment_method"] = "whisper_only"
        quality_report["whisper_used"] = bool(lines)
        quality_report["total_lines"] = len(lines)
        quality_report["overall_score"] = 50.0 if lines else 0.0
        if not lines:
            quality_report["issues"].append("Whisper-only mode produced no lines")
        return lines, metadata, quality_report

    if whisper_map_lrc:
        quality_report["alignment_method"] = "whisper_map_lrc"

    # 1. Try LRC first
    logger.debug(
        f"Fetching LRC lyrics... (target_duration={target_duration}, "
        f"evaluate={evaluate_sources})"
    )
    lrc_text, line_timings, source = _fetch_lrc_text_and_timings(
        title, artist, target_duration, vocals_path, evaluate_sources, filter_promos
    )
    quality_report["source"] = source

    # 2. Fetch Genius as fallback or for singer info
    genius_lines, metadata = _fetch_genius_with_quality_tracking(
        line_timings, title, artist, quality_report
    )
    if genius_lines is None and not line_timings:
        # No lyrics from any source - return placeholder
        lines, meta = _create_no_lyrics_placeholder(title, artist)
        return lines, meta, quality_report

    # Get LRC quality report if we have LRC
    if line_timings and lrc_text:
        quality_report["lyrics_quality"] = get_lyrics_quality_report(
            lrc_text, source, target_duration
        )

    # Cast issues to the expected type for helper functions
    issues_list: List[str] = quality_report["issues"]  # type: ignore[assignment]

    # 3. Apply vocal offset if available
    if vocals_path and line_timings:
        line_timings, _ = _detect_offset_with_issues(
            vocals_path, line_timings, lyrics_offset, issues_list
        )

    # 4. Create Line objects and apply timing
    if line_timings and lrc_text:
        lines = create_lines_from_lrc(
            lrc_text,
            romanize=False,
            title=title,
            artist=artist,
            filter_promos=filter_promos,
        )
        quality_report["alignment_method"] = "lrc_only"
        _apply_timing_to_lines(lines, line_timings)

        # 5. Refine word timing using audio
        if vocals_path and len(line_timings) > 1:
            lines, method = _refine_timing_with_quality(
                lines,
                vocals_path,
                line_timings,
                lrc_text,
                target_duration,
                issues_list,
            )
            quality_report["alignment_method"] = method

            # 5b. Apply Whisper if requested
            if use_whisper:
                lines, quality_report = _apply_whisper_with_quality(
                    lines,
                    vocals_path,
                    whisper_language,
                    whisper_model,
                    whisper_force_dtw,
                    quality_report,
                )
            elif whisper_map_lrc:
                try:
                    from .timing_evaluator import (
                        _whisper_lang_to_epitran,
                        transcribe_vocals,
                    )

                    transcription, _, detected_lang = transcribe_vocals(
                        vocals_path, whisper_language, whisper_model
                    )
                    if transcription:
                        lang = _whisper_lang_to_epitran(detected_lang)
                        lines, mapped, issues = _map_lrc_lines_to_whisper_segments(
                            lines, transcription, lang
                        )
                        if mapped:
                            quality_report["alignment_method"] = "whisper_map_lrc"
                            quality_report["whisper_used"] = True
                            quality_report["whisper_corrections"] = mapped
                        for issue in issues:
                            issues_list.append(issue)
                except Exception as e:
                    logger.warning(f"Whisper LRC mapping failed: {e}")
                    issues_list.append(f"Whisper LRC mapping failed: {e}")
    else:
        # Fallback: use Genius text
        if genius_lines:
            text_lines = [text for text, _ in genius_lines if text.strip()]
            fallback_text = "\n".join(text_lines)
        else:
            fallback_text = ""
        lines = create_lines_from_lrc(
            fallback_text,
            romanize=romanize,
            title=title,
            artist=artist,
            filter_promos=filter_promos,
        )

    # 6. Romanize
    if romanize:
        _romanize_lines(lines)

    # Apply singer info for duets
    if metadata and metadata.is_duet and genius_lines:
        _apply_singer_info(lines, genius_lines, metadata)

    # Calculate overall quality score
    quality_report["total_lines"] = len(lines)
    quality_report["overall_score"] = _calculate_quality_score(quality_report)

    logger.debug(
        f"Returning {len(lines)} lines (quality: {quality_report['overall_score']:.0f})"
    )
    return lines, metadata, quality_report


def _fetch_genius_with_quality_tracking(
    line_timings: Optional[List[Tuple[float, str]]],
    title: str,
    artist: str,
    quality_report: dict,
) -> Tuple[Optional[List[Tuple[str, str]]], Optional[SongMetadata]]:
    """Fetch Genius lyrics with quality tracking for fallback case."""
    from .genius import fetch_genius_lyrics_with_singers

    if not line_timings:
        logger.debug("No LRC found, fetching lyrics from Genius...")
        genius_lines, metadata = fetch_genius_lyrics_with_singers(title, artist)
        quality_report["alignment_method"] = "genius_fallback"
        quality_report["issues"].append("No synced LRC found, using Genius text")
        if not genius_lines:
            logger.warning("No lyrics found from any source, using placeholder")
            quality_report["issues"].append("No lyrics found from any source")
            quality_report["overall_score"] = 0.0
            return None, None
        return genius_lines, metadata
    else:
        return fetch_genius_lyrics_with_singers(title, artist)


def _apply_whisper_with_quality(
    lines: List[Line],
    vocals_path: str,
    whisper_language: Optional[str],
    whisper_model: str,
    whisper_force_dtw: bool,
    quality_report: dict,
) -> Tuple[List[Line], dict]:
    """Apply Whisper alignment and update quality report."""
    try:
        lines, whisper_fixes, whisper_metrics = _apply_whisper_alignment(
            lines, vocals_path, whisper_language, whisper_model, whisper_force_dtw
        )
        quality_report["whisper_used"] = True
        quality_report["whisper_corrections"] = len(whisper_fixes)
        if whisper_metrics:
            quality_report["dtw_metrics"] = whisper_metrics
        if whisper_fixes:
            quality_report["alignment_method"] = "whisper_hybrid"
    except Exception as e:
        logger.warning(f"Whisper alignment failed: {e}")
        quality_report["issues"].append(f"Whisper alignment failed: {e}")
    return lines, quality_report


class LyricsProcessor:
    """High-level lyrics processor with caching support."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "karaoke")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_lyrics(
        self,
        title: Optional[str] = None,
        artist: Optional[str] = None,
        romanize: bool = True,
        **kwargs,
    ) -> Tuple[List[Line], Optional[SongMetadata]]:
        """Get lyrics for a song.

        Args:
            title: Song title
            artist: Artist name
            romanize: Whether to romanize non-Latin scripts
            **kwargs: Additional options (vocals_path, lyrics_offset)

        Returns:
            Tuple of (lines, metadata)
        """
        if not title or not artist:
            placeholder_line = Line(words=[])
            placeholder_metadata = SongMetadata(
                singers=[],
                is_duet=False,
                title=title or "Unknown",
                artist=artist or "Unknown",
            )
            return [placeholder_line], placeholder_metadata

        lines, metadata = get_lyrics_simple(
            title=title,
            artist=artist,
            vocals_path=kwargs.get("vocals_path"),
            cache_dir=str(self.cache_dir),
            lyrics_offset=kwargs.get("lyrics_offset"),
            romanize=romanize,
        )
        return lines, metadata


def get_lyrics(
    title: str,
    artist: str,
    vocals_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[List[Line], Optional[SongMetadata]]:
    """Get lyrics for a song (convenience function).

    Args:
        title: Song title
        artist: Artist name
        vocals_path: Path to vocals audio (optional)
        cache_dir: Cache directory (optional)

    Returns:
        Tuple of (lines, metadata)
    """
    return get_lyrics_simple(
        title=title,
        artist=artist,
        vocals_path=vocals_path,
        cache_dir=cache_dir,
    )
