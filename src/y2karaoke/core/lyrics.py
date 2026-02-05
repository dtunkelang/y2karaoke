"""Lyrics public API.

This module provides the main interface for lyrics fetching and processing:
- Fetches lyrics from Genius (canonical text + singer info)
- Gets LRC timing from syncedlyrics
- Aligns text to audio for word-level timing
- Applies romanization for non-Latin scripts
"""

from __future__ import annotations

import logging
import re
from statistics import median
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from .models import Word, Line, SongMetadata
from .romanization import romanize_line
from .lrc import (
    parse_lrc_timestamp,
    parse_lrc_with_timing,
    create_lines_from_lrc,
    create_lines_from_lrc_timings,
    split_long_lines,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .timing_evaluator import TranscriptionSegment, TranscriptionWord


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
    whisper_model: Optional[str],
    whisper_force_dtw: bool,
    whisper_aggressive: bool = False,
    prefer_whisper_timing_map: bool = False,
) -> Tuple[List[Line], List[str], Dict[str, float]]:
    """Apply Whisper alignment to lines. Returns (lines, fixes_list)."""
    from .timing_evaluator import (
        align_lrc_text_to_whisper_timings,
        correct_timing_with_whisper,
    )

    model_size = whisper_model or "base"
    try:
        if prefer_whisper_timing_map:
            lines, whisper_fixes, whisper_metrics = align_lrc_text_to_whisper_timings(
                lines,
                vocals_path,
                language=whisper_language,
                model_size=model_size,
                aggressive=whisper_aggressive,
            )
        else:
            lines, whisper_fixes, whisper_metrics = correct_timing_with_whisper(
                lines,
                vocals_path,
                language=whisper_language,
                model_size=model_size,
                aggressive=whisper_aggressive,
                force_dtw=whisper_force_dtw,
            )
    except TypeError:
        if prefer_whisper_timing_map:
            lines, whisper_fixes, whisper_metrics = align_lrc_text_to_whisper_timings(
                lines,
                vocals_path,
                language=whisper_language,
                model_size=model_size,
                aggressive=whisper_aggressive,
            )
        else:
            lines, whisper_fixes, whisper_metrics = correct_timing_with_whisper(
                lines,
                vocals_path,
                language=whisper_language,
                model_size=model_size,
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


def _norm_token(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w']+", "", text)
    return text


def _build_whisper_word_list(
    transcription: List[TranscriptionSegment],
) -> List[TranscriptionWord]:
    words: List[TranscriptionWord] = []
    for seg in sorted(transcription, key=lambda s: s.start):
        if not seg.words:
            continue
        for t_word in seg.words:
            if t_word.start is not None:
                words.append(t_word)
    words.sort(key=lambda w: w.start)
    return words


def _select_window_sequence(
    window_words: List[TranscriptionWord],
    line_text: str,
    language: str,
    target_len: int,
    first_token: Optional[str],
    phonetic_similarity,
) -> List[TranscriptionWord]:
    if not window_words:
        return []
    best_seq = window_words
    best_sim = -1.0
    best_score = -1.0
    lengths = [target_len - 1, target_len, target_len + 1]
    lengths = [length for length in lengths if length > 0]
    max_start = len(window_words)
    start_candidates = range(max_start)
    if first_token:
        anchor_idx = None
        anchor_sim = 0.0
        for idx, w in enumerate(window_words):
            sim = phonetic_similarity(first_token, w.text, language)
            if sim > anchor_sim:
                anchor_sim = sim
                anchor_idx = idx
        if anchor_idx is not None and anchor_sim >= 0.6:
            start_candidates = range(
                max(anchor_idx - 1, 0), min(anchor_idx + 1, max_start - 1) + 1
            )
    for start_idx in start_candidates:
        for length in lengths:
            end_idx = start_idx + length
            if end_idx > len(window_words):
                continue
            seq = window_words[start_idx:end_idx]
            seq_text = " ".join(w.text for w in seq)
            sim = phonetic_similarity(line_text, seq_text, language)
            length_penalty = 0.02 * abs(length - target_len)
            score = sim - length_penalty
            if score > best_score:
                best_score = score
                best_sim = sim
                best_seq = seq
    if best_sim < 0.2:
        return window_words
    return best_seq


def _slots_from_sequence(
    seq_words: List[TranscriptionWord], seq_end: Optional[float]
) -> Optional[Tuple[List[float], List[float]]]:
    if not seq_words:
        return None
    slots: List[float] = []
    spokens: List[float] = []
    for i, w in enumerate(seq_words):
        if w.start is None:
            return None
        if i + 1 < len(seq_words) and seq_words[i + 1].start is not None:
            slot = seq_words[i + 1].start - w.start
        elif w.end is not None and w.end > w.start:
            slot = w.end - w.start
        elif seq_end is not None and seq_end > w.start:
            slot = seq_end - w.start
        else:
            slot = 0.02
        slot = max(slot, 0.02)
        if w.end is not None and w.end > w.start:
            spoken = min(w.end - w.start, slot)
        else:
            spoken = slot * 0.85
        slots.append(slot)
        spokens.append(max(spoken, 0.02))
    return slots, spokens


def _resample_slots_to_line(
    slots: List[float],
    spokens: List[float],
    line_length: int,
) -> Tuple[List[float], List[float]]:
    if not slots or line_length <= 0:
        return [], []
    if line_length == 1:
        slot = median(slots)
        spoken = median(spokens) if spokens else slot * 0.85
        return [max(slot, 0.02)], [max(spoken, 0.02)]
    seg_count = len(slots)
    resampled: List[float] = []
    resampled_spoken: List[float] = []
    for i in range(line_length):
        pos = i * (seg_count - 1) / (line_length - 1)
        lo = int(pos)
        hi = min(lo + 1, seg_count - 1)
        frac = pos - lo
        slot = slots[lo] * (1 - frac) + slots[hi] * frac
        spoken = spokens[lo] * (1 - frac) + spokens[hi] * frac
        resampled.append(max(slot, 0.02))
        resampled_spoken.append(max(spoken, 0.02))
    return resampled, resampled_spoken


def _whisper_durations_for_line(  # noqa: C901
    line_words: List[Word],
    seg_words: Optional[List[TranscriptionWord]],
    seg_end: Optional[float],
    language: str,
    phonetic_similarity,
) -> Optional[Tuple[List[float], List[float]]]:
    if not seg_words:
        return None
    seg_tokens = [_norm_token(w.text) for w in seg_words]
    seg_starts = [w.start for w in seg_words if w.start is not None]
    if not seg_starts:
        return None
    seq_slots = _slots_from_sequence(seg_words, seg_end)
    if not seq_slots:
        return None
    seg_slots, seg_spoken = seq_slots
    default_slot = median(seg_slots) if seg_slots else 0.1
    default_spoken = median(seg_spoken) if seg_spoken else default_slot * 0.85
    slots: List[float] = []
    spokens: List[float] = []
    min_word_sim = 0.5

    def _token_similarity(token: str, seg_token: str) -> float:
        if not token or not seg_token:
            return 0.0
        if (
            seg_token == token
            or seg_token.startswith(token)
            or token.startswith(seg_token)
        ):
            return 1.0
        return phonetic_similarity(token, seg_token, language)

    # Monotonic DP alignment so repeated words cannot match out of order.
    line_tokens = [_norm_token(w.text) for w in line_words]
    n = len(line_tokens)
    m = len(seg_tokens)
    if n == 0 or m == 0:
        return None
    gap_line = 0.6
    gap_seg = 0.3
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] - gap_line
        back[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] - gap_seg
        back[0][j] = 2
    for i in range(1, n + 1):
        token = line_tokens[i - 1]
        for j in range(1, m + 1):
            seg_token = seg_tokens[j - 1]
            sim = _token_similarity(token, seg_token)
            match_cost = 2.0 if sim < min_word_sim else 1.0 - sim
            score_match = dp[i - 1][j - 1] - match_cost
            score_line_skip = dp[i - 1][j] - gap_line
            score_seg_skip = dp[i][j - 1] - gap_seg
            if score_match >= score_line_skip and score_match >= score_seg_skip:
                dp[i][j] = score_match
                back[i][j] = 0
            elif score_line_skip >= score_seg_skip:
                dp[i][j] = score_line_skip
                back[i][j] = 1
            else:
                dp[i][j] = score_seg_skip
                back[i][j] = 2
    mapping: List[Optional[int]] = [None] * n
    i = n
    j = m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and back[i][j] == 0:
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or back[i][j] == 1):
            i -= 1
        else:
            j -= 1

    matched = sum(1 for idx in mapping if idx is not None)
    match_ratio = matched / max(len(line_words), 1)
    if match_ratio < 0.4:
        return _resample_slots_to_line(seg_slots, seg_spoken, len(line_words))

    for idx in mapping:
        if idx is None:
            slots.append(default_slot)
            spokens.append(default_spoken)
        else:
            slots.append(max(seg_slots[idx], 0.02))
            spokens.append(max(seg_spoken[idx], 0.02))
    return slots, spokens


def _apply_weighted_slots_to_line(
    line_words: List[Word],
    desired_start: float,
    line_duration: float,
    slot_durations: Optional[List[float]],
    spoken_durations: Optional[List[float]],
) -> List[Word]:
    if not line_words:
        return []
    if not slot_durations or not spoken_durations:
        weights = [1.0 / len(line_words)] * len(line_words)
        slot_durations = [1.0] * len(line_words)
        spoken_durations = [0.85] * len(line_words)
    else:
        total = sum(slot_durations) or 1.0
        weights = [d / total for d in slot_durations]

    cursor = desired_start
    weighted_words = []
    for i, (word_obj, weight) in enumerate(zip(line_words, weights)):
        seg_dur = line_duration * weight
        slot = slot_durations[i]
        spoken = spoken_durations[i]
        spoken_ratio = min(max(spoken / max(slot, 0.02), 0.5), 1.0)
        start = cursor
        end = start + max(seg_dur * spoken_ratio, 0.02)
        weighted_words.append(
            Word(
                text=word_obj.text,
                start_time=start,
                end_time=end,
                singer=word_obj.singer,
            )
        )
        cursor += seg_dur
    return weighted_words


def _shift_words(words: List[Word], shift: float) -> List[Word]:
    return [
        Word(
            text=w.text,
            start_time=w.start_time + shift,
            end_time=w.end_time + shift,
            singer=w.singer,
        )
        for w in words
    ]


def _find_best_segment_for_line(
    line: Line,
    sorted_segments: List[TranscriptionSegment],
    seg_idx: int,
    lookahead: int,
    language: str,
    phonetic_similarity,
) -> Tuple[Optional[int], float, int]:
    best_idx = None
    best_sim = 0.0
    window_end = min(seg_idx + lookahead, len(sorted_segments))
    for idx in range(seg_idx, window_end):
        seg = sorted_segments[idx]
        sim = phonetic_similarity(line.text, seg.text, language)
        if sim > best_sim:
            best_sim = sim
            best_idx = idx
    return best_idx, best_sim, window_end


def _map_line_words_to_segment(line: Line, seg: TranscriptionSegment) -> List[Word]:
    duration = max(seg.end - seg.start, 0.2)
    orig_start = line.words[0].start_time
    orig_end = line.words[-1].end_time
    orig_duration = max(orig_end - orig_start, 0.2)
    mapped_duration = duration

    new_words: List[Word] = []
    if orig_duration > 0 and len(line.words) > 0:
        for word in line.words:
            rel_start = (word.start_time - orig_start) / orig_duration
            rel_end = (word.end_time - orig_start) / orig_duration
            start = seg.start + rel_start * mapped_duration
            end = seg.start + rel_end * mapped_duration
            new_words.append(
                Word(
                    text=word.text,
                    start_time=start,
                    end_time=max(end, start + 0.02),
                    singer=word.singer,
                )
            )
    else:
        spacing = mapped_duration / max(len(line.words), 1)
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
    return new_words


def _select_window_words_for_line(
    all_words: List[TranscriptionWord],
    line: Line,
    desired_start: Optional[float],
    next_lrc_start: Optional[float],
    language: str,
    phonetic_similarity,
) -> Optional[List[TranscriptionWord]]:
    if desired_start is None or next_lrc_start is None or not all_words:
        return None
    window_start = desired_start - 1.0
    window_words = [w for w in all_words if window_start <= w.start < next_lrc_start]
    if not window_words:
        return None
    first_token = line.words[0].text if line.words else None
    min_count = max(3, len(line.words) // 2)
    min_prob = 0.5

    def _pick_sequence(candidates: List[TranscriptionWord]):
        seq = _select_window_sequence(
            candidates,
            line.text,
            language,
            len(line.words),
            first_token,
            phonetic_similarity,
        )
        sim = phonetic_similarity(line.text, " ".join(w.text for w in seq), language)
        return seq, sim

    def _preferred_sequence(candidates: List[TranscriptionWord]):
        filtered = [
            w for w in candidates if w.probability is None or w.probability >= min_prob
        ]
        seq_all, sim_all = _pick_sequence(candidates)
        if len(filtered) >= min_count:
            seq_f, sim_f = _pick_sequence(filtered)
            if sim_f >= 0.55 and sim_f >= sim_all - 0.02:
                return seq_f, sim_f
        return seq_all, sim_all

    initial_seq, initial_sim = _preferred_sequence(window_words)
    if len(window_words) >= min_count and initial_sim >= 0.55:
        return initial_seq

    # Expand window earlier to catch words that precede a late LRC start.
    expanded_start = max(desired_start - 4.0, 0.0)
    if expanded_start < window_start:
        expanded_words = [
            w for w in all_words if expanded_start <= w.start < next_lrc_start
        ]
        if expanded_words:
            expanded_seq, expanded_sim = _preferred_sequence(expanded_words)
            if expanded_sim > initial_sim + 0.05 or initial_sim < 0.45:
                return expanded_seq
    return initial_seq


def _line_duration_from_lrc(
    line: Line,
    desired_start: float,
    lrc_line_starts: Optional[List[float]],
    line_idx: int,
) -> float:
    if lrc_line_starts and line_idx + 1 < len(lrc_line_starts):
        return max(lrc_line_starts[line_idx + 1] - desired_start, 0.5)
    return max(line.words[-1].end_time - line.words[0].start_time, 0.5)


def _apply_lrc_weighted_timing(  # noqa: C901
    line: Line,
    desired_start: float,
    next_lrc_start: Optional[float],
    lrc_line_starts: Optional[List[float]],
    line_idx: int,
    all_words: List[TranscriptionWord],
    seg: Optional[TranscriptionSegment],
    language: str,
    phonetic_similarity,
    line_duration_override: Optional[float] = None,
) -> Tuple[List[Word], float]:
    line_duration = (
        line_duration_override
        if line_duration_override is not None
        else _line_duration_from_lrc(line, desired_start, lrc_line_starts, line_idx)
    )
    window_words = _select_window_words_for_line(
        all_words,
        line,
        desired_start,
        next_lrc_start,
        language,
        phonetic_similarity,
    )

    def _anchor_desired_start(
        desired_start: float,
        line_duration: float,
        offsets_scaled: List[float],
        align_by_index: bool = False,
    ) -> float:
        if not window_words or not offsets_scaled:
            return desired_start
        window_start = (
            window_words[0].start if window_words[0].start is not None else None
        )
        require_forward = (
            window_start is not None and desired_start + 0.2 < window_start
        )
        require_backward = (
            window_start is not None and desired_start - 0.2 > window_start
        )
        best_score: Optional[float] = None
        best_start: Optional[float] = None
        for i, word_obj in enumerate(line.words):
            token = _norm_token(word_obj.text)
            if not token or i >= len(offsets_scaled):
                continue
            if align_by_index and i < len(window_words):
                candidates = [window_words[i]]
            else:
                candidates = window_words
            for ww in candidates:
                if ww.start is None:
                    continue
                sim = phonetic_similarity(token, ww.text, language)
                if sim < 0.6:
                    continue
                expected = desired_start + offsets_scaled[i]
                delta = abs(expected - ww.start)
                score = sim - min(delta / max(line_duration, 0.01), 1.0) * 0.2
                candidate_start = ww.start - offsets_scaled[i]
                forward_shift = candidate_start - desired_start
                if forward_shift > 1.0:
                    allow_large_shift = (
                        i == 0
                        and offsets_scaled[i] <= line_duration * 0.05
                        and forward_shift <= 2.0
                    )
                    if not allow_large_shift:
                        continue
                if offsets_scaled[i] >= line_duration * 0.6:
                    if candidate_start - desired_start > 0.8:
                        continue
                allow_early_anchor = (
                    len(window_words) <= 3 and offsets_scaled[i] >= line_duration * 0.5
                )
                if not allow_early_anchor and i == len(line.words) - 1:
                    allow_early_anchor = len(window_words) <= 3
                if (
                    window_start is not None
                    and candidate_start < window_start - 0.05
                    and not allow_early_anchor
                ):
                    continue
                if (
                    allow_early_anchor
                    and window_start is not None
                    and candidate_start < window_start - 1.5
                ):
                    continue
                if require_forward and candidate_start < desired_start:
                    continue
                if require_backward and candidate_start > desired_start:
                    continue
                if best_score is None or score > best_score:
                    best_score = score
                    best_start = candidate_start
        if best_start is None:
            return desired_start
        shift = best_start - desired_start
        max_shift = 2.5
        if abs(shift) > max_shift:
            best_start = desired_start + max_shift * (1 if shift > 0 else -1)
        return max(best_start, 0.0)

    direct_offsets: Optional[List[float]] = None
    if (
        window_words is not None
        and len(window_words) == len(line.words)
        and next_lrc_start is not None
    ):
        first_start = window_words[0].start
        last_start = window_words[-1].start
        if (
            first_start is not None
            and last_start is not None
            and last_start > first_start
        ):
            offsets = [t_word.start - first_start for t_word in window_words]  # type: ignore[operator]
            if all(b > a for a, b in zip(offsets, offsets[1:])):
                direct_offsets = offsets

    if direct_offsets and next_lrc_start is not None:
        if line_duration_override is not None:
            line_duration = max(line_duration_override, 0.5)
        else:
            line_duration = max(next_lrc_start - desired_start, 0.5)
        if window_words and window_words[0].start is not None:
            whisper_span = None
            if window_words[-1].end is not None:
                whisper_span = window_words[-1].end - window_words[0].start
            elif window_words[-1].start is not None:
                whisper_span = window_words[-1].start - window_words[0].start
            if whisper_span is not None and whisper_span > 0.2:
                if next_lrc_start is not None and desired_start < window_words[0].start:
                    max_span = max(next_lrc_start - window_words[0].start - 0.1, 0.5)
                    whisper_span = min(whisper_span, max_span)
                if whisper_span < line_duration * 0.95:
                    line_duration = max(0.85 * whisper_span + 0.15 * line_duration, 0.5)
                lrc_start = None
                if lrc_line_starts and line_idx < len(lrc_line_starts):
                    lrc_start = lrc_line_starts[line_idx]
                elif line.words:
                    lrc_start = line.words[0].start_time
                forward_shift = None
                if lrc_start is not None and window_words[0].start is not None:
                    forward_shift = window_words[0].start - lrc_start
                if (
                    forward_shift is not None
                    and forward_shift >= 0.8
                    and whisper_span > 0.2
                ):
                    tightened = max(whisper_span * 1.05, 0.5)
                    if next_lrc_start is not None:
                        tightened = min(
                            tightened, max(next_lrc_start - desired_start, 0.5)
                        )
                    line_duration = min(line_duration, tightened)
        last_end = (
            window_words[-1].end
            if window_words and window_words[-1].end is not None
            else (window_words[-1].start if window_words else None)
        )
        if last_end is not None and first_start is not None:
            whisper_total = max(last_end - first_start, 0.01)
        else:
            whisper_total = max(direct_offsets[-1], 0.01)
        offsets_scaled = [
            (offset / whisper_total) * line_duration for offset in direct_offsets
        ]
        desired_start = _anchor_desired_start(
            desired_start,
            line_duration,
            offsets_scaled,
            align_by_index=True,
        )
        mapped_words: List[Word] = []
        for i, (word_obj, offset) in enumerate(zip(line.words, direct_offsets)):
            start = desired_start + (offset / whisper_total) * line_duration
            if i + 1 < len(direct_offsets):
                next_offset = direct_offsets[i + 1]
                slot = (next_offset - offset) / whisper_total * line_duration
            else:
                slot = line_duration - (offset / whisper_total) * line_duration
            spoken_ratio = 0.85
            if window_words is not None:
                ww = window_words[i]
                if ww.end is not None and ww.start is not None and ww.end > ww.start:
                    spoken_ratio = min(
                        max((ww.end - ww.start) / max(slot, 0.02), 0.5), 1.0
                    )
            end = start + max(slot * spoken_ratio, 0.02)
            mapped_words.append(
                Word(
                    text=word_obj.text,
                    start_time=start,
                    end_time=end,
                    singer=word_obj.singer,
                )
            )
        return mapped_words, desired_start

    if window_words and len(window_words) == len(line.words):
        whisper_timing = _slots_from_sequence(
            window_words, window_words[-1].end if window_words else None
        )
    else:
        whisper_timing = _whisper_durations_for_line(
            line.words,
            window_words if window_words else (seg.words if seg else None),
            (
                (window_words[-1].end if window_words else None)
                if window_words
                else (seg.end if seg else None)
            ),
            language,
            phonetic_similarity,
        )
    slot_durations = whisper_timing[0] if whisper_timing else None
    spoken_durations = whisper_timing[1] if whisper_timing else None
    if window_words and window_words[0].start is not None:
        whisper_span = None
        if window_words[-1].end is not None:
            whisper_span = window_words[-1].end - window_words[0].start
        elif window_words[-1].start is not None:
            whisper_span = window_words[-1].start - window_words[0].start
        if whisper_span is not None and whisper_span > 0.2:
            if next_lrc_start is not None and desired_start < window_words[0].start:
                max_span = max(next_lrc_start - window_words[0].start - 0.1, 0.5)
                whisper_span = min(whisper_span, max_span)
            if whisper_span < line_duration * 0.95:
                line_duration = max(0.85 * whisper_span + 0.15 * line_duration, 0.5)
            lrc_start = None
            if lrc_line_starts and line_idx < len(lrc_line_starts):
                lrc_start = lrc_line_starts[line_idx]
            elif line.words:
                lrc_start = line.words[0].start_time
            forward_shift = None
            if lrc_start is not None and window_words[0].start is not None:
                forward_shift = window_words[0].start - lrc_start
            if (
                forward_shift is not None
                and forward_shift >= 0.8
                and whisper_span > 0.2
            ):
                tightened = max(whisper_span * 1.05, 0.5)
                if next_lrc_start is not None:
                    tightened = min(tightened, max(next_lrc_start - desired_start, 0.5))
                line_duration = min(line_duration, tightened)
    if slot_durations:
        total = sum(slot_durations) or 1.0
        offsets_scaled = []
        cursor = 0.0
        for duration in slot_durations[: len(line.words)]:
            offsets_scaled.append((cursor / total) * line_duration)
            cursor += duration
        desired_start = _anchor_desired_start(
            desired_start,
            line_duration,
            offsets_scaled,
        )
    return (
        _apply_weighted_slots_to_line(
            line.words,
            desired_start,
            line_duration,
            slot_durations,
            spoken_durations,
        ),
        desired_start,
    )


def _clamp_line_end_to_next_start(
    line: Line,
    new_words: List[Word],
    lrc_line_starts: Optional[List[float]],
    line_idx: int,
) -> List[Word]:
    if not new_words:
        return new_words
    line_start = new_words[0].start_time
    orig_duration = max(line.words[-1].end_time - line.words[0].start_time, 0.5)
    max_end = line_start + max(1.5, orig_duration * 1.5)
    if lrc_line_starts and line_idx + 1 < len(lrc_line_starts):
        next_start = lrc_line_starts[line_idx + 1]
        max_end = min(max_end, next_start - 0.1)
    if new_words[-1].end_time > max_end and max_end > line_start:
        span = new_words[-1].end_time - line_start
        scale = (max_end - line_start) / span if span > 0 else 1.0
        new_words = [
            Word(
                text=w.text,
                start_time=line_start + (w.start_time - line_start) * scale,
                end_time=line_start + (w.end_time - line_start) * scale,
                singer=w.singer,
            )
            for w in new_words
        ]
    return new_words


def _create_lines_from_whisper(
    transcription: List[TranscriptionSegment],
) -> List[Line]:
    """Create Line objects directly from Whisper transcription."""
    from .models import Line

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


def _map_lrc_lines_to_whisper_segments(  # noqa: C901
    lines: List[Line],
    transcription: List[TranscriptionSegment],
    language: str,
    lrc_line_starts: Optional[List[float]] = None,
    min_similarity: float = 0.35,
    min_similarity_fallback: float = 0.2,
    max_time_offset: float = 4.0,
    lookahead: int = 6,
) -> Tuple[List[Line], int, List[str]]:  # noqa: C901
    """Map LRC lines onto Whisper segment timing without reordering."""
    from .timing_evaluator import _phonetic_similarity

    if not lines or not transcription:
        return lines, 0, []

    sorted_segments = sorted(transcription, key=lambda s: s.start)
    all_words = _build_whisper_word_list(transcription)
    adjusted: List[Line] = []
    fixes = 0
    issues: List[str] = []
    seg_idx = 0
    last_end = None
    min_gap = 0.01
    prev_text = None

    for line_idx, line in enumerate(lines):
        if not line.words:
            adjusted.append(line)
            continue

        best_idx, best_sim, window_end = _find_best_segment_for_line(
            line,
            sorted_segments,
            seg_idx,
            lookahead,
            language,
            _phonetic_similarity,
        )

        text_norm = line.text.strip().lower() if line.text else ""
        gap_required = 0.21 if prev_text and text_norm == prev_text else min_gap

        desired_start = (
            lrc_line_starts[line_idx]
            if lrc_line_starts and line_idx < len(lrc_line_starts)
            else None
        )
        next_lrc_start = (
            lrc_line_starts[line_idx + 1]
            if lrc_line_starts and line_idx + 1 < len(lrc_line_starts)
            else None
        )
        if (
            best_idx is None
            or best_sim < min_similarity_fallback
            or (
                desired_start is not None
                and best_idx is not None
                and abs(sorted_segments[best_idx].start - desired_start)
                > max_time_offset
            )
        ):
            new_words: List[Word]
            window_fallback = False
            if desired_start is not None and next_lrc_start is not None and all_words:
                window_words = _select_window_words_for_line(
                    all_words,
                    line,
                    desired_start,
                    next_lrc_start,
                    language,
                    _phonetic_similarity,
                )
                if window_words:
                    if window_words[0].start is not None:
                        whisper_start = window_words[0].start
                        start_delta = desired_start - whisper_start
                        if start_delta > 0.4:
                            desired_start = max(
                                whisper_start, desired_start - min(start_delta, 0.8)
                            )
                        elif start_delta < -0.4:
                            desired_start = min(
                                whisper_start, desired_start + min(-start_delta, 2.5)
                            )
                    whisper_duration = None
                    if (
                        window_words[-1].end is not None
                        and window_words[0].start is not None
                    ):
                        whisper_duration = max(
                            window_words[-1].end - window_words[0].start, 0.5
                        )
                    elif (
                        window_words[-1].start is not None
                        and window_words[0].start is not None
                    ):
                        whisper_duration = max(
                            window_words[-1].start - window_words[0].start, 0.5
                        )
                    window_text = " ".join(w.text for w in window_words)
                    window_sim = _phonetic_similarity(line.text, window_text, language)
                    if window_sim >= min_similarity_fallback:
                        new_words, desired_start = _apply_lrc_weighted_timing(
                            line,
                            desired_start,
                            next_lrc_start,
                            lrc_line_starts,
                            line_idx,
                            all_words,
                            None,
                            language,
                            _phonetic_similarity,
                            line_duration_override=whisper_duration,
                        )
                        window_fallback = True
                        issues.append(
                            f"Used window-only mapping for line '{line.text[:30]}...' "
                            f"(segment offset {sorted_segments[best_idx].start - desired_start:+.2f}s)"
                            if best_idx is not None and desired_start is not None
                            else f"Used window-only mapping for line '{line.text[:30]}...'"
                        )
            if not window_fallback:
                new_words = list(line.words)
            if desired_start is not None and new_words:
                shift = desired_start - new_words[0].start_time
                new_words = _shift_words(new_words, shift)
            new_words = _clamp_line_end_to_next_start(
                line, new_words, lrc_line_starts, line_idx
            )
            if last_end is not None and new_words:
                if new_words[0].start_time < last_end + gap_required:
                    shift = (last_end + gap_required) - new_words[0].start_time
                    new_words = _shift_words(new_words, shift)
            if new_words:
                adjusted.append(Line(words=new_words, singer=line.singer))
                last_end = new_words[-1].end_time
                if window_fallback:
                    fixes += 1
            else:
                adjusted.append(line)
                last_end = line.end_time
            if not window_fallback:
                if best_sim < min_similarity_fallback and best_idx is not None:
                    issues.append(
                        f"Skipped Whisper mapping for line '{line.text[:30]}...' "
                        f"(sim={best_sim:.2f})"
                    )
                if (
                    desired_start is not None
                    and best_idx is not None
                    and abs(sorted_segments[best_idx].start - desired_start)
                    > max_time_offset
                ):
                    issues.append(
                        f"Skipped Whisper mapping for line '{line.text[:30]}...' "
                        f"(segment offset {sorted_segments[best_idx].start - desired_start:+.2f}s)"
                    )
            prev_text = text_norm
            continue

        if best_idx is None:
            adjusted.append(line)
            prev_text = text_norm
            continue
        seg = sorted_segments[best_idx]
        forced_fallback = False
        if last_end is not None and seg.start < last_end + gap_required:
            next_idx: Optional[int] = None
            best_future_idx: Optional[int] = None
            best_future_sim: Optional[float] = None
            for idx in range(best_idx, window_end):
                seg_candidate = sorted_segments[idx]
                if seg_candidate.start < last_end + gap_required:
                    continue
                next_idx = idx
                sim_candidate = _phonetic_similarity(
                    line.text, seg_candidate.text, language
                )
                if best_future_sim is None or sim_candidate > best_future_sim:
                    best_future_sim = sim_candidate
                    best_future_idx = idx
            if best_future_idx is not None and best_future_sim is not None:
                if best_future_sim >= min_similarity_fallback:
                    best_idx = best_future_idx
                    seg = sorted_segments[best_idx]
                else:
                    if next_idx is None:
                        adjusted.append(line)
                        prev_text = text_norm
                        continue
                    best_idx = next_idx
                    seg = sorted_segments[best_idx]
                    forced_fallback = True
        new_words = _map_line_words_to_segment(line, seg)

        # If we have an LRC line start, redistribute timing within the line
        # using Whisper word durations as proportions, while preserving the
        # line's total duration.
        if desired_start is not None and line.words:
            new_words, desired_start = _apply_lrc_weighted_timing(
                line,
                desired_start,
                next_lrc_start,
                lrc_line_starts,
                line_idx,
                all_words,
                seg,
                language,
                _phonetic_similarity,
            )

        if desired_start is not None and new_words:
            shift = desired_start - new_words[0].start_time
            new_words = _shift_words(new_words, shift)

        # Clamp line duration so it doesn't explode far past the next line.
        new_words = _clamp_line_end_to_next_start(
            line, new_words, lrc_line_starts, line_idx
        )

        if (
            last_end is not None
            and new_words
            and new_words[0].start_time < last_end + gap_required
        ):
            shift = (last_end + gap_required) - new_words[0].start_time
            new_words = _shift_words(new_words, shift)
        adjusted.append(Line(words=new_words, singer=line.singer))
        fixes += 1
        if best_sim < min_similarity:
            issues.append(
                f"Low similarity mapping for line '{line.text[:30]}...' (sim={best_sim:.2f})"
            )
        if forced_fallback:
            issues.append(
                f"Forced forward mapping for line '{line.text[:30]}...' to avoid overlap"
            )
        last_end = new_words[-1].end_time if new_words else last_end
        seg_idx = best_idx + 1 if best_idx is not None else seg_idx
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
    if target_duration and lrc_duration and abs(target_duration - lrc_duration) > 8:
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
    elif quality_report.get("dtw_metrics"):
        base_score = _score_from_dtw_metrics(quality_report["dtw_metrics"])
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


def _score_from_dtw_metrics(metrics: dict) -> float:
    """Heuristic score derived from Whisper/DTW alignment metrics."""
    matched_ratio = float(metrics.get("matched_ratio", 0.0))
    avg_similarity = float(metrics.get("avg_similarity", 0.0))
    line_coverage = float(metrics.get("line_coverage", 0.0))

    score = 40.0
    score += matched_ratio * 25.0
    score += avg_similarity * 20.0
    score += line_coverage * 10.0
    return max(20.0, min(100.0, score))


def _fetch_lrc_text_and_timings(
    title: str,
    artist: str,
    target_duration: Optional[int] = None,
    vocals_path: Optional[str] = None,
    evaluate_sources: bool = False,
    filter_promos: bool = True,
    offline: bool = False,
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
        if evaluate_sources and vocals_path and not offline:
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
                title, artist, target_duration, tolerance=8, offline=offline
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

            lrc_text, is_synced, source = fetch_lyrics_multi_source(
                title, artist, offline=offline
            )
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


def get_lyrics_simple(  # noqa: C901
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
    whisper_map_lrc_dtw: bool = False,
    lyrics_file: Optional[Path] = None,
    whisper_language: Optional[str] = None,
    whisper_model: Optional[str] = None,
    whisper_force_dtw: bool = False,
    whisper_aggressive: bool = False,
    offline: bool = False,
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
        whisper_map_lrc_dtw: If True, map LRC text onto Whisper timing using phonetic DTW
        lyrics_file: Optional local lyrics file (plain text or .lrc)
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

        model_size = whisper_model or "base"
        transcription, _, detected_lang = transcribe_vocals(
            vocals_path, whisper_language, model_size, whisper_aggressive
        )
        if not transcription:
            logger.warning(
                "No Whisper transcription available; using placeholder lyrics"
            )
            return _create_no_lyrics_placeholder(title, artist)
        lines = _create_lines_from_whisper(transcription)
        whisper_metadata = SongMetadata(
            singers=[],
            is_duet=False,
            title=title,
            artist=artist,
        )
        if romanize:
            _romanize_lines(lines)
        logger.debug(f"Returning {len(lines)} lines from Whisper-only mode")
        return lines, whisper_metadata

    file_lines: List[str] = []
    file_lrc_text: Optional[str] = None
    file_line_timings: Optional[List[Tuple[float, str]]] = None
    if lyrics_file:
        file_lrc_text, file_line_timings, file_lines = _load_lyrics_file(
            lyrics_file, filter_promos
        )
        if file_lines or file_lrc_text:
            logger.info(f"Using lyrics from file: {lyrics_file}")

    # 1. Try LRC first (preferred source), with duration validation if provided
    logger.debug(
        f"Fetching LRC lyrics... (target_duration={target_duration}, "
        f"evaluate={evaluate_sources})"
    )
    lrc_text, line_timings, _source = _fetch_lrc_text_and_timings(
        title=title,
        artist=artist,
        target_duration=target_duration,
        vocals_path=vocals_path,
        evaluate_sources=evaluate_sources,
        filter_promos=filter_promos,
        offline=offline,
    )
    if file_lrc_text and file_line_timings:
        lrc_text = file_lrc_text
        line_timings = file_line_timings
    if target_duration and lrc_text:
        from .sync import get_lrc_duration

        lrc_duration = get_lrc_duration(lrc_text)
        if lrc_duration and abs(target_duration - lrc_duration) > 8:
            logger.warning(
                "LRC duration mismatch: keeping text but ignoring LRC timings"
            )
            line_timings = None

    # 2. Fetch Genius as fallback or for singer info
    genius_lines: Optional[List[Tuple[str, str]]] = None
    metadata: Optional[SongMetadata] = None
    if not line_timings and not lrc_text and not file_lines:
        if offline:
            logger.warning("Offline mode: no cached lyrics available")
            return _create_no_lyrics_placeholder(title, artist)
        logger.debug("No LRC found, fetching lyrics from Genius...")
        genius_lines, metadata = fetch_genius_lyrics_with_singers(title, artist)
        if not genius_lines:
            logger.warning("No lyrics found from any source, using placeholder")
            return _create_no_lyrics_placeholder(title, artist)
    else:
        # Fetch Genius for singer/duet metadata only
        genius_lines, metadata = fetch_genius_lyrics_with_singers(title, artist)

    # 3. Apply vocal offset if available
    if vocals_path and line_timings:
        line_timings, _ = _detect_and_apply_offset(
            vocals_path, line_timings, lyrics_offset
        )

    # 4. Create Line objects
    has_lrc_timing = bool(line_timings)
    if lrc_text or file_lines:
        if line_timings and file_lines:
            lines = create_lines_from_lrc_timings(line_timings, file_lines)
        elif line_timings and lrc_text:
            lines = create_lines_from_lrc(
                lrc_text,
                romanize=False,
                title=title,
                artist=artist,
                filter_promos=filter_promos,
            )
            _apply_timing_to_lines(lines, line_timings)
        else:
            text_lines = file_lines or _extract_text_lines_from_lrc(lrc_text or "")
            lines = _create_lines_from_plain_text(text_lines)

        # 5. Refine word timing using audio
        if vocals_path and line_timings and len(line_timings) > 1:
            lines = _refine_timing_with_audio(
                lines, vocals_path, line_timings, lrc_text or "", target_duration
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
                    whisper_aggressive,
                    prefer_whisper_timing_map=not has_lrc_timing,
                )
            except Exception as e:
                logger.warning(f"Whisper alignment failed: {e}")
        elif vocals_path and whisper_map_lrc:
            try:
                if whisper_map_lrc_dtw:
                    from .timing_evaluator import align_lrc_text_to_whisper_timings

                    model_size = whisper_model or "small"
                    lines, alignments, metrics = align_lrc_text_to_whisper_timings(
                        lines,
                        vocals_path,
                        language=whisper_language,
                        model_size=model_size,
                        aggressive=whisper_aggressive,
                    )
                    logger.info(
                        f"DTW-mapped {len(alignments)} LRC line(s) onto Whisper timing"
                    )
                    if metrics:
                        logger.debug(f"DTW metrics: {metrics}")
                else:
                    from .timing_evaluator import (
                        _whisper_lang_to_epitran,
                        transcribe_vocals,
                    )

                    model_size = whisper_model or "small"
                    transcription, _, detected_lang = transcribe_vocals(
                        vocals_path, whisper_language, model_size, whisper_aggressive
                    )
                    if transcription:
                        lang = _whisper_lang_to_epitran(detected_lang)
                        lines, mapped, issues = _map_lrc_lines_to_whisper_segments(
                            lines, transcription, lang
                        )
                        if mapped:
                            logger.info(
                                f"Mapped {mapped} LRC line(s) onto Whisper timing"
                            )
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


def get_lyrics_with_quality(  # noqa: C901
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
    whisper_map_lrc_dtw: bool = False,
    lyrics_file: Optional[Path] = None,
    whisper_language: Optional[str] = None,
    whisper_model: Optional[str] = None,
    whisper_force_dtw: bool = False,
    whisper_aggressive: bool = False,
    offline: bool = False,
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
    issues_list: List[str] = quality_report["issues"]  # type: ignore[assignment]

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
            whisper_map_lrc_dtw=whisper_map_lrc_dtw,
            lyrics_file=lyrics_file,
            whisper_language=whisper_language,
            whisper_model=whisper_model,
            whisper_force_dtw=whisper_force_dtw,
            whisper_aggressive=whisper_aggressive,
            offline=offline,
        )
        quality_report["alignment_method"] = "whisper_only"
        quality_report["whisper_used"] = bool(lines)
        quality_report["total_lines"] = len(lines)
        quality_report["overall_score"] = 50.0 if lines else 0.0
        if not lines:
            issues_list.append("Whisper-only mode produced no lines")
        return lines, metadata, quality_report

    if whisper_map_lrc:
        quality_report["alignment_method"] = (
            "whisper_map_lrc_dtw" if whisper_map_lrc_dtw else "whisper_map_lrc"
        )

    file_lines: List[str] = []
    file_lrc_text: Optional[str] = None
    file_line_timings: Optional[List[Tuple[float, str]]] = None
    if lyrics_file:
        file_lrc_text, file_line_timings, file_lines = _load_lyrics_file(
            lyrics_file, filter_promos
        )
        if file_lrc_text or file_lines:
            quality_report["source"] = f"lyrics_file:{lyrics_file}"

    # 1. Try LRC first
    logger.debug(
        f"Fetching LRC lyrics... (target_duration={target_duration}, "
        f"evaluate={evaluate_sources})"
    )
    lrc_text, line_timings, source = _fetch_lrc_text_and_timings(
        title=title,
        artist=artist,
        target_duration=target_duration,
        vocals_path=vocals_path,
        evaluate_sources=evaluate_sources,
        filter_promos=filter_promos,
        offline=offline,
    )
    if file_lrc_text and file_line_timings:
        lrc_text = file_lrc_text
        line_timings = file_line_timings
        source = "lyrics_file_lrc"
    if not quality_report["source"]:
        quality_report["source"] = source
    if target_duration and lrc_text:
        from .sync import get_lrc_duration

        lrc_duration = get_lrc_duration(lrc_text)
        if lrc_duration and abs(target_duration - lrc_duration) > 8:
            issues_list.append(
                "LRC duration mismatch: keeping text but ignoring LRC timings"
            )
            line_timings = None

    # 2. Fetch Genius as fallback or for singer info
    if (lrc_text or file_lines) and not line_timings:
        # LRC text exists but timings are invalid; use Genius for metadata only
        if offline:
            genius_lines, metadata = None, None
        else:
            from .genius import fetch_genius_lyrics_with_singers

            genius_lines, metadata = fetch_genius_lyrics_with_singers(title, artist)
    else:
        if offline:
            genius_lines, metadata = None, None
        else:
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

    # 3. Apply vocal offset if available
    if vocals_path and line_timings:
        line_timings, _ = _detect_offset_with_issues(
            vocals_path, line_timings, lyrics_offset, issues_list
        )

    has_lrc_timing = bool(line_timings)
    # 4. Create Line objects and apply timing
    if lrc_text or file_lines:
        if line_timings and file_lines:
            lines = create_lines_from_lrc_timings(line_timings, file_lines)
            quality_report["alignment_method"] = "lrc_only"
        elif line_timings and lrc_text:
            lines = create_lines_from_lrc(
                lrc_text,
                romanize=False,
                title=title,
                artist=artist,
                filter_promos=filter_promos,
            )
            quality_report["alignment_method"] = "lrc_only"
            _apply_timing_to_lines(lines, line_timings)
        else:
            text_lines = file_lines or _extract_text_lines_from_lrc(lrc_text or "")
            lines = _create_lines_from_plain_text(text_lines)

        # 5. Refine word timing using audio
        if (
            vocals_path
            and line_timings
            and len(line_timings) > 1
            and not whisper_map_lrc
        ):
            lines, method = _refine_timing_with_quality(
                lines,
                vocals_path,
                line_timings,
                lrc_text or "",
                target_duration,
                issues_list,
            )
            quality_report["alignment_method"] = method

        # 5b. Apply Whisper if requested (even when no LRC timings)
        if vocals_path and use_whisper:
            lines, quality_report = _apply_whisper_with_quality(
                lines,
                vocals_path,
                whisper_language,
                whisper_model,
                whisper_force_dtw,
                whisper_aggressive,
                quality_report,
                prefer_whisper_timing_map=not has_lrc_timing,
            )
        elif vocals_path and whisper_map_lrc:
            try:
                if whisper_map_lrc_dtw:
                    from .timing_evaluator import align_lrc_text_to_whisper_timings

                    model_size = whisper_model or "small"
                    lines, alignments, metrics = align_lrc_text_to_whisper_timings(
                        lines,
                        vocals_path,
                        language=whisper_language,
                        model_size=model_size,
                        aggressive=whisper_aggressive,
                    )
                    quality_report["alignment_method"] = "whisper_map_lrc_dtw"
                    quality_report["whisper_used"] = True
                    quality_report["whisper_corrections"] = len(alignments)
                    if metrics:
                        quality_report["dtw_metrics"] = metrics
                else:
                    from .timing_evaluator import (
                        _whisper_lang_to_epitran,
                        transcribe_vocals,
                    )

                    model_size = whisper_model or "small"
                    transcription, _, detected_lang = transcribe_vocals(
                        vocals_path, whisper_language, model_size, whisper_aggressive
                    )
                    if transcription:
                        lang = _whisper_lang_to_epitran(detected_lang)
                        lrc_starts = (
                            [ts for ts, _ in line_timings] if line_timings else None
                        )
                        lines, mapped, issues = _map_lrc_lines_to_whisper_segments(
                            lines, transcription, lang, lrc_line_starts=lrc_starts
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
        elif vocals_path and not has_lrc_timing:
            lines, quality_report = _apply_whisper_with_quality(
                lines,
                vocals_path,
                whisper_language,
                whisper_model,
                whisper_force_dtw,
                whisper_aggressive,
                quality_report,
                prefer_whisper_timing_map=True,
            )
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
    whisper_model: Optional[str],
    whisper_force_dtw: bool,
    whisper_aggressive: bool = False,
    quality_report: Optional[dict] = None,
    prefer_whisper_timing_map: bool = False,
) -> Tuple[List[Line], dict]:
    """Apply Whisper alignment and update quality report."""
    if quality_report is None:
        quality_report = {"issues": []}
    try:
        lines, whisper_fixes, whisper_metrics = _apply_whisper_alignment(
            lines,
            vocals_path,
            whisper_language,
            whisper_model,
            whisper_force_dtw,
            whisper_aggressive,
            prefer_whisper_timing_map=prefer_whisper_timing_map,
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
