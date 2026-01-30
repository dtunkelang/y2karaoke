"""Shorten long instrumental breaks in karaoke tracks.

This module detects and shortens long instrumental breaks to improve
the karaoke experience. It uses audio analysis to find natural cut points
and creates smooth transitions.
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
from pydub import AudioSegment

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class InstrumentalBreak:
    """Represents an instrumental break in the audio."""
    start: float  # Start time in seconds
    end: float    # End time in seconds

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class BreakEdit:
    """Represents an edit made to shorten a break."""
    original_start: float
    original_end: float
    new_end: float  # Where the break now ends (after shortening)
    time_removed: float  # How much time was removed

    @property
    def original_duration(self) -> float:
        return self.original_end - self.original_start

    @property
    def new_duration(self) -> float:
        return self.new_end - self.original_start


def detect_instrumental_breaks(
    vocals_path: str,
    min_break_duration: float = 5.0,
    energy_threshold: float = 0.02,
    sample_rate: int = 22050,
) -> List[InstrumentalBreak]:
    """Detect instrumental breaks by finding low-energy sections in vocals.

    Args:
        vocals_path: Path to isolated vocals audio
        min_break_duration: Minimum duration (seconds) to consider a break
        energy_threshold: RMS energy threshold below which is considered silence
        sample_rate: Sample rate for analysis

    Returns:
        List of InstrumentalBreak objects
    """
    import librosa

    # Load vocals
    y, sr = librosa.load(vocals_path, sr=sample_rate)

    # Calculate RMS energy in small windows
    hop_length = 512
    frame_length = 2048
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Convert frame indices to time
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # Find sections below threshold
    is_silent = rms < energy_threshold

    # Find contiguous silent regions
    breaks = []
    in_break = False
    break_start = 0

    for i, (t, silent) in enumerate(zip(times, is_silent)):
        if silent and not in_break:
            in_break = True
            break_start = t
        elif not silent and in_break:
            in_break = False
            break_end = t
            duration = break_end - break_start
            if duration >= min_break_duration:
                breaks.append(InstrumentalBreak(start=break_start, end=break_end))

    # Handle case where audio ends in a break
    if in_break:
        break_end = times[-1]
        duration = break_end - break_start
        if duration >= min_break_duration:
            breaks.append(InstrumentalBreak(start=break_start, end=break_end))

    logger.debug(f"Detected {len(breaks)} instrumental breaks")
    for i, b in enumerate(breaks):
        logger.debug(f"  Break {i}: {b.start:.1f}s - {b.end:.1f}s ({b.duration:.1f}s)")

    return breaks


def find_beat_near(
    instrumental_path: str,
    target_time: float,
    search_window: float = 2.0,
    sample_rate: int = 22050,
) -> float:
    """Find the nearest beat to a target time.

    Args:
        instrumental_path: Path to instrumental audio
        target_time: Time to search near (seconds)
        search_window: How far to search in each direction (seconds)
        sample_rate: Sample rate for analysis

    Returns:
        Time of nearest beat (seconds)
    """
    import librosa

    # Load a portion of the audio around the target time
    start_time = max(0, target_time - search_window)
    duration = search_window * 2

    y, sr = librosa.load(
        instrumental_path,
        sr=sample_rate,
        offset=start_time,
        duration=duration
    )

    # Detect beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr) + start_time

    if len(beat_times) == 0:
        return target_time

    # Find nearest beat to target
    nearest_idx = np.argmin(np.abs(beat_times - target_time))
    return beat_times[nearest_idx]


def shorten_break(
    instrumental_path: str,
    break_info: InstrumentalBreak,
    keep_start: float = 5.0,
    keep_end: float = 3.0,
    crossfade_duration: float = 1.0,
    align_to_beats: bool = True,
) -> Tuple[AudioSegment, AudioSegment, float, float]:
    """Shorten a single instrumental break.

    Strategy:
    1. Keep the first `keep_start` seconds of the break
    2. Keep the last `keep_end` seconds of the break
    3. Crossfade between them
    4. Optionally align cuts to beat boundaries

    Args:
        instrumental_path: Path to instrumental audio
        break_info: The break to shorten
        keep_start: Seconds to keep at start of break
        keep_end: Seconds to keep at end of break
        crossfade_duration: Duration of crossfade (seconds)
        align_to_beats: Whether to align cuts to beat boundaries

    Returns:
        Tuple of (before_segment, shortened_break, after_start_time, time_removed)
    """
    # Calculate cut points
    cut_out_start = break_info.start + keep_start
    cut_out_end = break_info.end - keep_end

    # If the break isn't long enough to cut, return None
    min_cut = keep_start + keep_end + crossfade_duration
    if break_info.duration < min_cut:
        logger.debug(f"Break too short to cut ({break_info.duration:.1f}s < {min_cut:.1f}s)")
        return None, None, break_info.end, 0.0

    # Align to beats if requested
    if align_to_beats:
        cut_out_start = find_beat_near(instrumental_path, cut_out_start)
        cut_out_end = find_beat_near(instrumental_path, cut_out_end)

        # Make sure we still have room for crossfade
        if cut_out_end - cut_out_start < crossfade_duration:
            cut_out_end = cut_out_start + crossfade_duration

    time_removed = cut_out_end - cut_out_start - crossfade_duration

    logger.debug(f"Shortening break: cut {cut_out_start:.1f}s to {cut_out_end:.1f}s "
                 f"(removing {time_removed:.1f}s)")

    return cut_out_start, cut_out_end, time_removed


def shorten_instrumental_breaks(
    instrumental_path: str,
    vocals_path: str,
    output_path: str,
    max_break_duration: float = 20.0,
    keep_start: float = 5.0,
    keep_end: float = 3.0,
    crossfade_ms: int = 1000,
    skip_intro: bool = True,
    intro_threshold: float = 10.0,
) -> Tuple[str, List[BreakEdit]]:
    """Shorten all long instrumental breaks in an audio file.

    Args:
        instrumental_path: Path to instrumental audio
        vocals_path: Path to vocals (for detecting breaks)
        output_path: Where to save the shortened audio
        max_break_duration: Breaks longer than this will be shortened
        keep_start: Seconds to keep at start of each break
        keep_end: Seconds to keep at end of each break
        crossfade_ms: Crossfade duration in milliseconds
        skip_intro: Whether to skip breaks at the very beginning
        intro_threshold: Breaks starting before this time are considered intro

    Returns:
        Tuple of (output_path, list of BreakEdit objects for timing adjustment)
    """
    # Detect breaks
    breaks = detect_instrumental_breaks(vocals_path, min_break_duration=5.0)

    # Filter to long breaks that need shortening
    long_breaks = [b for b in breaks if b.duration > max_break_duration]

    # Optionally skip intro breaks
    if skip_intro:
        long_breaks = [b for b in long_breaks if b.start >= intro_threshold]

    if not long_breaks:
        logger.info("No long instrumental breaks to shorten")
        return instrumental_path, []

    logger.info(f"Found {len(long_breaks)} long instrumental break(s) to shorten")

    # Load full audio
    audio = AudioSegment.from_file(instrumental_path)

    # Process each break, tracking edits
    edits: List[BreakEdit] = []
    cumulative_removed = 0.0

    # Build new audio by concatenating segments
    segments = []
    last_end_ms = 0

    for break_info in long_breaks:
        # Adjust break times for previous edits
        adjusted_start = break_info.start - cumulative_removed
        adjusted_end = break_info.end - cumulative_removed

        # Calculate cut points (in original time, then adjust)
        cut_out_start = break_info.start + keep_start
        cut_out_end = break_info.end - keep_end

        # Make sure we have enough to cut
        if cut_out_end - cut_out_start < crossfade_ms / 1000:
            logger.debug(f"Break at {break_info.start:.1f}s too short to cut effectively")
            continue

        # Align to beats
        cut_out_start = find_beat_near(instrumental_path, cut_out_start)
        cut_out_end = find_beat_near(instrumental_path, cut_out_end)

        # Convert to ms (adjusted for previous edits)
        cut_start_ms = int((cut_out_start - cumulative_removed) * 1000)
        cut_end_ms = int((cut_out_end - cumulative_removed) * 1000)

        # Add segment before the cut
        if cut_start_ms > last_end_ms:
            segments.append(audio[last_end_ms:cut_start_ms])

        # Create crossfade between cut points
        fade_out_segment = audio[cut_start_ms:cut_start_ms + crossfade_ms].fade_out(crossfade_ms)
        fade_in_segment = audio[cut_end_ms:cut_end_ms + crossfade_ms].fade_in(crossfade_ms)
        crossfaded = fade_out_segment.overlay(fade_in_segment)
        segments.append(crossfaded)

        # Track the edit
        time_removed = (cut_out_end - cut_out_start) - (crossfade_ms / 1000)
        edit = BreakEdit(
            original_start=break_info.start,
            original_end=break_info.end,
            new_end=break_info.start + keep_start + (crossfade_ms / 1000) + keep_end,
            time_removed=time_removed,
        )
        edits.append(edit)

        cumulative_removed += time_removed
        last_end_ms = cut_end_ms + crossfade_ms

        logger.info(f"  Shortened break at {break_info.start:.1f}s: "
                    f"{break_info.duration:.1f}s → {edit.new_duration:.1f}s "
                    f"(removed {time_removed:.1f}s)")

    # Add remaining audio after last cut
    if last_end_ms < len(audio):
        segments.append(audio[last_end_ms:])

    # Concatenate all segments
    if segments:
        shortened_audio = segments[0]
        for seg in segments[1:]:
            shortened_audio = shortened_audio + seg
    else:
        shortened_audio = audio

    # Export
    shortened_audio.export(output_path, format="wav")

    total_removed = sum(e.time_removed for e in edits)
    logger.info(f"Total time removed: {total_removed:.1f}s")

    return output_path, edits


def adjust_lyrics_timing(
    lines: list,
    edits: List[BreakEdit],
    keep_start: float = 5.0,
    keep_end: float = 3.0,
    crossfade_duration: float = 1.0,
) -> list:
    """Adjust lyrics timing based on break edits.

    Args:
        lines: List of Line objects with word timing
        edits: List of BreakEdit objects describing removed time
        keep_start: Seconds kept at start of each break
        keep_end: Seconds kept at end of each break
        crossfade_duration: Duration of crossfade in seconds

    Returns:
        New list of Line objects with adjusted timing
    """
    from .models import Line, Word

    if not edits:
        return lines

    adjusted_lines = []

    for line in lines:
        if not line.words:
            adjusted_lines.append(line)
            continue

        new_words = []
        for word in line.words:
            # Calculate how much time to subtract based on which edits occurred before this word
            time_adjustment = 0.0
            for edit in edits:
                cut_start = edit.original_start + keep_start
                cut_end = edit.original_end - keep_end

                if word.start_time >= edit.original_end:
                    # Word is after this break entirely, subtract full removed time
                    time_adjustment += edit.time_removed
                elif word.start_time >= cut_end:
                    # Word is in the "keep_end" portion (last 3 seconds of break)
                    # These get shifted to right after the crossfade
                    # New position = keep_start + crossfade + (word_time - cut_end)
                    # Adjustment = word_time - new_position = word_time - (original_start + keep_start + crossfade + word_time - cut_end)
                    #            = cut_end - original_start - keep_start - crossfade
                    time_adjustment += (cut_end - edit.original_start - keep_start - crossfade_duration)
                elif word.start_time > cut_start:
                    # Word is in the cut section (should be rare - lyrics during instrumental)
                    # Place at the crossfade point
                    time_adjustment += (word.start_time - cut_start)

            new_word = Word(
                text=word.text,
                start_time=word.start_time - time_adjustment,
                end_time=word.end_time - time_adjustment,
                singer=word.singer,
            )
            new_words.append(new_word)

        adjusted_lines.append(Line(words=new_words, singer=line.singer))

    return adjusted_lines
