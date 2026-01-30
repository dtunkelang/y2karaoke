"""Audio analysis for alignment."""

from typing import List, Tuple, Optional
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


def detect_audio_silence_regions(
    vocals_path: str,
    min_silence_duration: float = 5.0,
    energy_threshold_percentile: float = 15.0,
    sample_rate: int = 22050,
) -> List[Tuple[float, float]]:
    """Detect silence/low-energy regions in vocals audio.

    Args:
        vocals_path: Path to vocals audio file
        min_silence_duration: Minimum duration to consider a silence region
        energy_threshold_percentile: RMS percentile below which is silence
        sample_rate: Sample rate for analysis

    Returns:
        List of (start, end) tuples for each silence region
    """
    try:
        import librosa

        y, sr = librosa.load(vocals_path, sr=sample_rate)

        hop_length = 512
        frame_length = 2048
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

        # Use absolute threshold based on noise floor to peak ratio
        # This is more reliable than percentile for detecting instrumental breaks
        # where some guitar might bleed through but vocals are absent
        noise_floor = np.percentile(rms, 5)
        peak_level = np.percentile(rms, 95)
        # Threshold at noise_floor + 10% of dynamic range - catches instrumental bleed
        threshold = noise_floor + 0.10 * (peak_level - noise_floor)
        logger.debug(f"Silence detection: noise={noise_floor:.4f}, peak={peak_level:.4f}, threshold={threshold:.4f}")

        # Find silence regions
        is_silent = rms < threshold
        silences = []
        in_silence = False
        silence_start = 0

        for i, (t, silent) in enumerate(zip(times, is_silent)):
            if silent and not in_silence:
                in_silence = True
                silence_start = t
            elif not silent and in_silence:
                in_silence = False
                silence_end = t
                duration = silence_end - silence_start
                if duration >= min_silence_duration:
                    silences.append((silence_start, silence_end))

        # Handle case ending in silence
        if in_silence:
            silence_end = times[-1]
            duration = silence_end - silence_start
            if duration >= min_silence_duration:
                silences.append((silence_start, silence_end))

        logger.debug(f"Detected {len(silences)} silence regions in audio")
        for i, (start, end) in enumerate(silences):
            logger.debug(f"  Silence {i}: {start:.1f}s - {end:.1f}s ({end-start:.1f}s)")

        return silences

    except Exception as e:
        logger.warning(f"Failed to detect audio silences: {e}")
        return []


def detect_lrc_gaps(
    line_timings: List[Tuple[float, str]],
    min_gap_duration: float = 5.0,
) -> List[Tuple[float, float]]:
    """Detect large gaps between LRC lines.

    Args:
        line_timings: List of (timestamp, text) from parsed LRC
        min_gap_duration: Minimum gap to consider

    Returns:
        List of (start, end) tuples for each gap
    """
    if len(line_timings) < 2:
        return []

    gaps = []
    for i in range(len(line_timings) - 1):
        current_time = line_timings[i][0]
        next_time = line_timings[i + 1][0]
        gap_duration = next_time - current_time

        if gap_duration >= min_gap_duration:
            gaps.append((current_time, next_time))
            logger.debug(f"LRC gap {len(gaps)}: {current_time:.1f}s - {next_time:.1f}s ({gap_duration:.1f}s)")

    return gaps


def calculate_gap_adjustments(
    lrc_gaps: List[Tuple[float, float]],
    audio_silences: List[Tuple[float, float]],
    tolerance: float = 10.0,
) -> List[Tuple[float, float]]:
    """Match LRC gaps to audio silences and calculate timing adjustments.

    Args:
        lrc_gaps: Gaps detected in LRC
        audio_silences: Silence regions in audio
        tolerance: Maximum difference in start time to consider a match

    Returns:
        List of (gap_end_time, cumulative_adjustment) tuples
        adjustment is positive if audio silence is longer than LRC gap
    """
    adjustments = []
    cumulative_adj = 0.0

    for lrc_start, lrc_end in lrc_gaps:
        lrc_duration = lrc_end - lrc_start

        # Find matching audio silence (accounting for previous adjustments)
        adjusted_lrc_start = lrc_start + cumulative_adj

        best_match = None
        best_overlap = 0

        for audio_start, audio_end in audio_silences:
            # Check if this audio silence overlaps with the adjusted LRC gap position
            if abs(audio_start - adjusted_lrc_start) < tolerance:
                audio_duration = audio_end - audio_start
                # Use overlap as match quality metric
                overlap_start = max(audio_start, adjusted_lrc_start)
                overlap_end = min(audio_end, lrc_end + cumulative_adj)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = (audio_start, audio_end, audio_duration)

        if best_match:
            audio_start, audio_end, audio_duration = best_match
            duration_diff = audio_duration - lrc_duration

            if abs(duration_diff) > 1.0:  # Only adjust for differences > 1s
                logger.info(f"Gap at LRC {lrc_start:.1f}s: LRC={lrc_duration:.1f}s, audio={audio_duration:.1f}s, diff={duration_diff:+.1f}s")
                cumulative_adj += duration_diff
                adjustments.append((lrc_end, cumulative_adj))
        else:
            logger.debug(f"No audio match for LRC gap at {lrc_start:.1f}s")

    if adjustments:
        logger.info(f"Total timing adjustment: {cumulative_adj:+.1f}s")

    return adjustments


def adjust_timing_for_duration_mismatch(
    lines: list,
    line_timings: List[Tuple[float, str]],
    vocals_path: str,
    lrc_duration: Optional[int] = None,
    audio_duration: Optional[int] = None,
) -> list:
    """Adjust lyrics timing based on duration mismatches between LRC and audio.

    This handles cases where the LRC was created for a different version of the
    song (e.g., radio edit vs. album version with extended instrumental breaks).

    Strategy:
    1. Detect large gaps in LRC and corresponding silences in audio
    2. For each gap, calculate if there's a duration difference
    3. Apply cumulative adjustments to lines after each gap

    Args:
        lines: List of Line objects with timing
        line_timings: Original LRC line timings
        vocals_path: Path to vocals audio
        lrc_duration: Duration implied by LRC (optional)
        audio_duration: Actual audio duration (optional)

    Returns:
        Lines with adjusted timing
    """
    from .models import Line, Word

    # Skip if durations are close enough
    if lrc_duration and audio_duration:
        diff = abs(audio_duration - lrc_duration)
        if diff <= 10:
            logger.debug(f"Duration difference ({diff}s) within tolerance, no gap adjustment needed")
            return lines

    # Detect gaps in LRC
    lrc_gaps = detect_lrc_gaps(line_timings, min_gap_duration=10.0)
    if not lrc_gaps:
        logger.debug("No significant LRC gaps detected")
        return lines

    # Detect silence regions in audio
    audio_silences = detect_audio_silence_regions(vocals_path, min_silence_duration=10.0)
    if not audio_silences:
        logger.debug("No significant audio silences detected")
        return lines

    # Match LRC gaps to audio silences and calculate adjustments
    # The key insight: we need to align where vocals RESUME after each break
    # LRC gap end = when LRC expects next line
    # Audio silence end = when vocals actually resume
    # Adjustment = audio_silence_end - lrc_gap_end
    adjustments = []  # List of (lrc_gap_end, cumulative_adjustment)
    cumulative_adj = 0.0

    for lrc_start, lrc_end in lrc_gaps:
        lrc_gap_duration = lrc_end - lrc_start

        # Find audio silence that corresponds to this LRC gap
        # Account for previous adjustments when matching
        adjusted_lrc_start = lrc_start + cumulative_adj
        best_match = None
        best_distance = float('inf')

        for audio_start, audio_end in audio_silences:
            # Match based on where singing should resume (end of gap/silence)
            adjusted_lrc_end = lrc_end + cumulative_adj
            distance = abs(audio_end - adjusted_lrc_end)
            # Also check start alignment
            start_distance = abs(audio_start - adjusted_lrc_start)

            # Use minimum of start or end distance
            match_distance = min(distance, start_distance)
            if match_distance < best_distance and match_distance < 20.0:
                best_distance = match_distance
                best_match = (audio_start, audio_end)

        if best_match:
            audio_start, audio_end = best_match

            # Calculate adjustment: where do vocals actually resume vs where LRC expects
            # If audio_end > lrc_end + cumulative_adj: audio has longer silence, shift lyrics later
            # If audio_end < lrc_end + cumulative_adj: audio has shorter silence, shift lyrics earlier
            adjusted_lrc_end = lrc_end + cumulative_adj
            end_diff = audio_end - adjusted_lrc_end

            if abs(end_diff) > 1.0:  # Only adjust for differences > 1s
                logger.info(f"Gap at LRC {lrc_start:.1f}s: LRC resumes at {lrc_end:.1f}s, "
                           f"audio resumes at {audio_end:.1f}s, shift={end_diff:+.1f}s")
                cumulative_adj += end_diff
                adjustments.append((lrc_end, cumulative_adj))
        else:
            logger.debug(f"No audio silence match for LRC gap at {lrc_start:.1f}s")

    if not adjustments:
        logger.debug("No timing adjustments calculated")
        return lines

    logger.info(f"Total timing adjustment: {cumulative_adj:+.1f}s")

    # Apply adjustments to lines
    adjusted_lines = []

    for line in lines:
        if not line.words:
            adjusted_lines.append(line)
            continue

        # Find which adjustment applies to this line
        line_start = line.start_time
        adjustment = 0.0

        for gap_end, adj in adjustments:
            if line_start >= gap_end:
                adjustment = adj

        if adjustment == 0.0:
            adjusted_lines.append(line)
            continue

        # Apply adjustment to all words in this line
        new_words = []
        for word in line.words:
            new_words.append(Word(
                text=word.text,
                start_time=word.start_time + adjustment,
                end_time=word.end_time + adjustment,
                singer=word.singer,
            ))

        adjusted_lines.append(Line(words=new_words, singer=line.singer))

        line_text = " ".join(w.text for w in line.words[:3])
        logger.debug(f"Adjusted '{line_text}...' by {adjustment:+.1f}s")

    return adjusted_lines


def detect_song_start(audio_path: str, min_duration: float = 0.3) -> float:
    """
    Detect where vocals actually start in the audio using onset detection.

    Uses librosa onset detection focused on vocal frequency range (100Hz-4kHz)
    combined with energy threshold validation for robust detection.
    """
    try:
        import librosa
        import numpy as np

        # Load at 22050Hz for better frequency resolution
        y, sr = librosa.load(audio_path, sr=22050)

        # Focus on vocal frequency range (100Hz - 4kHz) to filter out
        # low bass/drums and high-frequency artifacts
        y_vocal_band = _bandpass_filter(y, sr, low_freq=100, high_freq=4000)

        # Onset detection with backtracking for precise timing
        hop_length = 512
        onset_env = librosa.onset.onset_strength(
            y=y_vocal_band,
            sr=sr,
            hop_length=hop_length,
            aggregate=np.median  # More robust to outliers
        )

        # Find onsets with backtracking to get actual start times
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
            backtrack=True,
            units='time'
        )

        if len(onsets) == 0:
            logger.warning("No onsets detected, falling back to RMS method")
            return _detect_song_start_rms(y, sr, min_duration)

        # Validate onset with sustained energy check
        # The first onset should be followed by sustained vocal activity
        rms = librosa.feature.rms(y=y_vocal_band, hop_length=hop_length)[0]
        noise_floor = np.percentile(rms, 10)
        peak_level = np.percentile(rms, 90)
        energy_threshold = noise_floor + 0.1 * (peak_level - noise_floor)

        for onset_time in onsets:
            # Check if there's sustained energy after this onset
            onset_frame = int(onset_time * sr / hop_length)
            min_frames = int(min_duration * sr / hop_length)

            if onset_frame + min_frames >= len(rms):
                continue

            # Check for sustained activity in the frames following the onset
            frames_above = rms[onset_frame:onset_frame + min_frames] > energy_threshold
            if np.mean(frames_above) > 0.6:  # At least 60% of frames above threshold
                logger.debug(f"Detected vocal onset at {onset_time:.2f}s")
                return onset_time

        # If no validated onset, use the first onset anyway
        logger.debug(f"Using first onset at {onsets[0]:.2f}s (not strongly validated)")
        return onsets[0]

    except Exception as e:
        logger.warning(f"Could not detect song start: {e}")
        return 0.0


def _bandpass_filter(y, sr: int, low_freq: int = 100, high_freq: int = 4000):
    """Apply a bandpass filter to isolate vocal frequencies."""
    import numpy as np
    from scipy import signal

    # Design butterworth bandpass filter
    nyquist = sr / 2
    low = low_freq / nyquist
    high = min(high_freq / nyquist, 0.99)  # Stay below Nyquist

    # Use 4th order filter for good selectivity without ringing
    b, a = signal.butter(4, [low, high], btype='band')

    # Apply filter with zero-phase (filtfilt) for no time delay
    y_filtered = signal.filtfilt(b, a, y)

    return y_filtered


def _detect_song_start_rms(y, sr: int, min_duration: float = 0.3) -> float:
    """
    Fallback RMS-based detection for when onset detection fails.
    """
    import librosa
    import numpy as np

    frame_length = int(0.05 * sr)
    hop_length = frame_length // 2

    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    noise_floor = np.percentile(rms, 10)
    peak_level = np.percentile(rms, 95)
    threshold = noise_floor + 0.15 * (peak_level - noise_floor)

    above = rms > threshold
    min_frames = int(min_duration * sr / hop_length)

    consecutive = 0
    for i, active in enumerate(above):
        if active:
            consecutive += 1
            if consecutive >= min_frames:
                start_frame = i - consecutive + 1
                start_time = start_frame * hop_length / sr
                logger.debug(f"RMS fallback: detected vocal start at {start_time:.2f}s")
                return start_time
        else:
            consecutive = 0

    logger.warning("No sustained vocal activity detected, assuming start at 0")
    return 0.0
