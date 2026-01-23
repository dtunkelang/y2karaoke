"""Audio analysis for alignment."""

from ..utils.logging import get_logger

logger = get_logger(__name__)


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
