"""Audio analysis utilities for timing evaluation."""

from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path

from ..utils.logging import get_logger
from .components.alignment.timing_models import AudioFeatures

logger = get_logger(__name__)


def _load_librosa():
    import librosa

    return librosa


def _get_audio_features_cache_path(vocals_path: str) -> Optional[str]:
    """Get the cache file path for audio features."""
    vocals_file = Path(vocals_path)
    if not vocals_file.exists():
        return None

    cache_name = f"{vocals_file.stem}_audio_features.npz"
    return str(vocals_file.parent / cache_name)


def _load_audio_features_cache(cache_path: str) -> Optional[AudioFeatures]:
    """Load cached audio features if available."""
    cache_file = Path(cache_path)
    if not cache_file.exists():
        return None

    try:
        data = np.load(cache_file, allow_pickle=True)

        # Reconstruct AudioFeatures
        features = AudioFeatures(
            onset_times=data["onset_times"],
            silence_regions=list(data["silence_regions"]),
            vocal_start=float(data["vocal_start"]),
            vocal_end=float(data["vocal_end"]),
            duration=float(data["duration"]),
            energy_envelope=data["energy_envelope"],
            energy_times=data["energy_times"],
        )

        logger.debug(
            f"Loaded cached audio features: {len(features.onset_times)} onsets"
        )
        return features

    except Exception as e:
        logger.debug(f"Failed to load audio features cache: {e}")
        return None


def _save_audio_features_cache(cache_path: str, features: AudioFeatures) -> None:
    """Save audio features to cache."""
    try:
        np.savez(
            cache_path,
            onset_times=features.onset_times,
            silence_regions=np.array(features.silence_regions, dtype=object),
            vocal_start=features.vocal_start,
            vocal_end=features.vocal_end,
            duration=features.duration,
            energy_envelope=features.energy_envelope,
            energy_times=features.energy_times,
        )
        logger.debug(f"Saved audio features to cache: {cache_path}")

    except Exception as e:
        logger.debug(f"Failed to save audio features cache: {e}")


def extract_audio_features(
    vocals_path: str,
    min_silence_duration: float = 0.5,
) -> Optional[AudioFeatures]:
    """Extract audio features for timing evaluation.

    Results are cached to disk alongside the vocals file to avoid
    expensive re-extraction on subsequent runs.

    Args:
        vocals_path: Path to vocals audio file
        min_silence_duration: Minimum duration (seconds) to consider as silence

    Returns:
        AudioFeatures object or None if extraction fails
    """
    # Check cache first
    cache_path = _get_audio_features_cache_path(vocals_path)
    if cache_path:
        cached = _load_audio_features_cache(cache_path)
        if cached:
            return cached

    try:
        librosa = _load_librosa()

        # Load audio
        y, sr = librosa.load(vocals_path, sr=22050)
        duration = len(y) / sr
        hop_length = 512

        # Onset detection
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=hop_length, backtrack=True, units="frames"
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

        # RMS energy for silence detection
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        rms_times = librosa.frames_to_time(
            np.arange(len(rms)), sr=sr, hop_length=hop_length
        )

        # Compute silence threshold
        noise_floor = np.percentile(rms, 10)
        peak_level = np.percentile(rms, 90)
        silence_threshold = noise_floor + 0.15 * (peak_level - noise_floor)

        # Find silence regions
        is_silent = rms < silence_threshold
        silence_regions = _find_silence_regions(
            is_silent, rms_times, min_silence_duration
        )

        # Find vocal start and end
        vocal_start = _find_vocal_start(onset_times, rms, rms_times, silence_threshold)
        vocal_end = _find_vocal_end(rms, rms_times, silence_threshold)

        features = AudioFeatures(
            onset_times=onset_times,
            silence_regions=silence_regions,
            vocal_start=vocal_start,
            vocal_end=vocal_end,
            duration=duration,
            energy_envelope=rms,
            energy_times=rms_times,
        )

        # Save to cache
        if cache_path:
            _save_audio_features_cache(cache_path, features)

        return features

    except Exception as e:
        logger.error(f"Failed to extract audio features: {e}")
        return None


def _find_silence_regions(
    is_silent: np.ndarray,
    times: np.ndarray,
    min_duration: float,
) -> List[Tuple[float, float]]:
    """Find regions of sustained silence."""
    regions = []
    in_silence = False
    silence_start = 0.0

    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            in_silence = True
            silence_start = times[i]
        elif not silent and in_silence:
            in_silence = False
            silence_end = times[i]
            if silence_end - silence_start >= min_duration:
                regions.append((silence_start, silence_end))

    # Handle trailing silence
    if in_silence:
        silence_end = times[-1]
        if silence_end - silence_start >= min_duration:
            regions.append((silence_start, silence_end))

    return regions


def _compute_silence_overlap(
    start: float, end: float, silence_regions: List[Tuple[float, float]]
) -> float:
    """Compute total silence duration between two timestamps."""
    if start >= end or not silence_regions:
        return 0.0

    overlap = 0.0
    for silence_start, silence_end in silence_regions:
        if silence_end <= start or silence_start >= end:
            continue
        overlap += min(end, silence_end) - max(start, silence_start)
    return max(overlap, 0.0)


def _is_time_in_silence(
    time: float, silence_regions: List[Tuple[float, float]]
) -> bool:
    """Return True if the given time falls inside any silence region."""
    for silence_start, silence_end in silence_regions:
        if silence_start <= time <= silence_end:
            return True
    return False


def _find_vocal_start(
    onset_times: np.ndarray,
    rms: np.ndarray,
    rms_times: np.ndarray,
    threshold: float,
    min_duration: float = 0.3,
) -> float:
    """Find where vocals actually start."""
    if len(onset_times) == 0:
        return 0.0

    # Validate onsets with sustained energy
    for onset_time in onset_times:
        onset_idx = np.searchsorted(rms_times, onset_time)
        if onset_idx >= len(rms):
            continue

        # Check for sustained energy after onset
        min_frames = int(
            min_duration / (rms_times[1] - rms_times[0]) if len(rms_times) > 1 else 10
        )
        end_idx = min(onset_idx + min_frames, len(rms))

        if end_idx > onset_idx:
            frames_above = rms[onset_idx:end_idx] > threshold
            if np.mean(frames_above) > 0.5:
                return onset_time

    return onset_times[0] if len(onset_times) > 0 else 0.0


def _find_vocal_end(
    rms: np.ndarray,
    rms_times: np.ndarray,
    threshold: float,
    min_silence: float = 0.5,
) -> float:
    """Find where vocals end (last sustained activity)."""
    if len(rms) == 0:
        return 0.0

    # Find last frame above threshold with sustained activity before it
    for i in range(len(rms) - 1, -1, -1):
        if rms[i] > threshold:
            return rms_times[i]

    return rms_times[-1]


def _check_vocal_activity_in_range(
    start_time: float,
    end_time: float,
    audio_features: AudioFeatures,
) -> float:
    """Check how much vocal activity exists in a time range.

    Looks for TRUE SILENCE (energy near zero) vs any vocal activity.
    True silence has energy < 2% of peak. Anything above that is activity.

    Returns:
        Fraction of the time range that has vocal activity (0.0 to 1.0)
    """
    # Find frames in this range
    mask = (audio_features.energy_times >= start_time) & (
        audio_features.energy_times <= end_time
    )
    range_energy = audio_features.energy_envelope[mask]

    if len(range_energy) == 0:
        return 0.0

    # Normalize to peak energy
    peak_level = np.percentile(audio_features.energy_envelope, 99)
    if peak_level == 0:
        return 0.0

    range_energy_norm = range_energy / peak_level

    # True silence threshold: < 2% of peak energy
    # This catches actual pauses but not quiet singing
    silence_threshold = 0.02

    # Count frames above silence threshold (i.e., frames with any vocal activity)
    active_frames = np.sum(range_energy_norm > silence_threshold)
    activity_ratio = active_frames / len(range_energy) if len(range_energy) > 0 else 0.0
    return float(activity_ratio)


def _check_for_silence_in_range(
    start_time: float,
    end_time: float,
    audio_features: AudioFeatures,
    min_silence_duration: float = 0.5,
) -> bool:
    """Check if there's a significant silence region within a time range.

    Even if there's some vocal activity in the range, if there's also
    a sustained silence period, it indicates a real phrase boundary.

    Args:
        start_time: Start of range to check
        end_time: End of range to check
        audio_features: Audio features with energy envelope
        min_silence_duration: Minimum silence duration to detect

    Returns:
        True if there's a silence region >= min_silence_duration
    """
    times = audio_features.energy_times
    energy = audio_features.energy_envelope

    # Use 2% of peak as silence threshold
    peak_level = np.max(energy) if len(energy) > 0 else 1.0
    silence_threshold = 0.02 * peak_level

    # Find indices for the range
    start_idx = int(np.searchsorted(times, start_time))
    end_idx = int(np.searchsorted(times, end_time))

    if start_idx >= end_idx or start_idx >= len(energy):
        return False

    # Scan for silence regions
    in_silence = False
    silence_start_time = 0.0

    for i in range(start_idx, min(end_idx, len(times))):
        t = times[i]
        is_silent = energy[i] < silence_threshold

        if is_silent and not in_silence:
            in_silence = True
            silence_start_time = t
        elif not is_silent and in_silence:
            silence_duration = t - silence_start_time
            if silence_duration >= min_silence_duration:
                return True
            in_silence = False

    # Check if still in silence at end of range
    if in_silence:
        silence_duration = (
            min(end_time, times[min(end_idx, len(times) - 1)]) - silence_start_time
        )
        if silence_duration >= min_silence_duration:
            return True

    return False
