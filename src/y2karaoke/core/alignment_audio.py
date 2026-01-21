"""Audio analysis helpers for forced alignment."""

from ..utils.logging import get_logger

logger = get_logger(__name__)


def detect_song_start(audio_path: str, min_duration: float = 0.3) -> float:
    """
    Detect where vocals actually start in the audio.
    """
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(audio_path, sr=16000)

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
                    logger.info(f"Detected vocal start at {start_time:.2f}s")
                    return start_time
            else:
                consecutive = 0

        logger.warning("No sustained vocal activity detected, assuming start at 0")
        return 0.0

    except Exception as e:
        logger.warning(f"Could not detect song start: {e}")
        return 0.0


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    try:
        import librosa
        return librosa.get_duration(path=audio_path)
    except Exception as e:
        logger.warning(f"Could not get audio duration: {e}")
        return 180.0
