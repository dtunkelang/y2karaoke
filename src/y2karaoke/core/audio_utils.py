"""Audio-related utilities for karaoke generation."""

from pathlib import Path
from typing import Any
from pydub import AudioSegment

from ..utils.logging import get_logger

logger = get_logger(__name__)


def trim_audio_if_needed(
    audio_path: str,
    start_time: float,
    video_id: str,
    cache_manager: Any,
    force: bool = False,
) -> str:
    """Trim the audio from start_time onward, caching the result."""
    if start_time <= 0:
        return audio_path

    trimmed_name = f"trimmed_from_{start_time:.2f}s.wav"
    if not force and cache_manager.file_exists(video_id, trimmed_name):
        logger.info("📁 Using cached trimmed audio")
        return str(cache_manager.get_file_path(video_id, trimmed_name))

    logger.info(f"✂️ Trimming audio from {start_time:.2f}s")
    audio = AudioSegment.from_wav(audio_path)
    start_ms = int(start_time * 1000)
    if start_ms >= len(audio):
        logger.warning("Start time beyond audio length, using original")
        return audio_path

    trimmed = audio[start_ms:]
    trimmed_path = cache_manager.get_file_path(video_id, trimmed_name)
    trimmed.export(str(trimmed_path), format="wav")
    return str(trimmed_path)


def apply_audio_effects(
    audio_path: str,
    key_shift: int,
    tempo: float,
    video_id: str,
    cache_manager: Any,
    audio_processor: Any,
    force: bool = False,
) -> str:
    """Apply key shift and tempo effects to audio, caching the result."""
    if key_shift == 0 and tempo == 1.0:
        return audio_path

    effects_name = f"instrumental_key{key_shift:+d}_tempo{tempo:.2f}.wav"
    if not force and cache_manager.file_exists(video_id, effects_name):
        logger.info("📁 Using cached processed audio")
        return str(cache_manager.get_file_path(video_id, effects_name))

    logger.info(f"🎛️ Applying effects: key={key_shift:+d}, tempo={tempo:.2f}x")
    output_path = cache_manager.get_file_path(video_id, effects_name)
    return audio_processor.process_audio(audio_path, str(output_path), key_shift, tempo)
