"""Audio-related utilities for karaoke generation."""

from pathlib import Path
import subprocess
from typing import Any
from pydub import AudioSegment

from .cache_identity import select_matching_cached_stem
from ....config import DEFAULT_CACHE_DIR
from ....utils.logging import get_logger

logger = get_logger(__name__)


def _trim_audio_with_ffmpeg(
    audio_path: str, start_time: float, output_path: str
) -> bool:
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start_time:.3f}",
                "-i",
                audio_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                output_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return Path(output_path).exists()


def _candidate_video_cache_dirs(cache_manager: Any, video_id: str) -> list[Path]:
    raw_primary = cache_manager.get_video_cache_dir(video_id)
    candidates: list[Path] = []
    if isinstance(raw_primary, (str, Path)):
        candidates.append(Path(raw_primary))
    shared = DEFAULT_CACHE_DIR / video_id
    if shared not in candidates:
        candidates.append(shared)
    return candidates


def _select_cached_stem_from_dirs(
    cache_dirs: list[Path], pattern: str, *, audio_path: str
) -> Path | None:
    for cache_dir in cache_dirs:
        stem_path = select_matching_cached_stem(
            cache_dir.glob(pattern),
            audio_path=audio_path,
        )
        if stem_path is not None:
            return stem_path
    return None


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
        logger.debug("📁 Using cached trimmed audio")
        return str(cache_manager.get_file_path(video_id, trimmed_name))

    logger.info(f"✂️ Trimming audio from {start_time:.2f}s")
    trimmed_path = cache_manager.get_file_path(video_id, trimmed_name)
    if _trim_audio_with_ffmpeg(audio_path, start_time, str(trimmed_path)):
        return str(trimmed_path)

    audio = AudioSegment.from_wav(audio_path)
    start_ms = int(start_time * 1000)
    if start_ms >= len(audio):
        logger.warning("Start time beyond audio length, using original")
        return audio_path

    trimmed = audio[start_ms:]
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
    cache_suffix: str = "",
) -> str:
    """Apply key shift and tempo effects to audio, caching the result."""
    if key_shift == 0 and tempo == 1.0:
        return audio_path

    effects_name = f"audio{cache_suffix}_key{key_shift:+d}_tempo{tempo:.2f}.wav"
    if not force and cache_manager.file_exists(video_id, effects_name):
        logger.debug("📁 Using cached processed audio")
        return str(cache_manager.get_file_path(video_id, effects_name))

    logger.info(f"🎛️ Applying effects: key={key_shift:+d}, tempo={tempo:.2f}x")
    output_path = cache_manager.get_file_path(video_id, effects_name)
    return audio_processor.process_audio(audio_path, str(output_path), key_shift, tempo)


def separate_vocals(
    audio_path: str,
    video_id: str,
    separator: Any,
    cache_manager: Any,
    force: bool = False,
) -> dict[str, str]:
    """Separate vocals and instrumental from audio, using cache if available."""
    cache_dirs = _candidate_video_cache_dirs(cache_manager, video_id)
    audio_filename = Path(audio_path).name.lower()
    if any(
        marker in audio_filename
        for marker in ["vocals", "instrumental", "drums", "bass", "other"]
    ):
        vocals_path = _select_cached_stem_from_dirs(
            cache_dirs,
            "*[Vv]ocals*.wav",
            audio_path=audio_path,
        )
        instrumental_path = _select_cached_stem_from_dirs(
            cache_dirs,
            "*[Ii]nstrumental*.wav",
            audio_path=audio_path,
        )
        if vocals_path and instrumental_path:
            return {
                "vocals_path": str(vocals_path),
                "instrumental_path": str(instrumental_path),
            }
        raise RuntimeError(
            f"Found separated file but missing vocals/instrumental: {audio_path}"
        )

    if not force:
        # Preserve the existing cache-manager lookup for the active run cache,
        # then fall back to the shared default cache roots.
        vocals_path = select_matching_cached_stem(
            cache_manager.find_files(video_id, "*[Vv]ocals*.wav"),
            audio_path=audio_path,
        )
        if vocals_path is None:
            vocals_path = _select_cached_stem_from_dirs(
                cache_dirs[1:],
                "*[Vv]ocals*.wav",
                audio_path=audio_path,
            )
        instrumental_path = select_matching_cached_stem(
            cache_manager.find_files(video_id, "*[Ii]nstrumental*.wav"),
            audio_path=audio_path,
        )
        if instrumental_path is None:
            instrumental_path = _select_cached_stem_from_dirs(
                cache_dirs[1:],
                "*[Ii]nstrumental*.wav",
                audio_path=audio_path,
            )
        if vocals_path and instrumental_path:
            logger.info("📁 Using cached vocal separation")
            return {
                "vocals_path": str(vocals_path),
                "instrumental_path": str(instrumental_path),
            }

    cache_dir = cache_manager.get_video_cache_dir(video_id)
    return separator.separate_vocals(audio_path, str(cache_dir))
