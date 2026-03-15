"""Audio-related utilities for karaoke generation."""

from pathlib import Path
import re
import subprocess
from typing import Any
from pydub import AudioSegment

from .cache_identity import select_matching_cached_stem
from ....config import DEFAULT_CACHE_DIR
from ....utils.logging import get_logger

logger = get_logger(__name__)
_TRIMMED_AUDIO_RE = re.compile(r"trimmed_from_(\d+(?:\.\d+)?)s$", re.IGNORECASE)


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


def _select_cached_trimmed_audio(
    cache_dirs: list[Path],
    trimmed_name: str,
) -> Path | None:
    for cache_dir in cache_dirs:
        candidate = cache_dir / trimmed_name
        if candidate.exists():
            return candidate
    return None


def _parse_trimmed_start_time(audio_path: str) -> float | None:
    match = _TRIMMED_AUDIO_RE.match(Path(audio_path).stem)
    if match is None:
        return None
    return float(match.group(1))


def _select_full_length_cached_stem(
    cache_dirs: list[Path],
    pattern: str,
) -> Path | None:
    candidates: list[Path] = []
    for cache_dir in cache_dirs:
        for candidate in cache_dir.glob(pattern):
            if candidate.stem.lower().startswith("trimmed_from_"):
                continue
            candidates.append(candidate)
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0.0)
    return candidates[-1]


def _derive_trimmed_stem_paths(
    *,
    cache_dir: Path,
    start_time: float,
    vocals_stem: Path,
    instrumental_stem: Path,
    force: bool,
) -> dict[str, Path] | None:
    trimmed_vocals = (
        cache_dir / f"trimmed_from_{start_time:.2f}s_(Vocals)_htdemucs_ft.wav"
    )
    trimmed_instrumental = (
        cache_dir / f"trimmed_from_{start_time:.2f}s_instrumental.wav"
    )
    if force or not trimmed_vocals.exists():
        if not _trim_audio_with_ffmpeg(
            str(vocals_stem), start_time, str(trimmed_vocals)
        ):
            return None
    if force or not trimmed_instrumental.exists():
        if not _trim_audio_with_ffmpeg(
            str(instrumental_stem), start_time, str(trimmed_instrumental)
        ):
            return None
    return {
        "vocals_path": trimmed_vocals,
        "instrumental_path": trimmed_instrumental,
    }


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
    if not force:
        cached_trimmed = _select_cached_trimmed_audio(
            _candidate_video_cache_dirs(cache_manager, video_id)[1:],
            trimmed_name,
        )
        if cached_trimmed is not None:
            logger.debug("📁 Using shared cached trimmed audio")
            return str(cached_trimmed)

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
    trimmed_start_time = _parse_trimmed_start_time(audio_path)
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

    if trimmed_start_time is not None:
        full_vocals = _select_full_length_cached_stem(
            cache_dirs,
            "*[Vv]ocals*.wav",
        )
        full_instrumental = _select_full_length_cached_stem(
            cache_dirs,
            "*[Ii]nstrumental*.wav",
        )
        if full_vocals is not None and full_instrumental is not None:
            derived = _derive_trimmed_stem_paths(
                cache_dir=cache_dirs[0],
                start_time=trimmed_start_time,
                vocals_stem=full_vocals,
                instrumental_stem=full_instrumental,
                force=force,
            )
            if derived is not None:
                logger.info("📁 Reused full-song cached stems for trimmed clip")
                return {
                    "vocals_path": str(derived["vocals_path"]),
                    "instrumental_path": str(derived["instrumental_path"]),
                }

    cache_dir = cache_manager.get_video_cache_dir(video_id)
    return separator.separate_vocals(audio_path, str(cache_dir))
