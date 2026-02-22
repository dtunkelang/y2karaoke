"""Media resolution helpers for visual bootstrap workflows."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable, Optional


def extract_audio_from_video(
    video_path: Path,
    output_dir: Path,
    *,
    run_fn: Callable[..., Any] = subprocess.run,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / f"{video_path.stem}.extracted.wav"
    if audio_path.exists():
        return audio_path
    run_fn(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "44100",
            str(audio_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not audio_path.exists():
        raise RuntimeError(f"Audio extraction failed for {video_path}")
    return audio_path


def resolve_media_paths(
    downloader: Any,
    candidate_url: str,
    cached_video_path: Optional[Path],
    song_dir: Path,
    *,
    extract_audio_fn: Callable[[Path, Path], Path] = extract_audio_from_video,
    log_fn: Optional[Callable[[str], None]] = None,
) -> tuple[Path, Path]:
    if cached_video_path is None:
        vid_info = downloader.download_video(candidate_url)
        v_path = Path(vid_info["video_path"])
    else:
        v_path = cached_video_path

    a_path: Optional[Path] = None
    if cached_video_path is not None:
        try:
            a_path = extract_audio_fn(v_path, song_dir / "video")
            if log_fn:
                log_fn(f"Extracted audio from cached candidate video: {a_path}")
        except Exception as e:
            if log_fn:
                log_fn(
                    "Could not extract audio from cached video (%s); falling back to "
                    "direct audio download." % e
                )
    if a_path is None:
        aud_info = downloader.download_audio(candidate_url)
        a_path = Path(aud_info["audio_path"])
    return v_path, a_path
