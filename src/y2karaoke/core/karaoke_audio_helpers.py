"""Audio helper implementations used by KaraokeGenerator."""

import json
from typing import Any, List, Tuple

from ..utils.logging import get_logger

logger = get_logger(__name__)


def append_outro_line_impl(
    lines,
    outro_line: str,
    audio_path: str,
    *,
    min_tail: float = 0.5,
) -> None:
    """Append a final lyric line near the end of the audio."""
    if not lines or not outro_line.strip():
        return

    from moviepy import AudioFileClip
    from ..config import OUTRO_DELAY
    from .models import Line, Word

    with AudioFileClip(audio_path) as clip:
        audio_duration = clip.duration

    last_end = lines[-1].end_time
    end_time = max(last_end + min_tail, audio_duration - OUTRO_DELAY)
    duration = min(3.0, max(1.5, end_time - last_end))
    start_time = max(last_end + min_tail, end_time - duration)
    if end_time <= start_time + 0.2:
        return

    tokens = [t for t in outro_line.strip().split() if t]
    if not tokens:
        return
    spacing = duration / len(tokens)
    words = []
    for i, token in enumerate(tokens):
        start = start_time + i * spacing
        end = start + spacing * 0.9
        words.append(Word(text=token, start_time=start, end_time=end))
    lines.append(Line(words=words))


def shorten_breaks_impl(
    audio_path: str,
    vocals_path: str,
    instrumental_path: str,
    video_id: str,
    max_break_duration: float,
    *,
    cache_manager,
    force: bool = False,
    cache_suffix: str = "",
) -> Tuple[str, List[Any]]:
    """Shorten long instrumental breaks in the given audio track."""
    from .break_shortener import BreakEdit, shorten_instrumental_breaks

    shortened_name = f"shortened_breaks_{max_break_duration:.0f}s{cache_suffix}.wav"
    edits_name = f"shortened_breaks_{max_break_duration:.0f}s_edits.json"

    if not force and cache_manager.file_exists(video_id, shortened_name):
        edits_path = cache_manager.get_file_path(video_id, edits_name)
        if edits_path.exists():
            try:
                with open(edits_path) as f:
                    edits_data = json.load(f)
                edits = [
                    BreakEdit(
                        original_start=e["original_start"],
                        original_end=e["original_end"],
                        new_end=e["new_end"],
                        time_removed=e["time_removed"],
                        cut_start=e.get("cut_start", 0.0),
                    )
                    for e in edits_data
                ]
                logger.info(
                    f"📁 Using cached shortened audio ({len(edits)} break edits)"
                )
                return str(cache_manager.get_file_path(video_id, shortened_name)), edits
            except (
                OSError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as e:
                logger.debug(f"Could not load cached edits: {e}")
        else:
            logger.info("📁 Using cached shortened audio (no edits)")
            return str(cache_manager.get_file_path(video_id, shortened_name)), []

    logger.info(
        f"✂️ Shortening instrumental breaks longer than {max_break_duration:.0f}s..."
    )
    output_path = cache_manager.get_file_path(video_id, shortened_name)

    shortened_path, edits = shorten_instrumental_breaks(
        audio_path,
        vocals_path,
        str(output_path),
        max_break_duration=max_break_duration,
        beat_reference_path=instrumental_path,
    )

    if edits:
        edits_path = cache_manager.get_file_path(video_id, edits_name)
        if not edits_path.exists():
            edits_data = [
                {
                    "original_start": e.original_start,
                    "original_end": e.original_end,
                    "new_end": e.new_end,
                    "time_removed": e.time_removed,
                    "cut_start": e.cut_start,
                }
                for e in edits
            ]
            with open(edits_path, "w") as f:
                json.dump(edits_data, f)

    return shortened_path, edits
