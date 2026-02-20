"""High-level karaoke video writing using MoviePy."""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List
from moviepy import AudioFileClip, VideoClip

from ....config import VIDEO_WIDTH, VIDEO_HEIGHT, FPS, FONT_SIZE
from ....utils.logging import get_logger
from ....utils.fonts import get_font
from ....utils.validation import validate_line_order
from .frame_generation import FrameGenerator
from .progress import ConsoleProgressBar
from ...models import Line, SongMetadata
from .backgrounds_static import create_gradient_background

if TYPE_CHECKING:
    from .backgrounds import BackgroundSegment


logger = get_logger(__name__)


def render_karaoke_video(
    lines: list[Line],
    audio_path: str,
    output_path: str,
    title: Optional[str] = None,
    artist: Optional[str] = None,
    timing_offset: float = 0.0,
    background_segments: Optional[list[BackgroundSegment]] = None,
    song_metadata: Optional[SongMetadata] = None,
    show_progress: bool = True,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[int] = None,
    font_size: Optional[int] = None,
) -> str:
    """Render karaoke video using MoviePy (frame-by-frame)."""
    validate_line_order(lines)

    video_width = width or VIDEO_WIDTH
    video_height = height or VIDEO_HEIGHT
    video_fps = fps or FPS
    lyrics_font_size = font_size or FONT_SIZE

    logger.info("Rendering karaoke video...")
    logger.info(
        f"Resolution: {video_width}x{video_height}, FPS: {video_fps}, Font: {lyrics_font_size}px"
    )
    if timing_offset != 0:
        logger.info(f"Applying timing offset: {timing_offset:+.2f}s")

    audio = AudioFileClip(audio_path)
    audio_duration = audio.duration

    OUTRO_DURATION = 5.0
    last_lyrics_end = lines[-1].end_time if lines else 0
    duration = max(audio_duration, last_lyrics_end) + OUTRO_DURATION

    font = get_font(lyrics_font_size)
    static_background = create_gradient_background(video_width, video_height)
    is_duet = song_metadata.is_duet if song_metadata else False

    total_frames = int(duration * video_fps)
    progress_bar = ConsoleProgressBar(total_frames) if show_progress else None

    generator = FrameGenerator(
        lines=lines,
        timing_offset=timing_offset,
        video_width=video_width,
        video_height=video_height,
        font=font,
        static_background=static_background,
        background_segments=background_segments,
        audio_duration=audio_duration,
        title=title,
        artist=artist,
        is_duet=is_duet,
    )

    def make_frame(t):
        if progress_bar:
            progress_bar.update()

        return generator.generate_frame(t)

    logger.info(
        f"Creating video ({duration:.1f}s at {video_fps}fps, {total_frames} frames)..."
    )
    video = VideoClip(make_frame, duration=duration)
    video = video.with_fps(video_fps)
    video = video.with_audio(audio)

    logger.info(f"Writing video to {output_path}...")
    if show_progress:
        print()
    video.write_videofile(
        output_path,
        fps=video_fps,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset="medium",
        logger=None,
    )
    if show_progress:
        print()

    audio.close()
    video.close()
    logger.info(f"Done! Output: {output_path}")
    return output_path


def get_background_at_time(segments: Optional[List["BackgroundSegment"]], t: float):
    """Return the image of the segment active at time t, or None if no segment matches."""
    if not segments:
        return None
    for segment in segments:
        if segment.start_time <= t <= segment.end_time:
            return segment.image
    return None
