"""Background extraction and processing for karaoke videos."""

import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from moviepy import VideoFileClip
from scenedetect import detect, ContentDetector

from lyrics import Line


# Video dimensions (must match renderer.py)
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

# Background processing settings
DARKEN_FACTOR = 0.4  # How much to darken (0 = black, 1 = original)
BLUR_RADIUS = 3  # Slight blur to reduce distraction


@dataclass
class BackgroundSegment:
    """A background image and when it should be shown."""
    image: np.ndarray  # Processed image ready for rendering
    start_time: float  # When to start showing this background
    end_time: float  # When to stop showing this background


def _detect_scenes_subprocess(video_path: str, threshold: float) -> list[float]:
    """Run scene detection in a subprocess to avoid ffmpeg conflicts."""
    import subprocess
    import json
    import sys

    code = f'''
import json
from scenedetect import detect, ContentDetector
scenes = detect("{video_path}", ContentDetector(threshold={threshold}))
times = [scene[0].get_seconds() for scene in scenes]
print(json.dumps(times))
'''
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    if result.returncode == 0 and result.stdout.strip():
        return json.loads(result.stdout.strip())
    return []


def _is_valid_frame(frame: np.ndarray, min_brightness: int = 20) -> bool:
    """Check if a frame has enough content (not too dark/black)."""
    return frame.max() >= min_brightness


def extract_scene_frames(video_path: str, min_scenes: int = 5, max_scenes: int = 20) -> list[tuple[float, np.ndarray]]:
    """
    Extract frames at scene changes from a video.
    Skips black/dark frames to ensure valid backgrounds.

    Args:
        video_path: Path to the video file
        min_scenes: Minimum number of scenes to extract
        max_scenes: Maximum number of scenes to extract

    Returns:
        List of (timestamp, frame) tuples
    """
    print("Detecting scene changes...")

    # Run scene detection in subprocess to avoid ffmpeg conflicts with moviepy
    scene_times = []
    for t in [27.0, 20.0, 15.0, 10.0]:
        scene_times = _detect_scenes_subprocess(video_path, t)
        if len(scene_times) >= min_scenes:
            break

    print(f"Found {len(scene_times)} scene changes")

    # Open video with moviepy to extract frames
    clip = VideoFileClip(video_path)

    frames = []
    first_valid_frame = None

    # Try to get first frame, but skip if it's black
    frame = clip.get_frame(0.0)
    if _is_valid_frame(frame):
        frames.append((0.0, frame.astype(np.uint8)))
        first_valid_frame = frame.astype(np.uint8)

    # Get frame at start of each scene, skipping black frames
    for start_time in scene_times:
        if start_time < clip.duration and len(frames) < max_scenes:
            frame = clip.get_frame(start_time)
            if _is_valid_frame(frame):
                frame_uint8 = frame.astype(np.uint8)
                frames.append((start_time, frame_uint8))
                if first_valid_frame is None:
                    first_valid_frame = frame_uint8

    # If we don't have enough frames, sample evenly through the video
    if len(frames) < min_scenes and first_valid_frame is None:
        # Find the first valid frame by scanning through the video
        for t in range(0, int(clip.duration), 5):
            frame = clip.get_frame(float(t))
            if _is_valid_frame(frame):
                first_valid_frame = frame.astype(np.uint8)
                frames.append((float(t), first_valid_frame))
                break

    # Ensure we always have at least one valid frame at the start
    if frames and frames[0][0] > 0 and first_valid_frame is not None:
        # Insert the first valid frame at time 0
        frames.insert(0, (0.0, first_valid_frame))

    clip.close()

    print(f"Extracted {len(frames)} background frames")
    return frames


def process_frame(frame: np.ndarray) -> np.ndarray:
    """
    Process a frame to make it suitable as a karaoke background.
    Darkens and slightly blurs to ensure lyrics are readable.

    Args:
        frame: RGB numpy array

    Returns:
        Processed frame as RGB numpy array
    """
    # Convert to PIL for processing
    img = Image.fromarray(frame)

    # Resize to video dimensions
    img = img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)

    # Apply slight blur
    if BLUR_RADIUS > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))

    # Darken the image
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(DARKEN_FACTOR)

    return np.array(img)


def detect_song_sections(
    lines: list[Line],
    duration: float,
    min_sections: int = 5,
) -> list[tuple[float, float]]:
    """
    Detect song sections based on gaps in lyrics.
    Large gaps typically indicate verse/chorus boundaries.
    Falls back to time-based sections if not enough gaps are found.

    Args:
        lines: List of lyric lines with timing
        duration: Total song duration
        min_sections: Minimum number of sections to create

    Returns:
        List of (start_time, end_time) tuples for each section
    """
    if not lines:
        return [(0.0, duration)]

    sections = []
    section_start = 0.0

    # Threshold for detecting section boundaries (seconds)
    gap_threshold = 3.0

    for i in range(len(lines) - 1):
        current_end = lines[i].end_time
        next_start = lines[i + 1].start_time
        gap = next_start - current_end

        if gap >= gap_threshold:
            # End current section, start new one
            sections.append((section_start, current_end + gap / 2))
            section_start = current_end + gap / 2

    # Add final section
    sections.append((section_start, duration))

    # If not enough sections from gaps, create time-based sections
    if len(sections) < min_sections:
        section_duration = duration / min_sections
        sections = [
            (i * section_duration, (i + 1) * section_duration)
            for i in range(min_sections)
        ]

    return sections


def create_background_segments(
    video_path: str,
    lines: list[Line],
    duration: float,
) -> list[BackgroundSegment]:
    """
    Create background segments for a karaoke video.

    Args:
        video_path: Path to the source video
        lines: Lyric lines with timing
        duration: Total duration of the karaoke video

    Returns:
        List of BackgroundSegment objects
    """
    # Extract frames at scene changes
    scene_frames = extract_scene_frames(video_path)

    if not scene_frames:
        print("Warning: No frames extracted, falling back to gradient")
        return []

    # Process all frames and filter out any that are still too dark after processing
    print("Processing background frames...")
    processed_frames = []
    for ts, frame in scene_frames:
        processed = process_frame(frame)
        # Check if processed frame has enough brightness (not just black)
        if processed.max() >= 10:
            processed_frames.append((ts, processed))

    if not processed_frames:
        print("Warning: All frames too dark, falling back to gradient")
        return []

    print(f"Valid processed frames: {len(processed_frames)}")

    # Detect song sections
    sections = detect_song_sections(lines, duration)
    print(f"Detected {len(sections)} song sections")

    # Assign backgrounds to sections
    # Cycle through available valid frames if more sections than frames
    segments = []
    for i, (start, end) in enumerate(sections):
        frame_idx = i % len(processed_frames)
        _, frame = processed_frames[frame_idx]

        segments.append(BackgroundSegment(
            image=frame,
            start_time=start,
            end_time=end,
        ))

    return segments


def get_background_at_time(
    segments: list[BackgroundSegment],
    current_time: float,
    crossfade_duration: float = 1.0,
) -> np.ndarray:
    """
    Get the background image for a specific time, with crossfade support.

    Args:
        segments: List of background segments
        current_time: Current playback time
        crossfade_duration: Duration of crossfade between backgrounds

    Returns:
        Background image as numpy array
    """
    if not segments:
        return None

    # Find current segment
    current_segment = None
    next_segment = None

    for i, segment in enumerate(segments):
        if segment.start_time <= current_time < segment.end_time:
            current_segment = segment
            if i + 1 < len(segments):
                next_segment = segments[i + 1]
            break

    # If past all segments, use last one
    if current_segment is None:
        current_segment = segments[-1]

    # Check if we're in crossfade zone
    if next_segment and current_segment.end_time - current_time < crossfade_duration:
        # Calculate crossfade progress
        time_to_end = current_segment.end_time - current_time
        alpha = 1.0 - (time_to_end / crossfade_duration)

        # Blend images
        blended = cv2.addWeighted(
            current_segment.image, 1.0 - alpha,
            next_segment.image, alpha,
            0
        )
        return blended

    return current_segment.image


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python backgrounds.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    frames = extract_scene_frames(video_path)

    print(f"\nExtracted {len(frames)} frames:")
    for ts, frame in frames:
        print(f"  {ts:.1f}s - {frame.shape}")

    # Save first processed frame as test
    if frames:
        processed = process_frame(frames[0][1])
        Image.fromarray(processed).save("test_background.png")
        print("\nSaved test_background.png")
