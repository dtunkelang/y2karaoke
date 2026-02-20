"""Script to detect scene changes in a video using OpenCV.

Designed to be run as a subprocess to avoid conflicts (e.g. MoviePy vs OpenCV threads).
"""

import sys
import cv2
import numpy as np


def detect_scenes(video_path: str, threshold: float = 30.0) -> list[float]:
    """Detect timestamp of scene changes."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scenes = [0.0]  # Always include start
    prev_frame = None

    # Sample every 30 frames for speed
    for i in range(0, frame_count, 30):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if not ret:
            break

        if prev_frame is not None:
            # Calculate frame difference
            diff = cv2.absdiff(frame, prev_frame)
            mean_diff = np.mean(diff)

            if mean_diff > threshold:
                timestamp = i / fps
                scenes.append(timestamp)

        prev_frame = frame

    cap.release()
    return scenes


if __name__ == "__main__":
    if len(sys.argv) > 1:
        found_scenes = detect_scenes(sys.argv[1])
        print(",".join(map(str, found_scenes)))
