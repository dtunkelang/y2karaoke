"""Background processing for dynamic video backgrounds."""

import subprocess
import sys
from dataclasses import dataclass
from typing import List, TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from ..config import VIDEO_WIDTH, VIDEO_HEIGHT, DARKEN_FACTOR, BLUR_RADIUS
from ..exceptions import RenderError
from ..utils.logging import get_logger

if TYPE_CHECKING:
    from .models import Line

logger = get_logger(__name__)

@dataclass
class BackgroundSegment:
    """A background image segment with timing."""
    image: np.ndarray
    start_time: float
    end_time: float

class BackgroundProcessor:
    """Process video backgrounds for karaoke."""
    
    def __init__(self):
        self.width = VIDEO_WIDTH
        self.height = VIDEO_HEIGHT
    
    def create_background_segments(
        self, 
        video_path: str, 
        lines: List["Line"], 
        duration: float
    ) -> List[BackgroundSegment]:
        """Create background segments from video."""
        
        try:
            # Extract scene frames
            scene_frames = self._extract_scene_frames(video_path)
            
            if not scene_frames:
                logger.warning("No scene frames extracted")
                return []
            
            # Process frames
            processed_frames = []
            for timestamp, frame in scene_frames:
                processed_frame = self._process_frame(frame)
                processed_frames.append((timestamp, processed_frame))
            
            # Create segments
            segments = []
            for i, (timestamp, frame) in enumerate(processed_frames):
                start_time = timestamp
                
                # Determine end time
                if i + 1 < len(processed_frames):
                    end_time = processed_frames[i + 1][0]
                else:
                    end_time = duration
                
                segments.append(BackgroundSegment(
                    image=frame,
                    start_time=start_time,
                    end_time=end_time
                ))
            
            logger.debug(f"Created {len(segments)} background segments")
            return segments
            
        except Exception as e:
            logger.error(f"Background processing failed: {e}")
            return []
    
    def _extract_scene_frames(self, video_path: str) -> List[tuple]:
        """Extract frames at scene changes."""
        
        try:
            # Use subprocess to avoid moviepy conflicts
            scene_times = self._detect_scenes_subprocess(video_path)
            
            if not scene_times:
                # Fallback to regular intervals
                scene_times = [i * 30.0 for i in range(10)]  # Every 30 seconds
            
            # Extract frames at scene times
            frames = []
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise RenderError(f"Could not open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            for scene_time in scene_times[:10]:  # Limit to 10 scenes
                frame_number = int(scene_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                ret, frame = cap.read()
                if ret and self._is_valid_frame(frame):
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append((scene_time, frame_rgb))
            
            cap.release()
            return frames
            
        except Exception as e:
            logger.warning(f"Scene extraction failed: {e}")
            return []
    
    def _detect_scenes_subprocess(self, video_path: str) -> List[float]:
        """Detect scene changes using subprocess."""

        try:
            # Simple scene detection code - video_path passed safely via sys.argv
            code = '''
import sys
import cv2
import numpy as np

def detect_scenes(video_path, threshold=30.0):
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

if len(sys.argv) > 1:
    scenes = detect_scenes(sys.argv[1])
    logger.debug(",".join(map(str, scenes)))
'''

            result = subprocess.run(
                [sys.executable, '-c', code, video_path],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0 and result.stdout.strip():
                times = [float(x) for x in result.stdout.strip().split(',')]
                return times[:10]  # Limit scenes

        except Exception as e:
            logger.warning(f"Scene detection subprocess failed: {e}")

        return []
    
    def _is_valid_frame(self, frame: np.ndarray, min_brightness: int = 20) -> bool:
        """Check if frame has enough content."""
        return frame.max() >= min_brightness
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame for background use."""
        
        # Convert to PIL for processing
        img = Image.fromarray(frame)
        
        # Resize to video dimensions
        img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
        
        # Darken the image
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(DARKEN_FACTOR)
        
        # Apply slight blur
        img = img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
        
        # Convert back to numpy array
        return np.array(img)

# Backward compatibility function
def create_background_segments(video_path: str, lines: List["Line"], duration: float) -> List[BackgroundSegment]:
    """Create background segments (backward compatibility)."""
    processor = BackgroundProcessor()
    return processor.create_background_segments(video_path, lines, duration)
