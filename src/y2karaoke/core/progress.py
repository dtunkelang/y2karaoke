"""Progress bar utilities for video rendering."""

from ..config import VIDEO_WIDTH, VIDEO_HEIGHT


class RenderProgressBar:
    """Custom progress bar for video rendering."""

    def __init__(self, total_frames: int):
        self.total_frames = total_frames
        self.current_frame = 0
        self.last_percent = -1

    def __call__(self, gf, t):
        """Called by MoviePy for each frame."""
        self.current_frame += 1
        percent = int(100 * self.current_frame / self.total_frames)
        if percent != self.last_percent and percent % 5 == 0:
            bar_len = 30
            filled = int(bar_len * percent / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  Rendering: [{bar}] {percent}%", end="", flush=True)
            self.last_percent = percent


class ProgressLogger:
    """Custom logger for MoviePy that shows a progress bar."""

    def __init__(self, total_duration: float, fps: int):
        self.total_frames = int(total_duration * fps)
        self.last_percent = -1

    def bars_callback(self, bar, attr, value, old_value=None):
        """Callback for progress bars."""
        if attr == "index":
            percent = (
                int(100 * value / self.total_frames) if self.total_frames > 0 else 0
            )
            if percent != self.last_percent:
                bar_len = 30
                filled = int(bar_len * percent / 100)
                bar_str = "█" * filled + "░" * (bar_len - filled)
                print(f"\r  Rendering: [{bar_str}] {percent}%", end="", flush=True)
                self.last_percent = percent

    def callback(self, **kw):
        """General callback."""
        pass


def draw_progress_bar(
    draw,
    progress: float,
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
    bar_width: int = 600,
    bar_height: int = 12,
    border_radius: int = 6,
    Colors=None,
) -> None:
    """
    Draw a horizontal progress bar at the center of the screen.

    Args:
        draw: PIL ImageDraw object
        progress: Progress value from 0.0 to 1.0
        width: Video width (default from config)
        height: Video height (default from config)
        bar_width: Width of the progress bar
        bar_height: Height of the progress bar
        border_radius: Rounded corner radius
        Colors: Optional color config with PROGRESS_BG / PROGRESS_FG
    """
    if Colors is None:
        from ..config import Colors

    # Draw background
    draw.rounded_rectangle(
        [
            (width - bar_width) // 2,
            (height - bar_height) // 2,
            (width + bar_width) // 2,
            (height + bar_height) // 2,
        ],
        radius=border_radius,
        fill=Colors.PROGRESS_BG,
    )

    # Draw filled portion
    if progress > 0:
        fill_width = int(bar_width * min(progress, 1.0))
        if fill_width > 0:
            draw.rounded_rectangle(
                [
                    (width - bar_width) // 2,
                    (height - bar_height) // 2,
                    (width - bar_width) // 2 + fill_width,
                    (height + bar_height) // 2,
                ],
                radius=border_radius,
                fill=Colors.PROGRESS_FG,
            )
