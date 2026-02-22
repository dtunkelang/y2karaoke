"""Progress bar utilities for video rendering."""

from ....config import VIDEO_WIDTH, VIDEO_HEIGHT


class ConsoleProgressBar:
    """Console progress bar for long-running operations."""

    def __init__(self, total: int, prefix: str = "Rendering"):
        self.total = total
        self.current = 0
        self.last_percent = -1
        self.prefix = prefix

    def update(self) -> None:
        """Increment progress and print bar if percentage changed significantly."""
        self.current += 1
        percent = int(100 * self.current / self.total) if self.total > 0 else 0

        # Update every 2% to reduce I/O churn
        if percent != self.last_percent and percent % 2 == 0:
            bar_len = 30
            filled = int(bar_len * percent / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  {self.prefix}: [{bar}] {percent}%", end="", flush=True)
            self.last_percent = percent


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
        from ....config import Colors as default_colors

        Colors = default_colors

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
