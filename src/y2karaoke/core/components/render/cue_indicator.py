"""Cue indicator drawing primitives."""

import math

from PIL import ImageDraw

from ....config import Colors


def draw_cue_indicator(
    draw: ImageDraw.ImageDraw, x: int, y: int, time_until_start: float, font_size: int
) -> None:
    """Draw animated cue indicator (pulsing dots) to prepare singer."""
    dot_radius = max(4, font_size // 12)
    dot_spacing = dot_radius * 3
    total_width = dot_spacing * 2 + dot_radius * 2
    start_x = x - total_width - dot_spacing
    dots_to_show = min(3, max(1, int(time_until_start) + 1))
    pulse = 0.5 + 0.5 * math.sin(time_until_start * math.pi * 3)

    for i in range(3):
        dot_x = start_x + i * dot_spacing
        dot_y = y

        if i < dots_to_show:
            if i == dots_to_show - 1:
                radius = int(dot_radius * (0.8 + 0.4 * pulse))
            else:
                radius = dot_radius

            draw.ellipse(
                [dot_x - radius, dot_y - radius, dot_x + radius, dot_y + radius],
                fill=Colors.CUE_INDICATOR,
            )
        else:
            draw.ellipse(
                [
                    dot_x - dot_radius,
                    dot_y - dot_radius,
                    dot_x + dot_radius,
                    dot_y + dot_radius,
                ],
                outline=(100, 100, 100),
                width=1,
            )
