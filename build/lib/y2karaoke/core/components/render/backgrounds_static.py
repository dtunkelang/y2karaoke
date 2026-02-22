"""Static background and splash/logo rendering for karaoke videos."""

from typing import Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ....config import VIDEO_WIDTH, VIDEO_HEIGHT, Colors
from ....utils.fonts import get_font


def create_gradient_background(
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> np.ndarray:
    """Create a simple vertical gradient background image."""
    w = width or VIDEO_WIDTH
    h = height or VIDEO_HEIGHT
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)

    for y in range(h):
        ratio = y / h
        r = int(Colors.BG_TOP[0] * (1 - ratio) + Colors.BG_BOTTOM[0] * ratio)
        g = int(Colors.BG_TOP[1] * (1 - ratio) + Colors.BG_BOTTOM[1] * ratio)
        b = int(Colors.BG_TOP[2] * (1 - ratio) + Colors.BG_BOTTOM[2] * ratio)
        draw.line([(0, y), (w, y)], fill=(r, g, b))

    return np.array(img)


def draw_splash_screen(
    draw: ImageDraw.ImageDraw,
    title: str,
    artist: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    """Draw the intro splash screen with song title, artist, and y2karaoke logo."""
    video_width = width or VIDEO_WIDTH
    video_height = height or VIDEO_HEIGHT

    title_font = get_font(84)
    artist_font = get_font(48)
    logo_font = get_font(32)

    # Truncate long titles
    max_title_chars = 40
    display_title = (
        title if len(title) <= max_title_chars else title[: max_title_chars - 3] + "..."
    )

    # Center title
    title_bbox = title_font.getbbox(display_title)
    title_x = (video_width - (title_bbox[2] - title_bbox[0])) // 2
    title_y = video_height // 2 - 80

    # Center artist
    artist_text = f"by {artist}"
    artist_bbox = artist_font.getbbox(artist_text)
    artist_x = (video_width - (artist_bbox[2] - artist_bbox[0])) // 2
    artist_y = title_y + 100

    # y2karaoke branding at bottom
    logo_text = "y2karaoke"
    logo_bbox = logo_font.getbbox(logo_text)
    logo_x = (video_width - (logo_bbox[2] - logo_bbox[0])) // 2
    logo_y = video_height - 100

    draw.text((title_x, title_y), display_title, font=title_font, fill=Colors.HIGHLIGHT)
    draw.text((artist_x, artist_y), artist_text, font=artist_font, fill=Colors.TEXT)
    draw.text((logo_x, logo_y), logo_text, font=logo_font, fill=Colors.SUNG)


def draw_logo_screen(
    draw: ImageDraw.ImageDraw,
    font: Optional[ImageFont.ImageFont | ImageFont.FreeTypeFont] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    """Draw the y2karaoke logo screen for the outro."""
    video_width = width or VIDEO_WIDTH
    video_height = height or VIDEO_HEIGHT

    # Use provided font for logo, or default sizes
    logo_font = font or get_font(96)
    tagline_font = get_font(36)
    url_font = get_font(28)

    logo_text = "y2karaoke"
    tagline_text = "youtube to karaoke"
    url_text = "github.com/dtunkelang/y2karaoke"

    # Center the logo
    logo_bbox = logo_font.getbbox(logo_text)
    logo_width = logo_bbox[2] - logo_bbox[0]
    logo_x = (video_width - logo_width) // 2
    logo_y = video_height // 2 - 80

    # Center the tagline
    tagline_bbox = tagline_font.getbbox(tagline_text)
    tagline_width = tagline_bbox[2] - tagline_bbox[0]
    tagline_x = (video_width - tagline_width) // 2
    tagline_y = logo_y + 100

    # Center the URL
    url_bbox = url_font.getbbox(url_text)
    url_width = url_bbox[2] - url_bbox[0]
    url_x = (video_width - url_width) // 2
    url_y = tagline_y + 60

    # Draw text
    draw.text((logo_x, logo_y), logo_text, font=logo_font, fill=Colors.HIGHLIGHT)
    draw.text((tagline_x, tagline_y), tagline_text, font=tagline_font, fill=Colors.TEXT)
    draw.text((url_x, url_y), url_text, font=url_font, fill=Colors.SUNG)
