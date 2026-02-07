"""Validation utilities."""

import logging
import re
from pathlib import Path

from ..config import KEY_SHIFT_RANGE, TEMPO_RANGE
from ..exceptions import ValidationError

logger = logging.getLogger(__name__)


def validate_youtube_url(url: str) -> str:
    """Validate and normalize YouTube URL."""
    if not url:
        raise ValidationError("URL cannot be empty")

    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        if re.search(pattern, url):
            return url

    raise ValidationError(f"Invalid YouTube URL: {url}")


def validate_key_shift(key: int) -> int:
    """Validate key shift parameter."""
    min_key, max_key = KEY_SHIFT_RANGE
    if not min_key <= key <= max_key:
        raise ValidationError(f"Key shift must be between {min_key} and {max_key}")
    return key


def validate_tempo(tempo: float) -> float:
    """Validate tempo parameter."""
    min_tempo, max_tempo = TEMPO_RANGE
    if not min_tempo <= tempo <= max_tempo:
        raise ValidationError(f"Tempo must be between {min_tempo} and {max_tempo}")
    return tempo


def validate_offset(offset: float) -> float:
    """Validate timing offset."""
    if abs(offset) > 10.0:
        raise ValidationError("Timing offset must be between -10 and +10 seconds")
    return offset


def validate_output_path(path: str) -> Path:
    """Validate and normalize output path."""
    output_path = Path(path)

    # Check if parent directory exists or can be created
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ValidationError(f"Cannot create output directory: {e}")

    # Check file extension
    if output_path.suffix.lower() not in [".mp4", ".avi", ".mov"]:
        raise ValidationError("Output file must have .mp4, .avi, or .mov extension")

    return output_path


def fix_line_order(lines):
    """Fix non-monotonic line ordering by shifting lines forward.

    Returns the (possibly modified) list of lines.
    """
    from ..core.models import Line, Word

    prev_start = None
    fixed = list(lines)
    for idx, line in enumerate(fixed):
        if not getattr(line, "words", None):
            continue
        start = line.start_time
        if prev_start is not None and start < prev_start:
            shift = (prev_start - start) + 0.01
            logger.warning(
                "Line %d starts before previous line (%.2fs < %.2fs), shifting +%.2fs",
                idx + 1,
                start,
                prev_start,
                shift,
            )
            new_words = [
                Word(
                    text=w.text,
                    start_time=w.start_time + shift,
                    end_time=w.end_time + shift,
                    singer=w.singer,
                )
                for w in line.words
            ]
            fixed[idx] = Line(words=new_words, singer=line.singer)
            start = fixed[idx].start_time
        prev_start = start
    return fixed


def validate_line_order(lines) -> None:
    """Validate that lines are monotonic and have non-negative durations."""
    prev_start = None
    for idx, line in enumerate(lines):
        if not getattr(line, "words", None):
            continue
        start = line.start_time
        end = line.end_time
        if end < start:
            raise ValidationError(
                f"Line {idx + 1} has end before start ({start:.2f}s -> {end:.2f}s)"
            )
        if prev_start is not None and start < prev_start:
            raise ValidationError(
                f"Line {idx + 1} starts before previous line ({start:.2f}s < {prev_start:.2f}s)"
            )
        prev_start = start


def sanitize_filename(name: str) -> str:
    """Remove invalid characters from filename."""
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "", name)
    # Limit length
    return sanitized[:100].strip()
