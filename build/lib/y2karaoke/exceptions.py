"""Custom exceptions for Y2Karaoke."""


class Y2KaraokeError(Exception):
    """Base exception for Y2Karaoke."""

    pass


class DownloadError(Y2KaraokeError):
    """Error downloading from YouTube."""

    pass


class SeparationError(Y2KaraokeError):
    """Error separating audio stems."""

    pass


class LyricsError(Y2KaraokeError):
    """Error fetching or processing lyrics."""

    pass


class RenderError(Y2KaraokeError):
    """Error rendering video."""

    pass


class UploadError(Y2KaraokeError):
    """Error uploading to YouTube."""

    pass


class ValidationError(Y2KaraokeError):
    """Invalid input parameters."""

    pass


class CacheError(Y2KaraokeError):
    """Error with cache operations."""

    pass


class ConfigError(Y2KaraokeError):
    """Configuration error."""

    pass


class OCRError(Y2KaraokeError):
    """Error during OCR processing."""

    pass


class VisualRefinementError(Y2KaraokeError):
    """Error during visual timing refinement."""

    pass
