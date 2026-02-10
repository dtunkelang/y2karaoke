"""Compatibility module alias for audio downloader implementation."""

from .components.audio import downloader as _impl
import sys as _sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .components.audio.downloader import (
        YouTubeDownloader,
        download_audio,
        download_video,
    )

_sys.modules[__name__] = _impl
