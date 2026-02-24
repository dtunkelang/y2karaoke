"""Audio component facade."""

from .downloader import YouTubeDownloader, extract_video_id
from .separator import AudioSeparator, separate_vocals

__all__ = [
    "YouTubeDownloader",
    "extract_video_id",
    "AudioSeparator",
    "separate_vocals",
]

try:
    from .audio_effects import AudioProcessor
    from .audio_utils import (
        apply_audio_effects,
        separate_vocals as separate_vocals_cached,
        trim_audio_if_needed,
    )

    __all__ += [
        "separate_vocals_cached",
        "AudioProcessor",
        "trim_audio_if_needed",
        "apply_audio_effects",
    ]
except ImportError:
    pass
