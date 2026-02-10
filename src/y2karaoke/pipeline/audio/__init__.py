"""Audio/media subsystem facade."""

from ...core.downloader import YouTubeDownloader, extract_video_id
from ...core.separator import AudioSeparator, separate_vocals
from ...core.audio_effects import AudioProcessor
from ...core.audio_utils import (
    trim_audio_if_needed,
    apply_audio_effects,
    separate_vocals as separate_vocals_cached,
)

__all__ = [
    "YouTubeDownloader",
    "extract_video_id",
    "AudioSeparator",
    "separate_vocals",
    "separate_vocals_cached",
    "AudioProcessor",
    "trim_audio_if_needed",
    "apply_audio_effects",
]
