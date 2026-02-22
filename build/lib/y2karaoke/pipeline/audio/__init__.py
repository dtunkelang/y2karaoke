"""Audio/media subsystem facade."""

from ...core.components.audio import (
    AudioProcessor,
    AudioSeparator,
    YouTubeDownloader,
    apply_audio_effects,
    extract_video_id,
    separate_vocals,
    separate_vocals_cached,
    trim_audio_if_needed,
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
