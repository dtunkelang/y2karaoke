"""Core functionality modules."""

from .components.audio.downloader import (
    YouTubeDownloader,
    download_audio,
    download_video,
)
from .components.audio.separator import AudioSeparator, separate_vocals
from .components.audio.audio_effects import AudioProcessor, process_audio
from .models import SingerID, Word, Line, SongMetadata
from .lyrics import LyricsProcessor, get_lyrics
from .components.render.video_writer import render_karaoke_video
from .components.render.backgrounds import (
    BackgroundProcessor,
    BackgroundSegment,
    create_background_segments,
)
from .karaoke import KaraokeGenerator

__all__ = [
    "YouTubeDownloader",
    "download_audio",
    "download_video",
    "AudioSeparator",
    "separate_vocals",
    "AudioProcessor",
    "process_audio",
    "SingerID",
    "Word",
    "Line",
    "SongMetadata",
    "LyricsProcessor",
    "get_lyrics",
    "render_karaoke_video",
    "BackgroundProcessor",
    "BackgroundSegment",
    "create_background_segments",
    "KaraokeGenerator",
]
