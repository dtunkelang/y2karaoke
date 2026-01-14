"""Core functionality modules."""

from .downloader import YouTubeDownloader, download_audio, download_video
from .separator import AudioSeparator, separate_vocals
from .audio_effects import AudioProcessor, process_audio
from .lyrics import LyricsProcessor, Line, Word, SongMetadata, get_lyrics
from .renderer import VideoRenderer, render_karaoke_video
from .backgrounds import BackgroundProcessor, BackgroundSegment, create_background_segments
from .uploader import YouTubeUploader, upload_video
from .karaoke import KaraokeGenerator

__all__ = [
    'YouTubeDownloader', 'download_audio', 'download_video',
    'AudioSeparator', 'separate_vocals',
    'AudioProcessor', 'process_audio',
    'LyricsProcessor', 'Line', 'Word', 'SongMetadata', 'get_lyrics',
    'VideoRenderer', 'render_karaoke_video',
    'BackgroundProcessor', 'BackgroundSegment', 'create_background_segments',
    'YouTubeUploader', 'upload_video',
    'KaraokeGenerator',
]
