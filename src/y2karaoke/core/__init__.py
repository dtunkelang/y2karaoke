"""Core functionality modules."""

from .downloader import YouTubeDownloader, download_audio, download_video
from .separator import AudioSeparator, separate_vocals
from .audio_effects import AudioProcessor, process_audio
from .models import SingerID, Word, Line, SongMetadata
from .lyrics import LyricsProcessor, get_lyrics
from .renderer import VideoRenderer, render_karaoke_video
from .backgrounds import BackgroundProcessor, BackgroundSegment, create_background_segments
from .uploader import YouTubeUploader, upload_video
from .karaoke import KaraokeGenerator

__all__ = [
    'YouTubeDownloader', 'download_audio', 'download_video',
    'AudioSeparator', 'separate_vocals',
    'AudioProcessor', 'process_audio',
    'SingerID', 'Word', 'Line', 'SongMetadata',
    'LyricsProcessor', 'get_lyrics',
    'VideoRenderer', 'render_karaoke_video',
    'BackgroundProcessor', 'BackgroundSegment', 'create_background_segments',
    'YouTubeUploader', 'upload_video',
    'KaraokeGenerator',
]
