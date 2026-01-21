"""YouTube downloader with improved error handling."""

from pathlib import Path
from typing import Dict, Optional

import yt_dlp

from ..config import get_cache_dir
from ..exceptions import DownloadError
from ..utils.logging import get_logger
from .youtube_metadata import (
    sanitize_filename,
    validate_youtube_url,
    extract_video_id,
    _parse_artist_title_from_video_title,
    _parse_metadata_from_description,
    _clean_uploader_name,
    clean_title,
    extract_metadata_from_youtube,
)

logger = get_logger(__name__)


class YouTubeDownloader:
    """YouTube downloader with caching and error handling."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or get_cache_dir()

    # ------------------------
    # Metadata helper methods
    # ------------------------
    def get_video_title(self, url: str) -> str:
        """Return the title of a YouTube video without downloading the full video."""
        ydl_opts = {
            'quiet': True,
            'skip_download': True,
            'forcejson': True,
            'extract_flat': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get('title', url)

    def get_video_uploader(self, url: str) -> str:
        """Return the uploader/channel name for a YouTube video without downloading full video."""
        ydl_opts = {'quiet': True, 'skip_download': True, 'extract_flat': True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info.get('uploader', 'Unknown')
        except Exception:
            return "Unknown"

    # ------------------------
    # Download methods
    # ------------------------
    def download_audio(self, url: str, output_dir: Optional[Path] = None) -> Dict[str, str]:
        url = validate_youtube_url(url)
        video_id = extract_video_id(url)
        output_dir = Path(output_dir or self.cache_dir / video_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading audio from {url}")

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                ydl.download([url])

                # Find the file
                video_title = info.get('title', 'Unknown')
                safe_title = sanitize_filename(video_title)
                audio_path = output_dir / f"{safe_title}.wav"
                if not audio_path.exists():
                    wav_files = list(output_dir.glob("*.wav"))
                    if wav_files:
                        audio_path = wav_files[0]
                    else:
                        raise DownloadError("Downloaded audio file not found")

                # Extract metadata
                metadata = extract_metadata_from_youtube(url)
                artist = metadata['artist']
                cleaned_title = metadata['title']

                return {
                    'audio_path': str(audio_path),
                    'title': cleaned_title,
                    'artist': artist,
                    'video_id': video_id,
                }
        except Exception as e:
            raise DownloadError(f"Failed to download audio: {e}")

    def download_video(self, url: str, output_dir: Optional[Path] = None) -> Dict[str, str]:
        url = validate_youtube_url(url)
        video_id = extract_video_id(url)
        output_dir = Path(output_dir or self.cache_dir / video_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading video from {url}")

        ydl_opts = {
            'format': 'best[height<=720]',
            'outtmpl': str(output_dir / '%(title)s_video.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                ydl.download([url])

                video_files = list(output_dir.glob("*_video.*"))
                if not video_files:
                    raise DownloadError("Downloaded video file not found")
                video_path = video_files[0]

                # Extract metadata
                metadata = extract_metadata_from_youtube(url)
                artist = metadata['artist']
                cleaned_title = metadata['title']

                return {
                    'video_path': str(video_path),
                    'title': cleaned_title or info.get('title', 'Unknown'),
                    'artist': artist or 'Unknown',
                    'video_id': video_id,
                }
        except Exception as e:
            raise DownloadError(f"Failed to download video: {e}")


# ------------------------
# Convenience functions
# ------------------------
def download_audio(url: str, output_dir: Optional[Path] = None) -> Dict[str, str]:
    downloader = YouTubeDownloader()
    return downloader.download_audio(url, output_dir)


def download_video(url: str, output_dir: Optional[Path] = None) -> Dict[str, str]:
    downloader = YouTubeDownloader()
    return downloader.download_video(url, output_dir)
