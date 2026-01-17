"""YouTube downloader with improved error handling."""

import hashlib
import re
from pathlib import Path
from typing import Dict, Optional

import yt_dlp

from ..config import get_cache_dir
from ..exceptions import DownloadError
from ..utils.logging import get_logger
from ..utils.validation import sanitize_filename, validate_youtube_url

logger = get_logger(__name__)

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:embed/)([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    # Fallback: hash the URL
    return hashlib.md5(url.encode()).hexdigest()[:11]

def clean_title(title: str, artist: str = "") -> str:
    """Clean up YouTube video title for display."""
    patterns_to_remove = [
        r'\s*\(Official\s*(Music\s*)?Video\)',
        r'\s*\(Official\s*Audio\)',
        r'\s*\(Official\s*Lyric\s*Video\)',
        r'\s*\(Lyric\s*Video\)',
        r'\s*\(Lyrics?\)',
        r'\s*\(Audio\)',
        r'\s*\(Visualizer\)',
        r'\s*\(Remaster(ed)?\s*\d*\)',
        r'\s*\(Live\)',
        r'\s*\[Official\s*(Music\s*)?Video\]',
        r'\s*\[Official\s*Audio\]',
        r'\s*\[4K\]',
        r'\s*\[HD\]',
        r'\s*\[HQ\]',
        r'\s*\[\d+K\]',
        r'\s*\(4K\)',
        r'\s*\(HD\)',
        r'\s*\(HQ\)',
        r'\s*\(\d+K\)',
        r'\s*【[^】]*】',
        r'\s*ft\.?\s+[^(\[]+$',
        r'\s*\(Lyrics?\s*(and|&|\+)?\s*Translation\)',
        r'\s*\(Translation\)',
        r'\s*Lyrics?\s*(and|&|\+)?\s*Translation\s*$',
        r'\s*Translation\s*$',
        # Aspect ratios and year patterns
        r'\s*\d+:\d+\s*',  # 16:9, 4:3, etc.
        r'\s*-?\s*\d{4}\s*(Original\s*)?(Music|Musik)?\s*Video\s*',  # "2003 Original Music Video"
        r'\s*-?\s*Original\s*(Music|Musik)?\s*Video\s*',  # "Original Music Video"
        r'\s*by\s+[^-]+$',  # "by Artist Name" at end
        r'\s*M/?V\s*$',  # M/V or MV at end
        r'\s*\(M/?V\)',  # (M/V) or (MV)
    ]

    cleaned = title
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    # Extract song title if format is "Artist - Song Title" or "Song Title - Artist"
    if ' - ' in cleaned and artist:
        parts = cleaned.split(' - ', 1)
        if len(parts) == 2:
            artist_lower = artist.lower()
            # Handle "Artist - Song Title" format
            if artist_lower in parts[0].lower():
                cleaned = parts[1]
            # Handle "Song Title - Artist" format
            elif artist_lower in parts[1].lower():
                cleaned = parts[0]

    return cleaned.strip()

class YouTubeDownloader:
    """YouTube downloader with caching and error handling."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or get_cache_dir()
        
    def download_audio(self, url: str, output_dir: Optional[Path] = None) -> Dict[str, str]:
        """Download audio from YouTube URL."""
        url = validate_youtube_url(url)
        video_id = extract_video_id(url)
        
        if output_dir is None:
            output_dir = self.cache_dir / video_id
        
        output_dir = Path(output_dir)
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
                title = info.get('title', 'Unknown')
                uploader = info.get('uploader', 'Unknown')
                description = info.get('description', '')
                
                # Download the audio
                ydl.download([url])
                
                # Find the downloaded file
                safe_title = sanitize_filename(title)
                audio_path = output_dir / f"{safe_title}.wav"
                
                if not audio_path.exists():
                    # Try to find any wav file in the directory
                    wav_files = list(output_dir.glob("*.wav"))
                    if wav_files:
                        audio_path = wav_files[0]
                    else:
                        raise DownloadError("Downloaded audio file not found")
                
                # Extract artist and title from description or title
                artist = self._extract_artist(title, uploader, description)
                cleaned_title = self._extract_title(title, artist, description)
                
                logger.info(f"Downloaded: {cleaned_title} by {artist}")
                
                return {
                    'audio_path': str(audio_path),
                    'title': cleaned_title,
                    'artist': artist,
                    'video_id': video_id,
                }
                
        except Exception as e:
            raise DownloadError(f"Failed to download audio: {e}")
    
    def download_video(self, url: str, output_dir: Optional[Path] = None) -> Dict[str, str]:
        """Download video from YouTube URL."""
        url = validate_youtube_url(url)
        video_id = extract_video_id(url)
        
        if output_dir is None:
            output_dir = self.cache_dir / video_id
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading video from {url}")
        
        ydl_opts = {
            'format': 'best[height<=720]',  # Limit resolution for processing
            'outtmpl': str(output_dir / '%(title)s_video.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'Unknown')
                
                ydl.download([url])
                
                # Find the downloaded video file
                video_files = list(output_dir.glob("*_video.*"))
                if not video_files:
                    raise DownloadError("Downloaded video file not found")
                
                video_path = video_files[0]
                logger.info(f"Downloaded video: {video_path.name}")
                
                return {
                    'video_path': str(video_path),
                    'title': title,
                    'video_id': video_id,
                }
                
        except Exception as e:
            raise DownloadError(f"Failed to download video: {e}")
    
    def _extract_artist(self, title: str, uploader: str, description: str = "") -> str:
        """Extract artist name from title, description, or uploader."""
        # Strategy 1: Check if description is mostly promotional
        if description:
            lines = description.split('\n')[:5]  # Check first 5 lines
            promotional_count = 0
            for line in lines:
                if any(word in line.lower() for word in ['listen to', 'subscribe', 'merch', 'lnk.to', 'follow', 'watch the official']):
                    promotional_count += 1
            
            # If most lines are promotional, skip description and extract from title
            if promotional_count >= 2:
                logger.info("Description appears promotional, extracting from title/uploader")
                if ' - ' in title:
                    parts = title.split(' - ', 1)
                    if len(parts) == 2:
                        potential_artist = parts[0].strip()
                        if len(potential_artist) < 50 and not any(
                            word in potential_artist.lower() 
                            for word in ['official', 'video', 'audio', 'lyrics', 'lyric']
                        ):
                            logger.info(f"Extracted artist from title (- format): {potential_artist}")
                            return potential_artist
                
                # Fall back to uploader
                artist = uploader
                suffixes = ['Official', 'VEVO', 'Records', 'Music', 'Channel', '- Topic', ' - Topic', 'TV']
                for suffix in suffixes:
                    if artist.endswith(suffix):
                        artist = artist[:-len(suffix)].strip()
                logger.info(f"Using uploader as artist: {artist}")
                return artist or "Unknown"
        
        # Strategy 2: YouTube Music format in description ("Title · Artist")
        if description:
            lines = description.split('\n')
            non_url_lines = [line.strip() for line in lines 
                           if line.strip() 
                           and not line.strip().startswith('http') 
                           and '@' not in line
                           and 'provided to' not in line.lower()]
            
            if non_url_lines:
                line1 = non_url_lines[0]
                if '·' in line1:
                    parts = line1.split('·')
                    if len(parts) >= 2:
                        artist = parts[-1].strip()
                        if artist and 2 < len(artist) < 100:
                            logger.info(f"Extracted artist from description (· format): {artist}")
                            return artist
                
                # Check second line for artist name
                if len(non_url_lines) >= 2:
                    line2 = non_url_lines[1]
                    if line2 and 2 < len(line2) < 100 and not line2.startswith('#') and not line2.startswith('👉'):
                        artist = re.sub(r'\s*[\(\[].*?[\)\]]\s*', '', line2).strip()
                        if artist:
                            logger.info(f"Extracted artist from description (line 2): {artist}")
                            return artist
        
        # Strategy 3: "Artist - Title" format in video title
        if ' - ' in title:
            parts = title.split(' - ', 1)
            if len(parts) == 2:
                potential_artist = parts[0].strip()
                if len(potential_artist) < 50 and not any(
                    word in potential_artist.lower() 
                    for word in ['official', 'video', 'audio', 'lyrics', 'lyric']
                ):
                    logger.info(f"Extracted artist from title (- format): {potential_artist}")
                    return potential_artist
        
        # Strategy 4: Clean up uploader name
        artist = uploader
        suffixes = ['Official', 'VEVO', 'Records', 'Music', 'Channel', '- Topic', ' - Topic', 'TV']
        for suffix in suffixes:
            if artist.endswith(suffix):
                artist = artist[:-len(suffix)].strip()
        
        logger.info(f"Using uploader as artist: {artist}")
        return artist or "Unknown"
    
    def _extract_title(self, title: str, artist: str, description: str = "") -> str:
        """Extract song title from title or description."""
        # Strategy 1: Check if description is mostly promotional
        if description:
            lines = description.split('\n')[:5]  # Check first 5 lines
            promotional_count = 0
            for line in lines:
                if any(word in line.lower() for word in ['listen to', 'subscribe', 'merch', 'lnk.to', 'follow', 'watch the official']):
                    promotional_count += 1
            
            # If most lines are promotional, skip description and use video title
            if promotional_count >= 2:
                logger.info("Description appears promotional, using video title")
                cleaned = clean_title(title, artist)
                logger.info(f"Extracted title from video title: {cleaned}")
                return cleaned
        
        # Strategy 2: YouTube Music format in description ("Title · Artist")
        if description:
            lines = description.split('\n')
            for line in lines[:10]:
                line = line.strip()
                if not line or line.startswith('http') or line.startswith('#') or '@' in line:
                    continue
                if 'provided to' in line.lower() or 'youtube' in line.lower():
                    continue
                # Skip promotional lines
                if any(word in line.lower() for word in ['visit:', 'merchandise', 'releases', 'store', 'buy', 'download', 'listen to', 'lnk.to']):
                    continue
                
                if '·' in line:
                    parts = line.split('·')
                    if len(parts) >= 2:
                        song_title = parts[0].strip()
                        if song_title and 2 < len(song_title) < 100:
                            logger.info(f"Extracted title from description (· format): {song_title}")
                            return song_title
                
                # First non-metadata line might be title
                if line and len(line) < 100:
                    song_title = re.sub(r'\s*[\(\[].*?[\)\]]\s*', '', line).strip()
                    if song_title and len(song_title) > 2:
                        logger.info(f"Extracted title from description (line 1): {song_title}")
                        return song_title
        
        # Strategy 3: Clean the video title
        cleaned = clean_title(title, artist)
        logger.info(f"Extracted title from video title: {cleaned}")
        return cleaned

# Convenience functions for backward compatibility
def download_audio(url: str, output_dir: Optional[str] = None) -> Dict[str, str]:
    """Download audio from YouTube URL."""
    downloader = YouTubeDownloader()
    output_path = Path(output_dir) if output_dir else None
    return downloader.download_audio(url, output_path)

def download_video(url: str, output_dir: Optional[str] = None) -> Dict[str, str]:
    """Download video from YouTube URL."""
    downloader = YouTubeDownloader()
    output_path = Path(output_dir) if output_dir else None
    return downloader.download_video(url, output_path)
