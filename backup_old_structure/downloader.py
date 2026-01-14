"""YouTube audio downloader using yt-dlp."""

import os
import re
import yt_dlp


def sanitize_filename(name: str) -> str:
    """Remove invalid characters from filename."""
    return re.sub(r'[<>:"/\\|?*]', '', name)


def clean_title(title: str, artist: str = "") -> str:
    """
    Clean up YouTube video title for display.

    Removes common suffixes like (Official Video), [4K], etc.
    Also extracts song title if format is "Artist - Song Title".
    """
    # Common patterns to remove (case insensitive)
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
        r'\s*【[^】]*】',  # Japanese brackets
        r'\s*ft\.?\s+[^(\[]+$',  # Remove "ft. Artist" at end (optional)
    ]

    cleaned = title
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    # If title is "Artist - Song Title", extract just the song title
    if ' - ' in cleaned and artist:
        parts = cleaned.split(' - ', 1)
        # Check if first part matches artist name (fuzzy match)
        artist_lower = artist.lower().strip()
        first_part_lower = parts[0].lower().strip()
        if artist_lower in first_part_lower or first_part_lower in artist_lower:
            cleaned = parts[1]

    return cleaned.strip()


def download_audio(url: str, output_dir: str = ".") -> dict:
    """
    Download audio from YouTube URL.

    Returns:
        dict with keys: audio_path, title, artist, duration
    """
    os.makedirs(output_dir, exist_ok=True)

    # First, extract info without downloading
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        info = ydl.extract_info(url, download=False)

    title = info.get('title', 'Unknown')
    artist = info.get('artist') or info.get('uploader', 'Unknown')
    duration = info.get('duration', 0)

    # Clean up title for filename
    safe_title = sanitize_filename(title)
    output_path = os.path.join(output_dir, f"{safe_title}")

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': output_path,
        'quiet': False,
        'no_warnings': True,
    }

    print(f"Downloading: {title}")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    audio_path = f"{output_path}.wav"

    return {
        'audio_path': audio_path,
        'title': title,
        'artist': artist,
        'duration': duration,
    }


def download_video(url: str, output_dir: str = ".") -> dict:
    """
    Download video from YouTube URL.

    Returns:
        dict with keys: video_path, title, artist, duration
    """
    os.makedirs(output_dir, exist_ok=True)

    # First, extract info without downloading
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        info = ydl.extract_info(url, download=False)

    title = info.get('title', 'Unknown')
    artist = info.get('artist') or info.get('uploader', 'Unknown')
    duration = info.get('duration', 0)

    # Clean up title for filename
    safe_title = sanitize_filename(title)
    output_path = os.path.join(output_dir, f"{safe_title}_video.mp4")

    ydl_opts = {
        # Prefer h264/avc codec for OpenCV compatibility, avoid av1
        'format': 'bestvideo[height<=1080][vcodec^=avc]+bestaudio[ext=m4a]/bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best',
        'outtmpl': output_path.replace('.mp4', ''),
        'merge_output_format': 'mp4',
        'quiet': False,
        'no_warnings': True,
    }

    print(f"Downloading video: {title}")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # yt-dlp may add extension, find the actual file
    if not os.path.exists(output_path):
        base = output_path.replace('.mp4', '')
        for ext in ['.mp4', '.mkv', '.webm']:
            if os.path.exists(base + ext):
                output_path = base + ext
                break

    return {
        'video_path': output_path,
        'title': title,
        'artist': artist,
        'duration': duration,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python downloader.py <youtube_url>")
        sys.exit(1)

    result = download_audio(sys.argv[1], output_dir="./output")
    print(f"Downloaded: {result['audio_path']}")
    print(f"Title: {result['title']}")
    print(f"Artist: {result['artist']}")
    print(f"Duration: {result['duration']}s")
