"""YouTube audio downloader using yt-dlp."""

import os
import re
import yt_dlp


def sanitize_filename(name: str) -> str:
    """Remove invalid characters from filename."""
    return re.sub(r'[<>:"/\\|?*]', '', name)


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
