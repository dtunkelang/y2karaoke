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
