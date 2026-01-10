"""YouTube video uploader with OAuth authentication."""

import os
import pickle
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# YouTube API scopes
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

# Credentials cache location
CREDENTIALS_DIR = Path.home() / '.cache' / 'karaoke'
CREDENTIALS_FILE = CREDENTIALS_DIR / 'youtube_credentials.pickle'
CLIENT_SECRETS_FILE = CREDENTIALS_DIR / 'client_secrets.json'


def get_authenticated_service():
    """
    Get an authenticated YouTube API service.

    Uses cached credentials if available, otherwise initiates OAuth flow.

    Returns:
        YouTube API service object
    """
    credentials = None

    # Load cached credentials if they exist
    if CREDENTIALS_FILE.exists():
        with open(CREDENTIALS_FILE, 'rb') as f:
            credentials = pickle.load(f)

    # Refresh or get new credentials if needed
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            print("Refreshing YouTube credentials...")
            credentials.refresh(Request())
        else:
            if not CLIENT_SECRETS_FILE.exists():
                raise FileNotFoundError(
                    f"YouTube API credentials not found.\n\n"
                    f"To enable YouTube uploads:\n"
                    f"1. Go to https://console.cloud.google.com/\n"
                    f"2. Create a project and enable the YouTube Data API v3\n"
                    f"3. Create OAuth 2.0 credentials (Desktop app)\n"
                    f"4. Download the JSON file and save it as:\n"
                    f"   {CLIENT_SECRETS_FILE}\n"
                )

            print("Opening browser for YouTube authentication...")
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CLIENT_SECRETS_FILE),
                SCOPES
            )
            credentials = flow.run_local_server(port=0)

        # Cache credentials for future use
        CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
        with open(CREDENTIALS_FILE, 'wb') as f:
            pickle.dump(credentials, f)
        print("YouTube credentials saved.")

    return build('youtube', 'v3', credentials=credentials)


def upload_video(
    video_path: str,
    title: str,
    description: str,
    tags: list[str] = None,
) -> str:
    """
    Upload a video to YouTube as unlisted.

    Args:
        video_path: Path to the video file
        title: Video title
        description: Video description
        tags: Optional list of tags

    Returns:
        Shareable URL of the uploaded video
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    youtube = get_authenticated_service()

    body = {
        'snippet': {
            'title': title,
            'description': description,
            'tags': tags or [],
            'categoryId': '10',  # Music category
        },
        'status': {
            'privacyStatus': 'unlisted',
            'selfDeclaredMadeForKids': False,
        },
    }

    # Create media upload object
    media = MediaFileUpload(
        video_path,
        mimetype='video/mp4',
        resumable=True,
    )

    print(f"Uploading to YouTube: {title}")
    print("This may take a few minutes...")

    # Execute upload
    request = youtube.videos().insert(
        part=','.join(body.keys()),
        body=body,
        media_body=media,
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            progress = int(status.progress() * 100)
            print(f"Upload progress: {progress}%")

    video_id = response['id']
    video_url = f"https://youtu.be/{video_id}"

    print(f"\nUpload complete!")
    print(f"Unlisted video URL: {video_url}")

    return video_url


def generate_metadata(title: str, artist: str) -> tuple[str, str]:
    """
    Generate video title and description from song info.

    Args:
        title: Song title
        artist: Artist name

    Returns:
        Tuple of (video_title, video_description)
    """
    if artist and artist != "Unknown":
        video_title = f"{title} - {artist} (Karaoke)"
    else:
        video_title = f"{title} (Karaoke)"

    description = f"""Karaoke version of "{title}"{f' by {artist}' if artist and artist != 'Unknown' else ''}.

Features:
- Instrumental track (vocals removed)
- Word-by-word synchronized lyrics
- KaraFun-style highlighting

Generated with y2karaoke
https://github.com/dtunkelang/y2karaoke
"""

    return video_title, description


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python uploader.py <video_file> [title] [artist]")
        sys.exit(1)

    video_file = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else "Karaoke Video"
    artist = sys.argv[3] if len(sys.argv) > 3 else "Unknown"

    video_title, description = generate_metadata(title, artist)
    url = upload_video(video_file, video_title, description)
    print(f"\nShare this link: {url}")
