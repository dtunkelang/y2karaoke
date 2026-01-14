"""YouTube uploader with OAuth authentication."""

import pickle
from pathlib import Path
from typing import Dict, List, Optional

from ..config import get_credentials_dir, YOUTUBE_API_SCOPES
from ..exceptions import UploadError
from ..utils.logging import get_logger

logger = get_logger(__name__)

class YouTubeUploader:
    """Upload videos to YouTube with OAuth."""
    
    def __init__(self):
        self.credentials_dir = get_credentials_dir()
        self.credentials_file = self.credentials_dir / 'youtube_credentials.pickle'
        self.client_secrets_file = self.credentials_dir / 'client_secrets.json'
    
    def upload_video(
        self, 
        video_path: str, 
        title: str, 
        artist: str,
        tags: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Upload video to YouTube as unlisted."""
        
        if not Path(video_path).exists():
            raise UploadError(f"Video file not found: {video_path}")
        
        try:
            # Generate metadata
            video_title, description = self._generate_metadata(title, artist)
            
            # Get authenticated service
            youtube = self._get_authenticated_service()
            
            # Upload video
            video_id = self._upload_to_youtube(
                youtube, video_path, video_title, description, tags
            )
            
            video_url = f"https://youtu.be/{video_id}"
            
            logger.info(f"✅ Upload completed: {video_url}")
            
            return {
                'video_id': video_id,
                'url': video_url,
                'title': video_title
            }
            
        except Exception as e:
            raise UploadError(f"Upload failed: {e}")
    
    def _get_authenticated_service(self):
        """Get authenticated YouTube API service."""
        
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
            
            credentials = None
            
            # Load cached credentials
            if self.credentials_file.exists():
                with open(self.credentials_file, 'rb') as f:
                    credentials = pickle.load(f)
            
            # Refresh or get new credentials
            if not credentials or not credentials.valid:
                if credentials and credentials.expired and credentials.refresh_token:
                    logger.info("Refreshing YouTube credentials...")
                    credentials.refresh(Request())
                else:
                    if not self.client_secrets_file.exists():
                        raise UploadError(
                            f"YouTube API credentials not found.\n\n"
                            f"To enable YouTube uploads:\n"
                            f"1. Go to https://console.cloud.google.com/\n"
                            f"2. Create a project and enable YouTube Data API v3\n"
                            f"3. Create OAuth 2.0 credentials (Desktop app)\n"
                            f"4. Download JSON and save as:\n"
                            f"   {self.client_secrets_file}\n"
                        )
                    
                    logger.info("Opening browser for YouTube authentication...")
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self.client_secrets_file),
                        YOUTUBE_API_SCOPES
                    )
                    credentials = flow.run_local_server(port=0)
                
                # Cache credentials
                self.credentials_dir.mkdir(parents=True, exist_ok=True)
                with open(self.credentials_file, 'wb') as f:
                    pickle.dump(credentials, f)
                logger.info("YouTube credentials saved")
            
            return build('youtube', 'v3', credentials=credentials)
            
        except ImportError:
            raise UploadError("Google API libraries not available")
    
    def _upload_to_youtube(
        self, 
        youtube, 
        video_path: str, 
        title: str, 
        description: str,
        tags: Optional[List[str]] = None
    ) -> str:
        """Upload video to YouTube."""
        
        try:
            from googleapiclient.http import MediaFileUpload
            
            body = {
                'snippet': {
                    'title': title,
                    'description': description,
                    'tags': tags or ['karaoke', 'music'],
                    'categoryId': '10',  # Music category
                },
                'status': {
                    'privacyStatus': 'unlisted',
                    'selfDeclaredMadeForKids': False,
                },
            }
            
            # Create media upload
            media = MediaFileUpload(
                video_path,
                mimetype='video/mp4',
                resumable=True,
            )
            
            logger.info(f"Uploading: {title}")
            
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
                    logger.info(f"Upload progress: {progress}%")
            
            return response['id']
            
        except Exception as e:
            raise UploadError(f"YouTube upload failed: {e}")
    
    def _generate_metadata(self, title: str, artist: str) -> tuple:
        """Generate video title and description."""
        
        if artist and artist != "Unknown":
            video_title = f"{title} - {artist} (Karaoke)"
        else:
            video_title = f"{title} (Karaoke)"
        
        description = f"""Karaoke version of "{title}"{f' by {artist}' if artist and artist != 'Unknown' else ''}.

Features:
- Instrumental track (vocals removed)
- Word-by-word synchronized lyrics
- KaraFun-style highlighting

Generated with Y2Karaoke
https://github.com/dtunkelang/y2karaoke
"""
        
        return video_title, description

# Backward compatibility functions
def upload_video(video_path: str, title: str, description: str, tags: List[str] = None) -> str:
    """Upload video to YouTube (backward compatibility)."""
    uploader = YouTubeUploader()
    result = uploader.upload_video(video_path, title, "Unknown", tags)
    return result['url']

def generate_metadata(title: str, artist: str) -> tuple:
    """Generate metadata (backward compatibility)."""
    uploader = YouTubeUploader()
    return uploader._generate_metadata(title, artist)
