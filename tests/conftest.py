"""Test configuration and fixtures."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_youtube_url():
    """Sample YouTube URL for testing."""
    return "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

@pytest.fixture
def mock_audio_file(temp_dir):
    """Create a mock audio file."""
    audio_file = temp_dir / "test_audio.wav"
    audio_file.write_bytes(b"fake audio data")
    return audio_file

@pytest.fixture
def mock_video_file(temp_dir):
    """Create a mock video file."""
    video_file = temp_dir / "test_video.mp4"
    video_file.write_bytes(b"fake video data")
    return video_file
