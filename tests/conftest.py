"""Test configuration and fixtures.

Provides reusable fixtures for:
- Temporary files and directories
- MusicBrainz API responses
- YouTube metadata and search results
- LRC (synced lyrics) provider responses
- TrackInfo objects for testing
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run tests that require network access",
    )


def pytest_collection_modifyitems(config, items):
    run_network = config.getoption("--run-network") or os.getenv(
        "RUN_INTEGRATION_TESTS"
    ) == "1"
    if run_network:
        return

    skip_network = pytest.mark.skip(
        reason="requires network access (use --run-network or RUN_INTEGRATION_TESTS=1)"
    )
    for item in items:
        if "network" in item.keywords or "integration" in item.keywords:
            item.add_marker(skip_network)

# =============================================================================
# Basic Fixtures
# =============================================================================


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


# =============================================================================
# MusicBrainz Fixtures
# =============================================================================


@pytest.fixture
def musicbrainz_recording_queen():
    """MusicBrainz recording response for Queen - Bohemian Rhapsody."""
    return {
        "id": "ebf79ba5-085e-48d2-9eb8-2d992fbf0f6d",
        "title": "Bohemian Rhapsody",
        "length": 354000,  # milliseconds
        "disambiguation": "",
        "artist-credit": [
            {"artist": {"id": "0383dadf-2a4e-4d10-a46a-e9e041da8eb3", "name": "Queen"}}
        ],
        "release-list": [
            {
                "id": "a1b2c3d4",
                "title": "A Night at the Opera",
                "release-group": {"primary-type": "Album", "secondary-type-list": []},
            }
        ],
    }


@pytest.fixture
def musicbrainz_recording_beatles():
    """MusicBrainz recording response for The Beatles - Yesterday."""
    return {
        "id": "bc9c9f88-3c2c-4f80-8bd1-f38e3bdc0789",
        "title": "Yesterday",
        "length": 125000,
        "disambiguation": "",
        "artist-credit": [
            {
                "artist": {
                    "id": "b10bbbfc-cf9e-42e0-be17-e2c3e1d2600d",
                    "name": "The Beatles",
                }
            }
        ],
        "release-list": [
            {
                "id": "e5f6g7h8",
                "title": "Help!",
                "release-group": {"primary-type": "Album", "secondary-type-list": []},
            }
        ],
    }


@pytest.fixture
def musicbrainz_recording_live():
    """MusicBrainz recording response for a live version (should be penalized)."""
    return {
        "id": "live-version-id",
        "title": "Bohemian Rhapsody (Live at Wembley)",
        "length": 420000,
        "disambiguation": "live",
        "artist-credit": [
            {"artist": {"id": "0383dadf-2a4e-4d10-a46a-e9e041da8eb3", "name": "Queen"}}
        ],
        "release-list": [
            {
                "id": "live-album-id",
                "title": "Live at Wembley",
                "release-group": {
                    "primary-type": "Album",
                    "secondary-type-list": ["Live"],
                },
            }
        ],
    }


@pytest.fixture
def musicbrainz_search_results(musicbrainz_recording_queen, musicbrainz_recording_live):
    """MusicBrainz search results with multiple recordings."""
    return {
        "recording-list": [
            musicbrainz_recording_queen,
            musicbrainz_recording_live,
        ]
    }


@pytest.fixture
def mock_musicbrainz(musicbrainz_search_results):
    """Mock musicbrainzngs module for testing."""
    with patch("musicbrainzngs.search_recordings") as mock_search:
        mock_search.return_value = musicbrainz_search_results
        yield mock_search


# =============================================================================
# YouTube Fixtures
# =============================================================================


@pytest.fixture
def youtube_metadata_queen():
    """YouTube metadata for Queen - Bohemian Rhapsody official video."""
    return {
        "title": "Queen - Bohemian Rhapsody (Official Video)",
        "uploader": "Queen Official",
        "channel": "Queen Official",
        "duration": 354,
        "id": "fJ9rUzIMcZQ",
        "view_count": 1500000000,
    }


@pytest.fixture
def youtube_metadata_cover():
    """YouTube metadata for a cover version."""
    return {
        "title": "Bohemian Rhapsody - Amazing Cover by Random Artist",
        "uploader": "RandomCoverChannel",
        "channel": "RandomCoverChannel",
        "duration": 360,
        "id": "xyz789abc",
        "view_count": 50000,
    }


@pytest.fixture
def youtube_search_response_queen():
    """Mock YouTube search HTML response containing Queen videos."""
    return """
    "videoRenderer":{"videoId":"fJ9rUzIMcZQ","title":{"runs":[{"text":"Queen - Bohemian Rhapsody (Official Video)"}]},"simpleText":"5:54"}
    "videoRenderer":{"videoId":"abc123def","title":{"runs":[{"text":"Queen - Bohemian Rhapsody (Live at Wembley)"}]},"simpleText":"7:00"}
    "videoRenderer":{"videoId":"xyz789ghi","title":{"runs":[{"text":"Bohemian Rhapsody Cover"}]},"simpleText":"5:50"}
    """


@pytest.fixture
def mock_youtube_dlp(youtube_metadata_queen):
    """Mock yt_dlp for testing YouTube metadata extraction."""
    mock_ydl = MagicMock()
    mock_ydl.__enter__ = Mock(return_value=mock_ydl)
    mock_ydl.__exit__ = Mock(return_value=False)
    mock_ydl.extract_info = Mock(return_value=youtube_metadata_queen)

    with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
        yield mock_ydl


@pytest.fixture
def mock_youtube_search(youtube_search_response_queen):
    """Mock requests.get for YouTube search."""
    mock_response = Mock()
    mock_response.text = youtube_search_response_queen
    mock_response.raise_for_status = Mock()

    with patch("requests.get", return_value=mock_response) as mock_get:
        yield mock_get


# =============================================================================
# LRC (Synced Lyrics) Fixtures
# =============================================================================


@pytest.fixture
def lrc_bohemian_rhapsody():
    """Synced LRC lyrics for Bohemian Rhapsody (simplified)."""
    return """[ar:Queen]
[ti:Bohemian Rhapsody]
[al:A Night at the Opera]
[length:05:54]

[00:00.00]Is this the real life?
[00:04.00]Is this just fantasy?
[00:08.00]Caught in a landslide
[00:11.00]No escape from reality
[00:15.00]Open your eyes
[00:19.00]Look up to the skies and see
[00:25.00]I'm just a poor boy
[00:27.00]I need no sympathy
[00:31.00]Because I'm easy come, easy go
[00:35.00]Little high, little low
[00:39.00]Any way the wind blows
[00:43.00]Doesn't really matter to me, to me
[05:50.00]Nothing really matters
[05:54.00]"""


@pytest.fixture
def lrc_yesterday():
    """Synced LRC lyrics for Yesterday (simplified)."""
    return """[ar:The Beatles]
[ti:Yesterday]
[al:Help!]
[length:02:05]

[00:00.00]Yesterday
[00:04.00]All my troubles seemed so far away
[00:10.00]Now it looks as though they're here to stay
[00:16.00]Oh, I believe in yesterday
[02:00.00]Yesterday
[02:05.00]"""


@pytest.fixture
def lrc_invalid_timing():
    """LRC with invalid/sparse timing (should fail validation)."""
    return """[ar:Artist]
[ti:Song]
[00:00.00]First line
[05:00.00]Last line after huge gap
"""


@pytest.fixture
def lrc_no_metadata():
    """LRC without metadata tags."""
    return """[00:00.00]First line of lyrics
[00:05.00]Second line of lyrics
[00:10.00]Third line of lyrics
[03:00.00]Last line
"""


@pytest.fixture
def mock_lrc_provider(lrc_bohemian_rhapsody):
    """Mock syncedlyrics/lyriq for testing LRC fetching."""
    with patch("y2karaoke.core.sync.fetch_lyrics_multi_source") as mock_fetch:
        mock_fetch.return_value = (lrc_bohemian_rhapsody, True, "lyriq")
        yield mock_fetch


@pytest.fixture
def mock_lrc_provider_not_found():
    """Mock LRC provider that returns no results."""
    with patch("y2karaoke.core.sync.fetch_lyrics_multi_source") as mock_fetch:
        mock_fetch.return_value = (None, False, None)
        yield mock_fetch


# =============================================================================
# TrackInfo Fixtures
# =============================================================================


@pytest.fixture
def track_info_queen():
    """TrackInfo for Queen - Bohemian Rhapsody."""
    from y2karaoke.core.track_identifier import TrackInfo

    return TrackInfo(
        artist="Queen",
        title="Bohemian Rhapsody",
        duration=354,
        youtube_url="https://www.youtube.com/watch?v=fJ9rUzIMcZQ",
        youtube_duration=354,
        source="musicbrainz",
        lrc_duration=354,
        lrc_validated=True,
        identification_quality=95.0,
        quality_issues=[],
        sources_tried=["musicbrainz", "lyriq"],
        fallback_used=False,
    )


@pytest.fixture
def track_info_beatles():
    """TrackInfo for The Beatles - Yesterday."""
    from y2karaoke.core.track_identifier import TrackInfo

    return TrackInfo(
        artist="The Beatles",
        title="Yesterday",
        duration=125,
        youtube_url="https://www.youtube.com/watch?v=NrgmdOz227I",
        youtube_duration=125,
        source="syncedlyrics",
        lrc_duration=125,
        lrc_validated=True,
        identification_quality=90.0,
        quality_issues=[],
        sources_tried=["lyriq"],
        fallback_used=False,
    )


@pytest.fixture
def track_info_youtube_fallback():
    """TrackInfo from YouTube fallback (no LRC found)."""
    from y2karaoke.core.track_identifier import TrackInfo

    return TrackInfo(
        artist="Unknown Artist",
        title="Some Song",
        duration=200,
        youtube_url="https://www.youtube.com/watch?v=abc123",
        youtube_duration=200,
        source="youtube",
        lrc_duration=None,
        lrc_validated=False,
        identification_quality=50.0,
        quality_issues=["No synced lyrics found", "Artist identification uncertain"],
        sources_tried=["musicbrainz", "lyriq", "musixmatch"],
        fallback_used=True,
    )


# =============================================================================
# Combined Service Mocks
# =============================================================================


@pytest.fixture
def mock_all_external_services(
    mock_musicbrainz, mock_youtube_dlp, mock_youtube_search, mock_lrc_provider
):
    """Mock all external services for fully isolated testing."""
    return {
        "musicbrainz": mock_musicbrainz,
        "youtube_dlp": mock_youtube_dlp,
        "youtube_search": mock_youtube_search,
        "lrc_provider": mock_lrc_provider,
    }


@pytest.fixture
def mock_track_identifier_dependencies():
    """Mock all dependencies for TrackIdentifier testing."""
    with patch("y2karaoke.core.track_identifier.musicbrainzngs") as mock_mb:
        with patch("y2karaoke.core.sync.fetch_lyrics_multi_source") as mock_lrc:
            with patch("y2karaoke.core.sync.get_lrc_duration") as mock_duration:
                with patch("y2karaoke.core.sync.validate_lrc_quality") as mock_validate:
                    with patch("requests.get") as mock_requests:
                        # Set up default returns
                        mock_mb.search_recordings.return_value = {"recording-list": []}
                        mock_lrc.return_value = (None, False, None)
                        mock_duration.return_value = None
                        mock_validate.return_value = (True, None)
                        mock_requests.return_value = Mock(
                            text="", raise_for_status=Mock()
                        )

                        yield {
                            "musicbrainz": mock_mb,
                            "fetch_lyrics": mock_lrc,
                            "get_duration": mock_duration,
                            "validate_quality": mock_validate,
                            "requests": mock_requests,
                        }


# =============================================================================
# Lyrics Line Fixtures
# =============================================================================


@pytest.fixture
def sample_lyrics_lines():
    """Sample parsed lyrics lines for testing."""
    from y2karaoke.core.lyrics import Line, Word

    return [
        Line(
            words=[
                Word(text="Is", start_time=0.0, end_time=0.5),
                Word(text="this", start_time=0.5, end_time=1.0),
                Word(text="the", start_time=1.0, end_time=1.3),
                Word(text="real", start_time=1.3, end_time=1.8),
                Word(text="life?", start_time=1.8, end_time=2.5),
            ]
        ),
        Line(
            words=[
                Word(text="Is", start_time=4.0, end_time=4.5),
                Word(text="this", start_time=4.5, end_time=5.0),
                Word(text="just", start_time=5.0, end_time=5.5),
                Word(text="fantasy?", start_time=5.5, end_time=7.0),
            ]
        ),
    ]


@pytest.fixture
def sample_lyrics_lines_with_break():
    """Sample lyrics lines with a long instrumental break."""
    from y2karaoke.core.lyrics import Line, Word

    return [
        Line(
            words=[
                Word(text="First", start_time=0.0, end_time=0.5),
                Word(text="verse", start_time=0.5, end_time=1.5),
            ]
        ),
        # 60 second gap here - instrumental break
        Line(
            words=[
                Word(text="After", start_time=61.5, end_time=62.0),
                Word(text="break", start_time=62.0, end_time=63.0),
            ]
        ),
    ]


# =============================================================================
# Cache Fixtures
# =============================================================================


@pytest.fixture
def mock_cache_manager(temp_dir):
    """Mock CacheManager for testing caching behavior."""
    from y2karaoke.utils.cache import CacheManager

    return CacheManager(temp_dir)


@pytest.fixture
def populated_cache(mock_cache_manager):
    """CacheManager with pre-populated test data."""
    video_id = "test_video_123"

    # Create cache directory structure
    video_dir = mock_cache_manager.get_video_cache_dir(video_id)
    video_dir.mkdir(parents=True, exist_ok=True)

    # Add some cached files
    (video_dir / "audio.wav").write_bytes(b"fake audio")
    (video_dir / "vocals.wav").write_bytes(b"fake vocals")
    (video_dir / "instrumental.wav").write_bytes(b"fake instrumental")

    # Save metadata
    mock_cache_manager.save_metadata(
        video_id,
        {
            "title": "Test Song",
            "artist": "Test Artist",
            "duration": 200,
        },
    )

    return mock_cache_manager, video_id
