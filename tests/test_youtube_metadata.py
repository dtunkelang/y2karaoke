import hashlib
import sys
import types

import pytest

from y2karaoke.core import youtube_metadata


def test_validate_youtube_url_accepts_valid_urls():
    assert (
        youtube_metadata.validate_youtube_url("https://www.youtube.com/watch?v=abc123def45")
        == "https://www.youtube.com/watch?v=abc123def45"
    )
    assert (
        youtube_metadata.validate_youtube_url("https://youtu.be/abc123def45")
        == "https://youtu.be/abc123def45"
    )


def test_validate_youtube_url_rejects_invalid_url():
    with pytest.raises(ValueError):
        youtube_metadata.validate_youtube_url("https://example.com/video")


def test_extract_video_id_patterns():
    assert (
        youtube_metadata.extract_video_id("https://www.youtube.com/watch?v=abc123def45")
        == "abc123def45"
    )
    assert (
        youtube_metadata.extract_video_id("https://youtu.be/abc123def45")
        == "abc123def45"
    )
    assert (
        youtube_metadata.extract_video_id("https://www.youtube.com/embed/abc123def45")
        == "abc123def45"
    )


def test_extract_video_id_fallback_hash():
    url = "https://example.com/not-youtube"
    expected = hashlib.md5(url.encode()).hexdigest()[:11]
    assert youtube_metadata.extract_video_id(url) == expected


def test_parse_artist_title_from_video_title():
    artist, title = youtube_metadata._parse_artist_title_from_video_title(
        "Artist - Song (Official Music Video)"
    )
    assert artist == "Artist"
    assert title == "Song"

    artist, title = youtube_metadata._parse_artist_title_from_video_title(
        "Song by Artist"
    )
    assert artist == "Artist"
    assert title == "Song"


def test_parse_metadata_from_description():
    description = "Song · Artist\nAlbum · 2020\n"
    artist, title = youtube_metadata._parse_metadata_from_description(
        description, "Fallback"
    )
    assert artist == "Artist"
    assert title == "Song"


def test_clean_uploader_name():
    assert youtube_metadata._clean_uploader_name("Artist Official") == "Artist"
    assert youtube_metadata._clean_uploader_name("Official Artist") == "Artist"


def test_clean_title_strips_artist():
    assert (
        youtube_metadata.clean_title("Artist - Song Title", artist="Artist")
        == "Song Title"
    )


def test_extract_metadata_from_youtube_prefers_artist_track(monkeypatch):
    info = {
        "title": "Video Title",
        "uploader": "Uploader",
        "description": "",
        "artist": "Meta Artist",
        "track": "Meta Track",
        "id": "abc123def45",
    }

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download=False):
            return info

    monkeypatch.setitem(
        sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL)
    )

    result = youtube_metadata.extract_metadata_from_youtube(
        "https://www.youtube.com/watch?v=abc123def45"
    )
    assert result["artist"] == "Meta Artist"
    assert result["title"] == "Meta Track"
    assert result["video_id"] == "abc123def45"


def test_extract_metadata_from_youtube_parses_title(monkeypatch):
    info = {
        "title": "Artist - Song (Official Video)",
        "uploader": "Artist",
        "description": "",
        "id": "abc123def45",
    }

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download=False):
            return info

    monkeypatch.setitem(
        sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL)
    )

    result = youtube_metadata.extract_metadata_from_youtube(
        "https://www.youtube.com/watch?v=abc123def45"
    )
    assert result["artist"] == "Artist"
    assert result["title"] == "Song"
    assert result["video_id"] == "abc123def45"
