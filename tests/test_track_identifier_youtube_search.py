import logging
import sys
from types import SimpleNamespace

import pytest

from y2karaoke.core.track_identifier import TrackIdentifier
from y2karaoke.exceptions import Y2KaraokeError


def _make_response(text: str):
    class DummyResponse:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            return None

    return DummyResponse(text)


def _youtube_snippet(video_id: str, title: str, duration: str | None) -> str:
    duration_text = f'"simpleText":"{duration}"' if duration else ""
    return (
        f'"videoRenderer":{{"videoId":"{video_id}",'
        f'"title":{{"runs":[{{"text":"{title}"}}]}}'
        f',"lengthText":{{{duration_text}}}}}'
    )


def test_search_youtube_verified_scores_and_filters(monkeypatch):
    identifier = TrackIdentifier()

    response_text = "\n".join(
        [
            _youtube_snippet("abc123def45", "Artist - Song (Live)", "5:00"),
            _youtube_snippet("ghi456jkl78", "Artist - Song (Official Audio)", "5:01"),
            _youtube_snippet("mno789pqr01", "Other - Different", "4:50"),
        ]
    )

    def fake_get(*_args, **_kwargs):
        return _make_response(response_text)

    monkeypatch.setattr("requests.get", fake_get)

    result = identifier._search_youtube_verified(
        "Artist Song", 300, expected_artist="Artist", expected_title="Song"
    )

    assert result is not None
    assert result["url"].endswith("ghi456jkl78")
    assert result["duration"] == 301


def test_search_youtube_verified_prefers_audio_title(monkeypatch):
    identifier = TrackIdentifier()

    response_text = "\n".join(
        [
            _youtube_snippet("vid11122233", "Artist - Song (Official Video)", "3:30"),
            _youtube_snippet("aud11122233", "Artist - Song (Official Audio)", "3:30"),
        ]
    )

    def fake_get(*_args, **_kwargs):
        return _make_response(response_text)

    monkeypatch.setattr("requests.get", fake_get)

    result = identifier._search_youtube_verified(
        "Artist Song", 210, expected_artist="Artist", expected_title="Song"
    )

    assert result is not None
    assert result["url"].endswith("aud11122233")


def test_search_youtube_verified_returns_none_for_no_match(monkeypatch):
    identifier = TrackIdentifier()

    response_text = _youtube_snippet("abc123def45", "Completely Different", "4:00")

    def fake_get(*_args, **_kwargs):
        return _make_response(response_text)

    monkeypatch.setattr("requests.get", fake_get)

    result = identifier._search_youtube_verified(
        "Artist Song", 240, expected_artist="Artist", expected_title="Song"
    )

    assert result is None


def test_search_youtube_by_duration_prefers_audio_fallback(monkeypatch):
    identifier = TrackIdentifier()

    results = [
        {"url": "https://www.youtube.com/watch?v=first", "duration": 400},
        {"url": "https://www.youtube.com/watch?v=audio", "duration": 305},
    ]

    def fake_single(*_args, **_kwargs):
        return results.pop(0)

    monkeypatch.setattr(identifier, "_search_youtube_single", fake_single)

    result = identifier._search_youtube_by_duration("Artist Song", 300)

    assert result["url"].endswith("audio")


def test_search_youtube_by_duration_uses_audio_when_no_results(monkeypatch):
    identifier = TrackIdentifier()

    calls = []

    def fake_single(query, *_args, **_kwargs):
        calls.append(query)
        if "audio" in query:
            return {"url": "https://www.youtube.com/watch?v=audio", "duration": 123}
        return None

    monkeypatch.setattr(identifier, "_search_youtube_single", fake_single)

    result = identifier._search_youtube_by_duration("Artist Song", 0)

    assert result["url"].endswith("audio")
    assert calls == ["Artist Song", "Artist Song audio"]


def test_search_youtube_single_returns_first_when_no_duration(monkeypatch):
    identifier = TrackIdentifier()

    candidates = [
        {"video_id": "abc123def45", "title": "Song Live", "duration": None},
        {
            "video_id": "ghi456jkl78",
            "title": "Song Official Audio",
            "duration": None,
        },
    ]

    def fake_get(*_args, **_kwargs):
        return _make_response("ignored")

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr(
        identifier, "_extract_youtube_candidates", lambda _text: candidates
    )

    result = identifier._search_youtube_single("Song live", 300)

    assert result["url"].endswith("abc123def45")
    assert result["duration"] is None


def test_search_youtube_single_filters_non_studio_when_no_duration(monkeypatch):
    identifier = TrackIdentifier()

    candidates = [
        {"video_id": "abc123def45", "title": "Song Live", "duration": None},
        {
            "video_id": "ghi456jkl78",
            "title": "Song Official Audio",
            "duration": None,
        },
    ]

    def fake_get(*_args, **_kwargs):
        return _make_response("ignored")

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr(
        identifier, "_extract_youtube_candidates", lambda _text: candidates
    )

    result = identifier._search_youtube_single("Song", 300)

    assert result["url"].endswith("ghi456jkl78")
    assert result["duration"] is None


def test_search_youtube_single_keeps_non_studio_when_query_requests(monkeypatch):
    identifier = TrackIdentifier()

    candidates = [
        {"video_id": "abc123def45", "title": "Song Live at Wembley", "duration": 300},
        {"video_id": "ghi456jkl78", "title": "Song Official Audio", "duration": 305},
    ]

    def fake_get(*_args, **_kwargs):
        return _make_response("ignored")

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr(
        identifier, "_extract_youtube_candidates", lambda _text: candidates
    )

    result = identifier._search_youtube_single("Song live", 300)

    assert result["url"].endswith("abc123def45")
    assert result["duration"] == 300


def test_search_youtube_single_prefers_official_audio(monkeypatch):
    identifier = TrackIdentifier()

    candidates = [
        {"video_id": "vid11122233", "title": "Song (Official Video)", "duration": 210},
        {
            "video_id": "aud11122233",
            "title": "Song (Official Audio)",
            "duration": 210,
        },
    ]

    def fake_get(*_args, **_kwargs):
        return _make_response("ignored")

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr(
        identifier, "_extract_youtube_candidates", lambda _text: candidates
    )

    result = identifier._search_youtube_single("Song", 210)

    assert result["url"].endswith("aud11122233")
    assert result["duration"] == 210


def test_search_youtube_single_warns_on_large_mismatch(monkeypatch, caplog):
    identifier = TrackIdentifier()

    candidates = [
        {"video_id": "abc123def45", "title": "Song", "duration": 400},
        {"video_id": "ghi456jkl78", "title": "Song", "duration": 410},
    ]

    def fake_get(*_args, **_kwargs):
        return _make_response("ignored")

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr(
        identifier, "_extract_youtube_candidates", lambda _text: candidates
    )

    with caplog.at_level(logging.WARNING):
        result = identifier._search_youtube_single("Song", 200)

    assert result["url"].endswith("abc123def45")
    assert "No YouTube video within" in caplog.text


def test_extract_youtube_candidates_parses_hours():
    identifier = TrackIdentifier()

    response_text = _youtube_snippet("abc123def45", "Long Track", "1:02:03")

    candidates = identifier._extract_youtube_candidates(response_text)

    assert candidates[0]["duration"] == 3723


def test_get_youtube_metadata_success(monkeypatch):
    identifier = TrackIdentifier()

    class DummyYDL:
        def __init__(self, _opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, _url, download=False):
            return {"title": "Test Song", "uploader": "Uploader", "duration": 123}

    dummy_module = SimpleNamespace(YoutubeDL=DummyYDL)
    monkeypatch.setitem(sys.modules, "yt_dlp", dummy_module)

    title, uploader, duration = identifier._get_youtube_metadata(
        "https://www.youtube.com/watch?v=abc123def45"
    )

    assert title == "Test Song"
    assert uploader == "Uploader"
    assert duration == 123


def test_get_youtube_metadata_error(monkeypatch):
    identifier = TrackIdentifier()

    class DummyYDL:
        def __init__(self, _opts):
            pass

        def __enter__(self):
            raise RuntimeError("boom")

        def __exit__(self, exc_type, exc, tb):
            return False

    dummy_module = SimpleNamespace(YoutubeDL=DummyYDL)
    monkeypatch.setitem(sys.modules, "yt_dlp", dummy_module)

    with pytest.raises(Y2KaraokeError):
        identifier._get_youtube_metadata("https://www.youtube.com/watch?v=abc123def45")
