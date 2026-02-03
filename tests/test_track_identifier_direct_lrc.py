import types

from y2karaoke.core.track_identifier import TrackIdentifier


def test_try_direct_lrc_search_swaps_query(monkeypatch):
    identifier = TrackIdentifier()

    def fake_fetch(query, artist, synced_only=True):
        if query == "wrong order":
            return None, False, ""
        return "[00:01.00]hi", True, "provider"

    monkeypatch.setattr(
        "y2karaoke.core.sync.fetch_lyrics_multi_source",
        fake_fetch,
    )
    monkeypatch.setattr(
        "y2karaoke.core.sync.get_lrc_duration",
        lambda *_: 180,
    )
    monkeypatch.setattr(identifier, "_extract_lrc_metadata", lambda *_: (None, None))
    monkeypatch.setattr(identifier, "_parse_query", lambda *_: ("Artist", "Title"))
    monkeypatch.setattr(
        identifier,
        "_search_youtube_verified",
        lambda *a, **k: {"url": "https://youtube.com/watch?v=1", "duration": 180},
    )

    result = identifier._try_direct_lrc_search("wrong order")

    assert result is not None
    assert result.artist == "Artist"
    assert result.title == "Title"


def test_try_direct_lrc_search_rejects_short_lrc(monkeypatch):
    identifier = TrackIdentifier()

    monkeypatch.setattr(
        "y2karaoke.core.sync.fetch_lyrics_multi_source",
        lambda *a, **k: ("[00:01.00]hi", True, "provider"),
    )
    monkeypatch.setattr(
        "y2karaoke.core.sync.get_lrc_duration",
        lambda *_: 30,
    )

    assert identifier._try_direct_lrc_search("artist title") is None


def test_try_direct_lrc_search_uses_musicbrainz_when_unknown(monkeypatch):
    identifier = TrackIdentifier()

    monkeypatch.setattr(
        "y2karaoke.core.sync.fetch_lyrics_multi_source",
        lambda *a, **k: ("[00:01.00]hi", True, "provider"),
    )
    monkeypatch.setattr(
        "y2karaoke.core.sync.get_lrc_duration",
        lambda *_: 180,
    )
    monkeypatch.setattr(identifier, "_extract_lrc_metadata", lambda *_: (None, None))
    monkeypatch.setattr(identifier, "_parse_query", lambda *_: ("Unknown", "Title"))
    monkeypatch.setattr(
        identifier,
        "_lookup_musicbrainz_for_query",
        lambda *a, **k: ("MB Artist", "MB Title"),
    )
    monkeypatch.setattr(
        identifier,
        "_search_youtube_verified",
        lambda *a, **k: {"url": "https://youtube.com/watch?v=2", "duration": 180},
    )

    result = identifier._try_direct_lrc_search("some query")

    assert result is not None
    assert result.artist == "MB Artist"
    assert result.title == "MB Title"


def test_extract_lrc_metadata_parses_tags():
    identifier = TrackIdentifier()
    lrc_text = "[ar:Test Artist]\n[ti:Test Title]\n[00:01.00]Line"

    artist, title = identifier._extract_lrc_metadata(lrc_text)

    assert artist == "Test Artist"
    assert title == "Test Title"


def test_try_direct_lrc_search_falls_back_to_query_when_artist_search_fails(
    monkeypatch,
):
    identifier = TrackIdentifier()

    monkeypatch.setattr(
        "y2karaoke.core.sync.fetch_lyrics_multi_source",
        lambda *a, **k: ("[00:01.00]hi", True, "provider"),
    )
    monkeypatch.setattr(
        "y2karaoke.core.sync.get_lrc_duration",
        lambda *_: 180,
    )
    monkeypatch.setattr(
        identifier, "_extract_lrc_metadata", lambda *_: ("Artist", "Title")
    )

    calls = []

    def fake_search(query, duration, artist, title):
        calls.append(query)
        if len(calls) == 1:
            return None
        return {"url": "https://youtube.com/watch?v=3", "duration": 180}

    monkeypatch.setattr(identifier, "_search_youtube_verified", fake_search)

    result = identifier._try_direct_lrc_search("original query")

    assert result is not None
    assert calls == ["Artist Title", "original query"]


def test_try_direct_lrc_search_returns_none_when_no_youtube_match(monkeypatch):
    identifier = TrackIdentifier()

    monkeypatch.setattr(
        "y2karaoke.core.sync.fetch_lyrics_multi_source",
        lambda *a, **k: ("[00:01.00]hi", True, "provider"),
    )
    monkeypatch.setattr(
        "y2karaoke.core.sync.get_lrc_duration",
        lambda *_: 180,
    )
    monkeypatch.setattr(
        identifier, "_extract_lrc_metadata", lambda *_: ("Artist", "Title")
    )
    monkeypatch.setattr(identifier, "_search_youtube_verified", lambda *a, **k: None)

    assert identifier._try_direct_lrc_search("query") is None


def test_try_direct_lrc_search_infers_artist_when_missing(monkeypatch):
    identifier = TrackIdentifier()

    monkeypatch.setattr(
        "y2karaoke.core.sync.fetch_lyrics_multi_source",
        lambda *a, **k: ("[00:01.00]hi", True, "provider"),
    )
    monkeypatch.setattr(
        "y2karaoke.core.sync.get_lrc_duration",
        lambda *_: 180,
    )
    monkeypatch.setattr(identifier, "_extract_lrc_metadata", lambda *_: (None, None))
    monkeypatch.setattr(identifier, "_parse_query", lambda *_: (None, None))
    monkeypatch.setattr(
        identifier, "_lookup_musicbrainz_for_query", lambda *a, **k: (None, None)
    )
    monkeypatch.setattr(
        identifier, "_infer_artist_from_query", lambda *a, **k: "Guessed Artist"
    )
    monkeypatch.setattr(
        identifier,
        "_search_youtube_verified",
        lambda *a, **k: {"url": "https://youtube.com/watch?v=4", "duration": 180},
    )

    result = identifier._try_direct_lrc_search("some query")

    assert result is not None
    assert result.artist == "Guessed Artist"
    assert result.title == "some query"
