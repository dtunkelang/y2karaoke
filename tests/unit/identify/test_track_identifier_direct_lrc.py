from y2karaoke.core.components.identify.implementation import TrackIdentifier


def test_try_direct_lrc_search_swaps_query():
    def fake_fetch(query, artist, synced_only=True):
        if query == "wrong order":
            return None, False, ""
        return "[00:01.00]hi", True, "provider"

    identifier = TrackIdentifier(
        fetch_lyrics_multi_source_fn=fake_fetch,
        get_lrc_duration_fn=lambda *_: 180,
        extract_lrc_metadata_fn=lambda *_: (None, None),
        parse_query_fn=lambda *_: ("Artist", "Title"),
        search_youtube_verified_fn=lambda *a, **k: {
            "url": "https://youtube.com/watch?v=1",
            "duration": 180,
        },
    )

    result = identifier._try_direct_lrc_search("wrong order")

    assert result is not None
    assert result.artist == "Artist"
    assert result.title == "Title"


def test_try_direct_lrc_search_rejects_short_lrc():
    identifier = TrackIdentifier(
        fetch_lyrics_multi_source_fn=lambda *a, **k: ("[00:01.00]hi", True, "provider"),
        get_lrc_duration_fn=lambda *_: 30,
    )

    assert identifier._try_direct_lrc_search("artist title") is None


def test_try_direct_lrc_search_uses_musicbrainz_when_unknown():
    identifier = TrackIdentifier(
        fetch_lyrics_multi_source_fn=lambda *a, **k: ("[00:01.00]hi", True, "provider"),
        get_lrc_duration_fn=lambda *_: 180,
        extract_lrc_metadata_fn=lambda *_: (None, None),
        parse_query_fn=lambda *_: ("Unknown", "Title"),
        lookup_musicbrainz_for_query_fn=lambda *a, **k: ("MB Artist", "MB Title"),
        search_youtube_verified_fn=lambda *a, **k: {
            "url": "https://youtube.com/watch?v=2",
            "duration": 180,
        },
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


def test_try_direct_lrc_search_falls_back_to_query_when_artist_search_fails():
    calls = []

    def fake_search(query, duration, artist, title):
        calls.append(query)
        if len(calls) == 1:
            return None
        return {"url": "https://youtube.com/watch?v=3", "duration": 180}

    identifier = TrackIdentifier(
        fetch_lyrics_multi_source_fn=lambda *a, **k: ("[00:01.00]hi", True, "provider"),
        get_lrc_duration_fn=lambda *_: 180,
        extract_lrc_metadata_fn=lambda *_: ("Artist", "Title"),
        search_youtube_verified_fn=fake_search,
    )

    result = identifier._try_direct_lrc_search("original query")

    assert result is not None
    assert calls == ["Artist Title", "original query"]


def test_try_direct_lrc_search_returns_none_when_no_youtube_match():
    identifier = TrackIdentifier(
        fetch_lyrics_multi_source_fn=lambda *a, **k: ("[00:01.00]hi", True, "provider"),
        get_lrc_duration_fn=lambda *_: 180,
        extract_lrc_metadata_fn=lambda *_: ("Artist", "Title"),
        search_youtube_verified_fn=lambda *a, **k: None,
    )

    assert identifier._try_direct_lrc_search("query") is None


def test_try_direct_lrc_search_returns_none_when_no_lrc():
    identifier = TrackIdentifier(
        fetch_lyrics_multi_source_fn=lambda *a, **k: (None, False, ""),
    )

    assert identifier._try_direct_lrc_search("artist title") is None


def test_try_direct_lrc_search_infers_artist_when_missing():
    identifier = TrackIdentifier(
        fetch_lyrics_multi_source_fn=lambda *a, **k: ("[00:01.00]hi", True, "provider"),
        get_lrc_duration_fn=lambda *_: 180,
        extract_lrc_metadata_fn=lambda *_: (None, None),
        parse_query_fn=lambda *_: (None, None),
        lookup_musicbrainz_for_query_fn=lambda *a, **k: (None, None),
        infer_artist_from_query_fn=lambda *a, **k: "Guessed Artist",
        search_youtube_verified_fn=lambda *a, **k: {
            "url": "https://youtube.com/watch?v=4",
            "duration": 180,
        },
    )

    result = identifier._try_direct_lrc_search("some query")

    assert result is not None
    assert result.artist == "Guessed Artist"
    assert result.title == "some query"
