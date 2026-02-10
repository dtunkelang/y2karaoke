import y2karaoke.core.track_identifier as ti


def test_infer_artist_from_query_basic():
    identifier = ti.TrackIdentifier()
    artist = identifier._infer_artist_from_query(
        "Daft Punk One More Time", "One More Time"
    )
    assert artist == "Daft Punk"


def test_infer_artist_from_query_returns_none_for_short():
    identifier = ti.TrackIdentifier()
    artist = identifier._infer_artist_from_query("A B", "B")
    assert artist is None


def test_lookup_musicbrainz_for_query_selects_best(monkeypatch):
    identifier = ti.TrackIdentifier()

    def fake_search_recordings(recording, limit):
        return {
            "recording-list": [
                {
                    "length": "200000",
                    "artist-credit": [{"artist": {"name": "Good Artist"}}],
                    "title": "Good Title",
                },
                {
                    "length": "500000",
                    "artist-credit": [{"artist": {"name": "Too Long"}}],
                    "title": "Too Long",
                },
                {
                    "length": "210000",
                    "artist-credit": [{"artist": {"name": "Bad"}}],
                    "title": "No Match",
                },
            ]
        }

    identifier._mb_search_recordings = fake_search_recordings

    artist, title = identifier._lookup_musicbrainz_for_query(
        "Good Artist Good Title", 205
    )

    assert artist == "Good Artist"
    assert title == "Good Title"


def test_lookup_musicbrainz_for_query_handles_exception(monkeypatch):
    identifier = ti.TrackIdentifier()

    def fake_search_recordings(recording, limit):
        raise RuntimeError("boom")

    identifier._mb_search_recordings = fake_search_recordings

    artist, title = identifier._lookup_musicbrainz_for_query("Query", 200)

    assert artist is None
    assert title is None


def test_lookup_musicbrainz_for_query_skips_invalid_records(monkeypatch):
    identifier = ti.TrackIdentifier()

    def fake_search_recordings(recording, limit):
        return {
            "recording-list": [
                {
                    "length": None,
                    "artist-credit": [{"artist": {"name": "Artist"}}],
                    "title": "Title",
                },
                {
                    "length": "800000",
                    "artist-credit": [{"artist": {"name": "Artist"}}],
                    "title": "Title",
                },
                {
                    "length": "200000",
                    "artist-credit": [],
                    "title": "Title",
                },
                {
                    "length": "200000",
                    "artist-credit": [{"artist": {"name": "Artist"}}],
                    "title": None,
                },
                {
                    "length": "200000",
                    "artist-credit": [{"artist": {"name": "Artist"}}],
                    "title": "Other",
                },
            ]
        }

    identifier._mb_search_recordings = fake_search_recordings

    artist, title = identifier._lookup_musicbrainz_for_query("Query Words", 200)

    assert artist is None
    assert title is None


def test_lookup_musicbrainz_for_query_skips_too_long(monkeypatch):
    identifier = ti.TrackIdentifier()

    def fake_search_recordings(recording, limit):
        return {
            "recording-list": [
                {
                    "length": "750000",
                    "artist-credit": [{"artist": {"name": "Artist"}}],
                    "title": "Title",
                }
            ]
        }

    identifier._mb_search_recordings = fake_search_recordings

    artist, title = identifier._lookup_musicbrainz_for_query("Artist Title", 740)

    assert artist is None
    assert title is None


def test_query_musicbrainz_prioritizes_title_match(monkeypatch):
    identifier = ti.TrackIdentifier()

    def fake_search_recordings(recording, artist=None, limit=25):
        return {
            "recording-list": [
                {"title": "Other", "artist-credit": [{"artist": {"name": "Artist"}}]},
                {"title": "Song", "artist-credit": [{"artist": {"name": "Artist"}}]},
            ]
        }

    identifier._mb_search_recordings = fake_search_recordings
    monkeypatch.setattr(identifier, "_score_recording_studio_likelihood", lambda rec: 0)

    results = identifier._query_musicbrainz(
        query="Artist Song",
        artist_hint="Artist",
        title_hint="The Song",
    )

    assert results[0]["title"] == "Song"


def test_query_musicbrainz_exact_title_match_beats_partial(monkeypatch):
    identifier = ti.TrackIdentifier()

    def fake_search_recordings(recording, artist=None, limit=25):
        return {
            "recording-list": [
                {
                    "title": "Song Extended",
                    "artist-credit": [{"artist": {"name": "Artist"}}],
                },
                {"title": "Song", "artist-credit": [{"artist": {"name": "Artist"}}]},
            ]
        }

    identifier._mb_search_recordings = fake_search_recordings
    monkeypatch.setattr(identifier, "_score_recording_studio_likelihood", lambda rec: 0)

    results = identifier._query_musicbrainz(
        query="Artist Song",
        artist_hint="Artist",
        title_hint="Song",
    )

    assert results[0]["title"] == "Song"


def test_query_musicbrainz_retries_on_transient_error(monkeypatch):
    identifier = ti.TrackIdentifier()
    calls = {"count": 0}

    def fake_search_recordings(recording, artist=None, limit=25):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("connection reset")
        return {"recording-list": [{"title": "Song"}]}

    identifier._mb_search_recordings = fake_search_recordings
    monkeypatch.setattr(identifier, "_score_recording_studio_likelihood", lambda rec: 0)
    identifier._sleep = lambda *_: None

    results = identifier._query_musicbrainz(
        query="Artist Song",
        artist_hint=None,
        title_hint="Song",
        max_retries=1,
    )

    assert calls["count"] == 2
    assert results


def test_query_musicbrainz_returns_empty_on_non_transient(monkeypatch):
    identifier = ti.TrackIdentifier()

    def fake_search_recordings(recording, artist=None, limit=25):
        raise RuntimeError("bad request")

    identifier._mb_search_recordings = fake_search_recordings

    results = identifier._query_musicbrainz(
        query="Artist Song",
        artist_hint=None,
        title_hint="Song",
        max_retries=0,
    )

    assert results == []
