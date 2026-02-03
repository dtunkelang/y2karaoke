import y2karaoke.core.track_identifier as ti


def test_infer_artist_from_query_basic():
    identifier = ti.TrackIdentifier()
    artist = identifier._infer_artist_from_query("Daft Punk One More Time", "One More Time")
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

    monkeypatch.setattr(ti.musicbrainzngs, "search_recordings", fake_search_recordings)

    artist, title = identifier._lookup_musicbrainz_for_query("Good Artist Good Title", 205)

    assert artist == "Good Artist"
    assert title == "Good Title"


def test_lookup_musicbrainz_for_query_handles_exception(monkeypatch):
    identifier = ti.TrackIdentifier()

    def fake_search_recordings(recording, limit):
        raise RuntimeError("boom")

    monkeypatch.setattr(ti.musicbrainzngs, "search_recordings", fake_search_recordings)

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

    monkeypatch.setattr(ti.musicbrainzngs, "search_recordings", fake_search_recordings)

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

    monkeypatch.setattr(ti.musicbrainzngs, "search_recordings", fake_search_recordings)

    artist, title = identifier._lookup_musicbrainz_for_query("Artist Title", 740)

    assert artist is None
    assert title is None
