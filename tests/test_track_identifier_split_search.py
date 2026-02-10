import y2karaoke.core.components.identify.implementation as ti


def test_try_split_search_selects_best_candidate(monkeypatch):
    identifier = ti.TrackIdentifier()

    monkeypatch.setattr(
        identifier,
        "_try_artist_title_splits",
        lambda query: [("Artist1", "Title1"), ("Artist2", "Title2")],
    )

    monkeypatch.setattr(
        identifier,
        "_query_musicbrainz",
        lambda query, artist, title: [{"dummy": True}],
    )

    def fake_find_best(recordings, query, artist_hint):
        if artist_hint == "Artist1":
            return (200, "Artist1", "Title1")
        return (210, "Artist2", "Title2")

    monkeypatch.setattr(identifier, "_find_best_with_artist_hint", fake_find_best)
    monkeypatch.setattr(
        identifier,
        "_score_split_candidate",
        lambda c, a, t, q: 20 if a == "Artist2" else 10,
    )
    monkeypatch.setattr(
        identifier, "_check_lrc_and_duration", lambda title, artist: (True, 200)
    )

    result = identifier._try_split_search("Artist1 Title1")

    assert result == (210, "Artist2", "Title2")


def test_try_split_search_returns_none_when_no_candidates(monkeypatch):
    identifier = ti.TrackIdentifier()

    monkeypatch.setattr(
        identifier, "_try_artist_title_splits", lambda query: [("Artist", "Title")]
    )
    monkeypatch.setattr(
        identifier, "_query_musicbrainz", lambda query, artist, title: []
    )

    assert identifier._try_split_search("Artist Title") is None


def test_try_split_search_skips_when_no_best_candidate(monkeypatch):
    identifier = ti.TrackIdentifier()

    monkeypatch.setattr(
        identifier, "_try_artist_title_splits", lambda query: [("Artist", "Title")]
    )
    monkeypatch.setattr(
        identifier,
        "_query_musicbrainz",
        lambda query, artist, title: [{"dummy": True}],
    )
    monkeypatch.setattr(
        identifier,
        "_find_best_with_artist_hint",
        lambda recordings, query, artist: None,
    )

    assert identifier._try_split_search("Artist Title") is None
