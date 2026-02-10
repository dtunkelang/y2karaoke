import y2karaoke.core.components.identify.implementation as ti


def test_try_split_search_selects_best_candidate():
    def fake_find_best(recordings, query, artist_hint):
        if artist_hint == "Artist1":
            return (200, "Artist1", "Title1")
        return (210, "Artist2", "Title2")

    identifier = ti.TrackIdentifier(
        try_artist_title_splits_fn=lambda query: [
            ("Artist1", "Title1"),
            ("Artist2", "Title2"),
        ],
        query_musicbrainz_fn=lambda query, artist, title: [{"dummy": True}],
        find_best_with_artist_hint_fn=fake_find_best,
        score_split_candidate_fn=lambda c, a, t, q: 20 if a == "Artist2" else 10,
        check_lrc_and_duration_fn=lambda title, artist: (True, 200),
    )

    result = identifier._try_split_search("Artist1 Title1")

    assert result == (210, "Artist2", "Title2")


def test_try_split_search_returns_none_when_no_candidates():
    identifier = ti.TrackIdentifier(
        try_artist_title_splits_fn=lambda query: [("Artist", "Title")],
        query_musicbrainz_fn=lambda query, artist, title: [],
    )

    assert identifier._try_split_search("Artist Title") is None


def test_try_split_search_skips_when_no_best_candidate():
    identifier = ti.TrackIdentifier(
        try_artist_title_splits_fn=lambda query: [("Artist", "Title")],
        query_musicbrainz_fn=lambda query, artist, title: [{"dummy": True}],
        find_best_with_artist_hint_fn=lambda recordings, query, artist: None,
    )

    assert identifier._try_split_search("Artist Title") is None
