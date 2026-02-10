import y2karaoke.core.components.identify.implementation as ti


def test_is_likely_cover_recording_detects_cover():
    identifier = ti.TrackIdentifier()
    query_words = {"bohemian", "rhapsody", "queen"}

    assert identifier._is_likely_cover_recording(
        "Bohemian Rhapsody (Queen)", "Doctor Octoroc", query_words
    )


def test_select_best_from_artist_matches_prefers_mb_consensus():
    identifier = ti.TrackIdentifier()
    artist_matches = [
        {"duration": 250, "artist": "Artist", "title": "Song"},
        {"duration": 310, "artist": "Artist", "title": "Song"},
    ]

    best = identifier._select_best_from_artist_matches(
        artist_matches, lrc_duration=200, mb_consensus=300
    )

    assert best == (310, "Artist", "Song")


def test_find_best_title_only_prefers_lrc_available():
    recordings = [
        {
            "title": "My Song",
            "length": "200000",
            "artist-credit": [{"artist": {"name": "Artist A"}}],
        },
        {
            "title": "My Song",
            "length": "205000",
            "artist-credit": [{"artist": {"name": "Artist B"}}],
        },
    ]

    def fake_check(title, artist):
        return (artist == "Artist B"), 205

    identifier = ti.TrackIdentifier(check_lrc_and_duration_fn=fake_check)

    result = identifier._find_best_title_only(recordings, "My Song")

    assert result == (205, "Artist B", "My Song")


def test_find_best_title_only_returns_none_for_no_match():
    identifier = ti.TrackIdentifier()

    recordings = [
        {
            "title": "Other Song",
            "length": "200000",
            "artist-credit": [{"artist": {"name": "Artist A"}}],
        }
    ]

    assert identifier._find_best_title_only(recordings, "My Song") is None


def test_find_best_title_only_fallbacks_to_consensus_artist():
    identifier = ti.TrackIdentifier(
        check_lrc_and_duration_fn=lambda *a, **k: (False, None)
    )

    recordings = [
        {
            "title": "My Song",
            "length": "200000",
            "artist-credit": [{"artist": {"name": "Artist A"}}],
        },
        {
            "title": "My Song",
            "length": "201000",
            "artist-credit": [{"artist": {"name": "Artist A"}}],
        },
        {
            "title": "My Song",
            "length": "205000",
            "artist-credit": [{"artist": {"name": "Artist B"}}],
        },
    ]

    result = identifier._find_best_title_only(recordings, "My Song")

    assert result == (200, "Artist A", "My Song")
