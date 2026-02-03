import y2karaoke.core.track_identifier as ti


def test_build_url_candidates_dedup_and_order():
    identifier = ti.TrackIdentifier()
    recordings = [
        {
            "title": "Song",
            "artist-credit": [{"artist": {"name": "Artist"}}],
        },
        {
            "title": "Song",
            "artist-credit": [{"artist": {"name": "Artist"}}],
        },
        {
            "title": "Other",
            "artist-credit": [{"artist": {"name": "Other Artist"}}],
        },
    ]

    candidates = identifier._build_url_candidates(
        yt_uploader="Uploader",
        parsed_artist="Artist",
        parsed_title="Song",
        recordings=recordings,
    )

    assert candidates[0] == {"artist": "Uploader", "title": "Song"}
    assert candidates[1] == {"artist": "Artist", "title": "Song"}
    assert {"artist": "Other Artist", "title": "Other"} in candidates
    assert len(candidates) == 3


def test_build_url_candidates_skips_duplicate_uploader_artist():
    identifier = ti.TrackIdentifier()
    candidates = identifier._build_url_candidates(
        yt_uploader="Artist",
        parsed_artist="Artist",
        parsed_title="Song",
        recordings=[],
    )

    assert candidates == [{"artist": "Artist", "title": "Song"}]


def test_find_fallback_artist_title_prefers_non_uploader_candidate():
    identifier = ti.TrackIdentifier()
    unique_candidates = [
        {"artist": "Uploader", "title": "Song"},
        {"artist": "Candidate", "title": "Alt"},
    ]

    artist, title = identifier._find_fallback_artist_title(
        unique_candidates=unique_candidates,
        yt_uploader="Uploader",
        parsed_artist=None,
        parsed_title="Song",
        yt_title="YT Song",
    )

    assert artist == "Candidate"
    assert title == "Alt"


def test_find_fallback_artist_title_uses_split_with_lrc(monkeypatch):
    identifier = ti.TrackIdentifier()

    def fake_try_splits(_):
        return [("Split Artist", "Split Title"), ("Alt", "Alt Title")]

    def fake_check_lrc(title, artist):
        return True, 200

    monkeypatch.setattr(identifier, "_try_artist_title_splits", fake_try_splits)
    monkeypatch.setattr(identifier, "_check_lrc_and_duration", fake_check_lrc)

    artist, title = identifier._find_fallback_artist_title(
        unique_candidates=[],
        yt_uploader="",
        parsed_artist=None,
        parsed_title="Split Artist - Split Title",
        yt_title="YT Title",
    )

    assert artist == "Split Artist"
    assert title == "Split Title"


def test_find_fallback_artist_title_uses_uploader_or_unknown():
    identifier = ti.TrackIdentifier()

    artist, title = identifier._find_fallback_artist_title(
        unique_candidates=[],
        yt_uploader="Uploader",
        parsed_artist=None,
        parsed_title="",
        yt_title="YT Title",
    )

    assert artist == "Uploader"
    assert title == "YT Title"

    artist, title = identifier._find_fallback_artist_title(
        unique_candidates=[],
        yt_uploader="",
        parsed_artist=None,
        parsed_title="",
        yt_title="YT Title",
    )

    assert artist == "Unknown"
    assert title == "YT Title"
