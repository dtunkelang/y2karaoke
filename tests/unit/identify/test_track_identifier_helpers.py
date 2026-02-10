import y2karaoke.core.components.lyrics.sync as sync
import y2karaoke.core.components.identify.implementation as ti


def test_parse_query_separators_and_by():
    identifier = ti.TrackIdentifier()
    assert identifier._parse_query("Artist - Title") == ("Artist", "Title")
    assert identifier._parse_query("Artist – Title") == ("Artist", "Title")
    assert identifier._parse_query("Artist — Title") == ("Artist", "Title")
    assert identifier._parse_query("Artist: Title") == ("Artist", "Title")
    assert identifier._parse_query("Title by Artist") == ("Artist", "Title")
    assert identifier._parse_query("JustTitle") == (None, "JustTitle")


def test_try_artist_title_splits_variants():
    identifier = ti.TrackIdentifier()
    splits = identifier._try_artist_title_splits("beatles yesterday")
    assert splits[0] == ("beatles", "yesterday")
    assert ("yesterday", "beatles") in splits
    assert identifier._try_artist_title_splits("single") == []


def test_parse_youtube_title_removes_suffix_and_metadata(monkeypatch):
    identifier = ti.TrackIdentifier()
    monkeypatch.setattr(
        ti.TrackIdentifier, "_try_parse_artist_from_title", lambda *_: None
    )

    artist, title = identifier._parse_youtube_title(
        "Artist - Song (Live) (Official Video)"
    )
    assert artist == "Artist"
    assert title == "Song"

    artist, title = identifier._parse_youtube_title(
        "Artist - (Don't Fear) The Reaper (Official Video)"
    )
    assert artist == "Artist"
    assert title == "(Don't Fear) The Reaper"


def test_check_lrc_and_duration_sync_unavailable(monkeypatch):
    identifier = ti.TrackIdentifier()
    monkeypatch.setattr(sync, "SYNCEDLYRICS_AVAILABLE", False)

    called = {"fetch": 0}

    def fake_fetch(*args, **kwargs):
        called["fetch"] += 1
        return ("", False, None)

    monkeypatch.setattr(sync, "fetch_lyrics_multi_source", fake_fetch)

    assert identifier._check_lrc_and_duration("Title", "Artist") == (False, None)
    assert called["fetch"] == 0


def test_check_lrc_and_duration_not_synced(monkeypatch):
    identifier = ti.TrackIdentifier()
    monkeypatch.setattr(sync, "SYNCEDLYRICS_AVAILABLE", True)
    monkeypatch.setattr(
        sync, "fetch_lyrics_multi_source", lambda *a, **k: ("", False, "source")
    )

    def bad_validate(*args, **kwargs):
        raise AssertionError("validate_lrc_quality should not be called")

    monkeypatch.setattr(sync, "validate_lrc_quality", bad_validate)
    assert identifier._check_lrc_and_duration("Title", "Artist") == (False, None)


def test_check_lrc_and_duration_invalid_quality(monkeypatch):
    identifier = ti.TrackIdentifier()
    monkeypatch.setattr(sync, "SYNCEDLYRICS_AVAILABLE", True)
    monkeypatch.setattr(
        sync,
        "fetch_lyrics_multi_source",
        lambda *a, **k: ("[00:01.00]Hi", True, "source"),
    )
    monkeypatch.setattr(sync, "validate_lrc_quality", lambda *a, **k: (False, "bad"))
    monkeypatch.setattr(sync, "get_lrc_duration", lambda *_: 123)

    assert identifier._check_lrc_and_duration("Title", "Artist") == (False, None)


def test_check_lrc_and_duration_valid(monkeypatch):
    identifier = ti.TrackIdentifier()
    monkeypatch.setattr(sync, "SYNCEDLYRICS_AVAILABLE", True)
    monkeypatch.setattr(
        sync,
        "fetch_lyrics_multi_source",
        lambda *a, **k: ("[00:01.00]Hi", True, "source"),
    )
    monkeypatch.setattr(sync, "validate_lrc_quality", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(sync, "get_lrc_duration", lambda *_: 123)

    assert identifier._check_lrc_and_duration("Title", "Artist") == (True, 123)


def test_check_lrc_and_duration_handles_exception(monkeypatch):
    identifier = ti.TrackIdentifier()
    monkeypatch.setattr(sync, "SYNCEDLYRICS_AVAILABLE", True)

    def raise_error(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(sync, "fetch_lyrics_multi_source", raise_error)

    assert identifier._check_lrc_and_duration("Title", "Artist") == (False, None)
