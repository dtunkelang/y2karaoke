import y2karaoke.core.components.identify.implementation as ti


def test_try_parse_artist_from_title_exact_match(monkeypatch):
    identifier = ti.TrackIdentifier()

    def fake_search_artists(artist, limit):
        if artist == "The White Stripes":
            return {"artist-list": [{"name": "The White Stripes"}]}
        return {"artist-list": []}

    monkeypatch.setattr(ti.musicbrainzngs, "search_artists", fake_search_artists)

    result = identifier._try_parse_artist_from_title(
        "The White Stripes fell in love with a girl"
    )

    assert result == ("The White Stripes", "fell in love with a girl")


def test_try_parse_artist_from_title_substring_match(monkeypatch):
    identifier = ti.TrackIdentifier()

    monkeypatch.setattr(
        ti.musicbrainzngs,
        "search_artists",
        lambda *a, **k: {"artist-list": [{"name": "The White Stripes"}]},
    )

    result = identifier._try_parse_artist_from_title(
        "White Stripes fell in love with a girl"
    )

    assert result == ("The White Stripes", "fell in love with a girl")
