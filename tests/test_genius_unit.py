import pytest
from bs4 import BeautifulSoup

from y2karaoke.core import genius


def test_extract_genius_title_artist():
    html = "<html><head><title>Artist - Song Lyrics | Genius Lyrics</title></head></html>"
    title, artist = genius._extract_genius_title_artist(
        BeautifulSoup(html, "html.parser"), "Fallback"
    )
    assert artist == "Artist"
    assert "Song" in title


def test_extract_genius_title_artist_handles_en_dash():
    html = "<html><head><title>Artist – Song | Genius Lyrics</title></head></html>"
    title, artist = genius._extract_genius_title_artist(
        BeautifulSoup(html, "html.parser"), "Fallback"
    )
    assert artist == "Artist"
    assert title == "Song"


def test_parse_genius_html_extracts_lines_and_singers():
    html = """
    <div data-lyrics-container="true">
      [Verse 1: Alice]<br/>
      Hello there<br/>
      [Chorus: Bob]<br/>
      Sing along<br/>
    </div>
    """
    lines, metadata = genius.parse_genius_html(html, "Test Artist")
    assert lines is not None
    assert metadata is not None
    assert metadata.is_duet is True
    assert len(metadata.singers) >= 2
    assert lines[0][0] == "Hello there"
    assert lines[0][1] == "Alice"


def test_parse_genius_html_handles_empty():
    lines, metadata = genius.parse_genius_html("<div></div>", "Artist")
    assert lines is None
    assert metadata is None


def test_fetch_genius_lyrics_with_singers_candidate_url(monkeypatch):
    html = """
    <div data-lyrics-container="true">
      Line one<br/>
      Line two<br/>
    </div>
    """

    def fake_fetch_html(url, headers=None, timeout=5):
        assert "genius.com" in url
        return html

    monkeypatch.setattr(genius, "fetch_html", fake_fetch_html)
    monkeypatch.setattr(genius, "fetch_json", lambda *a, **k: None)

    lines, metadata = genius.fetch_genius_lyrics_with_singers("Song", "Artist")
    assert lines is not None
    assert metadata is not None
    assert len(lines) == 2


def test_fetch_genius_lyrics_with_singers_search_fallback(monkeypatch):
    html = """
    <div data-lyrics-container="true">
      Line one<br/>
    </div>
    """

    def fake_fetch_html(url, headers=None, timeout=5):
        if "api/search" in url:
            return None
        if "artist-song-lyrics" in url:
            return html
        return None

    def fake_fetch_json(url, headers=None, timeout=5):
        return {
            "response": {
                "sections": [
                    {
                        "type": "song",
                        "hits": [
                            {
                                "result": {
                                    "url": "https://genius.com/artist-song-lyrics"
                                }
                            }
                        ],
                    }
                ]
            }
        }

    monkeypatch.setattr(genius, "fetch_html", fake_fetch_html)
    monkeypatch.setattr(genius, "fetch_json", fake_fetch_json)

    lines, metadata = genius.fetch_genius_lyrics_with_singers("Song", "Artist")
    assert lines is not None
    assert metadata is not None
    assert len(lines) == 1


def test_fetch_genius_lyrics_with_singers_returns_none_when_no_url(monkeypatch):
    monkeypatch.setattr(genius, "fetch_html", lambda *a, **k: None)
    monkeypatch.setattr(genius, "fetch_json", lambda *a, **k: None)

    lines, metadata = genius.fetch_genius_lyrics_with_singers("Song", "Artist")
    assert lines is None
    assert metadata is None


def test_fetch_genius_lyrics_with_singers_search_skips_non_song(monkeypatch):
    monkeypatch.setattr(genius, "fetch_html", lambda *a, **k: None)

    def fake_fetch_json(*args, **kwargs):
        return {
            "response": {
                "sections": [
                    {"type": "album", "hits": []},
                    {
                        "type": "song",
                        "hits": [
                            {"result": {"url": "https://genius.com/artists/Artist"}},
                            {"result": {"url": "https://genius.com/artist-song-translation"}},
                            {"result": {"url": "https://genius.com/artist-song"}},
                        ],
                    },
                ]
            }
        }

    monkeypatch.setattr(genius, "fetch_json", fake_fetch_json)

    lines, metadata = genius.fetch_genius_lyrics_with_singers("Song", "Artist")
    assert lines is None
    assert metadata is None


def test_fetch_genius_lyrics_with_singers_handles_missing_html(monkeypatch):
    calls = {"count": 0}

    def fake_fetch_html(url, headers=None, timeout=5):
        calls["count"] += 1
        if calls["count"] == 1:
            return "<html></html>"
        return None

    monkeypatch.setattr(genius, "fetch_html", fake_fetch_html)
    monkeypatch.setattr(genius, "fetch_json", lambda *a, **k: None)

    lines, metadata = genius.fetch_genius_lyrics_with_singers("Song", "Artist")
    assert lines is None
    assert metadata is None
