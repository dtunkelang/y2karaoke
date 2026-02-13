import types  # noqa: F401

import pytest  # noqa: F401

from y2karaoke.core.components.identify import title_resolver


def test_normalize_string_removes_stop_words_and_punct():
    assert (
        title_resolver.normalize_string("The Artist & The Song (Live)")
        == "artist song live"
    )


def test_strip_parentheses_removes_bracketed_text():
    assert (
        title_resolver._strip_parentheses("Song Title (Remastered) [Live]")
        == "Song Title"
    )


def test_guess_musicbrainz_candidates_handles_errors(monkeypatch):
    def raise_error(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(title_resolver.musicbrainzngs, "search_recordings", raise_error)

    candidates = title_resolver.guess_musicbrainz_candidates(
        "Artist - Song", fallback_artist="Fallback Artist", fallback_title="Fallback"
    )
    assert candidates[-1]["artist"] == "Fallback Artist"
    assert candidates[-1]["title"] == "Fallback"


def test_guess_musicbrainz_candidates_filters_bad_results(monkeypatch):
    def fake_search_recordings(*args, **kwargs):
        return {
            "recording-list": [
                "not-a-dict",
                {"title": "", "artist-credit": []},
                {
                    "title": "Test Song (Live)",
                    "artist-credit": [{"artist": {"name": "Test Artist"}}],
                },
            ]
        }

    monkeypatch.setattr(
        title_resolver.musicbrainzngs, "search_recordings", fake_search_recordings
    )

    candidates = title_resolver.guess_musicbrainz_candidates("Test Artist - Test Song")
    assert len(candidates) == 1
    assert candidates[0]["artist"] == "Test Artist"
    assert candidates[0]["title"] == "Test Song"


def test_resolve_artist_title_from_youtube_fallback(monkeypatch):
    monkeypatch.setattr(
        title_resolver, "guess_musicbrainz_candidates", lambda *a, **k: []
    )

    artist, title = title_resolver.resolve_artist_title_from_youtube(
        "Unknown - Unknown",
        fallback_artist="Fallback Artist",
        fallback_title="Fallback Title",
    )

    assert artist == "Fallback Artist"
    assert title == "Fallback Title"


def test_resolve_artist_title_from_youtube_prefers_artist_match(monkeypatch):
    candidates = [
        {"artist": "Other Artist", "title": "Song", "score": 0.0, "extra_words": set()},
        {"artist": "Match Artist", "title": "Song", "score": 0.0, "extra_words": set()},
    ]
    monkeypatch.setattr(
        title_resolver, "guess_musicbrainz_candidates", lambda *a, **k: candidates
    )

    artist, title = title_resolver.resolve_artist_title_from_youtube(
        "Match Artist - Song (Official Video)",
        youtube_artist="Match Artist",
        fallback_artist="Fallback Artist",
        fallback_title="Fallback Title",
    )

    assert artist == "Match Artist"
    assert title == "Song"
