"""Tests for title resolution from YouTube to canonical artist/title."""

import pytest
from unittest.mock import patch, MagicMock

from y2karaoke.core.title_resolver import (
    normalize_string,
    _strip_parentheses,
    guess_musicbrainz_candidates,
    resolve_artist_title_from_youtube,
    STOP_WORDS,
)


class TestNormalizeString:
    def test_lowercase(self):
        assert "hello" in normalize_string("HELLO")

    def test_removes_special_chars(self):
        result = normalize_string("hello!@#world")
        assert "hello" in result
        assert "world" in result
        assert "@" not in result

    def test_removes_stop_words(self):
        result = normalize_string("The Quick and Brown Fox")
        assert "the" not in result.split()
        assert "and" not in result.split()
        assert "quick" in result
        assert "brown" in result
        assert "fox" in result

    def test_preserves_spaces_between_words(self):
        result = normalize_string("Hello World")
        assert " " in result

    def test_empty_string(self):
        assert normalize_string("") == ""

    def test_only_stop_words(self):
        result = normalize_string("the a an")
        assert result == ""

    def test_unicode_normalization(self):
        # Casefold handles unicode
        result = normalize_string("HELLO")
        assert result == "hello"


class TestStripParentheses:
    def test_removes_parentheses(self):
        assert _strip_parentheses("Song (Remix)") == "Song"

    def test_removes_brackets(self):
        assert _strip_parentheses("Song [Official]") == "Song"

    def test_removes_multiple(self):
        result = _strip_parentheses("Song (Remix) [HD]")
        assert result == "Song"

    def test_preserves_non_parenthetical(self):
        assert _strip_parentheses("Just A Song") == "Just A Song"

    def test_empty_after_strip(self):
        assert _strip_parentheses("(Everything)") == ""

    def test_nested_content(self):
        result = _strip_parentheses("Song (feat. Artist)")
        assert result == "Song"


class TestStopWords:
    def test_common_stop_words_present(self):
        assert "the" in STOP_WORDS
        assert "a" in STOP_WORDS
        assert "an" in STOP_WORDS
        assert "and" in STOP_WORDS
        assert "&" in STOP_WORDS


class TestGuessMusicbrainzCandidates:
    @patch("y2karaoke.core.title_resolver.search_recordings")
    def test_returns_candidates(self, mock_search):
        mock_search.return_value = {
            "recording-list": [
                {
                    "title": "My Song",
                    "artist-credit": [{"artist": {"name": "Artist Name"}}],
                }
            ]
        }

        candidates = guess_musicbrainz_candidates("My Song Artist Name")

        assert len(candidates) >= 1
        assert any(c["title"] == "My Song" for c in candidates)
        assert any(c["artist"] == "Artist Name" for c in candidates)

    @patch("y2karaoke.core.title_resolver.search_recordings")
    def test_includes_fallback(self, mock_search):
        mock_search.return_value = {"recording-list": []}

        candidates = guess_musicbrainz_candidates(
            "query", fallback_artist="Fallback Artist", fallback_title="Fallback Title"
        )

        assert any(c["artist"] == "Fallback Artist" for c in candidates)
        assert any(c["title"] == "Fallback Title" for c in candidates)

    @patch("y2karaoke.core.title_resolver.search_recordings")
    def test_handles_api_error(self, mock_search):
        mock_search.side_effect = Exception("API Error")

        candidates = guess_musicbrainz_candidates(
            "query", fallback_artist="Fallback", fallback_title="Song"
        )

        # Should still return fallback
        assert len(candidates) >= 1
        assert candidates[-1]["artist"] == "Fallback"

    @patch("y2karaoke.core.title_resolver.search_recordings")
    def test_skips_non_dict_results(self, mock_search):
        mock_search.return_value = {
            "recording-list": [
                "not a dict",  # Invalid
                {
                    "title": "Valid Song",
                    "artist-credit": [{"artist": {"name": "Valid Artist"}}],
                },
            ]
        }

        candidates = guess_musicbrainz_candidates("query")

        # Should only have one valid candidate (plus fallback if any)
        valid_candidates = [c for c in candidates if c.get("title") == "Valid Song"]
        assert len(valid_candidates) == 1

    @patch("y2karaoke.core.title_resolver.search_recordings")
    def test_strips_parentheses_from_title(self, mock_search):
        mock_search.return_value = {
            "recording-list": [
                {
                    "title": "Song (Remastered 2020)",
                    "artist-credit": [{"artist": {"name": "Artist"}}],
                }
            ]
        }

        candidates = guess_musicbrainz_candidates("Song Artist")

        # Parenthetical should be stripped
        assert any(c["title"] == "Song" for c in candidates)

    @patch("y2karaoke.core.title_resolver.search_recordings")
    def test_multiple_artists_joined(self, mock_search):
        mock_search.return_value = {
            "recording-list": [
                {
                    "title": "Duet Song",
                    "artist-credit": [
                        {"artist": {"name": "Artist A"}},
                        {"artist": {"name": "Artist B"}},
                    ],
                }
            ]
        }

        candidates = guess_musicbrainz_candidates("Duet Song")

        # Artists should be joined with &
        assert any("Artist A & Artist B" in c["artist"] for c in candidates)


class TestResolveArtistTitleFromYoutube:
    @patch("y2karaoke.core.title_resolver.guess_musicbrainz_candidates")
    def test_returns_best_candidate(self, mock_candidates):
        mock_candidates.return_value = [
            {"artist": "Good Artist", "title": "Good Title", "score": 10.0},
            {"artist": "Bad Artist", "title": "Bad Title", "score": 1.0},
        ]

        artist, title = resolve_artist_title_from_youtube("Good Artist - Good Title")

        # Should prefer higher scored candidate
        assert artist == "Good Artist"
        assert title == "Good Title"

    @patch("y2karaoke.core.title_resolver.guess_musicbrainz_candidates")
    def test_cleans_youtube_title(self, mock_candidates):
        mock_candidates.return_value = [
            {"artist": "Artist", "title": "Song", "score": 5.0}
        ]

        resolve_artist_title_from_youtube(
            "Artist - Song (Official Music Video) [HD]"
        )

        # Check that cleaned title was passed to candidates
        call_args = mock_candidates.call_args[0][0]
        assert "Official Music Video" not in call_args
        assert "HD" not in call_args

    @patch("y2karaoke.core.title_resolver.guess_musicbrainz_candidates")
    def test_returns_fallback_when_no_candidates(self, mock_candidates):
        mock_candidates.return_value = []

        artist, title = resolve_artist_title_from_youtube(
            "Unknown Video",
            fallback_artist="Fallback Artist",
            fallback_title="Fallback Title",
        )

        assert artist == "Fallback Artist"
        assert title == "Fallback Title"

    @patch("y2karaoke.core.title_resolver.guess_musicbrainz_candidates")
    def test_returns_unknown_when_no_fallback(self, mock_candidates):
        mock_candidates.return_value = []

        artist, title = resolve_artist_title_from_youtube("Unknown Video")

        assert artist == "Unknown"
        assert title == "Unknown"

    @patch("y2karaoke.core.title_resolver.guess_musicbrainz_candidates")
    def test_boosts_matching_youtube_artist(self, mock_candidates):
        mock_candidates.return_value = [
            {"artist": "Channel Name", "title": "Song A", "score": 1.0},
            {"artist": "Other Artist", "title": "Song B", "score": 1.0},
        ]

        artist, title = resolve_artist_title_from_youtube(
            "Song Title",
            youtube_artist="Channel Name",
        )

        # Should prefer candidate matching YouTube uploader
        assert artist == "Channel Name"

    @patch("y2karaoke.core.title_resolver.guess_musicbrainz_candidates")
    def test_strips_parentheses_from_result(self, mock_candidates):
        mock_candidates.return_value = [
            {"artist": "Artist", "title": "Song (Live Version)", "score": 5.0}
        ]

        artist, title = resolve_artist_title_from_youtube("Artist Song")

        assert title == "Song"  # Parenthetical stripped

    @patch("y2karaoke.core.title_resolver.guess_musicbrainz_candidates")
    def test_handles_malformed_candidate(self, mock_candidates):
        mock_candidates.return_value = [
            "not a dict",  # Malformed
            {"wrong_key": "value"},  # Missing required keys
            {"artist": "Valid", "title": "Valid Song", "score": 5.0},
        ]

        artist, title = resolve_artist_title_from_youtube("Valid Song")

        # Should skip malformed and use valid candidate
        assert artist == "Valid"
        assert title == "Valid Song"
