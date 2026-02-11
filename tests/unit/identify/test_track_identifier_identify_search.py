"""Tests for TrackIdentifier identify_from_search orchestration."""

import logging
from unittest.mock import patch

import pytest

from y2karaoke.core.components.identify.implementation import TrackIdentifier, TrackInfo
from y2karaoke.exceptions import Y2KaraokeError


class TestIdentifyFromSearchMocked:
    """Tests for identify_from_search with mocked external services."""

    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._try_direct_lrc_search"
    )
    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._query_musicbrainz"
    )
    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._find_best_with_artist_hint"
    )
    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._search_youtube_by_duration"
    )
    @patch("y2karaoke.core.components.lyrics.sync.fetch_lyrics_for_duration")
    def test_successful_identification_with_artist(
        self,
        mock_fetch_lyrics,
        mock_yt_search,
        mock_find_best,
        mock_mb_query,
        mock_direct_lrc,
    ):
        """Successfully identifies track with artist hint."""
        mock_direct_lrc.return_value = None
        mock_mb_query.return_value = [
            {
                "title": "Bohemian Rhapsody",
                "length": 354000,
                "artist-credit": [{"artist": {"name": "Queen"}}],
            }
        ]
        mock_find_best.return_value = (354, "Queen", "Bohemian Rhapsody")
        mock_fetch_lyrics.return_value = ("lrc text", True, "lyriq", 354)
        mock_yt_search.return_value = {
            "url": "https://youtube.com/watch?v=fJ9rUzIMcZQ",
            "duration": 354,
        }

        identifier = TrackIdentifier()
        result = identifier.identify_from_search("Queen - Bohemian Rhapsody")

        assert result.artist == "Queen"
        assert result.title == "Bohemian Rhapsody"
        assert result.duration == 354
        assert result.source == "musicbrainz"

    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._try_direct_lrc_search"
    )
    def test_direct_lrc_success(self, mock_direct_lrc):
        """Uses direct LRC search when successful."""
        mock_direct_lrc.return_value = TrackInfo(
            artist="Queen",
            title="Bohemian Rhapsody",
            duration=354,
            youtube_url="https://youtube.com/watch?v=fJ9rUzIMcZQ",
            youtube_duration=354,
            source="syncedlyrics",
            lrc_duration=354,
            lrc_validated=True,
        )

        identifier = TrackIdentifier()
        result = identifier.identify_from_search("bohemian rhapsody queen")

        assert result.artist == "Queen"
        assert result.title == "Bohemian Rhapsody"
        assert result.source == "syncedlyrics"

    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._try_direct_lrc_search"
    )
    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._query_musicbrainz"
    )
    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._fallback_youtube_search"
    )
    def test_fallback_to_youtube(self, mock_fallback, mock_mb_query, mock_direct_lrc):
        """Falls back to YouTube when MusicBrainz returns no results."""
        mock_direct_lrc.return_value = None
        mock_mb_query.return_value = []
        mock_fallback.return_value = TrackInfo(
            artist="Unknown",
            title="Some Song",
            duration=200,
            youtube_url="https://youtube.com/watch?v=abc123",
            youtube_duration=200,
            source="youtube",
        )

        identifier = TrackIdentifier()
        result = identifier.identify_from_search("some obscure song")

        assert result.source == "youtube"
        mock_fallback.assert_called_once()

    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._try_direct_lrc_search"
    )
    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._query_musicbrainz"
    )
    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._parse_query"
    )
    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._find_best_title_only"
    )
    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._try_split_search"
    )
    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._fallback_youtube_search"
    )
    def test_fallback_to_youtube_when_no_candidate(
        self,
        mock_fallback,
        mock_try_split,
        mock_find_title,
        mock_parse_query,
        mock_mb_query,
        mock_direct_lrc,
    ):
        """Falls back to YouTube when no suitable candidate is found."""
        mock_direct_lrc.return_value = None
        mock_parse_query.return_value = (None, "Some Title")
        mock_mb_query.return_value = [{"dummy": True}]
        mock_find_title.return_value = None
        mock_try_split.return_value = None
        mock_fallback.return_value = TrackInfo(
            artist="Unknown",
            title="Some Song",
            duration=200,
            youtube_url="https://youtube.com/watch?v=abc123",
            youtube_duration=200,
            source="youtube",
        )

        identifier = TrackIdentifier()
        result = identifier.identify_from_search("some obscure song")

        assert result.source == "youtube"
        mock_fallback.assert_called_once()

    @patch("y2karaoke.core.components.lyrics.sync.fetch_lyrics_for_duration")
    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._search_youtube_by_duration"
    )
    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._try_split_search"
    )
    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._find_best_title_only"
    )
    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._query_musicbrainz"
    )
    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._parse_query"
    )
    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._try_direct_lrc_search"
    )
    def test_title_only_split_search_and_youtube_fallback(
        self,
        mock_direct_lrc,
        mock_parse,
        mock_mb_query,
        mock_find_title,
        mock_try_split,
        mock_search_youtube,
        mock_fetch_lyrics,
        caplog,
    ):
        """Falls back to split search and query-based YouTube lookup."""
        mock_direct_lrc.return_value = None
        mock_parse.return_value = (None, "Some Title")
        mock_mb_query.return_value = [{"dummy": True}]
        mock_find_title.return_value = None
        mock_try_split.return_value = (200, "Artist", "Some Title")
        mock_fetch_lyrics.return_value = (None, None, None, 250)
        mock_search_youtube.side_effect = [
            None,
            {"url": "https://youtube.com/watch?v=abc123", "duration": 200},
        ]

        identifier = TrackIdentifier()
        with caplog.at_level(logging.WARNING):
            result = identifier.identify_from_search("Some Title")

        assert result.source == "musicbrainz"
        assert result.lrc_validated is False
        assert "doesn't match canonical" in caplog.text.lower()
        assert mock_search_youtube.call_count == 2

    @patch("y2karaoke.core.components.lyrics.sync.fetch_lyrics_for_duration")
    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._search_youtube_by_duration"
    )
    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._find_best_title_only"
    )
    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._query_musicbrainz"
    )
    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._parse_query"
    )
    @patch(
        "y2karaoke.core.components.identify.implementation.TrackIdentifier._try_direct_lrc_search"
    )
    def test_raises_when_no_youtube_results(
        self,
        mock_direct_lrc,
        mock_parse,
        mock_mb_query,
        mock_find_title,
        mock_search_youtube,
        mock_fetch_lyrics,
    ):
        """Raises when no YouTube results can be found."""
        mock_direct_lrc.return_value = None
        mock_parse.return_value = (None, "Some Title")
        mock_mb_query.return_value = [{"dummy": True}]
        mock_find_title.return_value = (200, "Artist", "Some Title")
        mock_fetch_lyrics.return_value = (None, None, None, None)
        mock_search_youtube.return_value = None

        identifier = TrackIdentifier()

        with pytest.raises(Y2KaraokeError):
            identifier.identify_from_search("Some Title")
