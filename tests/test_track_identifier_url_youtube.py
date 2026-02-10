"""Tests for track identification URL/youtube and integration flows."""

import logging
import pytest
from unittest.mock import Mock, patch

from y2karaoke.core.track_identifier import TrackIdentifier, TrackInfo
from y2karaoke.exceptions import Y2KaraokeError


class TestIdentifyFromUrlMocked:
    """Tests for identify_from_url with mocked external services."""

    @patch("y2karaoke.core.track_identifier.TrackIdentifier._get_youtube_metadata")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._query_musicbrainz")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._find_best_lrc_by_duration")
    def test_successful_identification(
        self, mock_find_lrc, mock_mb_query, mock_yt_metadata
    ):
        """Successfully identifies track from URL."""
        mock_yt_metadata.return_value = ("Queen - Bohemian Rhapsody", "Queen", 354)
        mock_mb_query.return_value = [
            {
                "title": "Bohemian Rhapsody",
                "length": 354000,
                "artist-credit": [{"artist": {"name": "Queen"}}],
            }
        ]
        mock_find_lrc.return_value = ("Queen", "Bohemian Rhapsody", 354)

        identifier = TrackIdentifier()
        result = identifier.identify_from_url("https://youtube.com/watch?v=fJ9rUzIMcZQ")

        assert result.artist == "Queen"
        assert result.title == "Bohemian Rhapsody"
        assert result.youtube_duration == 354

    @patch("y2karaoke.core.track_identifier.TrackIdentifier._get_youtube_metadata")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._query_musicbrainz")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._find_best_lrc_by_duration")
    def test_explicit_artist_title_override(
        self, mock_find_lrc, mock_mb_query, mock_yt_metadata
    ):
        """Uses explicit artist/title when provided."""
        mock_yt_metadata.return_value = ("Some Random Title", "RandomUser", 200)
        mock_mb_query.return_value = []
        mock_find_lrc.return_value = ("The Beatles", "Yesterday", 200)

        identifier = TrackIdentifier()
        result = identifier.identify_from_url(
            "https://youtube.com/watch?v=abc123",
            artist_hint="The Beatles",
            title_hint="Yesterday",
        )

        # Should use provided artist/title for search, not YouTube metadata
        mock_mb_query.assert_called_once_with(
            "The Beatles Yesterday", "The Beatles", "Yesterday"
        )

    @patch("y2karaoke.core.track_identifier.TrackIdentifier._get_youtube_metadata")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._query_musicbrainz")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._find_best_lrc_by_duration")
    def test_fallback_when_no_lrc(self, mock_find_lrc, mock_mb_query, mock_yt_metadata):
        """Falls back to parsed info when no LRC found."""
        mock_yt_metadata.return_value = ("Artist - Song Title", "ArtistChannel", 180)
        mock_mb_query.return_value = []
        mock_find_lrc.return_value = None

        identifier = TrackIdentifier()
        result = identifier.identify_from_url("https://youtube.com/watch?v=xyz789")

        assert result.source == "youtube"
        assert result.youtube_duration == 180

    @patch("y2karaoke.core.track_identifier.TrackIdentifier._get_youtube_metadata")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._query_musicbrainz")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._find_best_lrc_by_duration")
    @patch(
        "y2karaoke.core.track_identifier.TrackIdentifier._search_matching_youtube_video"
    )
    def test_non_studio_mismatch_uses_alternative_video(
        self, mock_alt, mock_find_lrc, mock_mb_query, mock_yt_metadata, caplog
    ):
        """Uses alternative YouTube video when non-studio duration mismatches."""
        mock_yt_metadata.return_value = ("Artist - Song (Live)", "Uploader", 200)
        mock_mb_query.return_value = []
        mock_find_lrc.return_value = ("Artist", "Song", 240)
        mock_alt.return_value = TrackInfo(
            artist="Artist",
            title="Song",
            duration=200,
            youtube_url="https://youtube.com/watch?v=alt123",
            youtube_duration=200,
            source="youtube",
            lrc_duration=240,
            lrc_validated=False,
        )

        identifier = TrackIdentifier()
        with caplog.at_level(logging.WARNING):
            result = identifier.identify_from_url("https://youtube.com/watch?v=xyz789")

        assert result.youtube_url == "https://youtube.com/watch?v=alt123"
        assert "non-studio" in caplog.text.lower()
        mock_alt.assert_called_once()

    @patch("y2karaoke.core.track_identifier.TrackIdentifier._get_youtube_metadata")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._query_musicbrainz")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._find_best_lrc_by_duration")
    @patch(
        "y2karaoke.core.track_identifier.TrackIdentifier._search_matching_youtube_video"
    )
    def test_non_studio_mismatch_returns_lrc_candidate_when_no_alt(
        self, mock_alt, mock_find_lrc, mock_mb_query, mock_yt_metadata, caplog
    ):
        """Returns LRC candidate when no alternative YouTube match is found."""
        mock_yt_metadata.return_value = ("Artist - Song (Live)", "Uploader", 200)
        mock_mb_query.return_value = []
        mock_find_lrc.return_value = ("Artist", "Song", 240)
        mock_alt.return_value = None

        identifier = TrackIdentifier()
        with caplog.at_level(logging.WARNING):
            result = identifier.identify_from_url("https://youtube.com/watch?v=xyz789")

        assert result.source == "syncedlyrics"
        assert result.lrc_validated is False
        assert "non-studio" in caplog.text.lower()


class TestCheckLrcAndDuration:
    """Tests for _check_lrc_and_duration caching behavior."""

    def test_caches_results(self):
        """Results are cached to avoid redundant lookups."""
        identifier = TrackIdentifier()

        with patch("y2karaoke.core.sync.fetch_lyrics_multi_source") as mock_fetch:
            mock_fetch.return_value = ("[00:00.00]Line\n[03:30.00]End", True, "lyriq")

            with patch("y2karaoke.core.sync.get_lrc_duration") as mock_duration:
                mock_duration.return_value = 210

                with patch("y2karaoke.core.sync.validate_lrc_quality") as mock_validate:
                    mock_validate.return_value = (True, None)

                    # First call
                    result1 = identifier._check_lrc_and_duration("Song", "Artist")
                    # Second call - should use cache
                    result2 = identifier._check_lrc_and_duration("Song", "Artist")

                    assert result1 == result2
                    # fetch_lyrics should only be called once
                    assert mock_fetch.call_count == 1

    def test_cache_key_is_lowercase(self):
        """Cache key is case-insensitive."""
        identifier = TrackIdentifier()

        with patch("y2karaoke.core.sync.fetch_lyrics_multi_source") as mock_fetch:
            mock_fetch.return_value = ("[00:00.00]Line", True, "lyriq")

            with patch("y2karaoke.core.sync.get_lrc_duration") as mock_duration:
                mock_duration.return_value = 210

                with patch("y2karaoke.core.sync.validate_lrc_quality") as mock_validate:
                    mock_validate.return_value = (True, None)

                    # Different cases should hit same cache entry
                    result1 = identifier._check_lrc_and_duration("SONG", "ARTIST")
                    result2 = identifier._check_lrc_and_duration("song", "artist")

                    assert result1 == result2
                    assert mock_fetch.call_count == 1


class TestFindBestWithArtistHint:
    """Tests for _find_best_with_artist_hint - matching recordings with artist hint."""

    def test_exact_artist_match(self):
        """Finds recording with exact artist match."""
        identifier = TrackIdentifier()
        recordings = [
            {
                "title": "Yesterday",
                "length": 125000,
                "artist-credit": [{"artist": {"name": "The Beatles"}}],
            },
            {
                "title": "Yesterday",
                "length": 180000,
                "artist-credit": [{"artist": {"name": "Some Cover Band"}}],
            },
        ]

        result = identifier._find_best_with_artist_hint(
            recordings, "beatles yesterday", "beatles"
        )

        assert result is not None
        assert result[1] == "The Beatles"  # artist
        assert result[2] == "Yesterday"  # title

    def test_partial_artist_match(self):
        """Matches partial artist name."""
        identifier = TrackIdentifier()
        recordings = [
            {
                "title": "Let It Be",
                "length": 243000,
                "artist-credit": [{"artist": {"name": "The Beatles"}}],
            }
        ]

        # "beatles" should match "The Beatles"
        result = identifier._find_best_with_artist_hint(
            recordings, "beatles let it be", "beatles"
        )

        assert result is not None
        assert result[1] == "The Beatles"

    def test_skips_long_recordings(self):
        """Skips recordings longer than 12 minutes."""
        identifier = TrackIdentifier()
        recordings = [
            {
                "title": "Song",
                "length": 800000,  # 800 seconds = 13+ minutes
                "artist-credit": [{"artist": {"name": "Artist"}}],
            }
        ]

        result = identifier._find_best_with_artist_hint(
            recordings, "artist song", "artist"
        )

        assert result is None


class TestFindBestLrcByDuration:
    """Tests for _find_best_lrc_by_duration - finding best LRC match."""

    def test_prefers_better_duration_match(self):
        """Prefers candidate with closer duration match."""
        identifier = TrackIdentifier()
        candidates = [
            {"artist": "Artist1", "title": "Song"},
            {"artist": "Artist2", "title": "Song"},
        ]

        with patch.object(identifier, "_check_lrc_and_duration") as mock_check:
            # First candidate: duration way off
            # Second candidate: duration close
            mock_check.side_effect = [
                (True, 300),  # 100s off from target
                (True, 205),  # 5s off from target
            ]

            result = identifier._find_best_lrc_by_duration(
                candidates, target_duration=200, title_hint="Song"
            )

            assert result is not None
            assert result[0] == "Artist2"
            assert result[1] == "Song"
            assert result[2] == 205

    def test_prefers_better_title_match(self):
        """Prefers candidate with better title similarity."""
        identifier = TrackIdentifier()
        candidates = [
            {"artist": "Artist", "title": "Different Song"},
            {"artist": "Artist", "title": "Expected Song Title"},
        ]

        with patch.object(identifier, "_check_lrc_and_duration") as mock_check:
            mock_check.return_value = (True, 200)  # Both have same duration

            result = identifier._find_best_lrc_by_duration(
                candidates, target_duration=200, title_hint="Expected Song Title"
            )

            assert result is not None
            assert result[1] == "Expected Song Title"

    def test_returns_none_when_no_lrc(self):
        """Returns None when no candidates have LRC."""
        identifier = TrackIdentifier()
        candidates = [
            {"artist": "Artist", "title": "Song"},
        ]

        with patch.object(identifier, "_check_lrc_and_duration") as mock_check:
            mock_check.return_value = (False, None)

            result = identifier._find_best_lrc_by_duration(
                candidates, target_duration=200, title_hint="Song"
            )

            assert result is None

    def test_falls_back_when_duration_unknown(self, caplog):
        """Falls back to candidate when LRC duration is unknown."""
        identifier = TrackIdentifier()
        candidates = [
            {"artist": "Artist", "title": "Song"},
        ]

        with patch.object(identifier, "_check_lrc_and_duration") as mock_check:
            mock_check.return_value = (True, None)

            with caplog.at_level(logging.WARNING):
                result = identifier._find_best_lrc_by_duration(
                    candidates, target_duration=200, title_hint="Song"
                )

        assert result == ("Artist", "Song", 200)
        assert "unknown duration" in caplog.text.lower()

    def test_warns_on_low_title_similarity(self, caplog):
        """Warns when best match has low title similarity."""
        identifier = TrackIdentifier()
        candidates = [
            {"artist": "Artist", "title": "Completely Different"},
        ]

        with patch.object(identifier, "_check_lrc_and_duration") as mock_check:
            mock_check.return_value = (True, 230)

            with caplog.at_level(logging.WARNING):
                result = identifier._find_best_lrc_by_duration(
                    candidates,
                    target_duration=200,
                    title_hint="Unrelated Title",
                    tolerance=40,
                )

        assert result == ("Artist", "Completely Different", 230)
        assert "low title similarity" in caplog.text.lower()

    def test_handles_empty_title_hint(self):
        """Handles empty title hint without word overlap."""
        identifier = TrackIdentifier()
        candidates = [
            {"artist": "Artist", "title": "Song"},
        ]

        with patch.object(identifier, "_check_lrc_and_duration") as mock_check:
            mock_check.return_value = (True, 200)

            result = identifier._find_best_lrc_by_duration(
                candidates, target_duration=200, title_hint=""
            )

        assert result == ("Artist", "Song", 200)

    def test_warns_on_large_duration_difference(self, caplog):
        """Warns when best duration difference exceeds tolerance."""
        identifier = TrackIdentifier()
        candidates = [
            {"artist": "Artist", "title": "Song"},
        ]

        with patch.object(identifier, "_check_lrc_and_duration") as mock_check:
            mock_check.return_value = (True, 260)

            with caplog.at_level(logging.WARNING):
                result = identifier._find_best_lrc_by_duration(
                    candidates, target_duration=200, title_hint="Song", tolerance=10
                )

        assert result == ("Artist", "Song", 260)
        assert "duration difference" in caplog.text.lower()


class TestExtractYoutubeCandidates:
    """Tests for _extract_youtube_candidates - parsing YouTube search results."""

    def test_extracts_video_id_and_title(self):
        """Extracts video ID and title from YouTube response."""
        identifier = TrackIdentifier()
        # Simplified mock response structure
        response = """
        "videoRenderer":{"videoId":"dQw4w9WgXcQ","title":{"runs":[{"text":"Rick Astley - Never Gonna Give You Up"}
        """

        candidates = identifier._extract_youtube_candidates(response)

        assert len(candidates) >= 1
        assert candidates[0]["video_id"] == "dQw4w9WgXcQ"
        assert "Never Gonna Give You Up" in candidates[0]["title"]

    def test_extracts_duration(self):
        """Extracts duration when available."""
        identifier = TrackIdentifier()
        # Mock response with duration
        response = """
        "videoRenderer":{"videoId":"dQw4w9WgXcQ","title":{"runs":[{"text":"Song Title"}],"simpleText":"3:32"
        """

        candidates = identifier._extract_youtube_candidates(response)

        assert len(candidates) == 1
        assert candidates[0]["duration"] == 212


class TestSearchYoutubeByDuration:
    """Tests for _search_youtube_by_duration - YouTube search with duration matching."""

    @patch("requests.get")
    def test_filters_non_studio_versions(self, mock_get):
        """Filters out non-studio versions from results."""
        identifier = TrackIdentifier()

        # Mock response with both studio and live versions
        mock_response = Mock()
        mock_response.text = """
        "videoRenderer":{"videoId":"abc123def45","title":{"runs":[{"text":"Song - Official Audio"}]},"simpleText":"3:30"}
        "videoRenderer":{"videoId":"def456ghi78","title":{"runs":[{"text":"Song Live at Concert"}]},"simpleText":"3:30"}
        """
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # The search should prefer the studio version
        # This tests the filtering logic
        result = identifier._search_youtube_single("Song", 210)
        assert result is not None
        assert result["url"].endswith("abc123def45")
        assert result["duration"] == 210


@pytest.mark.network
class TestIdentifyFromSearchIntegration:
    """Integration tests for identify_from_search (requires network)."""

    @pytest.mark.slow
    def test_well_known_song(self):
        """Integration test with a well-known song."""
        identifier = TrackIdentifier()

        # This is a real test - will make network requests
        # Skip if marked slow and running quick tests
        result = identifier.identify_from_search("Queen - Bohemian Rhapsody")

        assert result.artist is not None
        assert "Queen" in result.artist or "queen" in result.artist.lower()
        assert result.title is not None
        assert result.youtube_url is not None
        assert result.youtube_url.startswith("https://")


@pytest.mark.network
class TestIdentifyFromUrlIntegration:
    """Integration tests for identify_from_url (requires network)."""

    @pytest.mark.slow
    def test_official_video(self):
        """Integration test with official music video URL."""
        identifier = TrackIdentifier()

        # Queen - Bohemian Rhapsody official video
        result = identifier.identify_from_url(
            "https://www.youtube.com/watch?v=fJ9rUzIMcZQ"
        )

        assert result.artist is not None
        assert result.title is not None
        assert result.youtube_duration > 0
