"""Tests for track identification pipeline.

Tests cover both Path A (search string) and Path B (YouTube URL) identification,
as well as helper methods for parsing, normalization, and scoring.
"""

import logging
import pytest
from unittest.mock import Mock, patch, MagicMock

from y2karaoke.core.track_identifier import TrackIdentifier, TrackInfo
from y2karaoke.exceptions import Y2KaraokeError


class TestParseQuery:
    """Tests for _parse_query method - parsing search strings into artist/title."""

    def test_hyphen_separator(self):
        """Standard 'Artist - Title' format."""
        identifier = TrackIdentifier()
        artist, title = identifier._parse_query("Queen - Bohemian Rhapsody")
        assert artist == "Queen"
        assert title == "Bohemian Rhapsody"

    def test_en_dash_separator(self):
        """En-dash separator: 'Artist – Title'."""
        identifier = TrackIdentifier()
        artist, title = identifier._parse_query("Queen – Bohemian Rhapsody")
        assert artist == "Queen"
        assert title == "Bohemian Rhapsody"

    def test_em_dash_separator(self):
        """Em-dash separator: 'Artist — Title'."""
        identifier = TrackIdentifier()
        artist, title = identifier._parse_query("Queen — Bohemian Rhapsody")
        assert artist == "Queen"
        assert title == "Bohemian Rhapsody"

    def test_colon_separator(self):
        """Colon separator: 'Artist: Title'."""
        identifier = TrackIdentifier()
        artist, title = identifier._parse_query("Queen: Bohemian Rhapsody")
        assert artist == "Queen"
        assert title == "Bohemian Rhapsody"

    def test_by_format(self):
        """'Title by Artist' format."""
        identifier = TrackIdentifier()
        artist, title = identifier._parse_query("Bohemian Rhapsody by Queen")
        assert artist == "Queen"
        assert title == "Bohemian Rhapsody"

    def test_no_separator(self):
        """Query without any recognized separator."""
        identifier = TrackIdentifier()
        artist, title = identifier._parse_query("bohemian rhapsody queen")
        assert artist is None
        assert title == "bohemian rhapsody queen"

    def test_whitespace_handling(self):
        """Handles extra whitespace around separators."""
        identifier = TrackIdentifier()
        artist, title = identifier._parse_query("  Queen  -  Bohemian Rhapsody  ")
        assert artist == "Queen"
        assert title == "Bohemian Rhapsody"

    def test_multiple_separators(self):
        """Uses first separator when multiple are present."""
        identifier = TrackIdentifier()
        artist, title = identifier._parse_query("Artist - Title - Extra")
        assert artist == "Artist"
        assert title == "Title - Extra"


class TestTryArtistTitleSplits:
    """Tests for _try_artist_title_splits - generating possible splits for queries."""

    def test_two_words(self):
        """Two-word query generates one split."""
        identifier = TrackIdentifier()
        splits = identifier._try_artist_title_splits("beatles yesterday")
        assert ("beatles", "yesterday") in splits
        assert ("yesterday", "beatles") in splits

    def test_three_words(self):
        """Three-word query generates multiple splits."""
        identifier = TrackIdentifier()
        splits = identifier._try_artist_title_splits("queen bohemian rhapsody")
        expected_splits = [
            ("queen", "bohemian rhapsody"),
            ("queen bohemian", "rhapsody"),
            ("queen", "bohemian rhapsody"),
        ]
        # Check first word as artist is included
        assert ("queen", "bohemian rhapsody") in splits
        # Check last word as artist is included
        assert ("rhapsody", "queen bohemian") in splits

    def test_single_word(self):
        """Single word returns empty list."""
        identifier = TrackIdentifier()
        splits = identifier._try_artist_title_splits("yesterday")
        assert splits == []

    def test_empty_string(self):
        """Empty string returns empty list."""
        identifier = TrackIdentifier()
        splits = identifier._try_artist_title_splits("")
        assert splits == []


class TestNormalizeTitle:
    """Tests for _normalize_title - normalizing titles for comparison."""

    def test_lowercase(self):
        """Converts to lowercase."""
        identifier = TrackIdentifier()
        assert identifier._normalize_title("Bohemian Rhapsody") == "bohemian rhapsody"

    def test_removes_punctuation(self):
        """Removes common punctuation."""
        identifier = TrackIdentifier()
        assert identifier._normalize_title("Don't Stop Me Now!") == "don t stop me now"

    def test_normalizes_whitespace(self):
        """Normalizes multiple spaces to single space."""
        identifier = TrackIdentifier()
        assert identifier._normalize_title("Too   Many   Spaces") == "too many spaces"

    def test_stopword_removal(self):
        """Removes stopwords when requested."""
        identifier = TrackIdentifier()
        result = identifier._normalize_title(
            "The Sound of Silence", remove_stopwords=True
        )
        assert "the" not in result.split()
        assert "of" not in result.split()
        assert "sound" in result.split()
        assert "silence" in result.split()

    def test_keeps_stopwords_by_default(self):
        """Keeps stopwords when not explicitly removed."""
        identifier = TrackIdentifier()
        result = identifier._normalize_title("The Sound of Silence")
        assert "the" in result.split()
        assert "of" in result.split()

    def test_removes_parentheses(self):
        """Converts parentheses to spaces and normalizes."""
        identifier = TrackIdentifier()
        # Parentheses are replaced with spaces, then whitespace is normalized
        assert identifier._normalize_title("Title (Remastered)") == "title remastered"

    def test_multilingual_stopwords(self):
        """Removes stopwords from multiple languages."""
        identifier = TrackIdentifier()
        # Spanish
        result = identifier._normalize_title(
            "El Amor de Mi Vida", remove_stopwords=True
        )
        assert "el" not in result.split()
        assert "de" not in result.split()
        # French - uses "le", "la", "les", "de", etc.
        result = identifier._normalize_title(
            "Le Monde de la Vie", remove_stopwords=True
        )
        assert "le" not in result.split()
        assert "la" not in result.split()
        assert "de" not in result.split()


class TestIsLikelyNonStudio:
    """Tests for _is_likely_non_studio - detecting non-studio recordings."""

    def test_live_versions(self):
        """Detects live recordings."""
        identifier = TrackIdentifier()
        assert identifier._is_likely_non_studio("Bohemian Rhapsody (Live)")
        assert identifier._is_likely_non_studio("Song Title Live at Wembley")
        assert identifier._is_likely_non_studio("Song - Concert Recording")

    def test_acoustic_versions(self):
        """Detects acoustic versions."""
        identifier = TrackIdentifier()
        assert identifier._is_likely_non_studio("Song (Acoustic)")
        assert identifier._is_likely_non_studio("Unplugged Version")

    def test_remixes(self):
        """Detects remixes."""
        identifier = TrackIdentifier()
        assert identifier._is_likely_non_studio("Song (Club Remix)")
        assert identifier._is_likely_non_studio("Song - Extended Mix")

    def test_covers(self):
        """Detects cover versions."""
        identifier = TrackIdentifier()
        assert identifier._is_likely_non_studio("Song (Cover)")
        assert identifier._is_likely_non_studio("Song - Tribute")

    def test_karaoke(self):
        """Detects karaoke versions."""
        identifier = TrackIdentifier()
        assert identifier._is_likely_non_studio("Song (Karaoke)")
        assert identifier._is_likely_non_studio("Instrumental Version")

    def test_audio_effects(self):
        """Detects versions with audio effects."""
        identifier = TrackIdentifier()
        assert identifier._is_likely_non_studio("Song (Slowed + Reverb)")
        assert identifier._is_likely_non_studio("Song 8D Audio")
        assert identifier._is_likely_non_studio("Nightcore - Song")

    def test_tv_shows(self):
        """Detects TV show performances."""
        identifier = TrackIdentifier()
        assert identifier._is_likely_non_studio("Song - SNL Performance")
        assert identifier._is_likely_non_studio("Song Live on Letterman")
        assert identifier._is_likely_non_studio("Song - Tiny Desk Concert")

    def test_radio_edit_is_studio(self):
        """Radio edits are considered studio recordings."""
        identifier = TrackIdentifier()
        assert not identifier._is_likely_non_studio("Song (Radio Edit)")
        assert not identifier._is_likely_non_studio("Song - Single Version")
        assert not identifier._is_likely_non_studio("Song (Album Version)")

    def test_studio_versions(self):
        """Studio recordings are not flagged."""
        identifier = TrackIdentifier()
        assert not identifier._is_likely_non_studio("Bohemian Rhapsody")
        assert not identifier._is_likely_non_studio("Song - Official Audio")
        assert not identifier._is_likely_non_studio("Artist - Song Title")

    def test_festivals(self):
        """Detects festival performances."""
        identifier = TrackIdentifier()
        assert identifier._is_likely_non_studio("Song at Glastonbury")
        assert identifier._is_likely_non_studio("Coachella 2023 - Song")
        assert identifier._is_likely_non_studio("Song - Lollapalooza")


class TestExtractLrcMetadata:
    """Tests for _extract_lrc_metadata - extracting artist/title from LRC tags."""

    def test_extracts_artist_tag(self):
        """Extracts artist from [ar:...] tag."""
        identifier = TrackIdentifier()
        lrc = "[ar:Queen]\n[ti:Bohemian Rhapsody]\n[00:00.00]First line"
        artist, title = identifier._extract_lrc_metadata(lrc)
        assert artist == "Queen"
        assert title == "Bohemian Rhapsody"

    def test_extracts_title_tag(self):
        """Extracts title from [ti:...] tag."""
        identifier = TrackIdentifier()
        lrc = "[ti:Bohemian Rhapsody]\n[00:00.00]First line"
        artist, title = identifier._extract_lrc_metadata(lrc)
        assert artist is None
        assert title == "Bohemian Rhapsody"

    def test_no_metadata(self):
        """Handles LRC without metadata tags."""
        identifier = TrackIdentifier()
        lrc = "[00:00.00]First line\n[00:05.00]Second line"
        artist, title = identifier._extract_lrc_metadata(lrc)
        assert artist is None
        assert title is None

    def test_case_insensitive(self):
        """Tag matching is case-insensitive."""
        identifier = TrackIdentifier()
        lrc = "[AR:Queen]\n[TI:Bohemian Rhapsody]\n[00:00.00]Line"
        artist, title = identifier._extract_lrc_metadata(lrc)
        assert artist == "Queen"
        assert title == "Bohemian Rhapsody"

    def test_whitespace_in_values(self):
        """Handles whitespace in tag values."""
        identifier = TrackIdentifier()
        lrc = "[ar:  The Beatles  ]\n[ti:  Hey Jude  ]\n[00:00.00]Line"
        artist, title = identifier._extract_lrc_metadata(lrc)
        assert artist == "The Beatles"
        assert title == "Hey Jude"


class TestParseYoutubeTitle:
    """Tests for _parse_youtube_title - parsing video titles for artist/title."""

    def test_standard_format(self):
        """Standard 'Artist - Title' format in YouTube titles."""
        identifier = TrackIdentifier()
        artist, title = identifier._parse_youtube_title("Queen - Bohemian Rhapsody")
        assert artist == "Queen"
        assert title == "Bohemian Rhapsody"

    def test_removes_official_video(self):
        """Removes 'Official Video' suffix."""
        identifier = TrackIdentifier()
        artist, title = identifier._parse_youtube_title(
            "Queen - Bohemian Rhapsody (Official Video)"
        )
        assert artist == "Queen"
        assert title == "Bohemian Rhapsody"

    def test_removes_official_audio(self):
        """Removes 'Official Audio' suffix."""
        identifier = TrackIdentifier()
        artist, title = identifier._parse_youtube_title("Artist - Song Official Audio")
        assert title == "Song"

    def test_removes_hd_4k(self):
        """Removes HD/4K suffixes."""
        identifier = TrackIdentifier()
        artist, title = identifier._parse_youtube_title("Artist - Song HD")
        assert title == "Song"
        artist, title = identifier._parse_youtube_title("Artist - Song 4K")
        assert title == "Song"

    def test_removes_lyric_video(self):
        """Removes 'Lyric Video' suffix."""
        identifier = TrackIdentifier()
        artist, title = identifier._parse_youtube_title("Artist - Song (Lyric Video)")
        assert title == "Song"

    def test_pipe_separator(self):
        """Handles pipe separator."""
        identifier = TrackIdentifier()
        artist, title = identifier._parse_youtube_title("Artist | Song Title")
        assert artist == "Artist"
        assert title == "Song Title"

    def test_preserves_meaningful_parentheses(self):
        """Preserves parentheses that are part of the title."""
        identifier = TrackIdentifier()
        # "(Don't Fear) The Reaper" - parentheses are part of title
        artist, title = identifier._parse_youtube_title(
            "Blue Öyster Cult - (Don't Fear) The Reaper"
        )
        assert artist == "Blue Öyster Cult"
        assert title == "(Don't Fear) The Reaper"


class TestTrackInfo:
    """Tests for TrackInfo dataclass."""

    def test_basic_creation(self):
        """Creates TrackInfo with required fields."""
        info = TrackInfo(
            artist="Queen",
            title="Bohemian Rhapsody",
            duration=354,
            youtube_url="https://youtube.com/watch?v=fJ9rUzIMcZQ",
            youtube_duration=354,
            source="musicbrainz",
        )
        assert info.artist == "Queen"
        assert info.title == "Bohemian Rhapsody"
        assert info.duration == 354
        assert info.source == "musicbrainz"
        assert info.quality_issues == []
        assert info.sources_tried == []

    def test_optional_fields(self):
        """Creates TrackInfo with optional fields."""
        info = TrackInfo(
            artist="Queen",
            title="Bohemian Rhapsody",
            duration=354,
            youtube_url="https://youtube.com/watch?v=fJ9rUzIMcZQ",
            youtube_duration=354,
            source="syncedlyrics",
            lrc_duration=354,
            lrc_validated=True,
        )
        assert info.lrc_duration == 354
        assert info.lrc_validated is True

    def test_quality_fields(self):
        """Creates TrackInfo with quality reporting fields."""
        info = TrackInfo(
            artist="Queen",
            title="Bohemian Rhapsody",
            duration=354,
            youtube_url="https://youtube.com/watch?v=fJ9rUzIMcZQ",
            youtube_duration=354,
            source="youtube",
            identification_quality=75.0,
            quality_issues=["LRC not found"],
            fallback_used=True,
        )
        assert info.identification_quality == 75.0
        assert "LRC not found" in info.quality_issues
        assert info.fallback_used is True


class TestScoreRecordingStudioLikelihood:
    """Tests for _score_recording_studio_likelihood - scoring MusicBrainz recordings."""

    def test_studio_recording_high_score(self):
        """Studio recordings get high scores."""
        identifier = TrackIdentifier()
        recording = {
            "title": "Bohemian Rhapsody",
            "disambiguation": "",
            "release-list": [
                {"release-group": {"primary-type": "Album", "secondary-type-list": []}}
            ],
        }
        score = identifier._score_recording_studio_likelihood(recording)
        assert score > 100  # Base + album bonus

    def test_live_recording_penalized(self):
        """Live recordings get lower scores."""
        identifier = TrackIdentifier()
        recording = {
            "title": "Bohemian Rhapsody (Live)",
            "disambiguation": "live",
            "release-list": [],
        }
        score = identifier._score_recording_studio_likelihood(recording)
        assert score < 50  # Significant penalty

    def test_compilation_only_penalized(self):
        """Recordings only on compilations get lower scores."""
        identifier = TrackIdentifier()
        recording = {
            "title": "Bohemian Rhapsody",
            "disambiguation": "",
            "release-list": [
                {
                    "release-group": {
                        "primary-type": "Album",
                        "secondary-type-list": ["Compilation"],
                    }
                }
            ],
        }
        score = identifier._score_recording_studio_likelihood(recording)
        assert score == 70

    def test_demo_penalized(self):
        """Demo recordings get lower scores."""
        identifier = TrackIdentifier()
        recording = {
            "title": "Bohemian Rhapsody",
            "disambiguation": "demo",
            "release-list": [],
        }
        score = identifier._score_recording_studio_likelihood(recording)
        assert score < 50


class TestInferArtistFromQuery:
    """Tests for _infer_artist_from_query - inferring artist name."""

    def test_basic_inference(self):
        """Infers artist by removing title words."""
        identifier = TrackIdentifier()
        artist = identifier._infer_artist_from_query(
            "queen bohemian rhapsody", "Bohemian Rhapsody"
        )
        assert artist is not None
        assert "queen" in artist.lower()

    def test_no_title(self):
        """Returns None when title is None."""
        identifier = TrackIdentifier()
        artist = identifier._infer_artist_from_query("queen bohemian rhapsody", None)
        assert artist is None

    def test_remaining_too_short(self):
        """Returns None when remaining text is too short."""
        identifier = TrackIdentifier()
        artist = identifier._infer_artist_from_query(
            "a bohemian rhapsody", "Bohemian Rhapsody"
        )
        assert artist is None


class TestIdentifyFromSearchMocked:
    """Tests for identify_from_search with mocked external services."""

    @patch("y2karaoke.core.track_identifier.TrackIdentifier._try_direct_lrc_search")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._query_musicbrainz")
    @patch(
        "y2karaoke.core.track_identifier.TrackIdentifier._find_best_with_artist_hint"
    )
    @patch(
        "y2karaoke.core.track_identifier.TrackIdentifier._search_youtube_by_duration"
    )
    @patch("y2karaoke.core.sync.fetch_lyrics_for_duration")
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

    @patch("y2karaoke.core.track_identifier.TrackIdentifier._try_direct_lrc_search")
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

    @patch("y2karaoke.core.track_identifier.TrackIdentifier._try_direct_lrc_search")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._query_musicbrainz")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._fallback_youtube_search")
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

    @patch("y2karaoke.core.track_identifier.TrackIdentifier._try_direct_lrc_search")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._query_musicbrainz")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._parse_query")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._find_best_title_only")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._try_split_search")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._fallback_youtube_search")
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

    @patch("y2karaoke.core.sync.fetch_lyrics_for_duration")
    @patch(
        "y2karaoke.core.track_identifier.TrackIdentifier._search_youtube_by_duration"
    )
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._try_split_search")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._find_best_title_only")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._query_musicbrainz")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._parse_query")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._try_direct_lrc_search")
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

    @patch("y2karaoke.core.sync.fetch_lyrics_for_duration")
    @patch(
        "y2karaoke.core.track_identifier.TrackIdentifier._search_youtube_by_duration"
    )
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._find_best_title_only")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._query_musicbrainz")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._parse_query")
    @patch("y2karaoke.core.track_identifier.TrackIdentifier._try_direct_lrc_search")
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
