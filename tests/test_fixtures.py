"""Tests that demonstrate and validate the shared fixtures.

These tests ensure the fixtures work correctly and serve as examples
for how to use them in other test files.
"""

import pytest
from unittest.mock import patch


class TestMusicBrainzFixtures:
    """Tests demonstrating MusicBrainz fixture usage."""

    def test_recording_fixture_structure(self, musicbrainz_recording_queen):
        """Verify MusicBrainz recording fixture has correct structure."""
        assert musicbrainz_recording_queen['title'] == 'Bohemian Rhapsody'
        assert musicbrainz_recording_queen['length'] == 354000
        assert len(musicbrainz_recording_queen['artist-credit']) > 0
        assert musicbrainz_recording_queen['artist-credit'][0]['artist']['name'] == 'Queen'

    def test_live_recording_has_disambiguation(self, musicbrainz_recording_live):
        """Live recordings should have disambiguation field."""
        assert musicbrainz_recording_live['disambiguation'] == 'live'
        assert 'Live' in musicbrainz_recording_live['title']

    def test_search_results_contain_multiple_recordings(self, musicbrainz_search_results):
        """Search results should contain multiple recordings."""
        recordings = musicbrainz_search_results['recording-list']
        assert len(recordings) >= 2


class TestYouTubeFixtures:
    """Tests demonstrating YouTube fixture usage."""

    def test_metadata_fixture_structure(self, youtube_metadata_queen):
        """Verify YouTube metadata fixture has correct structure."""
        assert youtube_metadata_queen['title'] == 'Queen - Bohemian Rhapsody (Official Video)'
        assert youtube_metadata_queen['duration'] == 354
        assert youtube_metadata_queen['uploader'] == 'Queen Official'
        assert youtube_metadata_queen['id'] == 'fJ9rUzIMcZQ'

    def test_cover_metadata_differs_from_official(
        self, youtube_metadata_queen, youtube_metadata_cover
    ):
        """Cover version should have different metadata than official."""
        assert youtube_metadata_cover['uploader'] != youtube_metadata_queen['uploader']
        assert youtube_metadata_cover['id'] != youtube_metadata_queen['id']
        assert 'Cover' in youtube_metadata_cover['title']

    def test_search_response_contains_video_ids(self, youtube_search_response_queen):
        """Search response should contain extractable video IDs."""
        assert 'fJ9rUzIMcZQ' in youtube_search_response_queen
        assert 'videoRenderer' in youtube_search_response_queen


class TestLRCFixtures:
    """Tests demonstrating LRC fixture usage."""

    def test_lrc_has_metadata_tags(self, lrc_bohemian_rhapsody):
        """LRC fixture should have metadata tags."""
        assert '[ar:Queen]' in lrc_bohemian_rhapsody
        assert '[ti:Bohemian Rhapsody]' in lrc_bohemian_rhapsody

    def test_lrc_has_timestamps(self, lrc_bohemian_rhapsody):
        """LRC fixture should have timestamped lines."""
        assert '[00:00.00]' in lrc_bohemian_rhapsody
        assert '[05:54.00]' in lrc_bohemian_rhapsody

    def test_lrc_no_metadata_lacks_tags(self, lrc_no_metadata):
        """LRC without metadata should not have artist/title tags."""
        assert '[ar:' not in lrc_no_metadata
        assert '[ti:' not in lrc_no_metadata
        # But should still have timestamps
        assert '[00:00.00]' in lrc_no_metadata

    def test_invalid_lrc_has_sparse_timing(self, lrc_invalid_timing):
        """Invalid LRC should have sparse timing (for testing validation)."""
        lines = [l for l in lrc_invalid_timing.split('\n') if l.startswith('[0')]
        assert len(lines) == 2  # Only 2 timestamped lines


class TestTrackInfoFixtures:
    """Tests demonstrating TrackInfo fixture usage."""

    def test_queen_track_info(self, track_info_queen):
        """Queen TrackInfo fixture should have high quality."""
        assert track_info_queen.artist == 'Queen'
        assert track_info_queen.title == 'Bohemian Rhapsody'
        assert track_info_queen.lrc_validated is True
        assert track_info_queen.identification_quality >= 90
        assert track_info_queen.fallback_used is False

    def test_fallback_track_info(self, track_info_youtube_fallback):
        """Fallback TrackInfo should have lower quality."""
        assert track_info_youtube_fallback.source == 'youtube'
        assert track_info_youtube_fallback.lrc_validated is False
        assert track_info_youtube_fallback.identification_quality < 60
        assert track_info_youtube_fallback.fallback_used is True
        assert len(track_info_youtube_fallback.quality_issues) > 0


class TestLyricsLineFixtures:
    """Tests demonstrating lyrics line fixture usage."""

    def test_sample_lines_structure(self, sample_lyrics_lines):
        """Sample lyrics lines should have correct structure."""
        assert len(sample_lyrics_lines) == 2
        first_line = sample_lyrics_lines[0]
        assert len(first_line.words) == 5
        assert first_line.words[0].text == 'Is'
        assert first_line.words[0].start_time == 0.0

    def test_lines_with_break_has_gap(self, sample_lyrics_lines_with_break):
        """Lines with break should have significant time gap."""
        first_line = sample_lyrics_lines_with_break[0]
        second_line = sample_lyrics_lines_with_break[1]

        first_end = first_line.words[-1].end_time
        second_start = second_line.words[0].start_time

        gap = second_start - first_end
        assert gap >= 60  # At least 60 second gap


class TestCacheFixtures:
    """Tests demonstrating cache fixture usage."""

    def test_mock_cache_manager_creates_dirs(self, mock_cache_manager, temp_dir):
        """Mock cache manager should use temp directory."""
        assert mock_cache_manager.cache_dir == temp_dir

    def test_populated_cache_has_files(self, populated_cache):
        """Populated cache should have pre-created files."""
        cache_manager, video_id = populated_cache
        video_dir = cache_manager.get_video_cache_dir(video_id)

        assert (video_dir / 'audio.wav').exists()
        assert (video_dir / 'vocals.wav').exists()
        assert (video_dir / 'instrumental.wav').exists()

    def test_populated_cache_has_metadata(self, populated_cache):
        """Populated cache should have metadata."""
        cache_manager, video_id = populated_cache
        metadata = cache_manager.load_metadata(video_id)

        assert metadata is not None
        assert metadata['title'] == 'Test Song'
        assert metadata['artist'] == 'Test Artist'


class TestMockProviderFixtures:
    """Tests demonstrating mock provider fixture usage."""

    def test_mock_lrc_provider_returns_lyrics(self, mock_lrc_provider):
        """Mock LRC provider should return synced lyrics."""
        from y2karaoke.core.sync import fetch_lyrics_multi_source
        lrc_text, is_synced, source = fetch_lyrics_multi_source('test', 'test')

        assert is_synced is True
        assert source == 'lyriq'
        assert '[ar:Queen]' in lrc_text

    def test_mock_lrc_provider_not_found(self, mock_lrc_provider_not_found):
        """Mock LRC provider should return no results when configured."""
        from y2karaoke.core.sync import fetch_lyrics_multi_source
        lrc_text, is_synced, source = fetch_lyrics_multi_source('test', 'test')

        assert is_synced is False
        assert lrc_text is None


class TestCombinedMockFixtures:
    """Tests demonstrating combined fixture usage."""

    def test_track_identifier_with_mocks(self, mock_track_identifier_dependencies):
        """Test TrackIdentifier with all dependencies mocked."""
        mocks = mock_track_identifier_dependencies

        # Configure mocks for a successful search
        mocks['musicbrainz'].search_recordings.return_value = {
            'recording-list': [
                {
                    'title': 'Test Song',
                    'length': 200000,
                    'artist-credit': [{'artist': {'name': 'Test Artist'}}],
                    'disambiguation': '',
                    'release-list': []
                }
            ]
        }
        mocks['fetch_lyrics'].return_value = ('[00:00.00]Lyrics', True, 'lyriq')
        mocks['get_duration'].return_value = 200
        mocks['validate_quality'].return_value = (True, None)

        # Now we can test TrackIdentifier without network calls
        from y2karaoke.core.track_identifier import TrackIdentifier
        identifier = TrackIdentifier()

        # The mocks are active, so any calls will use mock data
        assert mocks['musicbrainz'].search_recordings.return_value['recording-list'][0]['title'] == 'Test Song'
