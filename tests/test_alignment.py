"""Tests for alignment.py - audio analysis and timing adjustment (focused)."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from y2karaoke.core.alignment import (
    detect_audio_silence_regions,
    detect_lrc_gaps,
    calculate_gap_adjustments,
    adjust_timing_for_duration_mismatch,
    _durations_within_tolerance,
    _calculate_adjustments,
    _apply_adjustments_to_lines,
    detect_song_start,
    _detect_song_start_rms,
)
from y2karaoke.core.models import Word, Line


class TestDetectAudioSilenceRegions:
    @patch("librosa.load")
    @patch("librosa.feature.rms")
    @patch("librosa.frames_to_time")
    def test_detects_silence_regions(self, mock_frames_to_time, mock_rms, mock_load):
        # Mock audio data
        mock_load.return_value = (np.array([0.1, 0.2, 0.3]), 22050)
        mock_rms.return_value = np.array([[0.1, 0.01, 0.3]])  # Low energy in middle
        mock_frames_to_time.return_value = np.array([0.0, 1.0, 2.0])

        silence_regions = detect_audio_silence_regions(
            "test.wav", min_silence_duration=0.5
        )

        assert isinstance(silence_regions, list)
        mock_load.assert_called_once()

    @patch("librosa.load")
    def test_handles_audio_load_error(self, mock_load):
        mock_load.side_effect = Exception("Audio load failed")

        silence_regions = detect_audio_silence_regions("nonexistent.wav")

        assert silence_regions == []

    @patch("librosa.load")
    @patch("librosa.feature.rms")
    @patch("librosa.frames_to_time")
    def test_custom_parameters(self, mock_frames_to_time, mock_rms, mock_load):
        mock_load.return_value = (np.array([0.1, 0.2]), 44100)
        mock_rms.return_value = np.array([[0.1, 0.2]])
        mock_frames_to_time.return_value = np.array([0.0, 1.0])

        silence_regions = detect_audio_silence_regions(
            "test.wav",
            min_silence_duration=3.0,
            energy_threshold_percentile=20.0,
            sample_rate=44100,
        )

        assert isinstance(silence_regions, list)
        mock_load.assert_called_with("test.wav", sr=44100)

    @patch("librosa.load")
    @patch("librosa.feature.rms")
    @patch("librosa.frames_to_time")
    def test_detects_trailing_silence(self, mock_frames_to_time, mock_rms, mock_load):
        mock_load.return_value = (np.array([0.1, 0.2, 0.3]), 22050)
        mock_rms.return_value = np.array([[0.2, 0.01, 0.01]])
        mock_frames_to_time.return_value = np.array([0.0, 1.0, 2.0])

        silence_regions = detect_audio_silence_regions(
            "test.wav", min_silence_duration=0.5
        )

        assert silence_regions == [(1.0, 2.0)]


class TestDetectLrcGaps:
    def test_detects_single_gap(self):
        line_timings = [(1.0, "hello"), (5.0, "world")]  # Gap of 4.0s

        gaps = detect_lrc_gaps(line_timings, min_gap_duration=3.0)

        assert len(gaps) == 1
        assert gaps[0] == (1.0, 5.0)

    def test_detects_multiple_gaps(self):
        line_timings = [(1.0, "hello"), (6.0, "world"), (12.0, "test")]

        gaps = detect_lrc_gaps(line_timings, min_gap_duration=4.0)

        assert len(gaps) == 2
        assert gaps[0] == (1.0, 6.0)  # 5s gap
        assert gaps[1] == (6.0, 12.0)  # 6s gap

    def test_no_gaps_above_threshold(self):
        line_timings = [(1.0, "hello"), (2.0, "world")]  # Small gap

        gaps = detect_lrc_gaps(line_timings, min_gap_duration=5.0)

        assert gaps == []

    def test_empty_list_returns_empty(self):
        gaps = detect_lrc_gaps([], min_gap_duration=5.0)
        assert gaps == []

    def test_single_element_returns_empty(self):
        line_timings = [(1.0, "hello")]

        gaps = detect_lrc_gaps(line_timings, min_gap_duration=5.0)

        assert gaps == []

    def test_custom_threshold(self):
        line_timings = [(1.0, "hello"), (3.0, "world")]  # 2s gap

        gaps = detect_lrc_gaps(line_timings, min_gap_duration=1.5)

        assert len(gaps) == 1
        assert gaps[0] == (1.0, 3.0)


class TestCalculateGapAdjustments:
    def test_matching_gap_and_silence(self):
        lrc_gaps = [(10.0, 15.0)]  # 5s gap
        audio_silences = [(9.0, 16.0)]  # 7s silence, overlapping

        adjustments = calculate_gap_adjustments(lrc_gaps, audio_silences, tolerance=5.0)

        assert isinstance(adjustments, list)

    def test_no_match_within_tolerance(self):
        lrc_gaps = [(10.0, 15.0)]
        audio_silences = [(50.0, 55.0)]  # Far from LRC gap

        adjustments = calculate_gap_adjustments(lrc_gaps, audio_silences, tolerance=5.0)

        assert adjustments == []

    def test_adjusts_when_audio_silence_is_longer(self):
        lrc_gaps = [(10.0, 15.0)]
        audio_silences = [(9.0, 18.0)]

        adjustments = calculate_gap_adjustments(lrc_gaps, audio_silences, tolerance=5.0)

        assert adjustments == [(15.0, 4.0)]


def test_calculate_adjustments_no_match_logs_debug():
    lrc_gaps = [(10.0, 15.0)]
    audio_silences = [(40.0, 45.0)]

    adjustments = _calculate_adjustments(lrc_gaps, audio_silences)

    assert adjustments == []

    def test_empty_gaps_returns_empty(self):
        adjustments = calculate_gap_adjustments([], [(10.0, 15.0)])
        assert adjustments == []

    def test_empty_silences_returns_empty(self):
        lrc_gaps = [(10.0, 15.0)]
        adjustments = calculate_gap_adjustments(lrc_gaps, [])
        assert adjustments == []

    def test_custom_tolerance(self):
        lrc_gaps = [(10.0, 15.0)]
        audio_silences = [(12.0, 17.0)]  # 2s offset

        # Should not match with tight tolerance
        adjustments = calculate_gap_adjustments(lrc_gaps, audio_silences, tolerance=1.0)
        assert adjustments == []

        # Should match with loose tolerance
        adjustments = calculate_gap_adjustments(lrc_gaps, audio_silences, tolerance=5.0)
        assert isinstance(adjustments, list)


class TestDurationsWithinTolerance:
    def test_within_tolerance_returns_true(self):
        assert _durations_within_tolerance(100, 105, tolerance=10) == True
        assert _durations_within_tolerance(100, 95, tolerance=10) == True

    def test_outside_tolerance_returns_false(self):
        assert _durations_within_tolerance(100, 120, tolerance=10) == False
        assert _durations_within_tolerance(100, 80, tolerance=10) == False

    def test_none_values_return_false(self):
        assert _durations_within_tolerance(None, 100, tolerance=10) == False
        assert _durations_within_tolerance(100, None, tolerance=10) == False
        assert _durations_within_tolerance(None, None, tolerance=10) == False

    def test_exact_match(self):
        assert _durations_within_tolerance(100, 100, tolerance=10) == True

    def test_zero_tolerance(self):
        assert _durations_within_tolerance(100, 100, tolerance=0) == True
        assert _durations_within_tolerance(100, 101, tolerance=0) == False


class TestApplyAdjustmentsToLines:
    def test_empty_lines_returns_empty(self):
        result = _apply_adjustments_to_lines([], [])
        assert result == []

    def test_no_adjustments_returns_unchanged(self):
        words = [Word("hello", 1.0, 1.5)]
        lines = [Line(words)]

        result = _apply_adjustments_to_lines(lines, [])

        assert len(result) == 1
        assert result[0].words[0].start_time == 1.0
        assert result[0].words[0].end_time == 1.5

    def test_preserves_word_text(self):
        words = [Word("hello", 1.0, 1.5, singer="test")]
        lines = [Line(words)]

        result = _apply_adjustments_to_lines(lines, [])

        assert result[0].words[0].text == "hello"
        assert result[0].words[0].singer == "test"

    def test_empty_words_line_unchanged(self):
        lines = [Line([])]  # Empty line

        result = _apply_adjustments_to_lines(lines, [])

        assert len(result) == 1
        assert len(result[0].words) == 0


class TestAdjustTimingForDurationMismatch:
    def test_function_exists(self):
        # Just test that the function exists and can be called
        # The actual function requires line_timings and vocals_path parameters
        from y2karaoke.core.alignment import adjust_timing_for_duration_mismatch

        assert callable(adjust_timing_for_duration_mismatch)

    @patch("y2karaoke.core.alignment.detect_lrc_gaps")
    @patch("y2karaoke.core.alignment.detect_audio_silence_regions")
    @patch("y2karaoke.core.alignment._durations_within_tolerance")
    def test_early_exit_when_durations_match(
        self, mock_tolerance, mock_silence, mock_gaps
    ):
        mock_tolerance.return_value = True  # Durations match

        words = [Word("hello", 10.0, 15.0)]
        lines = [Line(words)]
        line_timings = [(10.0, "hello")]

        result = adjust_timing_for_duration_mismatch(
            lines, line_timings, "vocals.wav", lrc_duration=100, audio_duration=100
        )

        assert result == lines  # Should return unchanged
        mock_gaps.assert_not_called()  # Should not proceed to gap detection


def test_adjust_timing_for_duration_mismatch_applies_adjustments(monkeypatch):
    lines = [Line(words=[Word(text="hi", start_time=12.0, end_time=12.5)])]
    line_timings = [(1.0, "hello"), (15.0, "world")]

    monkeypatch.setattr(
        "y2karaoke.core.alignment.detect_lrc_gaps", lambda *_a, **_k: [(1.0, 15.0)]
    )
    monkeypatch.setattr(
        "y2karaoke.core.alignment.detect_audio_silence_regions",
        lambda *_a, **_k: [(2.0, 20.0)],
    )
    monkeypatch.setattr(
        "y2karaoke.core.alignment._calculate_adjustments",
        lambda *_a, **_k: [(15.0, 2.0)],
    )
    monkeypatch.setattr(
        "y2karaoke.core.alignment._apply_adjustments_to_lines",
        lambda *_a, **_k: ["adjusted"],
    )

    result = adjust_timing_for_duration_mismatch(
        lines,
        line_timings,
        "vocals.wav",
        lrc_duration=100,
        audio_duration=120,
    )

    assert result == ["adjusted"]


class TestDetectSongStart:
    @patch("y2karaoke.core.alignment._detect_song_start_rms")
    @patch("librosa.load")
    @patch("librosa.onset.onset_detect")
    def test_detects_song_start_with_onsets(self, mock_onset, mock_load, mock_rms):
        # Use longer audio to avoid filter issues
        mock_load.return_value = (np.random.random(1000), 22050)
        mock_onset.return_value = np.array([0.5, 1.0, 1.5])
        mock_rms.return_value = 0.3  # Fallback value

        start_time = detect_song_start("test.wav")

        assert start_time >= 0.0
        mock_load.assert_called_once()

    @patch("librosa.load")
    def test_handles_audio_load_error(self, mock_load):
        mock_load.side_effect = Exception("Load failed")

        start_time = detect_song_start("nonexistent.wav")

        assert start_time == 0.0

    @patch("librosa.load")
    def test_custom_min_duration(self, mock_load):
        mock_load.return_value = (np.random.random(1000), 22050)

        start_time = detect_song_start("test.wav", min_duration=0.5)

        assert start_time >= 0.0

    @patch("y2karaoke.core.alignment._bandpass_filter")
    @patch("librosa.onset.onset_detect")
    @patch("librosa.onset.onset_strength")
    @patch("librosa.feature.rms")
    @patch("librosa.load")
    def test_detects_validated_onset(
        self,
        mock_load,
        mock_rms,
        mock_strength,
        mock_detect,
        mock_filter,
    ):
        mock_load.return_value = (np.random.random(2000), 22050)
        mock_filter.side_effect = lambda y, *_args, **_kwargs: y
        mock_strength.return_value = np.array([0.5, 0.6, 0.7])
        mock_detect.return_value = np.array([0.1])

        rms = np.zeros(30)
        rms[4:24] = 1.0
        mock_rms.return_value = np.array([rms])

        start_time = detect_song_start("test.wav")

        assert start_time == 0.1

    @patch("y2karaoke.core.alignment._bandpass_filter")
    @patch("librosa.onset.onset_detect")
    @patch("librosa.onset.onset_strength")
    @patch("librosa.feature.rms")
    @patch("librosa.load")
    def test_uses_first_onset_when_not_validated(
        self,
        mock_load,
        mock_rms,
        mock_strength,
        mock_detect,
        mock_filter,
    ):
        mock_load.return_value = (np.array([0.1, 0.2, 0.3]), 22050)
        mock_filter.side_effect = lambda y, *_args, **_kwargs: y
        mock_strength.return_value = np.array([0.5, 0.6, 0.7])
        mock_detect.return_value = np.array([0.5, 1.0])
        mock_rms.return_value = np.array([[0.01, 0.02, 0.01, 0.01, 0.01]])

        start_time = detect_song_start("test.wav")

        assert start_time == 0.5


class TestDetectSongStartRms:
    def test_detects_start_from_rms(self):
        # Create longer audio with low energy at start, then higher
        y = np.concatenate(
            [
                np.random.random(1000) * 0.01,  # Low energy start
                np.random.random(1000) * 0.3,  # Higher energy
            ]
        )
        sr = 22050

        start_time = _detect_song_start_rms(y, sr, min_duration=0.1)

        assert start_time >= 0.0
        assert isinstance(start_time, float)

    def test_handles_short_audio(self):
        y = np.array([0.1, 0.2])  # Very short audio
        sr = 22050

        start_time = _detect_song_start_rms(y, sr)

        assert start_time == 0.0  # Should return 0 for very short audio

    def test_min_duration_parameter(self):
        y = np.concatenate(
            [
                np.random.random(2000) * 0.01,  # Low energy
                np.random.random(2000) * 0.3,  # High energy
            ]
        )
        sr = 22050

        start_time = _detect_song_start_rms(y, sr, min_duration=0.5)

        assert start_time >= 0.0

    def test_all_high_energy_returns_zero(self):
        y = np.random.random(2000) * 0.5  # All high energy
        sr = 22050

        start_time = _detect_song_start_rms(y, sr)

        assert start_time == 0.0

    def test_no_sustained_activity_returns_zero(self):
        y = np.array([0.0, 0.0, 0.0, 0.0])
        sr = 100

        with patch("librosa.feature.rms") as mock_rms:
            mock_rms.return_value = np.array([[0.01, 0.02, 0.01, 0.02]])
            start_time = _detect_song_start_rms(y, sr, min_duration=0.5)

            assert start_time == 0.0

    def test_detects_activity_returns_start(self):
        y = np.array([0.0, 0.0, 0.0])
        sr = 100

        with patch("librosa.feature.rms") as mock_rms:
            mock_rms.return_value = np.array([[0.01, 0.5, 0.5, 0.5, 0.5, 0.5]])
            start_time = _detect_song_start_rms(y, sr, min_duration=0.1)

            assert start_time > 0.0
