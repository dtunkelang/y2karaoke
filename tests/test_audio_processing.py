"""Tests for audio processing modules (audio_effects and alignment)."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import tempfile
import os

from y2karaoke.core.audio_effects import AudioProcessor, process_audio
from y2karaoke.core.alignment import (
    detect_lrc_gaps,
    calculate_gap_adjustments,
    _durations_within_tolerance,
    _calculate_adjustments,
    _apply_adjustments_to_lines,
)
from y2karaoke.core.models import Word, Line
from y2karaoke.exceptions import ValidationError

# ------------------------------
# AudioProcessor Tests
# ------------------------------


class TestAudioProcessor:
    def test_init_default_sample_rate(self):
        processor = AudioProcessor()
        assert processor.sample_rate > 0

    def test_init_custom_sample_rate(self):
        processor = AudioProcessor(sample_rate=48000)
        assert processor.sample_rate == 48000

    def test_process_audio_file_not_found(self):
        processor = AudioProcessor()
        with pytest.raises(ValidationError, match="not found"):
            processor.process_audio(
                "/nonexistent/path.wav", "/output/path.wav", semitones=0
            )

    def test_process_audio_validates_key_shift(self):
        processor = AudioProcessor()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        try:
            with pytest.raises(ValidationError, match="Key shift"):
                processor.process_audio(temp_path, "/output.wav", semitones=100)
        finally:
            os.unlink(temp_path)

    def test_process_audio_validates_tempo(self):
        processor = AudioProcessor()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        try:
            with pytest.raises(ValidationError, match="Tempo"):
                processor.process_audio(temp_path, "/output.wav", tempo_multiplier=10.0)
        finally:
            os.unlink(temp_path)

    @patch("shutil.copy")
    def test_process_audio_no_effects_copies_file(self, mock_copy):
        processor = AudioProcessor()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        try:
            result = processor.process_audio(
                temp_path, "/tmp/output.wav", semitones=0, tempo_multiplier=1.0
            )
            mock_copy.assert_called_once()
            assert result == "/tmp/output.wav"
        finally:
            os.unlink(temp_path)

    @patch("y2karaoke.core.audio_effects.sf.write")
    @patch("y2karaoke.core.audio_effects.librosa.effects.pitch_shift")
    @patch("y2karaoke.core.audio_effects.librosa.load")
    def test_process_audio_pitch_shift(self, mock_load, mock_pitch_shift, mock_write):
        mock_load.return_value = (np.zeros(44100), 44100)
        mock_pitch_shift.return_value = np.zeros(44100)

        processor = AudioProcessor()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = os.path.join(tmpdir, "output.wav")
                processor.process_audio(temp_path, output_path, semitones=2)
                mock_pitch_shift.assert_called_once()
                mock_write.assert_called_once()
        finally:
            os.unlink(temp_path)

    @patch("y2karaoke.core.audio_effects.sf.write")
    @patch("y2karaoke.core.audio_effects.librosa.effects.time_stretch")
    @patch("y2karaoke.core.audio_effects.librosa.load")
    def test_process_audio_tempo_change(self, mock_load, mock_time_stretch, mock_write):
        mock_load.return_value = (np.zeros(44100), 44100)
        mock_time_stretch.return_value = np.zeros(44100)

        processor = AudioProcessor()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = os.path.join(tmpdir, "output.wav")
                processor.process_audio(temp_path, output_path, tempo_multiplier=1.5)
                mock_time_stretch.assert_called_once()
                mock_write.assert_called_once()
        finally:
            os.unlink(temp_path)

    def test_shift_key_delegates_to_process_audio(self):
        processor = AudioProcessor()
        with patch.object(processor, "process_audio") as mock_process:
            mock_process.return_value = "/output.wav"
            result = processor.shift_key("/input.wav", "/output.wav", 3)
            mock_process.assert_called_once_with(
                "/input.wav", "/output.wav", semitones=3
            )
            assert result == "/output.wav"

    def test_change_tempo_delegates_to_process_audio(self):
        processor = AudioProcessor()
        with patch.object(processor, "process_audio") as mock_process:
            mock_process.return_value = "/output.wav"
            result = processor.change_tempo("/input.wav", "/output.wav", 1.25)
            mock_process.assert_called_once_with(
                "/input.wav", "/output.wav", tempo_multiplier=1.25
            )
            assert result == "/output.wav"


class TestAudioEffectsConvenienceFunctions:
    @patch("y2karaoke.core.audio_effects.AudioProcessor.process_audio")
    def test_process_audio_function(self, mock_process):
        mock_process.return_value = "/output.wav"
        result = process_audio("/input.wav", "/output.wav", semitones=2)
        mock_process.assert_called_once()
        assert result == "/output.wav"


# ------------------------------
# Alignment Module Tests
# ------------------------------


class TestDetectLrcGaps:
    def test_empty_list_returns_empty(self):
        result = detect_lrc_gaps([])
        assert result == []

    def test_single_element_returns_empty(self):
        result = detect_lrc_gaps([(0.0, "text")])
        assert result == []

    def test_no_gaps_above_threshold(self):
        timings = [(0.0, "a"), (2.0, "b"), (4.0, "c")]
        result = detect_lrc_gaps(timings, min_gap_duration=5.0)
        assert result == []

    def test_detects_single_gap(self):
        timings = [(0.0, "a"), (10.0, "b")]  # 10s gap
        result = detect_lrc_gaps(timings, min_gap_duration=5.0)
        assert len(result) == 1
        assert result[0] == (0.0, 10.0)

    def test_detects_multiple_gaps(self):
        timings = [(0.0, "a"), (10.0, "b"), (25.0, "c")]  # 10s and 15s gaps
        result = detect_lrc_gaps(timings, min_gap_duration=5.0)
        assert len(result) == 2
        assert result[0] == (0.0, 10.0)
        assert result[1] == (10.0, 25.0)

    def test_gap_exactly_at_threshold_included(self):
        timings = [(0.0, "a"), (5.0, "b")]  # Exactly 5s gap
        result = detect_lrc_gaps(timings, min_gap_duration=5.0)
        assert len(result) == 1

    def test_custom_threshold(self):
        timings = [(0.0, "a"), (20.0, "b"), (50.0, "c")]
        # 15s threshold: both gaps (20s and 30s) exceed it
        result = detect_lrc_gaps(timings, min_gap_duration=15.0)
        assert len(result) == 2


class TestCalculateGapAdjustments:
    def test_empty_gaps_returns_empty(self):
        result = calculate_gap_adjustments([], [(0.0, 10.0)])
        assert result == []

    def test_empty_silences_returns_empty(self):
        result = calculate_gap_adjustments([(0.0, 10.0)], [])
        assert result == []

    def test_matching_gap_and_silence(self):
        lrc_gaps = [(10.0, 20.0)]  # 10s gap
        audio_silences = [(10.0, 25.0)]  # 15s silence - 5s longer
        result = calculate_gap_adjustments(lrc_gaps, audio_silences, tolerance=10.0)
        # Should detect the 5s difference
        assert len(result) == 1
        assert result[0][0] == 20.0  # gap end time
        assert result[0][1] == pytest.approx(5.0, abs=0.1)  # adjustment

    def test_no_match_within_tolerance(self):
        lrc_gaps = [(10.0, 20.0)]
        audio_silences = [(100.0, 110.0)]  # Far away
        result = calculate_gap_adjustments(lrc_gaps, audio_silences, tolerance=10.0)
        assert result == []


class TestDurationsWithinTolerance:
    def test_none_values_return_false(self):
        assert _durations_within_tolerance(None, 100) is False
        assert _durations_within_tolerance(100, None) is False
        assert _durations_within_tolerance(None, None) is False

    def test_within_tolerance_returns_true(self):
        assert _durations_within_tolerance(100, 105, tolerance=10) is True
        assert _durations_within_tolerance(100, 100, tolerance=10) is True

    def test_outside_tolerance_returns_false(self):
        assert _durations_within_tolerance(100, 120, tolerance=10) is False


class TestCalculateAdjustmentsHelper:
    def test_empty_inputs(self):
        assert _calculate_adjustments([], []) == []

    def test_matching_gaps(self):
        lrc_gaps = [(10.0, 20.0)]
        audio_silences = [(10.0, 25.0)]
        result = _calculate_adjustments(lrc_gaps, audio_silences)
        assert len(result) == 1
        # Audio resumes at 25, LRC at 20, diff = +5
        assert result[0][0] == 20.0
        assert result[0][1] == pytest.approx(5.0, abs=0.1)

    def test_cumulative_adjustments(self):
        lrc_gaps = [(10.0, 15.0), (30.0, 35.0)]
        audio_silences = [(10.0, 20.0), (35.0, 45.0)]  # Both 5s longer
        result = _calculate_adjustments(lrc_gaps, audio_silences)
        # First adjustment: +5s, Second: cumulative +10s
        assert len(result) == 2


class TestApplyAdjustmentsToLines:
    def test_empty_lines_returns_empty(self):
        result = _apply_adjustments_to_lines([], [(10.0, 5.0)])
        assert result == []

    def test_empty_adjustments_returns_same_lines(self):
        word = Word(text="hello", start_time=0.0, end_time=1.0)
        line = Line(words=[word])
        result = _apply_adjustments_to_lines([line], [])
        assert len(result) == 1
        assert result[0].words[0].start_time == 0.0

    def test_applies_adjustment_after_gap(self):
        word1 = Word(text="before", start_time=5.0, end_time=6.0)
        word2 = Word(text="after", start_time=25.0, end_time=26.0)
        lines = [Line(words=[word1]), Line(words=[word2])]

        # Adjustment of +5s applies after gap ends at 20s
        adjustments = [(20.0, 5.0)]
        result = _apply_adjustments_to_lines(lines, adjustments)

        # First line (at 5s) should not be adjusted
        assert result[0].words[0].start_time == 5.0
        # Second line (at 25s > 20s) should be adjusted by +5s
        assert result[1].words[0].start_time == 30.0

    def test_preserves_word_text_and_singer(self):
        word = Word(text="hello", start_time=25.0, end_time=26.0, singer="singer1")
        line = Line(words=[word], singer="singer1")
        adjustments = [(20.0, 5.0)]

        result = _apply_adjustments_to_lines([line], adjustments)

        assert result[0].words[0].text == "hello"
        assert result[0].words[0].singer == "singer1"
        assert result[0].singer == "singer1"

    def test_empty_words_line_unchanged(self):
        line = Line(words=[])
        result = _apply_adjustments_to_lines([line], [(10.0, 5.0)])
        assert len(result) == 1
        assert result[0].words == []


# ------------------------------
# Integration Tests
# ------------------------------


class TestAlignmentIntegration:
    def test_full_adjustment_flow(self):
        """Test the complete flow from gap detection to line adjustment."""
        # Setup: LRC has a 10s gap, audio has a 15s silence (5s longer)
        lrc_timings = [(0.0, "first"), (5.0, "second"), (25.0, "third")]

        # Detect gaps (20s gap between 5s and 25s)
        gaps = detect_lrc_gaps(lrc_timings, min_gap_duration=10.0)
        assert len(gaps) == 1
        assert gaps[0] == (5.0, 25.0)

        # Create lines matching the timings
        lines = [
            Line(words=[Word(text="first", start_time=0.0, end_time=3.0)]),
            Line(words=[Word(text="second", start_time=5.0, end_time=8.0)]),
            Line(words=[Word(text="third", start_time=25.0, end_time=28.0)]),
        ]

        # Simulate audio silence being longer (ends at 30s instead of 25s)
        audio_silences = [(5.0, 30.0)]
        adjustments = _calculate_adjustments(gaps, audio_silences)

        # Apply adjustments
        adjusted = _apply_adjustments_to_lines(lines, adjustments)

        # Lines before gap should be unchanged
        assert adjusted[0].words[0].start_time == 0.0
        assert adjusted[1].words[0].start_time == 5.0

        # Line after gap should be shifted by +5s
        assert adjusted[2].words[0].start_time == 30.0
