"""Tests for word timing refinement module."""

import pytest
import numpy as np

from y2karaoke.core.refine import (
    refine_word_timing,
    _refine_line_timing,
    _detect_vocal_end,
    _find_best_onset_for_word,
    _fix_unmatched_word_starts,
    _build_refined_words,
    _match_words_to_onsets,
)
from y2karaoke.core.models import Word, Line

# ------------------------------
# Helper function tests
# ------------------------------


class TestFindBestOnsetForWord:
    def test_finds_closest_onset(self):
        onsets = np.array([1.0, 2.0, 3.0, 4.0])
        # Word expected at 2.1s, closest onset is at 2.0s (index 1)
        idx, score = _find_best_onset_for_word(
            word_idx=0,
            expected_start=2.1,
            sorted_onsets=onsets,
            min_next_onset_idx=0,
            n_words=1,
            n_onsets=4,
        )
        assert idx == 1
        assert score == pytest.approx(0.1, abs=0.01)

    def test_respects_min_next_onset_idx(self):
        onsets = np.array([1.0, 2.0, 3.0, 4.0])
        # Word expected at 1.5s but min_next_onset_idx=2, so must use onset at 3.0 or later
        idx, score = _find_best_onset_for_word(
            word_idx=0,
            expected_start=1.5,
            sorted_onsets=onsets,
            min_next_onset_idx=2,
            n_words=1,
            n_onsets=4,
        )
        assert idx == 2  # 3.0s onset

    def test_penalizes_early_onsets(self):
        onsets = np.array([1.0, 2.5])
        # Word expected at 2.0s
        # Onset at 1.0s is 1.0s early (>0.5s) so gets +2.0 penalty
        # Onset at 2.5s is 0.5s late, no penalty
        idx, score = _find_best_onset_for_word(
            word_idx=0,
            expected_start=2.0,
            sorted_onsets=onsets,
            min_next_onset_idx=0,
            n_words=1,
            n_onsets=2,
        )
        # 2.5s onset should win (score 0.5) vs 1.0s (score 1.0+2.0=3.0)
        assert idx == 1

    def test_no_match_when_distance_too_large(self):
        onsets = np.array([10.0])  # Very far from expected
        idx, score = _find_best_onset_for_word(
            word_idx=0,
            expected_start=1.0,
            sorted_onsets=onsets,
            min_next_onset_idx=0,
            n_words=1,
            n_onsets=1,
        )
        # Distance is 9.0 > 2.0, should skip
        assert idx is None


class TestFixUnmatchedWordStarts:
    def test_all_matched_unchanged(self):
        words = [
            Word(text="hello", start_time=0.0, end_time=0.5),
            Word(text="world", start_time=0.5, end_time=1.0),
        ]
        word_starts = [0.0, 0.5]
        result = _fix_unmatched_word_starts(word_starts, words)
        assert result == [0.0, 0.5]

    def test_interpolates_from_next_word(self):
        words = [
            Word(text="hello", start_time=0.0, end_time=0.5),
            Word(text="world", start_time=0.5, end_time=1.0),
        ]
        word_starts = [None, 1.0]  # First word unmatched
        result = _fix_unmatched_word_starts(word_starts, words)
        # First word should be interpolated before 1.0
        assert result[0] < 1.0
        assert result[1] == 1.0

    def test_interpolates_from_prev_word(self):
        words = [
            Word(text="hello", start_time=0.0, end_time=0.5),
            Word(text="world", start_time=0.5, end_time=1.0),
        ]
        word_starts = [0.0, None]  # Second word unmatched
        result = _fix_unmatched_word_starts(word_starts, words)
        assert result[0] == 0.0
        # Second word should be after first
        assert result[1] > result[0]

    def test_ensures_monotonic_increasing(self):
        words = [
            Word(text="a", start_time=0.0, end_time=0.3),
            Word(text="b", start_time=0.3, end_time=0.6),
            Word(text="c", start_time=0.6, end_time=0.9),
        ]
        # Non-monotonic starts
        word_starts = [0.5, 0.3, 0.7]
        result = _fix_unmatched_word_starts(word_starts, words)
        # Result should be monotonically increasing
        assert result[0] < result[1] < result[2]

    def test_falls_back_to_original_timing(self):
        words = [Word(text="only", start_time=5.0, end_time=6.0)]
        word_starts = [None]
        result = _fix_unmatched_word_starts(word_starts, words)
        # Should fall back to original start time
        assert result[0] == 5.0


class TestBuildRefinedWords:
    def test_builds_words_with_timing(self):
        words = [
            Word(text="hello", start_time=0.0, end_time=0.5),
            Word(text="world", start_time=0.5, end_time=1.0),
        ]
        word_starts = [0.0, 0.5]
        result = _build_refined_words(
            words, word_starts, line_start=0.0, vocal_end=1.0, respect_boundaries=True
        )
        assert len(result) == 2
        assert result[0].text == "hello"
        assert result[1].text == "world"
        assert result[0].start_time == 0.0
        assert result[1].start_time == 0.5

    def test_last_word_ends_at_vocal_end(self):
        words = [Word(text="hello", start_time=0.0, end_time=0.5)]
        word_starts = [0.0]
        result = _build_refined_words(
            words, word_starts, line_start=0.0, vocal_end=2.0, respect_boundaries=True
        )
        # Last word should end at vocal_end
        assert result[0].end_time == pytest.approx(2.0, abs=0.1)

    def test_enforces_minimum_duration(self):
        words = [
            Word(text="a", start_time=0.0, end_time=0.05),
            Word(text="b", start_time=0.05, end_time=0.1),
        ]
        # Very close start times
        word_starts = [0.0, 0.05]
        result = _build_refined_words(
            words, word_starts, line_start=0.0, vocal_end=1.0, respect_boundaries=True
        )
        # Each word should have at least min_duration (0.1)
        assert result[0].end_time - result[0].start_time >= 0.1

    def test_respects_line_boundaries(self):
        words = [Word(text="hello", start_time=-0.5, end_time=0.5)]
        word_starts = [-0.5]  # Before line start
        result = _build_refined_words(
            words, word_starts, line_start=0.0, vocal_end=1.0, respect_boundaries=True
        )
        # Start should be clamped to line_start
        assert result[0].start_time >= 0.0

    def test_preserves_singer(self):
        words = [Word(text="hello", start_time=0.0, end_time=0.5, singer="singer1")]
        word_starts = [0.0]
        result = _build_refined_words(
            words, word_starts, line_start=0.0, vocal_end=1.0, respect_boundaries=True
        )
        assert result[0].singer == "singer1"


class TestMatchWordsToOnsets:
    def test_empty_onsets_returns_original(self):
        words = [Word(text="hello", start_time=0.0, end_time=0.5)]
        result = _match_words_to_onsets(
            words, np.array([]), line_start=0.0, vocal_end=1.0, respect_boundaries=True
        )
        assert len(result) == 1
        assert result[0].text == "hello"

    def test_matches_words_to_closest_onsets(self):
        words = [
            Word(text="hello", start_time=0.0, end_time=0.5),
            Word(text="world", start_time=0.5, end_time=1.0),
        ]
        onsets = np.array([0.1, 0.6])
        result = _match_words_to_onsets(
            words, onsets, line_start=0.0, vocal_end=1.0, respect_boundaries=True
        )
        assert len(result) == 2
        # First word should be at ~0.1
        assert result[0].start_time == pytest.approx(0.1, abs=0.1)
        # Second word should be at ~0.6
        assert result[1].start_time == pytest.approx(0.6, abs=0.1)

    def test_preserves_word_order(self):
        words = [
            Word(text="a", start_time=0.0, end_time=0.3),
            Word(text="b", start_time=0.3, end_time=0.6),
            Word(text="c", start_time=0.6, end_time=0.9),
        ]
        onsets = np.array([0.1, 0.4, 0.7])
        result = _match_words_to_onsets(
            words, onsets, line_start=0.0, vocal_end=1.0, respect_boundaries=True
        )
        # Word starts should be monotonically increasing
        assert result[0].start_time < result[1].start_time < result[2].start_time


class TestDetectVocalEnd:
    def test_returns_line_end_when_no_rms_data(self):
        rms = np.array([])
        rms_times = np.array([])
        result = _detect_vocal_end(
            line_start=0.0,
            line_end=5.0,
            rms=rms,
            rms_times=rms_times,
            silence_threshold=0.1,
            word_count=3,
        )
        assert result == 5.0

    def test_detects_vocal_end_from_sustained_silence(self):
        # Create RMS data with vocals then silence
        # High energy for first 20 frames, then silence
        rms = np.concatenate([np.ones(20) * 0.5, np.ones(30) * 0.01])
        # Each frame is ~0.023s (512 samples at 22050 Hz)
        rms_times = np.arange(len(rms)) * 0.023

        result = _detect_vocal_end(
            line_start=0.0,
            line_end=2.0,
            rms=rms,
            rms_times=rms_times,
            silence_threshold=0.1,
            word_count=3,
            min_silence_duration=0.4,
        )
        # Vocal end should be around frame 20 * 0.023 = 0.46s
        assert result < 1.0  # Should detect end before line_end

    def test_respects_minimum_word_duration(self):
        # All silence - but we have 10 words, so need ~1.5s minimum
        rms = np.ones(50) * 0.01
        rms_times = np.arange(len(rms)) * 0.023

        result = _detect_vocal_end(
            line_start=0.0,
            line_end=5.0,
            rms=rms,
            rms_times=rms_times,
            silence_threshold=0.1,
            word_count=10,  # Need 10 * 0.15 = 1.5s minimum
        )
        # Should not trim below minimum for words
        assert result >= 1.5


class TestRefineLineTiming:
    def test_empty_line_returns_unchanged(self):
        line = Line(words=[])
        result = _refine_line_timing(
            line,
            onset_times=np.array([1.0, 2.0]),
            rms=np.ones(100) * 0.5,
            rms_times=np.arange(100) * 0.023,
            silence_threshold=0.1,
            respect_boundaries=True,
        )
        assert result.words == []

    def test_preserves_singer_info(self):
        words = [Word(text="hello", start_time=1.0, end_time=2.0)]
        line = Line(words=words, singer="singer1")
        result = _refine_line_timing(
            line,
            onset_times=np.array([1.0]),
            rms=np.ones(100) * 0.5,
            rms_times=np.arange(100) * 0.023,
            silence_threshold=0.1,
            respect_boundaries=True,
        )
        assert result.singer == "singer1"


class TestRefineWordTiming:
    @pytest.mark.filterwarnings("ignore:PySoundFile failed")
    def test_returns_original_on_nonexistent_file(self):
        """Test that refine_word_timing returns original lines when file doesn't exist."""
        words = [Word(text="hello", start_time=0.0, end_time=0.5)]
        lines = [Line(words=words)]

        result = refine_word_timing(lines, "/nonexistent/path.wav")

        # Should return original lines on error
        assert len(result) == 1
        assert result[0].words[0].text == "hello"
        assert result[0].words[0].start_time == 0.0

    @pytest.mark.filterwarnings("ignore:PySoundFile failed")
    def test_handles_empty_lines_list(self):
        """Test handling of empty lines list."""
        # With empty lines, it should return empty even if it tries to load audio
        result = refine_word_timing([], "/nonexistent/path.wav")
        assert result == []

    def test_refine_word_timing_success_path(self, monkeypatch):
        words = [
            Word(text="hello", start_time=0.0, end_time=0.5),
            Word(text="world", start_time=0.5, end_time=1.0),
        ]
        lines = [Line(words=words)]

        class FakeLibrosa:
            @staticmethod
            def load(_path, sr=22050):
                return np.zeros(2048), sr

            class onset:
                @staticmethod
                def onset_detect(*_args, **_kwargs):
                    return np.array([0, 512])

            @staticmethod
            def frames_to_time(frames, sr=22050, hop_length=512):
                return np.array(frames) * (hop_length / sr)

            class feature:
                @staticmethod
                def rms(*_args, **_kwargs):
                    return np.array([[0.1, 0.4, 0.2, 0.05]])

        monkeypatch.setattr(
            "y2karaoke.core.refine._load_librosa",
            lambda: FakeLibrosa,
        )

        refined = refine_word_timing(lines, "/tmp/vocals.wav")
        assert len(refined) == 1
        assert len(refined[0].words) == 2
