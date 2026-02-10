"""Tests for timing_evaluator module - focused on core functionality."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from y2karaoke.core.components.alignment.timing_models import (
    TimingIssue,
    AudioFeatures,
    TimingReport,
    TranscriptionWord,
    TranscriptionSegment,
)
from y2karaoke.core.audio_analysis import (
    extract_audio_features,
    _get_audio_features_cache_path,
    _load_audio_features_cache,
    _save_audio_features_cache,
)
from y2karaoke.core.phonetic_utils import (
    _normalize_text_for_matching,
    _text_similarity_basic,
)
from y2karaoke.core.components.alignment.timing_evaluator import (
    evaluate_timing,
    _find_closest_onset,
)
from y2karaoke.core.models import Word, Line


class TestTimingIssue:
    def test_creation(self):
        issue = TimingIssue(
            issue_type="early_line",
            line_index=0,
            lyrics_time=10.0,
            audio_time=9.5,
            delta=0.5,
            severity="minor",
            description="Test issue",
        )
        assert issue.issue_type == "early_line"
        assert issue.line_index == 0
        assert issue.delta == 0.5


class TestAudioFeatures:
    def test_creation(self):
        features = AudioFeatures(
            onset_times=np.array([1.0, 2.0, 3.0]),
            silence_regions=[(0.0, 0.5), (4.0, 5.0)],
            vocal_start=0.5,
            vocal_end=10.0,
            duration=10.0,
            energy_envelope=np.array([0.1, 0.2, 0.3]),
            energy_times=np.array([0.0, 1.0, 2.0]),
        )
        assert len(features.onset_times) == 3
        assert len(features.silence_regions) == 2
        assert features.vocal_start == 0.5


class TestTimingReport:
    def test_creation(self):
        report = TimingReport(
            source_name="test",
            overall_score=85.0,
            line_alignment_score=80.0,
            pause_alignment_score=90.0,
            issues=[],
            summary="Good timing",
            avg_line_offset=0.1,
            std_line_offset=0.2,
            matched_onsets=5,
            total_lines=6,
        )
        assert report.overall_score == 85.0
        assert report.matched_onsets == 5


class TestTranscriptionModels:
    def test_transcription_word(self):
        word = TranscriptionWord(start=1.0, end=1.5, text="hello", probability=0.95)
        assert word.text == "hello"
        assert word.start == 1.0

    def test_transcription_segment(self):
        words = [
            TranscriptionWord(start=1.0, end=1.5, text="hello", probability=0.95),
            TranscriptionWord(start=1.6, end=2.0, text="world", probability=0.90),
        ]
        segment = TranscriptionSegment(
            start=1.0, end=2.0, text="hello world", words=words
        )
        assert segment.text == "hello world"
        assert len(segment.words) == 2


class TestFindClosestOnset:
    def test_finds_closest_onset(self):
        onsets = np.array([1.0, 2.0, 3.0, 5.0])
        target_time = 2.1

        closest, delta = _find_closest_onset(target_time, onsets)

        assert closest == 2.0
        assert delta == pytest.approx(0.1, abs=1e-6)

    def test_empty_onsets_returns_none(self):
        onsets = np.array([])
        target_time = 2.0

        closest, delta = _find_closest_onset(target_time, onsets)

        assert closest is None
        assert delta == 0.0

    def test_finds_earliest_when_tied(self):
        onsets = np.array([1.0, 3.0])
        target_time = 2.0

        closest, delta = _find_closest_onset(target_time, onsets)

        assert closest == 1.0
        assert delta == 1.0  # target - closest = 2.0 - 1.0 = 1.0


class TestEvaluateTiming:
    def test_empty_lines_returns_zero_scores(self):
        lines = []
        features = AudioFeatures(
            onset_times=np.array([1.0, 2.0]),
            silence_regions=[],
            vocal_start=0.0,
            vocal_end=10.0,
            duration=10.0,
            energy_envelope=np.array([0.1, 0.2]),
            energy_times=np.array([0.0, 1.0]),
        )

        report = evaluate_timing(lines, features)

        # Empty lines gives 0% line alignment but 100% pause alignment (no gaps to check)
        # Overall score is weighted: 0.7 * 0 + 0.3 * 100 = 30.0
        assert report.overall_score == 30.0
        assert report.matched_onsets == 0
        assert report.total_lines == 0

    def test_perfect_alignment_high_score(self):
        words = [Word("hello", 1.0, 1.5)]
        lines = [Line(words)]
        features = AudioFeatures(
            onset_times=np.array([1.0, 2.0, 3.0]),
            silence_regions=[],
            vocal_start=0.0,
            vocal_end=10.0,
            duration=10.0,
            energy_envelope=np.array([0.1, 0.2, 0.3]),
            energy_times=np.array([0.0, 1.0, 2.0]),
        )

        report = evaluate_timing(lines, features)

        assert report.overall_score > 50.0
        assert report.matched_onsets == 1
        assert report.total_lines == 1

    def test_poor_alignment_creates_issues(self):
        words = [Word("hello", 1.0, 1.5)]
        lines = [Line(words)]
        features = AudioFeatures(
            onset_times=np.array([2.0, 3.0]),  # Far from line start
            silence_regions=[],
            vocal_start=0.0,
            vocal_end=10.0,
            duration=10.0,
            energy_envelope=np.array([0.1, 0.2]),
            energy_times=np.array([0.0, 1.0]),
        )

        report = evaluate_timing(lines, features)

        assert len(report.issues) > 0
        assert report.issues[0].severity in ["moderate", "severe"]

    def test_low_onset_coverage_reduces_score(self):
        words = [Word("hello", 1.0, 1.5)]
        lines = [Line(words)]
        features = AudioFeatures(
            onset_times=np.array([]),
            silence_regions=[],
            vocal_start=0.0,
            vocal_end=10.0,
            duration=10.0,
            energy_envelope=np.array([0.1, 0.2]),
            energy_times=np.array([0.0, 1.0]),
        )

        report = evaluate_timing(lines, features)

        assert report.overall_score == pytest.approx(9.0, abs=1e-6)

    def test_partial_onset_coverage_scales_score(self):
        lines = [
            Line(words=[Word("first", 1.0, 1.5)]),
            Line(words=[Word("second", 20.0, 20.5)]),
        ]
        features = AudioFeatures(
            onset_times=np.array([1.0]),
            silence_regions=[],
            vocal_start=0.0,
            vocal_end=25.0,
            duration=25.0,
            energy_envelope=np.array([0.1, 0.2, 0.3]),
            energy_times=np.array([0.0, 1.0, 2.0]),
        )

        report = evaluate_timing(lines, features)

        assert report.overall_score == pytest.approx(42.25, abs=1e-6)

    def test_missing_pause_coverage_reduces_score(self):
        lines = [
            Line(words=[Word("first", 0.0, 1.0)]),
            Line(words=[Word("second", 1.5, 2.0)]),
        ]
        features = AudioFeatures(
            onset_times=np.array([0.0, 1.5]),
            silence_regions=[(2.0, 5.0)],
            vocal_start=0.0,
            vocal_end=6.0,
            duration=6.0,
            energy_envelope=np.array([0.1, 0.2, 0.3]),
            energy_times=np.array([0.0, 1.0, 2.0]),
        )

        report = evaluate_timing(lines, features)

        assert report.overall_score == pytest.approx(49.0, abs=1e-6)

    def test_out_of_order_lines_flagged(self):
        lines = [
            Line(words=[Word("a", 2.0, 2.5)]),
            Line(words=[Word("b", 1.0, 1.5)]),
        ]
        features = AudioFeatures(
            onset_times=np.array([1.0, 2.0]),
            silence_regions=[],
            vocal_start=0.0,
            vocal_end=3.0,
            duration=3.0,
            energy_envelope=np.array([0.1, 0.2, 0.3]),
            energy_times=np.array([0.0, 1.0, 2.0]),
        )

        report = evaluate_timing(lines, features)

        assert any(issue.issue_type == "out_of_order_line" for issue in report.issues)

    def test_negative_line_duration_flagged(self):
        lines = [Line(words=[Word("a", 2.0, 1.5)])]
        features = AudioFeatures(
            onset_times=np.array([1.0, 2.0]),
            silence_regions=[],
            vocal_start=0.0,
            vocal_end=3.0,
            duration=3.0,
            energy_envelope=np.array([0.1, 0.2, 0.3]),
            energy_times=np.array([0.0, 1.0, 2.0]),
        )

        report = evaluate_timing(lines, features)

        assert any(
            issue.issue_type == "negative_line_duration" for issue in report.issues
        )


class TestTextSimilarity:
    def test_identical_texts_perfect_score(self):
        score = _text_similarity_basic("hello world", "hello world")
        assert score == 1.0

    def test_completely_different_texts_low_score(self):
        score = _text_similarity_basic("hello", "goodbye")
        assert score < 0.5

    def test_partial_similarity(self):
        score = _text_similarity_basic("hello world", "hello there")
        assert 0.0 < score < 1.0


class TestNormalizeTextForMatching:
    def test_removes_punctuation(self):
        result = _normalize_text_for_matching("Hello, world!")
        assert result == "hello world"

    def test_handles_unicode(self):
        result = _normalize_text_for_matching("Café naïve")
        assert "cafe" in result.lower()

    def test_removes_extra_whitespace(self):
        result = _normalize_text_for_matching("  hello   world  ")
        assert result == "hello world"


class TestAudioFeaturesCache:
    def test_cache_path_none_for_nonexistent(self):
        path = _get_audio_features_cache_path("/nonexistent/vocals.wav")
        assert path is None

    def test_save_and_load_cache(self, tmp_path):
        cache_file = tmp_path / "test_features.npz"
        features = AudioFeatures(
            onset_times=np.array([1.0, 2.0]),
            silence_regions=[(0.0, 0.5)],
            vocal_start=0.0,
            vocal_end=10.0,
            duration=10.0,
            energy_envelope=np.array([0.1, 0.2]),
            energy_times=np.array([0.0, 1.0]),
        )

        _save_audio_features_cache(str(cache_file), features)
        loaded = _load_audio_features_cache(str(cache_file))

        assert loaded is not None
        assert np.array_equal(loaded.onset_times, features.onset_times)
        # Check silence regions length and content
        assert len(loaded.silence_regions) == len(features.silence_regions)
        if len(loaded.silence_regions) > 0:
            assert loaded.silence_regions[0][0] == features.silence_regions[0][0]
            assert loaded.silence_regions[0][1] == features.silence_regions[0][1]

    def test_load_nonexistent_cache_returns_none(self):
        result = _load_audio_features_cache("/nonexistent/cache.npz")
        assert result is None


class TestExtractAudioFeatures:
    @patch("y2karaoke.core.audio_analysis._load_audio_features_cache")
    @patch("y2karaoke.core.audio_analysis._get_audio_features_cache_path")
    def test_uses_cached_features(self, mock_get_path, mock_load_cache):
        cached_features = AudioFeatures(
            onset_times=np.array([1.0, 2.0]),
            silence_regions=[],
            vocal_start=0.0,
            vocal_end=10.0,
            duration=10.0,
            energy_envelope=np.array([0.1, 0.2]),
            energy_times=np.array([0.0, 1.0]),
        )
        mock_get_path.return_value = "/fake/cache/path.npz"
        mock_load_cache.return_value = cached_features

        features = extract_audio_features("vocals.wav")

        assert features == cached_features

    @patch("y2karaoke.core.audio_analysis._load_audio_features_cache")
    @patch("librosa.load")
    def test_handles_audio_load_error(self, mock_load, mock_load_cache):
        mock_load_cache.return_value = None
        mock_load.side_effect = Exception("Audio load failed")

        features = extract_audio_features("nonexistent.wav")

        assert features is None

    def test_extract_audio_features_success(self, monkeypatch, tmp_path):
        cache_path = tmp_path / "vocals.audio.npz"

        monkeypatch.setattr(
            "y2karaoke.core.audio_analysis._get_audio_features_cache_path",
            lambda _path: str(cache_path),
        )
        monkeypatch.setattr(
            "y2karaoke.core.audio_analysis._load_audio_features_cache",
            lambda _path: None,
        )

        saved = {}

        def fake_save(path, features):
            saved["path"] = path
            saved["features"] = features

        monkeypatch.setattr(
            "y2karaoke.core.audio_analysis._save_audio_features_cache",
            fake_save,
        )

        def fake_load(_path, sr=22050):
            y = np.array([0.0, 0.5, 0.0, 0.2], dtype=float)
            return y, sr

        def fake_onset_detect(**_kwargs):
            return np.array([1, 3])

        def fake_frames_to_time(frames, sr, hop_length):
            return np.array(frames, dtype=float) * (hop_length / sr)

        def fake_rms(**_kwargs):
            return np.array([[0.1, 0.3, 0.05, 0.0]], dtype=float)

        fake_librosa = SimpleNamespace(
            load=fake_load,
            onset=SimpleNamespace(onset_detect=fake_onset_detect),
            frames_to_time=fake_frames_to_time,
            feature=SimpleNamespace(rms=fake_rms),
        )

        monkeypatch.setattr(
            "y2karaoke.core.audio_analysis._load_librosa", lambda: fake_librosa
        )

        features = extract_audio_features("vocals.wav", min_silence_duration=0.1)

        assert features is not None
        assert saved["path"] == str(cache_path)
        assert features.duration == pytest.approx(4 / 22050)
