import numpy as np
import pytest

from y2karaoke.core.models import Line, Word
import y2karaoke.core.components.alignment.timing_evaluator as te


def _line(text, start, end):
    word = Word(text=text, start_time=start, end_time=end)
    return Line(words=[word])


def _audio_features(energy_times, energy_envelope, silence_regions=None):
    return te.AudioFeatures(
        onset_times=np.array([]),
        silence_regions=silence_regions or [],
        vocal_start=0.0,
        vocal_end=energy_times[-1] if len(energy_times) else 0.0,
        duration=energy_times[-1] if len(energy_times) else 0.0,
        energy_envelope=np.array(energy_envelope, dtype=float),
        energy_times=np.array(energy_times, dtype=float),
    )


def test_check_vocal_activity_in_range():
    features = _audio_features([0, 1, 2, 3], [0.0, 0.5, 1.0, 0.0])

    activity = te._check_vocal_activity_in_range(0.0, 2.0, features)

    assert activity == pytest.approx(2 / 3)


def test_check_for_silence_in_range_detects_silence():
    features = _audio_features([0, 1, 2, 3], [0.0, 0.0, 0.5, 0.5])

    assert te._check_for_silence_in_range(0.0, 2.0, features, min_silence_duration=1.0)


def test_check_for_silence_in_range_trailing_silence():
    features = _audio_features([0, 1, 2, 3], [1.0, 0.0, 0.0, 0.0])

    assert te._check_for_silence_in_range(1.0, 3.0, features, min_silence_duration=1.0)


def test_check_pause_alignment_spurious_gap():
    lines = [_line("a", 0.0, 1.0), _line("b", 3.0, 4.0)]
    features = _audio_features([0, 1, 2, 3, 4], [1.0, 1.0, 1.0, 1.0, 1.0])

    issues = te._check_pause_alignment(lines, features)

    assert any(issue.issue_type == "spurious_gap" for issue in issues)


def test_check_pause_alignment_missing_pause():
    lines = [_line("a", 0.0, 1.0), _line("b", 3.5, 4.0)]
    features = _audio_features([0, 1, 2, 3, 4], [0.0, 0.0, 0.0, 0.0, 0.0])

    issues = te._check_pause_alignment(lines, features)

    assert any(issue.issue_type == "missing_pause" for issue in issues)


def test_check_pause_alignment_unexpected_pause():
    lines = [_line("a", 0.0, 1.0), _line("b", 1.2, 2.0)]
    features = _audio_features(
        [0, 1, 2, 3, 4],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        silence_regions=[(2.0, 4.5)],
    )

    issues = te._check_pause_alignment(lines, features)

    assert any(issue.issue_type == "unexpected_pause" for issue in issues)


def test_calculate_pause_score_no_silence_regions():
    lines = [_line("a", 0.0, 1.0), _line("b", 1.2, 2.0)]
    features = _audio_features([0, 1, 2], [1.0, 1.0, 1.0], silence_regions=[])

    assert te._calculate_pause_score(lines, features) == 100.0


def test_calculate_pause_score_ignores_pre_vocal_silence():
    lines = [_line("a", 1.0, 1.5), _line("b", 2.0, 2.5)]
    features = _audio_features(
        [0, 1, 2, 3],
        [1.0, 1.0, 1.0, 1.0],
        silence_regions=[(0.0, 2.5)],
    )
    features.vocal_start = 2.0

    assert te._calculate_pause_score(lines, features) == 100.0


def test_evaluate_timing_builds_report():
    lines = [_line("a", 0.9, 1.2), _line("b", 2.4, 2.6)]
    features = te.AudioFeatures(
        onset_times=np.array([0.0, 2.0]),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=3.0,
        duration=3.0,
        energy_envelope=np.array([1.0, 1.0, 1.0]),
        energy_times=np.array([0.0, 1.5, 3.0]),
    )

    report = te.evaluate_timing(lines, features, source_name="test")

    assert report.source_name == "test"
    assert report.total_lines == 2
    assert "Timing quality" in report.summary
    assert any(
        issue.issue_type in ("early_line", "late_line") for issue in report.issues
    )
