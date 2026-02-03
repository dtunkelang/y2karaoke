import numpy as np

import y2karaoke.core.timing_evaluator as te
from y2karaoke.core.models import Line, Word


def _features(energy_times, energy_envelope, silence_regions=None, vocal_start=0.0):
    return te.AudioFeatures(
        onset_times=np.array([]),
        silence_regions=silence_regions or [],
        vocal_start=vocal_start,
        vocal_end=energy_times[-1] if len(energy_times) else 0.0,
        duration=energy_times[-1] if len(energy_times) else 0.0,
        energy_envelope=np.array(energy_envelope),
        energy_times=np.array(energy_times),
    )


def test_find_closest_onset_within_distance():
    onset_times = np.array([1.0, 3.0])
    onset, delta = te._find_closest_onset(2.0, onset_times, max_distance=2.0)
    assert onset == 1.0
    assert delta == 1.0


def test_find_closest_onset_out_of_range():
    onset_times = np.array([10.0])
    onset, delta = te._find_closest_onset(1.0, onset_times, max_distance=2.0)
    assert onset is None
    assert delta == 0.0


def test_check_vocal_activity_in_range():
    features = _features(
        energy_times=[0, 1, 2, 3, 4],
        energy_envelope=[0.0, 0.5, 0.6, 0.0, 0.0],
    )
    activity = te._check_vocal_activity_in_range(1.0, 3.0, features)
    assert activity > 0.5


def test_check_for_silence_in_range_detects():
    features = _features(
        energy_times=[0, 1, 2, 3],
        energy_envelope=[1.0, 0.0, 0.0, 1.0],
    )
    assert te._check_for_silence_in_range(0.5, 2.5, features, 0.5) is True


def test_check_for_silence_in_range_false():
    features = _features(
        energy_times=[0, 1, 2, 3],
        energy_envelope=[1.0, 1.0, 1.0, 1.0],
    )
    assert te._check_for_silence_in_range(0.5, 2.5, features, 0.5) is False


def test_calculate_pause_score_matches_silence():
    lines = [
        Line(words=[Word(text="a", start_time=0.0, end_time=1.0)]),
        Line(words=[Word(text="b", start_time=5.0, end_time=6.0)]),
    ]
    features = _features(
        energy_times=[0, 1, 2, 3, 4, 5, 6],
        energy_envelope=[1, 1, 0, 0, 0, 1, 1],
        silence_regions=[(1.5, 4.5)],
        vocal_start=0.0,
    )
    score = te._calculate_pause_score(lines, features)
    assert score == 100.0


def test_check_pause_alignment_flags_spurious_gap():
    lines = [
        Line(words=[Word(text="a", start_time=0.0, end_time=1.0)]),
        Line(words=[Word(text="b", start_time=3.0, end_time=4.0)]),
    ]
    features = _features(
        energy_times=[0, 1, 2, 3, 4],
        energy_envelope=[1.0, 1.0, 1.0, 1.0, 1.0],
        silence_regions=[],
    )
    issues = te._check_pause_alignment(lines, features)
    assert any(issue.issue_type == "spurious_gap" for issue in issues)


def test_check_pause_alignment_flags_unexpected_pause():
    lines = [
        Line(words=[Word(text="a", start_time=0.0, end_time=1.0)]),
        Line(words=[Word(text="b", start_time=2.0, end_time=3.0)]),
    ]
    features = _features(
        energy_times=[0, 1, 2, 3, 4],
        energy_envelope=[1.0, 0.0, 0.0, 1.0, 1.0],
        silence_regions=[(1.2, 3.4)],
        vocal_start=0.0,
    )
    issues = te._check_pause_alignment(lines, features)
    assert any(issue.issue_type == "unexpected_pause" for issue in issues)


def test_generate_summary_includes_issue_count():
    summary = te._generate_summary(55.0, 60.0, 50.0, 0.1, 0.2, 3, 10)
    assert "Issues found: 3" in summary
