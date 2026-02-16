import numpy as np

import y2karaoke.core.components.alignment.timing_evaluator as te
import y2karaoke.core.components.alignment.timing_evaluator_core as te_core
import y2karaoke.core.audio_analysis as aa
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
    onset, delta = te_core._find_closest_onset(2.0, onset_times, max_distance=2.0)
    assert onset == 1.0
    assert delta == 1.0


def test_find_closest_onset_out_of_range():
    onset_times = np.array([10.0])
    onset, delta = te_core._find_closest_onset(1.0, onset_times, max_distance=2.0)
    assert onset is None
    assert delta == 0.0


def test_check_vocal_activity_in_range():
    features = _features(
        energy_times=[0, 1, 2, 3, 4],
        energy_envelope=[0.0, 0.5, 0.6, 0.0, 0.0],
    )
    activity = aa._check_vocal_activity_in_range(1.0, 3.0, features)
    assert activity > 0.5


def test_check_for_silence_in_range_detects():
    features = _features(
        energy_times=[0, 1, 2, 3],
        energy_envelope=[1.0, 0.0, 0.0, 1.0],
    )
    assert aa._check_for_silence_in_range(0.5, 2.5, features, 0.5) is True


def test_check_for_silence_in_range_false():
    features = _features(
        energy_times=[0, 1, 2, 3],
        energy_envelope=[1.0, 1.0, 1.0, 1.0],
    )
    assert aa._check_for_silence_in_range(0.5, 2.5, features, 0.5) is False


def test_check_for_silence_in_range_short_range_returns_false():
    features = _features(
        energy_times=[0, 1, 2, 3],
        energy_envelope=[0.0, 0.0, 0.0, 0.0],
    )
    assert aa._check_for_silence_in_range(3.5, 3.6, features, 0.5) is False


def test_check_for_silence_in_range_detects_mid_range_silence():
    features = _features(
        energy_times=[0, 1, 2, 3, 4],
        energy_envelope=[1.0, 0.0, 0.0, 1.0, 1.0],
    )
    assert aa._check_for_silence_in_range(0.0, 3.5, features, 1.0) is True


def test_check_for_silence_in_range_detects_at_end():
    features = _features(
        energy_times=[0, 1, 2, 3, 4],
        energy_envelope=[1.0, 1.0, 0.0, 0.0, 0.0],
    )
    assert aa._check_for_silence_in_range(1.5, 4.0, features, 1.0) is True


def test_check_for_silence_in_range_short_silence_resets():
    features = _features(
        energy_times=[0, 1, 2, 3],
        energy_envelope=[1.0, 0.0, 1.0, 1.0],
    )
    assert aa._check_for_silence_in_range(0.0, 3.0, features, 1.5) is False


def test_check_for_silence_in_range_end_short_silence_returns_false():
    features = _features(
        energy_times=[0, 1, 2, 3],
        energy_envelope=[1.0, 1.0, 0.0, 0.0],
    )
    assert aa._check_for_silence_in_range(0.0, 3.0, features, 1.5) is False


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
    score = te_core._calculate_pause_score(lines, features)
    assert score == 100.0


def test_calculate_pause_score_skips_short_and_pre_vocal_silences():
    lines = [
        Line(words=[Word(text="a", start_time=3.5, end_time=4.0)]),
        Line(words=[Word(text="b", start_time=8.0, end_time=9.0)]),
    ]
    features = _features(
        energy_times=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        energy_envelope=[1.0] * 10,
        silence_regions=[(0.0, 1.5), (1.0, 3.5), (5.0, 7.5)],
        vocal_start=3.0,
    )

    score = te_core._calculate_pause_score(lines, features)
    assert score == 100.0


def test_calculate_pause_score_skips_empty_lines():
    lines = [
        Line(words=[]),
        Line(words=[Word(text="a", start_time=0.0, end_time=1.0)]),
        Line(words=[]),
        Line(words=[Word(text="b", start_time=4.0, end_time=5.0)]),
    ]
    features = _features(
        energy_times=[0, 1, 2, 3, 4, 5],
        energy_envelope=[1.0] * 6,
        silence_regions=[(1.5, 3.8)],
        vocal_start=0.0,
    )

    score = te_core._calculate_pause_score(lines, features)
    assert score == 0.0


def test_calculate_pause_score_no_gap_match_returns_zero():
    lines = [
        Line(words=[Word(text="a", start_time=0.0, end_time=1.0)]),
        Line(words=[Word(text="b", start_time=3.0, end_time=3.5)]),
    ]
    features = _features(
        energy_times=[0, 1, 2, 3, 4, 5, 6, 7],
        energy_envelope=[1.0] * 8,
        silence_regions=[(4.0, 7.0)],
        vocal_start=0.0,
    )

    score = te_core._calculate_pause_score(lines, features)
    assert score == 0.0


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
    issues = te_core._check_pause_alignment(lines, features)
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
    issues = te_core._check_pause_alignment(lines, features)
    assert any(issue.issue_type == "unexpected_pause" for issue in issues)


def test_check_pause_alignment_flags_missing_pause(monkeypatch):
    lines = [
        Line(words=[Word(text="a", start_time=0.0, end_time=1.0)]),
        Line(words=[Word(text="b", start_time=4.0, end_time=5.0)]),
    ]
    features = _features(
        energy_times=[0, 1, 2, 3, 4, 5],
        energy_envelope=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        silence_regions=[],
        vocal_start=0.0,
    )

    monkeypatch.setattr(
        te_core, "_check_vocal_activity_in_range", lambda *_a, **_k: 0.0
    )
    issues = te_core._check_pause_alignment(lines, features)

    assert any(issue.issue_type == "missing_pause" for issue in issues)


def test_check_pause_alignment_silence_covered_by_gap():
    lines = [
        Line(words=[Word(text="a", start_time=0.0, end_time=4.0)]),
        Line(words=[Word(text="b", start_time=7.0, end_time=8.0)]),
    ]
    features = _features(
        energy_times=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        energy_envelope=[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        silence_regions=[(5.0, 6.0)],
        vocal_start=0.0,
    )

    issues = te_core._check_pause_alignment(lines, features)
    assert not any(issue.issue_type == "unexpected_pause" for issue in issues)
    assert not any(issue.issue_type == "missing_pause" for issue in issues)


def test_check_pause_alignment_silence_covered_with_empty_lines(monkeypatch):
    lines = [
        Line(words=[]),
        Line(words=[Word(text="a", start_time=0.0, end_time=1.0)]),
        Line(words=[Word(text="b", start_time=5.0, end_time=6.0)]),
    ]
    features = _features(
        energy_times=[0, 1, 2, 3, 4, 5, 6],
        energy_envelope=[1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        silence_regions=[(2.0, 4.5)],
        vocal_start=0.0,
    )

    monkeypatch.setattr(
        te_core, "_check_vocal_activity_in_range", lambda *_a, **_k: 0.0
    )
    issues = te_core._check_pause_alignment(lines, features)
    assert not any(issue.issue_type == "unexpected_pause" for issue in issues)


def test_check_pause_alignment_skips_empty_lines():
    lines = [
        Line(words=[]),
        Line(words=[Word(text="a", start_time=1.0, end_time=2.0)]),
        Line(words=[]),
    ]
    features = _features(
        energy_times=[0, 1, 2, 3],
        energy_envelope=[1.0, 1.0, 1.0, 1.0],
    )
    issues = te_core._check_pause_alignment(lines, features)
    assert issues == []


def test_generate_summary_includes_issue_count():
    summary = te_core._generate_summary(55.0, 60.0, 50.0, 45.0, 0.1, 0.2, 3, 10)
    assert "Issues found: 3" in summary
