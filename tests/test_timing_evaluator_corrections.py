import numpy as np

import y2karaoke.core.timing_evaluator as te
from y2karaoke.core.models import Line, Word


def _features(onset_times, energy_times, energy_envelope):
    return te.AudioFeatures(
        onset_times=np.array(onset_times),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=energy_times[-1] if energy_times else 0.0,
        duration=energy_times[-1] if energy_times else 0.0,
        energy_envelope=np.array(energy_envelope),
        energy_times=np.array(energy_times),
    )


def test_correct_line_timestamps_shifts_to_onset():
    lines = [Line(words=[Word(text="hi", start_time=1.0, end_time=1.4)])]
    features = _features(
        onset_times=[2.0],
        energy_times=[0.0, 1.0, 2.0, 3.0],
        energy_envelope=[1.0, 1.0, 0.0, 0.0],
    )

    corrected, notes = te.correct_line_timestamps(lines, features, max_correction=3.0)

    assert corrected[0].words[0].start_time == 2.0
    assert corrected[0].words[0].end_time == 2.4
    assert notes


def test_find_phrase_end_returns_silence_start():
    features = _features(
        onset_times=[],
        energy_times=[0.0, 1.0, 2.0, 3.0],
        energy_envelope=[1.0, 0.0, 0.0, 1.0],
    )
    phrase_end = te._find_phrase_end(0.0, 3.0, features, min_silence_duration=0.5)
    assert phrase_end == 1.0


def test_find_phrase_end_falls_back_to_max_end():
    features = _features(
        onset_times=[],
        energy_times=[0.0, 1.0, 2.0],
        energy_envelope=[1.0, 1.0, 1.0],
    )
    phrase_end = te._find_phrase_end(0.0, 2.0, features, min_silence_duration=0.5)
    assert phrase_end == 2.0


def test_fix_spurious_gaps_merges_continuous_lines():
    lines = [
        Line(words=[Word(text="hello", start_time=0.0, end_time=0.5)]),
        Line(words=[Word(text="world", start_time=1.0, end_time=1.5)]),
    ]
    features = _features(
        onset_times=[],
        energy_times=[0.0, 0.5, 0.7, 0.8, 1.0, 1.5, 2.0, 2.5],
        energy_envelope=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
    )

    fixed, fixes = te.fix_spurious_gaps(lines, features)

    assert len(fixed) == 1
    assert len(fixes) == 1
    assert len(fixed[0].words) == 2
    assert fixed[0].words[0].start_time == 0.0
    assert fixed[0].words[1].start_time == 1.0


def test_fix_spurious_gaps_skips_large_gap():
    lines = [
        Line(words=[Word(text="hello", start_time=0.0, end_time=0.5)]),
        Line(words=[Word(text="world", start_time=3.0, end_time=3.5)]),
    ]
    features = _features(
        onset_times=[],
        energy_times=[0.0, 1.0, 2.0, 3.0, 4.0],
        energy_envelope=[1.0, 1.0, 1.0, 1.0, 1.0],
    )

    fixed, fixes = te.fix_spurious_gaps(lines, features)

    assert fixed == lines
    assert fixes == []
