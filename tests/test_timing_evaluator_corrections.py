import numpy as np

import y2karaoke.core.timing_evaluator as te
import y2karaoke.core.timing_evaluator_correction as te_corr
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


def test_fix_spurious_gaps_skips_large_gap(monkeypatch):
    lines = [
        Line(words=[Word(text="hello", start_time=0.0, end_time=0.5)]),
        Line(words=[Word(text="world", start_time=3.0, end_time=3.5)]),
    ]
    features = _features(
        onset_times=[],
        energy_times=[0.0, 1.0, 2.0, 3.0, 4.0],
        energy_envelope=[1.0, 1.0, 1.0, 1.0, 1.0],
    )

    monkeypatch.setattr(te_corr, "_check_vocal_activity_in_range", lambda *_a, **_k: 0.9)
    monkeypatch.setattr(te_corr, "_check_for_silence_in_range", lambda *_a, **_k: True)

    fixed, fixes = te.fix_spurious_gaps(lines, features)

    assert fixed == lines
    assert fixes == []


def test_fix_spurious_gaps_keeps_empty_line():
    lines = [
        Line(words=[]),
        Line(words=[Word(text="hi", start_time=1.0, end_time=1.5)]),
    ]
    features = _features(
        onset_times=[],
        energy_times=[0.0, 1.0, 2.0],
        energy_envelope=[1.0, 1.0, 1.0],
    )

    fixed, fixes = te.fix_spurious_gaps(lines, features)

    assert fixed[0] is lines[0]
    assert fixed[1] is lines[1]
    assert fixes == []


def test_fix_spurious_gaps_breaks_on_low_mid_activity(monkeypatch):
    lines = [
        Line(words=[Word(text="hello", start_time=0.0, end_time=0.5)]),
        Line(words=[Word(text="world", start_time=1.0, end_time=1.5)]),
    ]
    features = _features(
        onset_times=[],
        energy_times=[0.0, 0.5, 1.0, 1.5],
        energy_envelope=[1.0, 1.0, 1.0, 1.0],
    )

    monkeypatch.setattr(te_corr, "_check_vocal_activity_in_range", lambda *_a, **_k: 0.5)

    fixed, fixes = te.fix_spurious_gaps(lines, features)

    assert fixed == lines
    assert fixes == []


def test_fix_spurious_gaps_merge_uses_next_line_start_and_min_duration(monkeypatch):
    lines = [
        Line(words=[Word(text="a", start_time=0.0, end_time=0.4)]),
        Line(words=[Word(text="b", start_time=0.9, end_time=1.3)]),
        Line(words=[Word(text="c", start_time=5.0, end_time=5.4)]),
    ]
    features = _features(
        onset_times=[],
        energy_times=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        energy_envelope=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )

    def _activity_for_range(start, end, *_a, **_k):
        # Encourage merge for the short gap (0.4 -> 0.9) and discourage long gaps.
        return 0.8 if end - start <= 2.0 else 0.2

    monkeypatch.setattr(te_corr, "_check_vocal_activity_in_range", _activity_for_range)
    monkeypatch.setattr(te_corr, "_check_for_silence_in_range", lambda *_a, **_k: True)
    monkeypatch.setattr(te_corr, "_find_phrase_end", lambda *_a, **_k: 0.1)

    fixed, fixes = te.fix_spurious_gaps(lines, features)

    assert len(fixed) == 2
    assert len(fixes) == 1
    assert len(fixed[0].words) == 2


def test_transcription_segment_defaults_words_to_empty_list():
    segment = te.TranscriptionSegment(start=0.0, end=1.0, text="hi", words=None)
    assert segment.words == []
