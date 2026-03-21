import numpy as np
import pytest

from y2karaoke.core.components.alignment.timing_models import (
    AudioFeatures,
    TranscriptionWord,
)
from y2karaoke.core.components.whisper import (
    whisper_integration_align_experimental as wiaexp,
)
from y2karaoke.core.models import Line, Word


def test_reanchor_low_support_lines_to_later_onset_helper():
    baseline = [
        Line(words=[Word(text="prev", start_time=49.7, end_time=51.2)]),
        Line(
            words=[
                Word(text="No", start_time=53.53, end_time=53.92),
                Word(text="one's", start_time=53.92, end_time=54.31),
                Word(text="around", start_time=54.31, end_time=54.7),
                Word(text="to", start_time=54.7, end_time=55.1),
                Word(text="judge", start_time=55.1, end_time=55.5),
                Word(text="me", start_time=55.5, end_time=55.88),
            ]
        ),
        Line(words=[Word(text="next", start_time=56.29, end_time=60.58)]),
    ]
    mapped = list(baseline)
    whisper_words = [
        TranscriptionWord(text="Gå", start=60.0, end=60.24, probability=0.09),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([54.08, 54.77], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=200.0,
        duration=200.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

    updated, applied = wiaexp.reanchor_low_support_lines_to_later_onset(
        mapped,
        baseline,
        whisper_words,
        audio_features,
    )

    assert applied == 1
    assert updated[1].start_time == pytest.approx(54.08, abs=0.01)


def test_reanchor_low_support_lines_to_later_onset_blocks_when_lexical_support_exists():
    baseline = [
        Line(words=[Word(text="prev", start_time=49.7, end_time=51.2)]),
        Line(
            words=[
                Word(text="No", start_time=53.53, end_time=53.92),
                Word(text="one's", start_time=53.92, end_time=54.31),
                Word(text="around", start_time=54.31, end_time=54.7),
                Word(text="to", start_time=54.7, end_time=55.1),
                Word(text="judge", start_time=55.1, end_time=55.5),
                Word(text="me", start_time=55.5, end_time=55.88),
            ]
        ),
        Line(words=[Word(text="next", start_time=56.29, end_time=60.58)]),
    ]
    mapped = list(baseline)
    whisper_words = [
        TranscriptionWord(text="judge", start=54.9, end=55.2, probability=0.8),
        TranscriptionWord(text="me", start=55.2, end=55.5, probability=0.8),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([54.08, 54.77], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=200.0,
        duration=200.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

    updated, applied = wiaexp.reanchor_low_support_lines_to_later_onset(
        mapped,
        baseline,
        whisper_words,
        audio_features,
    )

    assert applied == 0
    assert updated[1].start_time == pytest.approx(53.53, abs=0.01)


def test_reanchor_low_support_lines_to_later_onset_blocks_already_shifted_line():
    baseline = [
        Line(words=[Word(text="prev", start_time=150.59, end_time=156.41)]),
        Line(
            words=[
                Word(text="No", start_time=156.57, end_time=156.9),
                Word(text="I", start_time=156.9, end_time=157.2),
                Word(text="can't", start_time=157.2, end_time=157.5),
                Word(text="sleep", start_time=157.5, end_time=157.8),
                Word(text="until", start_time=157.8, end_time=158.2),
                Word(text="I", start_time=158.2, end_time=158.5),
            ]
        ),
        Line(words=[Word(text="next", start_time=160.79, end_time=161.2)]),
    ]
    mapped = [
        baseline[0],
        Line(
            words=[
                Word(text="No", start_time=157.11, end_time=157.44),
                Word(text="I", start_time=157.44, end_time=157.74),
                Word(text="can't", start_time=157.74, end_time=158.04),
                Word(text="sleep", start_time=158.04, end_time=158.34),
                Word(text="until", start_time=158.34, end_time=158.74),
                Word(text="I", start_time=158.74, end_time=159.04),
            ]
        ),
        baseline[2],
    ]
    whisper_words = [
        TranscriptionWord(text="[VOCAL]", start=157.11, end=157.2, probability=0.9),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([157.11, 157.43], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=200.0,
        duration=200.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

    updated, applied = wiaexp.reanchor_low_support_lines_to_later_onset(
        mapped,
        baseline,
        whisper_words,
        audio_features,
    )

    assert applied == 0
    assert updated[1].start_time == pytest.approx(157.11, abs=0.01)


def test_reanchor_repeated_cadence_lines_borrows_later_pair_spacing():
    lines = [
        Line(words=[Word(text="No", start_time=53.53, end_time=55.88)]),
        Line(
            words=[
                Word(text="I", start_time=56.29, end_time=56.89),
                Word(text="can't", start_time=56.89, end_time=57.49),
                Word(text="see", start_time=57.49, end_time=58.09),
                Word(text="clearly", start_time=58.09, end_time=58.69),
                Word(text="when", start_time=58.69, end_time=59.29),
                Word(text="you're", start_time=59.29, end_time=59.89),
                Word(text="gone", start_time=59.89, end_time=60.58),
            ]
        ),
        Line(words=[Word(text="bridge", start_time=61.5, end_time=62.0)]),
        Line(words=[Word(text="No", start_time=109.5, end_time=112.39)]),
        Line(
            words=[
                Word(text="I", start_time=112.69, end_time=113.09),
                Word(text="can't", start_time=113.09, end_time=113.49),
                Word(text="see", start_time=113.49, end_time=113.89),
                Word(text="clearly", start_time=113.89, end_time=114.29),
                Word(text="when", start_time=114.29, end_time=114.69),
                Word(text="you're", start_time=114.69, end_time=115.09),
                Word(text="gone", start_time=115.09, end_time=115.4),
            ]
        ),
    ]
    lines[0].words[0].text = "No one's around to judge me"
    lines[1].words[0].text = "I can't see clearly when you're gone"
    lines[3].words[0].text = "No one's around to judge me"
    lines[4].words[0].text = "I can't see clearly when you're gone"

    adjusted, applied = wiaexp.reanchor_repeated_cadence_lines(lines)

    assert applied == 1
    assert adjusted[1].start_time == pytest.approx(56.72, abs=0.01)


def test_reanchor_late_supported_lines_to_earlier_whisper_moves_line_earlier():
    lines = [
        Line(words=[Word(text="prev", start_time=16.43, end_time=19.58)]),
        Line(
            words=[
                Word(text="I", start_time=20.28, end_time=20.48),
                Word(text="like", start_time=20.5, end_time=20.7),
                Word(text="your", start_time=20.72, end_time=20.92),
                Word(text="poom-poom", start_time=20.94, end_time=21.14),
            ]
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="I", start=19.44, end=19.82, probability=1.0),
        TranscriptionWord(text="like", start=19.82, end=20.02, probability=1.0),
        TranscriptionWord(text="you", start=20.02, end=20.16, probability=1.0),
    ]

    adjusted, applied = wiaexp.reanchor_late_supported_lines_to_earlier_whisper(
        lines, whisper_words
    )

    assert applied == 1
    assert adjusted[1].start_time == pytest.approx(19.63, abs=0.01)
    assert adjusted[1].end_time == pytest.approx(21.14, abs=0.01)


def test_reanchor_late_supported_lines_to_earlier_whisper_requires_prefix_support():
    lines = [
        Line(words=[Word(text="prev", start_time=10.0, end_time=10.8)]),
        Line(
            words=[
                Word(text="Ya", start_time=12.0, end_time=12.3),
                Word(text="vi", start_time=12.3, end_time=12.6),
                Word(text="que", start_time=12.6, end_time=12.9),
            ]
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="ya", start=11.2, end=11.5, probability=1.0),
        TranscriptionWord(text="solo", start=11.5, end=11.8, probability=1.0),
    ]

    adjusted, applied = wiaexp.reanchor_late_supported_lines_to_earlier_whisper(
        lines, whisper_words
    )

    assert applied == 0
    assert adjusted[1].start_time == pytest.approx(12.0, abs=0.01)


def test_shift_restored_low_support_runs_to_onset_moves_dense_run_together():
    baseline = [
        Line(words=[Word(text="prev", start_time=50.72, end_time=52.98)]),
        Line(
            words=[
                Word(text="No", start_time=53.53, end_time=53.92),
                Word(text="one's", start_time=53.92, end_time=54.31),
                Word(text="around", start_time=54.31, end_time=54.7),
                Word(text="to", start_time=54.7, end_time=55.1),
                Word(text="judge", start_time=55.1, end_time=55.5),
                Word(text="me", start_time=55.5, end_time=55.88),
            ]
        ),
        Line(
            words=[
                Word(text="I", start_time=56.29, end_time=56.69),
                Word(text="can't", start_time=56.69, end_time=57.09),
                Word(text="see", start_time=57.09, end_time=57.49),
                Word(text="clearly", start_time=57.49, end_time=57.89),
                Word(text="when", start_time=57.89, end_time=58.29),
                Word(text="you're", start_time=58.29, end_time=58.69),
                Word(text="gone", start_time=58.69, end_time=59.0),
            ]
        ),
        Line(
            words=[
                Word(text="I", start_time=60.78, end_time=61.2),
                Word(text="said", start_time=61.2, end_time=61.62),
                Word(text="ooh", start_time=61.62, end_time=62.04),
                Word(text="im", start_time=62.04, end_time=62.46),
                Word(text="blinded", start_time=62.46, end_time=63.4),
                Word(text="by", start_time=63.4, end_time=63.8),
                Word(text="the", start_time=63.8, end_time=64.2),
                Word(text="lights", start_time=64.2, end_time=66.52),
            ]
        ),
        Line(words=[Word(text="next", start_time=67.5, end_time=71.4)]),
    ]
    whisper_words = [
        TranscriptionWord(text="[VOCAL]", start=54.0, end=54.1, probability=0.9),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([54.08, 54.45, 54.63], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=200.0,
        duration=200.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

    adjusted, applied = wiaexp.shift_restored_low_support_runs_to_onset(
        baseline,
        baseline,
        whisper_words,
        audio_features,
    )

    assert applied == 2
    assert adjusted[1].start_time == pytest.approx(54.08, abs=0.01)
    assert adjusted[2].start_time == pytest.approx(56.84, abs=0.01)
    assert adjusted[3].start_time == pytest.approx(60.78, abs=0.01)
