import pytest
import numpy as np

from y2karaoke.core.components.alignment.timing_models import AudioFeatures
from y2karaoke.core.components.whisper import whisper_forced_tail_repairs as _tails
from y2karaoke.core.models import Line, Word


def test_extend_low_score_forced_line_tails_from_source_extends_sweet_caroline_tail():
    baseline_lines = [
        Line(
            words=[
                Word(text="I've", start_time=11.95, end_time=12.833),
                Word(text="been", start_time=12.833, end_time=13.717),
                Word(text="inclined", start_time=13.717, end_time=14.6),
            ]
        ),
        Line(
            words=[
                Word(text="To", start_time=15.382, end_time=15.826),
                Word(text="believe", start_time=15.826, end_time=16.715),
                Word(text="they", start_time=16.715, end_time=17.604),
                Word(text="never", start_time=17.604, end_time=18.715),
                Word(text="would", start_time=18.715, end_time=19.826),
            ]
        ),
        Line(
            words=[
                Word(text="But", start_time=20.42, end_time=21.149),
                Word(text="now", start_time=21.149, end_time=21.877),
                Word(text="I", start_time=21.877, end_time=22.606),
                Word(text="look", start_time=22.606, end_time=23.334),
                Word(text="at", start_time=23.334, end_time=24.063),
                Word(text="the", start_time=24.063, end_time=24.791),
                Word(text="night", start_time=24.791, end_time=25.52),
            ]
        ),
    ]
    forced_lines = [
        Line(
            words=[
                Word(text="I've", start_time=12.741, end_time=13.202),
                Word(text="been", start_time=13.202, end_time=13.664),
                Word(text="inclined", start_time=13.664, end_time=14.125),
            ]
        ),
        Line(
            words=[
                Word(text="To", start_time=16.656, end_time=16.957),
                Word(text="believe", start_time=16.997, end_time=18.06),
                Word(text="they", start_time=18.12, end_time=18.541),
                Word(text="never", start_time=18.581, end_time=18.962),
                Word(text="would", start_time=18.982, end_time=19.183),
            ]
        ),
        baseline_lines[2],
    ]
    aligned_segments = [
        {},
        {
            "words": [
                {"word": "To", "score": 0.633},
                {"word": "believe", "score": 0.782},
                {"word": "they", "score": 0.722},
                {"word": "never", "score": 0.8},
                {"word": "would", "score": 0.515},
            ]
        },
        {},
    ]

    repaired_lines, extended = _tails.extend_low_score_forced_line_tails_from_source(
        baseline_lines,
        forced_lines,
        aligned_segments,
    )

    assert extended == 1
    assert repaired_lines[1].start_time == pytest.approx(forced_lines[1].start_time)
    assert repaired_lines[1].end_time == pytest.approx(19.826, abs=0.01)


def test_extend_low_score_forced_line_tails_from_source_skips_high_score_tails():
    baseline_lines = [
        Line(
            words=[
                Word(text="To", start_time=15.382, end_time=15.826),
                Word(text="believe", start_time=15.826, end_time=16.715),
                Word(text="they", start_time=16.715, end_time=17.604),
                Word(text="never", start_time=17.604, end_time=18.715),
                Word(text="would", start_time=18.715, end_time=19.826),
            ]
        ),
    ]
    forced_lines = [
        Line(
            words=[
                Word(text="To", start_time=16.656, end_time=16.957),
                Word(text="believe", start_time=16.997, end_time=18.06),
                Word(text="they", start_time=18.12, end_time=18.541),
                Word(text="never", start_time=18.581, end_time=18.962),
                Word(text="would", start_time=18.982, end_time=19.183),
            ]
        ),
    ]
    aligned_segments = [
        {
            "words": [
                {"word": "To", "score": 0.633},
                {"word": "believe", "score": 0.782},
                {"word": "they", "score": 0.722},
                {"word": "never", "score": 0.8},
                {"word": "would", "score": 0.8},
            ]
        }
    ]

    repaired_lines, extended = _tails.extend_low_score_forced_line_tails_from_source(
        baseline_lines,
        forced_lines,
        aligned_segments,
    )

    assert extended == 0
    assert repaired_lines[0].end_time == pytest.approx(forced_lines[0].end_time)


def test_extend_final_held_tail_lines_from_activity_extends_stayin_alive_tail():
    baseline_lines = [
        Line(
            words=[
                Word(text="Ah,", start_time=1.3, end_time=1.8),
                Word(text="ha,", start_time=1.8, end_time=2.4),
                Word(text="ha,", start_time=2.4, end_time=3.0),
                Word(text="ha,", start_time=3.0, end_time=3.6),
                Word(text="stayin'", start_time=3.6, end_time=4.6),
                Word(text="alive", start_time=4.6, end_time=5.7),
            ]
        ),
        Line(
            words=[
                Word(text="Ah,", start_time=5.85, end_time=6.55),
                Word(text="ha,", start_time=6.55, end_time=7.25),
                Word(text="ha,", start_time=7.25, end_time=7.95),
                Word(text="ha,", start_time=7.95, end_time=8.65),
                Word(text="stayin'", start_time=8.65, end_time=10.6),
                Word(text="alive", start_time=10.6, end_time=16.0),
            ]
        ),
    ]
    forced_lines = [
        baseline_lines[0],
        Line(
            words=[
                Word(text="Ah,", start_time=5.844, end_time=6.164),
                Word(text="ha,", start_time=6.485, end_time=6.726),
                Word(text="ha,", start_time=7.046, end_time=7.307),
                Word(text="ha,", start_time=7.648, end_time=7.888),
                Word(text="stayin'", start_time=8.189, end_time=8.53),
                Word(text="alive", start_time=8.59, end_time=8.851),
            ]
        ),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=15.9,
        duration=16.0,
        energy_envelope=np.ones(200, dtype=float),
        energy_times=np.linspace(0.0, 16.0, 200),
    )

    repaired_lines, extended = _tails.extend_final_held_tail_lines_from_activity(
        baseline_lines,
        forced_lines,
        audio_features,
    )

    assert extended == 1
    assert repaired_lines[1].start_time == pytest.approx(5.844, abs=0.01)
    assert repaired_lines[1].end_time == pytest.approx(15.9, abs=0.01)


def test_extend_final_held_tail_lines_from_activity_skips_short_take_on_me_tail():
    baseline_lines = [
        Line(
            words=[
                Word(text="In", start_time=17.37, end_time=18.18),
                Word(text="a", start_time=18.18, end_time=18.99),
                Word(text="day", start_time=18.99, end_time=19.8),
                Word(text="or", start_time=19.8, end_time=20.61),
                Word(text="two", start_time=20.61, end_time=21.42),
            ]
        )
    ]
    forced_lines = [
        Line(
            words=[
                Word(text="In", start_time=17.2, end_time=17.36),
                Word(text="a", start_time=17.36, end_time=17.74),
                Word(text="day", start_time=17.74, end_time=18.02),
                Word(text="or", start_time=18.02, end_time=18.3),
                Word(text="two", start_time=18.3, end_time=18.71),
            ]
        )
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=21.92,
        duration=22.0,
        energy_envelope=np.ones(200, dtype=float),
        energy_times=np.linspace(0.0, 22.0, 200),
    )

    repaired_lines, extended = _tails.extend_final_held_tail_lines_from_activity(
        baseline_lines,
        forced_lines,
        audio_features,
    )

    assert extended == 0
    assert repaired_lines[0].end_time == pytest.approx(18.71, abs=0.01)
