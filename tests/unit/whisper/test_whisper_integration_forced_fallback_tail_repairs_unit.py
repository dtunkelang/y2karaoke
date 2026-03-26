import pytest

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
