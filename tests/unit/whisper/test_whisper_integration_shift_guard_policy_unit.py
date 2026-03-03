from y2karaoke.core.models import Line, Word
from y2karaoke.core.components.whisper.whisper_integration_shift_guard import (
    should_apply_baseline_constraint,
)


def _line(start: float) -> Line:
    return Line(words=[Word(text="x", start_time=start, end_time=start + 0.2)])


def test_should_apply_baseline_constraint_false_for_strong_global_shift():
    mapped = [_line(10.0), _line(20.0), _line(30.0)]
    baseline = [_line(7.0), _line(17.0), _line(27.0)]
    apply, median_shift = should_apply_baseline_constraint(
        mapped,
        baseline,
        matched_ratio=0.8,
        line_coverage=0.9,
    )
    assert apply is False
    assert median_shift == 3.0


def test_should_apply_baseline_constraint_true_when_shift_is_too_large():
    mapped = [_line(25.0), _line(40.0)]
    baseline = [_line(5.0), _line(20.0)]
    apply, median_shift = should_apply_baseline_constraint(
        mapped,
        baseline,
        matched_ratio=0.8,
        line_coverage=0.9,
    )
    assert apply is True
    assert median_shift == 20.0
