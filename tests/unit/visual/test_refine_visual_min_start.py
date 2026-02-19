import pytest

from y2karaoke.core.models import TargetLine
from y2karaoke.core.visual.refinement import _compute_line_min_start_time


def test_compute_line_min_start_time_uses_bounded_visibility_lookback():
    line = TargetLine(
        line_index=1,
        start=10.0,
        end=12.0,
        text="line",
        words=["line"],
        y=10,
        word_rois=[(0, 0, 2, 2)],
        visibility_start=20.0,
        visibility_end=30.0,
    )

    assert _compute_line_min_start_time(
        line,
        last_assigned_start=None,
        last_assigned_visibility_end=None,
    ) == pytest.approx(18.0, abs=1e-6)
    assert _compute_line_min_start_time(
        line,
        last_assigned_start=19.2,
        last_assigned_visibility_end=None,
    ) == pytest.approx(19.25, abs=1e-6)


def test_compute_line_min_start_time_skips_global_gate_for_overlapping_visibility():
    line = TargetLine(
        line_index=1,
        start=100.0,
        end=101.0,
        text="line",
        words=["line"],
        y=10,
        word_rois=[(0, 0, 2, 2)],
        visibility_start=216.0,
        visibility_end=223.3,
    )

    assert _compute_line_min_start_time(
        line,
        last_assigned_start=223.5,
        last_assigned_visibility_end=233.2,
    ) == pytest.approx(215.0, abs=1e-6)
