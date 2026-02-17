import numpy as np
import pytest
from typing import List, Dict, Any

from y2karaoke.core.visual.refinement import (
    _assign_line_level_word_timings,
    _detect_highlight_times,
    _detect_highlight_with_confidence,
    _build_line_refinement_jobs,
    _merge_line_refinement_jobs,
    _refine_line_with_frames,
    _slice_frames_for_window,
    refine_line_timings_at_low_fps,
)
from y2karaoke.core.models import TargetLine


def _create_word_vals(
    times: List[float], colors: List[List[float]]
) -> List[Dict[str, Any]]:
    return [
        {"t": t, "avg": np.array(c, dtype=np.float32), "mask": None, "lab": None}
        for t, c in zip(times, colors)
    ]


def test_detect_highlight_times_returns_none_for_short_sequence():
    vals = _create_word_vals([0.1] * 5, [[0, 0, 0]] * 5)
    s, e = _detect_highlight_times(vals)
    assert s is None
    assert e is None


def test_detect_highlight_times_returns_none_for_flat_color():
    times = np.linspace(0, 2, 20)
    colors = [[10.0, 128.0, 128.0]] * 20  # Constant color
    vals = _create_word_vals(times, colors)
    s, e = _detect_highlight_times(vals)
    assert s is None
    assert e is None


def test_detect_highlight_times_detects_simple_transition():
    # Simulate:
    # 0.0 - 1.0: Inactive (Color A - Bright)
    # 1.0 - 1.5: Transition
    # 1.5 - 2.0: Active (Color B - Dark/Color)

    times = np.linspace(0, 2.0, 41)  # 0.05s intervals
    colors = []

    # Logic assumes Peak (Bright) -> Valley (Dark)
    color_a = np.array([90.0, 128.0, 128.0])  # Bright
    color_b = np.array([10.0, 128.0, 128.0])  # Darker

    for t in times:
        if t <= 1.0:
            c = color_a
        elif t >= 1.5:
            c = color_b
        else:
            # Linear interp
            ratio = (t - 1.0) / 0.5
            c = color_a + (color_b - color_a) * ratio
        colors.append(c)

    vals = _create_word_vals(times, colors)

    s, e = _detect_highlight_times(vals)

    assert s is not None
    assert e is not None

    # Start trigger: Consistent departure from noise floor
    # Noise is 0 here. Departure from initial state happens at 1.05.
    assert 1.0 <= s <= 1.2

    # End trigger: When color becomes closer to final than initial (Crossover)
    # Transition 1.0 -> 1.5. Midpoint is 1.25.
    # So e should be around 1.25 or 1.3.
    assert 1.2 <= e <= 1.4


def test_detect_highlight_times_handles_noisy_data():
    np.random.seed(42)
    times = np.linspace(0, 2.0, 41)
    colors = []
    color_a = np.array([90.0, 128.0, 128.0])
    color_b = np.array([10.0, 128.0, 128.0])

    for t in times:
        if t <= 1.0:
            c = color_a
        elif t >= 1.5:
            c = color_b
        else:
            ratio = (t - 1.0) / 0.5
            c = color_a + (color_b - color_a) * ratio

        # Add noise
        noise = np.random.normal(0, 0.5, 3)
        colors.append(c + noise)

    vals = _create_word_vals(times, colors)
    s, e = _detect_highlight_times(vals)

    assert s is not None
    assert e is not None
    assert 1.0 <= s <= 1.25
    assert 1.2 <= e <= 1.45


def test_detect_highlight_with_confidence_reports_strength():
    times = np.linspace(0, 2.0, 41)
    colors = []
    color_a = np.array([95.0, 128.0, 128.0])
    color_b = np.array([8.0, 128.0, 128.0])
    for t in times:
        if t <= 1.0:
            c = color_a
        elif t >= 1.5:
            c = color_b
        else:
            ratio = (t - 1.0) / 0.5
            c = color_a + (color_b - color_a) * ratio
        colors.append(c)
    vals = _create_word_vals(times, colors)

    s, e, confidence = _detect_highlight_with_confidence(vals)

    assert s is not None
    assert e is not None
    assert 0.0 <= confidence <= 1.0
    assert confidence > 0.6


def test_build_line_refinement_jobs_skips_lines_without_rois():
    lines = [
        TargetLine(
            line_index=1,
            start=5.0,
            end=8.0,
            text="a",
            words=["a"],
            y=10,
            word_rois=[(0, 0, 2, 2)],
        ),
        TargetLine(
            line_index=2,
            start=9.0,
            end=11.0,
            text="b",
            words=["b"],
            y=20,
            word_rois=None,
        ),
    ]
    jobs = _build_line_refinement_jobs(lines)
    assert len(jobs) == 1
    _, v_start, v_end = jobs[0]
    assert v_start == 4.0
    assert v_end == 9.0


def test_merge_line_refinement_jobs_merges_overlaps_and_splits_distance():
    lines = [
        TargetLine(
            line_index=1,
            start=10.0,
            end=12.0,
            text="l1",
            words=["l1"],
            y=10,
            word_rois=[(0, 0, 2, 2)],
        ),
        TargetLine(
            line_index=2,
            start=12.5,
            end=14.0,
            text="l2",
            words=["l2"],
            y=20,
            word_rois=[(0, 0, 2, 2)],
        ),
        TargetLine(
            line_index=3,
            start=30.0,
            end=31.0,
            text="l3",
            words=["l3"],
            y=30,
            word_rois=[(0, 0, 2, 2)],
        ),
    ]
    jobs = _build_line_refinement_jobs(lines)
    groups = _merge_line_refinement_jobs(jobs)
    assert len(groups) == 2

    first_start, first_end, first_jobs = groups[0]
    assert first_start == 9.0
    assert first_end == 15.0
    assert len(first_jobs) == 2

    second_start, second_end, second_jobs = groups[1]
    assert second_start == 29.0
    assert second_end == 32.0
    assert len(second_jobs) == 1


def test_slice_frames_for_window_uses_sorted_time_bounds():
    frames = [
        (0.5, None, None),
        (1.0, None, None),
        (1.5, None, None),
        (2.0, None, None),
    ]
    times = [f[0] for f in frames]

    selected = _slice_frames_for_window(frames, times, v_start=1.0, v_end=1.5)
    assert [f[0] for f in selected] == [1.0, 1.5]


def test_refine_line_with_frames_populates_word_outputs(monkeypatch):
    line = TargetLine(
        line_index=1,
        start=1.0,
        end=2.0,
        text="hello",
        words=["hello"],
        y=10,
        word_rois=[(0, 0, 2, 2)],
    )
    roi = np.ones((3, 3, 3), dtype=np.uint8) * 120
    roi_lab = np.ones((3, 3, 3), dtype=np.float32) * 50.0
    frames = [(1.0, roi, roi_lab) for _ in range(12)]

    monkeypatch.setattr(
        "y2karaoke.core.visual.refinement._word_fill_mask",
        lambda word_roi, c_bg: np.ones((2, 2), dtype=np.uint8) * 255,
    )
    monkeypatch.setattr(
        "y2karaoke.core.visual.refinement._detect_highlight_with_confidence",
        lambda vals: (1.1, 1.3, 0.8),
    )

    _refine_line_with_frames(line, frames)
    assert line.word_starts == [1.1]
    assert line.word_ends == [1.3]
    assert line.word_confidences == [0.8]


def test_assign_line_level_word_timings_weights_longer_words():
    line = TargetLine(
        line_index=1,
        start=5.0,
        end=7.0,
        text="go extraordinary",
        words=["go", "extraordinary"],
        y=10,
        word_rois=[(0, 0, 10, 5), (12, 0, 40, 5)],
    )

    _assign_line_level_word_timings(
        line, line_start=5.2, line_end=6.8, line_confidence=0.7
    )

    assert line.word_starts is not None
    assert line.word_ends is not None
    assert line.word_confidences is not None
    short_dur = line.word_ends[0] - line.word_starts[0]
    long_dur = line.word_ends[1] - line.word_starts[1]
    assert long_dur > short_dur
    assert line.word_starts[0] == pytest.approx(5.2, abs=1e-6)
    assert line.word_ends[-1] == pytest.approx(6.8, abs=1e-6)
    assert 0.2 <= line.word_confidences[0] <= 0.5


def test_refine_line_with_frames_uses_line_level_fallback(monkeypatch):
    line = TargetLine(
        line_index=1,
        start=1.0,
        end=2.5,
        text="hello world",
        words=["hello", "world"],
        y=10,
        word_rois=[(0, 0, 2, 2), (3, 0, 2, 2)],
    )
    roi = np.ones((6, 6, 3), dtype=np.uint8) * 120
    roi_lab = np.ones((6, 6, 3), dtype=np.float32) * 50.0
    frames = [(1.0 + i * 0.1, roi, roi_lab) for i in range(15)]

    monkeypatch.setattr(
        "y2karaoke.core.visual.refinement._word_fill_mask",
        lambda word_roi, c_bg: np.ones(word_roi.shape[:2], dtype=np.uint8) * 255,
    )

    calls = {"count": 0}

    def _fake_detect(vals):
        calls["count"] += 1
        if calls["count"] <= 2:
            return None, None, 0.0
        return 1.2, 2.0, 0.65

    monkeypatch.setattr(
        "y2karaoke.core.visual.refinement._detect_highlight_with_confidence",
        _fake_detect,
    )

    _refine_line_with_frames(line, frames)

    assert line.word_starts is not None
    assert line.word_ends is not None
    assert line.word_confidences is not None
    assert len(line.word_starts) == 2
    assert line.word_starts[0] == pytest.approx(1.2, abs=1e-6)
    assert line.word_ends[-1] == pytest.approx(2.0, abs=1e-6)
    assert all(c is not None and 0.2 <= c <= 0.5 for c in line.word_confidences)


def test_refine_line_timings_at_low_fps_assigns_line_level_timings(
    monkeypatch, tmp_path
):
    line = TargetLine(
        line_index=1,
        start=1.0,
        end=3.0,
        text="hello world",
        words=["hello", "world"],
        y=10,
        word_rois=[(0, 0, 2, 2), (3, 0, 2, 2)],
    )
    frames = [
        (
            0.9 + i * 0.1,
            np.ones((6, 6, 3), dtype=np.uint8),
            np.ones((6, 6, 3), dtype=np.float32),
        )
        for i in range(20)
    ]

    class FakeCap:
        def isOpened(self):
            return True

        def release(self):
            return None

    monkeypatch.setattr(
        "y2karaoke.core.visual.refinement.cv2.VideoCapture",
        lambda _p: FakeCap(),
    )
    monkeypatch.setattr(
        "y2karaoke.core.visual.refinement._build_line_refinement_jobs",
        lambda _lines: [(line, 0.5, 3.5)],
    )
    monkeypatch.setattr(
        "y2karaoke.core.visual.refinement._merge_line_refinement_jobs",
        lambda jobs: [(0.5, 3.5, jobs)],
    )
    monkeypatch.setattr(
        "y2karaoke.core.visual.refinement._read_window_frames_sampled",
        lambda *a, **k: frames,
    )
    monkeypatch.setattr(
        "y2karaoke.core.visual.refinement._slice_frames_for_window",
        lambda *a, **k: frames,
    )
    monkeypatch.setattr(
        "y2karaoke.core.visual.refinement._detect_line_highlight_with_confidence",
        lambda *_a, **_k: (1.2, 2.1, 0.7),
    )

    called = {"n": 0}

    def _fake_assign(ln, line_start, line_end, line_confidence):
        called["n"] += 1
        assert ln is line
        assert line_start == pytest.approx(1.2, abs=1e-6)
        assert line_end == pytest.approx(2.1, abs=1e-6)
        assert line_confidence == pytest.approx(0.7, abs=1e-6)

    monkeypatch.setattr(
        "y2karaoke.core.visual.refinement._assign_line_level_word_timings",
        _fake_assign,
    )

    refine_line_timings_at_low_fps(
        tmp_path / "video.mp4",
        [line],
        (0, 0, 10, 10),
        sample_fps=6.0,
    )
    assert called["n"] == 1
