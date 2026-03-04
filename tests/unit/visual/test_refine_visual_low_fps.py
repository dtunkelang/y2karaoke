import numpy as np
import pytest

from y2karaoke.core.models import TargetLine
from y2karaoke.core.visual.refinement import refine_line_timings_at_low_fps


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
        lambda _lines, **_kwargs: [(line, 0.5, 3.5)],
    )
    monkeypatch.setattr(
        "y2karaoke.core.visual.refinement._merge_line_refinement_jobs",
        lambda jobs, **_kwargs: [(0.5, 3.5, jobs)],
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


def test_refine_line_timings_at_low_fps_uses_previous_start_for_min_start_time(
    monkeypatch, tmp_path
):
    line1 = TargetLine(
        line_index=1,
        start=10.0,
        end=30.0,
        text="line one",
        words=["line", "one"],
        y=10,
        word_rois=[(0, 0, 2, 2), (3, 0, 2, 2)],
    )
    line2 = TargetLine(
        line_index=2,
        start=12.0,
        end=31.0,
        text="line two",
        words=["line", "two"],
        y=20,
        word_rois=[(0, 0, 2, 2), (3, 0, 2, 2)],
    )
    frames = [
        (
            9.5 + i * 0.1,
            np.ones((6, 6, 3), dtype=np.uint8),
            np.ones((6, 6, 3), dtype=np.float32),
        )
        for i in range(30)
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
        lambda _lines, **_kwargs: [(line1, 9.0, 31.0), (line2, 9.0, 31.0)],
    )
    monkeypatch.setattr(
        "y2karaoke.core.visual.refinement._merge_line_refinement_jobs",
        lambda jobs, **_kwargs: [(9.0, 31.0, jobs)],
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
        "y2karaoke.core.visual.refinement._apply_persistent_block_highlight_order",
        lambda *_a, **_k: None,
    )

    seen_min_start_times = []

    def _fake_detect(
        _ln,
        _line_frames,
        _c_bg_line,
        *,
        min_start_time=None,
        require_full_cycle=False,
    ):
        seen_min_start_times.append(min_start_time)
        if len(seen_min_start_times) == 1:
            return 10.0, 30.0, 0.7
        return 12.0, 31.0, 0.7

    monkeypatch.setattr(
        "y2karaoke.core.visual.refinement._detect_line_highlight_with_confidence",
        _fake_detect,
    )

    refine_line_timings_at_low_fps(
        tmp_path / "video.mp4",
        [line1, line2],
        (0, 0, 10, 10),
        sample_fps=6.0,
    )

    assert seen_min_start_times[0] is None
    assert seen_min_start_times[1] == pytest.approx(10.05, abs=1e-6)


def test_refine_line_timings_at_low_fps_honors_visibility_start_floor(
    monkeypatch, tmp_path
):
    line1 = TargetLine(
        line_index=1,
        start=10.0,
        end=12.0,
        text="line one",
        words=["line", "one"],
        y=10,
        word_rois=[(0, 0, 2, 2), (3, 0, 2, 2)],
    )
    line2 = TargetLine(
        line_index=2,
        start=20.0,
        end=22.0,
        text="line two",
        words=["line", "two"],
        y=20,
        word_rois=[(0, 0, 2, 2), (3, 0, 2, 2)],
        visibility_start=20.0,
        visibility_end=30.0,
    )
    frames = [
        (
            9.5 + i * 0.1,
            np.ones((6, 6, 3), dtype=np.uint8),
            np.ones((6, 6, 3), dtype=np.float32),
        )
        for i in range(30)
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
        lambda _lines, **_kwargs: [(line1, 9.0, 13.0), (line2, 19.0, 31.0)],
    )
    monkeypatch.setattr(
        "y2karaoke.core.visual.refinement._merge_line_refinement_jobs",
        lambda jobs, **_kwargs: [(9.0, 31.0, jobs)],
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
        "y2karaoke.core.visual.refinement._apply_persistent_block_highlight_order",
        lambda *_a, **_k: None,
    )

    seen_min_start_times = []

    def _fake_detect(
        _ln,
        _line_frames,
        _c_bg_line,
        *,
        min_start_time=None,
        require_full_cycle=False,
    ):
        seen_min_start_times.append(min_start_time)
        if len(seen_min_start_times) == 1:
            return 10.0, 11.0, 0.7
        return 20.5, 21.5, 0.7

    monkeypatch.setattr(
        "y2karaoke.core.visual.refinement._detect_line_highlight_with_confidence",
        _fake_detect,
    )

    refine_line_timings_at_low_fps(
        tmp_path / "video.mp4",
        [line1, line2],
        (0, 0, 10, 10),
        sample_fps=6.0,
    )

    assert seen_min_start_times[1] == pytest.approx(18.0, abs=1e-6)
