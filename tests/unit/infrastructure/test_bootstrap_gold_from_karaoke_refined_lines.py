import sys
from pathlib import Path
import importlib.util

# Add project root to sys.path before other project imports
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.y2karaoke.core.models import TargetLine  # noqa: E402

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "bootstrap_gold_from_karaoke.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "bootstrap_gold_from_karaoke_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_bootstrap_refined_lines_skips_high_fps_when_word_level_low(
    monkeypatch, tmp_path
) -> None:
    class FakeCap:
        def get(self, prop):
            if prop == _MODULE.cv2.CAP_PROP_FRAME_COUNT:
                return 300.0
            if prop == _MODULE.cv2.CAP_PROP_FPS:
                return 30.0
            return 0.0

        def release(self):
            return None

    class Args:
        visual_fps = 2.0
        raw_ocr_cache_version = "3"
        artist = None
        title = None

    monkeypatch.setattr(_MODULE, "detect_lyric_roi", lambda *a, **k: (0, 0, 10, 10))
    monkeypatch.setattr(_MODULE.cv2, "VideoCapture", lambda _p: FakeCap())
    monkeypatch.setattr(
        _MODULE,
        "_collect_raw_frames_cached",
        lambda *a, **k: [
            {"time": 8.0, "words": [{"text": "hello", "x": 0, "y": 10, "w": 4, "h": 3}]}
        ],
    )
    monkeypatch.setattr(
        _MODULE,
        "reconstruct_lyrics_from_visuals",
        lambda *_a, **_k: [
            TargetLine(
                line_index=1,
                start=8.0,
                end=10.0,
                text="hello",
                words=["hello"],
                y=10,
                word_rois=[(0, 0, 4, 3)],
            )
        ],
    )
    called_high = {"n": 0}
    called_low = {"n": 0}
    monkeypatch.setattr(
        _MODULE,
        "refine_word_timings_at_high_fps",
        lambda *a, **k: called_high.__setitem__("n", called_high["n"] + 1),
    )
    monkeypatch.setattr(
        _MODULE,
        "refine_line_timings_at_low_fps",
        lambda *a, **k: called_low.__setitem__("n", called_low["n"] + 1),
    )

    out = _MODULE._bootstrap_refined_lines(
        tmp_path / "v.mp4",
        Args(),
        tmp_path,
        selected_metrics={"word_level_score": 0.0},
    )

    assert len(out) == 1
    assert called_high["n"] == 0
    assert called_low["n"] == 1


def test_bootstrap_refined_lines_runs_high_fps_when_word_level_good(
    monkeypatch, tmp_path
) -> None:
    class FakeCap:
        def get(self, prop):
            if prop == _MODULE.cv2.CAP_PROP_FRAME_COUNT:
                return 300.0
            if prop == _MODULE.cv2.CAP_PROP_FPS:
                return 30.0
            return 0.0

        def release(self):
            return None

    class Args:
        visual_fps = 2.0
        raw_ocr_cache_version = "3"
        artist = None
        title = None

    monkeypatch.setattr(_MODULE, "detect_lyric_roi", lambda *a, **k: (0, 0, 10, 10))
    monkeypatch.setattr(_MODULE.cv2, "VideoCapture", lambda _p: FakeCap())
    monkeypatch.setattr(
        _MODULE,
        "_collect_raw_frames_cached",
        lambda *a, **k: [
            {"time": 8.0, "words": [{"text": "hello", "x": 0, "y": 10, "w": 4, "h": 3}]}
        ],
    )
    monkeypatch.setattr(
        _MODULE,
        "reconstruct_lyrics_from_visuals",
        lambda *_a, **_k: [
            TargetLine(
                line_index=1,
                start=8.0,
                end=10.0,
                text="hello",
                words=["hello"],
                y=10,
                word_rois=[(0, 0, 4, 3)],
            )
        ],
    )
    called_high = {"n": 0}
    called_low = {"n": 0}
    monkeypatch.setattr(
        _MODULE,
        "refine_word_timings_at_high_fps",
        lambda *a, **k: called_high.__setitem__("n", called_high["n"] + 1),
    )
    monkeypatch.setattr(
        _MODULE,
        "refine_line_timings_at_low_fps",
        lambda *a, **k: called_low.__setitem__("n", called_low["n"] + 1),
    )

    out = _MODULE._bootstrap_refined_lines(
        tmp_path / "v.mp4",
        Args(),
        tmp_path,
        selected_metrics={"word_level_score": 0.6},
    )

    assert len(out) == 1
    assert called_high["n"] == 1
    assert called_low["n"] == 0


def test_nearest_known_word_indices() -> None:
    prev_known, next_known = _MODULE._nearest_known_word_indices([1, 4], 6)
    assert prev_known == [-1, 1, 1, 1, 4, 4]
    assert next_known == [1, 1, 4, 4, 4, -1]
