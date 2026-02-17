import sys
from pathlib import Path
import importlib.util
import pytest

# Add project root to sys.path before other project imports
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.y2karaoke.core.models import TargetLine  # noqa: E402
from src.y2karaoke.core.text_utils import (  # noqa: E402
    text_similarity,
    normalize_ocr_line,
    normalize_text_basic as normalize_text,
)

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


def test_text_similarity_is_case_and_punctuation_tolerant() -> None:
    s = text_similarity(
        "White shirt now red, my bloody nose", "white shirt now red my bloody nose"
    )
    assert s > 0.95


def test_normalize_text_handles_hyphens() -> None:
    # Ensure 'anti-hero' matches 'anti hero'
    assert normalize_text("anti-hero") == "anti hero"
    assert normalize_text("Anti-Hero!!!") == "anti hero"


def test_normalize_ocr_line_fixes_typos() -> None:
    assert normalize_ocr_line("the problei") == "the problem"
    assert normalize_ocr_line("have this thing") == "I have this thing"


def test_target_line_construction() -> None:
    line = TargetLine(
        line_index=1,
        start=10.0,
        end=15.0,
        text="Hello world",
        words=["Hello", "world"],
        y=100.0,
    )
    assert line.text == "Hello world"
    assert line.y == 100.0


def test_is_suitability_good_enough() -> None:
    assert _MODULE._is_suitability_good_enough(
        {"detectability_score": 0.7, "word_level_score": 0.2}, 0.45, 0.15
    )
    assert not _MODULE._is_suitability_good_enough(
        {"detectability_score": 0.4, "word_level_score": 0.2}, 0.45, 0.15
    )
    assert not _MODULE._is_suitability_good_enough(
        {"detectability_score": 0.7, "word_level_score": 0.1}, 0.45, 0.15
    )


def test_select_candidate_requires_inputs() -> None:
    class DummyDownloader:
        pass

    class Args:
        candidate_url = None
        work_dir = Path("/tmp/demo")
        report_json = None
        artist = None
        title = None
        max_candidates = 3
        suitability_fps = 1.0
        show_candidates = False
        allow_low_suitability = False
        min_detectability = 0.45
        min_word_level_score = 0.15

    with pytest.raises(ValueError):
        _MODULE._select_candidate(Args(), DummyDownloader(), Path("/tmp/demo"))


def test_select_candidate_prefers_best(monkeypatch, tmp_path) -> None:
    class Args:
        candidate_url = None
        work_dir = tmp_path
        report_json = None
        artist = "Artist"
        title = "Song"
        max_candidates = 3
        suitability_fps = 1.0
        show_candidates = False
        allow_low_suitability = False
        min_detectability = 0.45
        min_word_level_score = 0.15

    candidates = [
        {"url": "https://youtube.com/watch?v=1", "title": "a"},
        {"url": "https://youtube.com/watch?v=2", "title": "b"},
    ]
    ranked = [
        {
            "url": "https://youtube.com/watch?v=2",
            "video_path": str(tmp_path / "b.mp4"),
            "metrics": {
                "detectability_score": 0.81,
                "word_level_score": 0.32,
                "avg_ocr_confidence": 0.9,
            },
            "score": 0.81,
        },
        {
            "url": "https://youtube.com/watch?v=1",
            "video_path": str(tmp_path / "a.mp4"),
            "metrics": {
                "detectability_score": 0.51,
                "word_level_score": 0.2,
                "avg_ocr_confidence": 0.7,
            },
            "score": 0.51,
        },
    ]

    monkeypatch.setattr(
        _MODULE, "_search_karaoke_candidates", lambda *a, **k: candidates
    )
    monkeypatch.setattr(
        _MODULE, "_rank_candidates_by_suitability", lambda *a, **k: ranked
    )

    url, video_path, metrics = _MODULE._select_candidate(Args(), object(), tmp_path)
    assert url.endswith("v=2")
    assert video_path and video_path.name == "b.mp4"
    assert metrics["detectability_score"] == 0.81


def test_clamp_confidence() -> None:
    assert _MODULE._clamp_confidence(None) == 0.0
    assert _MODULE._clamp_confidence(0.4) == 0.4
    assert _MODULE._clamp_confidence(1.2) == 1.0
    assert _MODULE._clamp_confidence(-0.5) == 0.0


def test_select_candidate_with_rankings_explicit_url(tmp_path) -> None:
    class Args:
        candidate_url = "https://youtube.com/watch?v=abc123def45"
        work_dir = tmp_path
        report_json = None
        artist = None
        title = None
        max_candidates = 3
        suitability_fps = 1.0
        show_candidates = False
        allow_low_suitability = False
        min_detectability = 0.45
        min_word_level_score = 0.15

    url, video_path, metrics, ranked = _MODULE._select_candidate_with_rankings(
        Args(), object(), tmp_path
    )
    assert url == "https://youtube.com/watch?v=abc123def45"
    assert video_path is None
    assert metrics == {}
    assert ranked == []


def test_collect_raw_frames_uses_grab_retrieve_sampling(monkeypatch) -> None:
    class FakeOCR:
        def predict(self, roi):
            return [
                {
                    "rec_texts": ["hello"],
                    "rec_boxes": [[[0, 0], [1, 0], [1, 1], [0, 1]]],
                }
            ]

    class FakeCap:
        def __init__(self, total_frames: int = 10, src_fps: float = 10.0) -> None:
            self.total_frames = total_frames
            self.src_fps = src_fps
            self.pos = 0
            self.grab_calls = 0
            self.retrieve_calls = 0

        def get(self, prop):
            if prop == _MODULE.cv2.CAP_PROP_FPS:
                return self.src_fps
            if prop == _MODULE.cv2.CAP_PROP_POS_MSEC:
                return (self.pos / self.src_fps) * 1000.0
            return 0.0

        def set(self, prop, value):
            if prop == _MODULE.cv2.CAP_PROP_POS_MSEC:
                self.pos = int(round((value / 1000.0) * self.src_fps))
                return True
            return False

        def grab(self):
            if self.pos >= self.total_frames:
                return False
            self.pos += 1
            self.grab_calls += 1
            return True

        def retrieve(self):
            self.retrieve_calls += 1
            frame = _MODULE.np.zeros((4, 4, 3), dtype=_MODULE.np.uint8)
            return True, frame

        def release(self):
            return None

    captured = {"cap": None}

    def make_cap(_path):
        cap = FakeCap()
        captured["cap"] = cap
        return cap

    monkeypatch.setattr(_MODULE, "get_ocr_engine", lambda: FakeOCR())
    monkeypatch.setattr(_MODULE.cv2, "VideoCapture", make_cap)

    raw = _MODULE._collect_raw_frames(
        video_path=Path("/tmp/fake.mp4"),
        start=0.0,
        end=0.95,
        fps=2.0,
        roi_rect=(0, 0, 4, 4),
    )

    cap = captured["cap"]
    assert cap is not None
    assert cap.grab_calls == 10
    assert cap.retrieve_calls == 2
    assert len(raw) == 2
