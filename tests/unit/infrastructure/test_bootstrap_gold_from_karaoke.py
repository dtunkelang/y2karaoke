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


def test_collect_raw_frames_cached_respects_cache_version(
    tmp_path, monkeypatch
) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video-bytes")
    cache_dir = tmp_path / "cache"
    calls = {"n": 0}

    def fake_collect(video_path, start, end, fps, roi_rect):
        calls["n"] += 1
        return [{"time": 0.0, "words": [{"text": "x", "x": 0, "y": 0, "w": 1, "h": 1}]}]

    monkeypatch.setattr(_MODULE, "_collect_raw_frames", fake_collect)

    first = _MODULE._collect_raw_frames_cached(
        video_path=video_path,
        duration=10.0,
        fps=2.0,
        roi_rect=(0, 0, 10, 10),
        cache_dir=cache_dir,
        cache_version="vA",
    )
    second = _MODULE._collect_raw_frames_cached(
        video_path=video_path,
        duration=10.0,
        fps=2.0,
        roi_rect=(0, 0, 10, 10),
        cache_dir=cache_dir,
        cache_version="vA",
    )
    third = _MODULE._collect_raw_frames_cached(
        video_path=video_path,
        duration=10.0,
        fps=2.0,
        roi_rect=(0, 0, 10, 10),
        cache_dir=cache_dir,
        cache_version="vB",
    )

    assert first == second == third
    assert calls["n"] == 2


def test_extract_audio_from_video_uses_ffmpeg_once(tmp_path, monkeypatch) -> None:
    video_path = tmp_path / "candidate.mp4"
    video_path.write_bytes(b"video-bytes")
    output_dir = tmp_path / "out"
    calls = {"n": 0}

    def fake_run(cmd, check, stdout, stderr):
        calls["n"] += 1
        assert cmd[0] == "ffmpeg"
        Path(cmd[-1]).write_bytes(b"wav-bytes")
        return None

    monkeypatch.setattr(_MODULE.subprocess, "run", fake_run)

    first = _MODULE._extract_audio_from_video(video_path, output_dir)
    second = _MODULE._extract_audio_from_video(video_path, output_dir)
    assert first == second
    assert first.exists()
    assert calls["n"] == 1


def test_build_refined_lines_output_skips_artist_and_title() -> None:
    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=10.0,
            text="Song Title",
            words=["Song", "Title"],
            y=50,
            word_starts=[8.1, 8.7],
            word_ends=[8.5, 9.2],
            word_rois=[(0, 0, 1, 1), (1, 0, 1, 1)],
        ),
        TargetLine(
            line_index=2,
            start=11.0,
            end=13.0,
            text="real lyric",
            words=["real", "lyric"],
            y=60,
            word_starts=[11.1, 11.8],
            word_ends=[11.5, 12.4],
            word_rois=[(0, 0, 1, 1), (1, 0, 1, 1)],
        ),
    ]
    out = _MODULE._build_refined_lines_output(
        lines,
        artist="Artist Name",
        title="Song Title",
    )
    assert len(out) == 1
    assert out[0]["text"] == "real lyric"
    assert out[0]["line_index"] == 1


def test_build_refined_lines_output_falls_back_when_word_starts_missing() -> None:
    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=10.0,
            text="hello world",
            words=["hello", "world"],
            y=70,
            word_starts=None,
            word_ends=None,
            word_rois=[(0, 0, 1, 1), (1, 0, 1, 1)],
        )
    ]
    out = _MODULE._build_refined_lines_output(lines, artist=None, title=None)
    assert len(out) == 1
    words = out[0]["words"]
    assert len(words) == 2
    assert words[0]["confidence"] == 0.0
    assert words[0]["start"] >= 5.0
    assert words[0]["end"] > words[0]["start"]


def test_ensure_selected_suitability_rejects_below_threshold(tmp_path) -> None:
    class Args:
        suitability_fps = 1.0
        allow_low_suitability = False
        min_detectability = 0.45
        min_word_level_score = 0.15

    with pytest.raises(ValueError):
        _MODULE._ensure_selected_suitability(
            {"detectability_score": 0.2, "word_level_score": 0.1},
            v_path=tmp_path / "v.mp4",
            song_dir=tmp_path,
            args=Args(),
        )


def test_ensure_selected_suitability_analyzes_when_empty(monkeypatch, tmp_path) -> None:
    class Args:
        suitability_fps = 1.0
        allow_low_suitability = False
        min_detectability = 0.45
        min_word_level_score = 0.15

    expected = {"detectability_score": 0.9, "word_level_score": 0.8}
    monkeypatch.setattr(
        _MODULE, "analyze_visual_suitability", lambda *a, **k: (expected, (0, 0, 1, 1))
    )

    metrics = _MODULE._ensure_selected_suitability(
        {},
        v_path=tmp_path / "v.mp4",
        song_dir=tmp_path,
        args=Args(),
    )
    assert metrics == expected


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
        selected_metrics={"word_level_score": 0.2},
    )

    assert len(out) == 1
    assert called_high["n"] == 1
    assert called_low["n"] == 0


def test_nearest_known_word_indices() -> None:
    prev_known, next_known = _MODULE._nearest_known_word_indices([1, 4], 6)
    assert prev_known == [-1, 1, 1, 1, 4, 4]
    assert next_known == [1, 1, 4, 4, 4, -1]
