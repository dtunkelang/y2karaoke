from pathlib import Path

from y2karaoke.vision import suitability as _MODULE


def test_get_cache_key_includes_file_identity(tmp_path) -> None:
    a = tmp_path / "same-name.mp4"
    b_dir = tmp_path / "other"
    b_dir.mkdir()
    b = b_dir / "same-name.mp4"
    a.write_bytes(b"aaa")
    b.write_bytes(b"bbbb")

    key_a = _MODULE._get_cache_key(a, "raw_frames", fps=1.0)
    key_b = _MODULE._get_cache_key(b, "raw_frames", fps=1.0)
    assert key_a != key_b


def test_collect_raw_frames_with_confidence_uses_sampled_retrieve(monkeypatch) -> None:
    class FakeOCR:
        def predict(self, roi):
            return [
                {
                    "rec_texts": ["hello"],
                    "rec_boxes": [[[0, 0], [1, 0], [1, 1], [0, 1]]],
                    "rec_scores": [0.9],
                }
            ]

    class FakeCap:
        def __init__(self, total_frames: int = 10, src_fps: float = 10.0) -> None:
            self.total_frames = total_frames
            self.src_fps = src_fps
            self.pos = 0
            self.grab_calls = 0
            self.retrieve_calls = 0

        def isOpened(self):
            return True

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
    monkeypatch.setattr(
        _MODULE, "classify_word_state", lambda *a, **k: ("selected", 0.8)
    )

    raw = _MODULE.collect_raw_frames_with_confidence(
        video_path=Path("/tmp/fake.mp4"),
        start=0.0,
        end=0.95,
        fps=2.0,
        c_un=None,
        c_sel=None,
        roi_rect=(0, 0, 4, 4),
    )

    cap = captured["cap"]
    assert cap is not None
    assert cap.grab_calls == 10
    assert cap.retrieve_calls == 2
    assert len(raw) == 2
