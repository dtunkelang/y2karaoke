from pathlib import Path

from y2karaoke.vision import roi as _MODULE


def test_detect_lyric_roi_uses_sampled_grab_retrieve(monkeypatch) -> None:  # noqa: C901
    class FakeOCR:
        def predict(self, _frame):
            return [{"rec_boxes": [[[0, 0], [2, 0], [2, 2], [0, 2]]]}]

    class FakeCap:
        def __init__(self, total_frames: int = 300, src_fps: float = 10.0) -> None:
            self.total_frames = total_frames
            self.src_fps = src_fps
            self.pos = 0
            self.set_calls = 0
            self.grab_calls = 0
            self.retrieve_calls = 0

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == _MODULE.cv2.CAP_PROP_FPS:
                return self.src_fps
            if prop == _MODULE.cv2.CAP_PROP_FRAME_COUNT:
                return float(self.total_frames)
            if prop == _MODULE.cv2.CAP_PROP_FRAME_WIDTH:
                return 100.0
            if prop == _MODULE.cv2.CAP_PROP_FRAME_HEIGHT:
                return 80.0
            return 0.0

        def set(self, prop, value):
            if prop == _MODULE.cv2.CAP_PROP_POS_MSEC:
                self.pos = int(round((value / 1000.0) * self.src_fps))
                self.set_calls += 1
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
            frame = _MODULE.np.zeros((8, 8, 3), dtype=_MODULE.np.uint8)
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

    roi = _MODULE.detect_lyric_roi(Path("/tmp/fake.mp4"), sample_fps=2.0)

    cap = captured["cap"]
    assert cap is not None
    assert cap.set_calls == 1
    assert cap.grab_calls > 0
    assert cap.retrieve_calls > 0
    assert roi[2] > 0 and roi[3] > 0


def test_detect_lyric_roi_applies_left_clip_guardrail(monkeypatch) -> None:
    class FakeOCR:
        def predict(self, _frame):
            # Simulate OCR boxes concentrated away from the left edge.
            return [
                {
                    "rec_boxes": [
                        [[30, 20], [70, 20], [70, 32], [30, 32]],
                        [[32, 40], [68, 40], [68, 52], [32, 52]],
                    ]
                }
            ]

    class FakeCap:
        def __init__(self) -> None:
            self.pos = 0
            self.total_frames = 200
            self.src_fps = 10.0

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == _MODULE.cv2.CAP_PROP_FPS:
                return self.src_fps
            if prop == _MODULE.cv2.CAP_PROP_FRAME_COUNT:
                return float(self.total_frames)
            if prop == _MODULE.cv2.CAP_PROP_FRAME_WIDTH:
                return 100.0
            if prop == _MODULE.cv2.CAP_PROP_FRAME_HEIGHT:
                return 80.0
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
            return True

        def retrieve(self):
            frame = _MODULE.np.zeros((8, 8, 3), dtype=_MODULE.np.uint8)
            return True, frame

        def release(self):
            return None

    monkeypatch.setattr(_MODULE, "get_ocr_engine", lambda: FakeOCR())
    monkeypatch.setattr(_MODULE.cv2, "VideoCapture", lambda _p: FakeCap())

    roi = _MODULE.detect_lyric_roi(Path("/tmp/fake.mp4"), sample_fps=2.0)
    # Guardrail should force left bound to at most 12% of frame width.
    assert roi[0] <= 12
