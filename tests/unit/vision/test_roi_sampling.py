from pathlib import Path

from y2karaoke.vision import roi as _MODULE


class _FakeOCR:
    def __init__(self, rec_boxes):
        self._rec_boxes = rec_boxes

    def predict(self, _frame):
        return [{"rec_boxes": self._rec_boxes}]


class _FakeCap:
    def __init__(
        self,
        *,
        total_frames: int,
        src_fps: float,
        frame_width: float,
        frame_height: float,
    ) -> None:
        self.total_frames = total_frames
        self.src_fps = src_fps
        self.frame_width = frame_width
        self.frame_height = frame_height
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
            return self.frame_width
        if prop == _MODULE.cv2.CAP_PROP_FRAME_HEIGHT:
            return self.frame_height
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


def _install_roi_doubles(
    monkeypatch,
    *,
    rec_boxes,
    frame_width: float,
    frame_height: float,
    total_frames: int = 200,
    src_fps: float = 10.0,
):
    captured = {"cap": None}

    def make_cap(_path):
        cap = _FakeCap(
            total_frames=total_frames,
            src_fps=src_fps,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        captured["cap"] = cap
        return cap

    monkeypatch.setattr(_MODULE, "get_ocr_engine", lambda: _FakeOCR(rec_boxes))
    monkeypatch.setattr(_MODULE.cv2, "VideoCapture", make_cap)
    return captured


def test_detect_lyric_roi_uses_sampled_grab_retrieve(monkeypatch) -> None:  # noqa: C901
    captured = _install_roi_doubles(
        monkeypatch,
        rec_boxes=[[[0, 0], [2, 0], [2, 2], [0, 2]]],
        frame_width=100.0,
        frame_height=80.0,
        total_frames=300,
    )

    roi = _MODULE.detect_lyric_roi(Path("/tmp/fake.mp4"), sample_fps=2.0)

    cap = captured["cap"]
    assert cap is not None
    assert cap.set_calls == 1
    assert cap.grab_calls > 0
    assert cap.retrieve_calls > 0
    assert roi[2] > 0 and roi[3] > 0


def test_detect_lyric_roi_applies_left_clip_guardrail(monkeypatch) -> None:
    _install_roi_doubles(
        monkeypatch,
        rec_boxes=[
            [[30, 20], [70, 20], [70, 32], [30, 32]],
            [[32, 40], [68, 40], [68, 52], [32, 52]],
        ],
        frame_width=100.0,
        frame_height=80.0,
    )

    roi = _MODULE.detect_lyric_roi(Path("/tmp/fake.mp4"), sample_fps=2.0)
    # Guardrail should force left bound to at most 12% of frame width.
    assert roi[0] <= 12


def test_detect_lyric_roi_applies_top_clip_guardrail(monkeypatch) -> None:
    _install_roi_doubles(
        monkeypatch,
        rec_boxes=[
            [[40, 90], [120, 90], [120, 110], [40, 110]],
            [[44, 128], [126, 128], [126, 148], [44, 148]],
        ],
        frame_width=200.0,
        frame_height=200.0,
    )

    roi = _MODULE.detect_lyric_roi(Path("/tmp/fake.mp4"), sample_fps=2.0)
    # Guardrail should force top bound to at most 22% of frame height.
    assert roi[1] <= 44
