import numpy as np

from y2karaoke.core import backgrounds
from y2karaoke.core.models import Line, Word


def _make_line(text="hello", start=0.0, end=1.0):
    words = [Word(text=text, start_time=start, end_time=end)]
    return Line(words=words)


def test_is_valid_frame_checks_brightness():
    processor = backgrounds.BackgroundProcessor()
    dark = np.zeros((10, 10, 3), dtype=np.uint8)
    bright = np.full((10, 10, 3), 255, dtype=np.uint8)
    assert processor._is_valid_frame(dark) == False
    assert processor._is_valid_frame(bright) == True


def test_process_frame_outputs_expected_shape():
    processor = backgrounds.BackgroundProcessor()
    frame = np.full((20, 30, 3), 200, dtype=np.uint8)
    processed = processor._process_frame(frame)
    assert processed.shape == (processor.height, processor.width, 3)


def test_extract_scene_frames_handles_failure(monkeypatch):
    processor = backgrounds.BackgroundProcessor()

    monkeypatch.setattr(processor, "_detect_scenes_subprocess", lambda *a, **k: [0.0])
    monkeypatch.setattr(
        backgrounds.cv2,
        "VideoCapture",
        lambda *_: type(
            "Cap",
            (),
            {
                "isOpened": lambda self: False,
                "release": lambda self: None,
            },
        )(),
    )

    frames = processor._extract_scene_frames("video.mp4")
    assert frames == []


def test_create_background_segments_from_frames(monkeypatch):
    processor = backgrounds.BackgroundProcessor()
    frame = np.full((20, 30, 3), 200, dtype=np.uint8)

    monkeypatch.setattr(
        processor,
        "_extract_scene_frames",
        lambda *a, **k: [(0.0, frame), (5.0, frame)],
    )
    monkeypatch.setattr(processor, "_process_frame", lambda f: f)

    segments = processor.create_background_segments(
        "video.mp4", [_make_line()], duration=12.0
    )

    assert len(segments) == 2
    assert segments[0].start_time == 0.0
    assert segments[0].end_time == 5.0
    assert segments[1].end_time == 12.0


def test_create_background_segments_empty_on_error(monkeypatch):
    processor = backgrounds.BackgroundProcessor()
    monkeypatch.setattr(processor, "_extract_scene_frames", lambda *a, **k: [])

    segments = processor.create_background_segments(
        "video.mp4", [_make_line()], duration=12.0
    )
    assert segments == []


def test_create_background_segments_function_wrapper(monkeypatch):
    frame = np.full((20, 30, 3), 200, dtype=np.uint8)

    monkeypatch.setattr(
        backgrounds.BackgroundProcessor,
        "create_background_segments",
        lambda self, *a, **k: [backgrounds.BackgroundSegment(frame, 0.0, 1.0)],
    )

    segments = backgrounds.create_background_segments(
        "video.mp4", [_make_line()], duration=1.0
    )
    assert len(segments) == 1
