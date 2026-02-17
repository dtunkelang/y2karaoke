from pathlib import Path

from y2karaoke.core.visual import bootstrap_ocr as _MODULE


def test_raw_frames_cache_path_changes_with_version(tmp_path):
    video_path = tmp_path / "v.mp4"
    video_path.write_bytes(b"v")
    cache_dir = tmp_path / "cache"
    p1 = _MODULE.raw_frames_cache_path(
        video_path, cache_dir, 2.0, (0, 0, 10, 10), cache_version="a"
    )
    p2 = _MODULE.raw_frames_cache_path(
        video_path, cache_dir, 2.0, (0, 0, 10, 10), cache_version="b"
    )
    assert p1 != p2


def test_collect_raw_frames_cached_uses_cache(tmp_path):
    video_path = tmp_path / "v.mp4"
    video_path.write_bytes(b"v")
    cache_dir = tmp_path / "cache"
    calls = {"n": 0}

    def fake_collect(video_path, start, end, fps, roi_rect):
        calls["n"] += 1
        return [{"time": 0.0, "words": [{"text": "x", "x": 0, "y": 0, "w": 1, "h": 1}]}]

    first = _MODULE.collect_raw_frames_cached(
        video_path=video_path,
        duration=5.0,
        fps=2.0,
        roi_rect=(0, 0, 10, 10),
        cache_dir=cache_dir,
        cache_version="v1",
        collect_fn=fake_collect,
    )
    second = _MODULE.collect_raw_frames_cached(
        video_path=video_path,
        duration=5.0,
        fps=2.0,
        roi_rect=(0, 0, 10, 10),
        cache_dir=cache_dir,
        cache_version="v1",
        collect_fn=fake_collect,
    )
    assert first == second
    assert calls["n"] == 1
