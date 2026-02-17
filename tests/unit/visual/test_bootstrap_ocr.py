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


def test_collect_raw_frames_cached_filters_persistent_edge_overlay_tokens(tmp_path):
    video_path = tmp_path / "v.mp4"
    video_path.write_bytes(b"v")
    cache_dir = tmp_path / "cache"

    def fake_collect(video_path, start, end, fps, roi_rect):
        frames = []
        for i in range(30):
            frames.append(
                {
                    "time": float(i),
                    "words": [
                        {"text": "White", "x": 80, "y": 120, "w": 60, "h": 20},
                        {"text": "shirt", "x": 150, "y": 120, "w": 60, "h": 20},
                        {
                            "text": "SingKIN" if i % 2 == 0 else "SingKII",
                            "x": 500,
                            "y": 297,
                            "w": 60,
                            "h": 24,
                        },
                        {
                            "text": "KARAO" if i % 3 == 0 else "KARAOK",
                            "x": 507,
                            "y": 320,
                            "w": 46,
                            "h": 12,
                        },
                        {
                            "text": "KIN" if i % 4 == 0 else "KII",
                            "x": 534,
                            "y": 304,
                            "w": 22,
                            "h": 14,
                        },
                    ],
                }
            )
        return frames

    out = _MODULE.collect_raw_frames_cached(
        video_path=video_path,
        duration=30.0,
        fps=1.0,
        roi_rect=(0, 0, 560, 332),
        cache_dir=cache_dir,
        cache_version="v1",
        collect_fn=fake_collect,
    )
    texts = [w["text"] for f in out for w in f.get("words", [])]
    assert "White" in texts and "shirt" in texts
    assert all("SingK" not in t for t in texts)
    assert all("KARAO" not in t for t in texts)
    assert "KIN" not in texts
    assert "KII" not in texts


def test_collect_raw_frames_cached_keeps_edge_lyrics_without_variant_churn(tmp_path):
    video_path = tmp_path / "v.mp4"
    video_path.write_bytes(b"v")
    cache_dir = tmp_path / "cache"

    def fake_collect(video_path, start, end, fps, roi_rect):
        frames = []
        for i in range(30):
            frames.append(
                {
                    "time": float(i),
                    "words": [
                        {"text": "love", "x": 430, "y": 210, "w": 46, "h": 24},
                        {"text": "you", "x": 490, "y": 210, "w": 32, "h": 24},
                    ],
                }
            )
        return frames

    out = _MODULE.collect_raw_frames_cached(
        video_path=video_path,
        duration=30.0,
        fps=1.0,
        roi_rect=(0, 0, 560, 332),
        cache_dir=cache_dir,
        cache_version="v1",
        collect_fn=fake_collect,
    )
    texts = [w["text"] for f in out for w in f.get("words", [])]
    assert "love" in texts
    assert "you" in texts
