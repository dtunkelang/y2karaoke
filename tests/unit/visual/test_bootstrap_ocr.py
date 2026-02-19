from y2karaoke.core.visual import bootstrap_ocr as _MODULE
import pytest


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


def test_collect_raw_frames_cached_filters_stable_corner_logo_without_variants(
    tmp_path,
):
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
                        {"text": "You", "x": 72, "y": 120, "w": 36, "h": 20},
                        {"text": "know", "x": 118, "y": 120, "w": 50, "h": 20},
                        {"text": "Sing", "x": 500, "y": 294, "w": 42, "h": 18},
                        {"text": "King", "x": 506, "y": 314, "w": 44, "h": 14},
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
    assert "You" in texts and "know" in texts
    assert "Sing" not in texts
    assert "King" not in texts


def test_collect_raw_frames_cached_suppresses_intro_title_cards(tmp_path):
    video_path = tmp_path / "v.mp4"
    video_path.write_bytes(b"v")
    cache_dir = tmp_path / "cache"

    def fake_collect(video_path, start, end, fps, roi_rect):
        frames = []
        # Intro/title phase (few lines, non-lyric card text).
        for i in range(20):
            t = i * 0.5
            frames.append(
                {
                    "time": t,
                    "words": [
                        {"text": "Sing", "x": 170, "y": 150, "w": 70, "h": 40},
                        {"text": "King", "x": 245, "y": 150, "w": 90, "h": 40},
                        {"text": "Karaoke", "x": 185, "y": 198, "w": 145, "h": 25},
                    ],
                }
            )
        # Lyrics phase (4-line dense blocks).
        for i in range(20, 40):
            t = i * 0.5
            frames.append(
                {
                    "time": t,
                    "words": [
                        {"text": "You", "x": 80, "y": 60, "w": 30, "h": 18},
                        {"text": "come", "x": 118, "y": 60, "w": 48, "h": 18},
                        {"text": "over", "x": 90, "y": 100, "w": 42, "h": 18},
                        {"text": "and", "x": 140, "y": 100, "w": 34, "h": 18},
                        {"text": "start", "x": 102, "y": 140, "w": 44, "h": 18},
                        {"text": "up", "x": 154, "y": 140, "w": 22, "h": 18},
                        {"text": "with", "x": 112, "y": 180, "w": 38, "h": 18},
                        {"text": "me", "x": 158, "y": 180, "w": 26, "h": 18},
                    ],
                }
            )
        return frames

    out = _MODULE.collect_raw_frames_cached(
        video_path=video_path,
        duration=20.0,
        fps=2.0,
        roi_rect=(0, 0, 560, 332),
        cache_dir=cache_dir,
        cache_version="v1",
        collect_fn=fake_collect,
    )
    first_half = [w["text"] for f in out[:20] for w in f.get("words", [])]
    second_half = [w["text"] for f in out[20:] for w in f.get("words", [])]
    assert not first_half
    assert "You" in second_half and "come" in second_half


def test_collect_raw_frames_cached_filters_large_early_title_words(tmp_path):
    video_path = tmp_path / "v.mp4"
    video_path.write_bytes(b"v")
    cache_dir = tmp_path / "cache"

    def fake_collect(video_path, start, end, fps, roi_rect):
        frames = []
        for i in range(40):
            t = i * 0.5
            words = [{"text": "You", "x": 80, "y": 120, "w": 30, "h": 18}]
            if t <= 12.0:
                words.append({"text": "Shape", "x": 48, "y": 68, "w": 190, "h": 67})
            frames.append({"time": t, "words": words})
        return frames

    out = _MODULE.collect_raw_frames_cached(
        video_path=video_path,
        duration=20.0,
        fps=2.0,
        roi_rect=(0, 0, 560, 332),
        cache_dir=cache_dir,
        cache_version="v1",
        collect_fn=fake_collect,
    )
    texts = [w["text"] for f in out for w in f.get("words", [])]
    assert "Shape" not in texts
    assert "You" in texts


def test_build_line_boxes_groups_words_by_row():
    words = [
        {"text": "You", "x": 20, "y": 40, "w": 30, "h": 12},
        {"text": "come", "x": 58, "y": 42, "w": 40, "h": 12},
        {"text": "over", "x": 25, "y": 80, "w": 35, "h": 12},
    ]
    line_boxes = _MODULE._build_line_boxes(words)
    assert len(line_boxes) == 2
    assert line_boxes[0]["tokens"] == ["You", "come"]
    assert line_boxes[1]["tokens"] == ["over"]
    # Padded boxes should extend beyond raw word extents.
    assert line_boxes[0]["x"] < 20
    assert line_boxes[0]["w"] > (98 - 20)


def test_build_line_boxes_robust_to_single_outlier_height():
    words = [
        {"text": "One", "x": 20, "y": 40, "w": 30, "h": 12},
        {"text": "line", "x": 58, "y": 41, "w": 40, "h": 12},
        # Outlier OCR box that is much taller should not overly shift center.
        {"text": "noise", "x": 102, "y": 30, "w": 40, "h": 30},
    ]
    line_boxes = _MODULE._build_line_boxes(words)
    assert len(line_boxes) == 1
    lb = line_boxes[0]
    line_center_y = lb["y"] + lb["h"] / 2.0
    assert 42.0 <= line_center_y <= 48.0


@pytest.mark.skipif(_MODULE.cv2 is None or _MODULE.np is None, reason="opencv required")
def test_build_line_boxes_refines_to_text_mask_center():
    np = _MODULE.np
    cv2 = _MODULE.cv2

    roi = np.zeros((90, 220, 3), dtype=np.uint8)
    # Simulate one rendered lyric line.
    cv2.rectangle(roi, (60, 36), (174, 50), (255, 255, 255), thickness=-1)
    words = [
        # OCR words are biased upward relative to real glyph bounds.
        {"text": "And", "x": 60, "y": 29, "w": 34, "h": 13},
        {"text": "trust", "x": 98, "y": 30, "w": 44, "h": 12},
        {"text": "me", "x": 145, "y": 30, "w": 28, "h": 12},
    ]

    line_boxes = _MODULE._build_line_boxes(words, roi_nd=roi)
    assert len(line_boxes) == 1
    lb = line_boxes[0]

    # The refined box should cover the rendered text band with small margins.
    assert lb["y"] <= 36
    assert lb["y"] + lb["h"] >= 50
    center_y = lb["y"] + lb["h"] / 2.0
    assert 40.0 <= center_y <= 48.0
