import json

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


def test_raw_frames_cache_path_changes_with_ocr_fingerprint(tmp_path, monkeypatch):
    video_path = tmp_path / "v.mp4"
    video_path.write_bytes(b"v")
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(_MODULE, "get_ocr_cache_fingerprint", lambda: "ocr-a")
    p1 = _MODULE.raw_frames_cache_path(
        video_path, cache_dir, 2.0, (0, 0, 10, 10), cache_version="v1"
    )
    monkeypatch.setattr(_MODULE, "get_ocr_cache_fingerprint", lambda: "ocr-b")
    p2 = _MODULE.raw_frames_cache_path(
        video_path, cache_dir, 2.0, (0, 0, 10, 10), cache_version="v1"
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


def test_collect_raw_frames_cached_applies_intro_filters_on_cache_load(tmp_path):
    video_path = tmp_path / "v.mp4"
    video_path.write_bytes(b"v")
    cache_dir = tmp_path / "cache"
    roi_rect = (0, 0, 560, 332)
    cache_version = "v1"

    cached_frames = []
    for i in range(20):
        t = i * 0.5
        cached_frames.append(
            {
                "time": t,
                "words": [
                    {"text": "Sing", "x": 170, "y": 150, "w": 70, "h": 40},
                    {"text": "King", "x": 245, "y": 150, "w": 90, "h": 40},
                    {"text": "Karaoke", "x": 185, "y": 198, "w": 145, "h": 25},
                ],
            }
        )
    for i in range(20, 40):
        t = i * 0.5
        cached_frames.append(
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

    cache_path = _MODULE.raw_frames_cache_path(
        video_path, cache_dir, 2.0, roi_rect, cache_version=cache_version
    )
    cache_path.write_text(json.dumps(cached_frames))

    calls = {"n": 0}

    def fake_collect(*_args, **_kwargs):
        calls["n"] += 1
        return []

    out = _MODULE.collect_raw_frames_cached(
        video_path=video_path,
        duration=20.0,
        fps=2.0,
        roi_rect=roi_rect,
        cache_dir=cache_dir,
        cache_version=cache_version,
        collect_fn=fake_collect,
    )
    first_half = [w["text"] for f in out[:20] for w in f.get("words", [])]
    second_half = [w["text"] for f in out[20:] for w in f.get("words", [])]
    assert calls["n"] == 0
    assert not first_half
    assert "You" in second_half and "come" in second_half


def test_collect_raw_frames_cached_ignores_dense_pre8s_intro_when_finding_lyrics_start(
    tmp_path,
):
    video_path = tmp_path / "v.mp4"
    video_path.write_bytes(b"v")
    cache_dir = tmp_path / "cache"

    def fake_collect(video_path, start, end, fps, roi_rect):
        frames = []
        # Early dense credit card (should not be treated as lyric start).
        for i in range(12):
            t = 4.0 + i * 0.25
            frames.append(
                {
                    "time": t,
                    "words": [
                        {"text": "Karaoke", "x": 160, "y": 185, "w": 120, "h": 22},
                        {"text": "Version", "x": 285, "y": 185, "w": 90, "h": 22},
                        {"text": "O", "x": 190, "y": 215, "w": 14, "h": 18},
                        {"text": "Connell", "x": 210, "y": 215, "w": 78, "h": 18},
                        {"text": "Universal", "x": 165, "y": 245, "w": 90, "h": 18},
                        {"text": "Music", "x": 262, "y": 245, "w": 56, "h": 18},
                        {"text": "Ltd", "x": 326, "y": 245, "w": 30, "h": 18},
                        {"text": "Kobalt", "x": 365, "y": 245, "w": 62, "h": 18},
                        {"text": "Publishing", "x": 430, "y": 245, "w": 95, "h": 18},
                    ],
                }
            )
        # Real lyric phase.
        for i in range(48):
            t = 16.0 + i * 0.25
            frames.append(
                {
                    "time": t,
                    "words": [
                        {"text": "White", "x": 90, "y": 20, "w": 42, "h": 18},
                        {"text": "shirt", "x": 136, "y": 20, "w": 44, "h": 18},
                        {"text": "now", "x": 184, "y": 20, "w": 32, "h": 18},
                        {"text": "red", "x": 220, "y": 20, "w": 28, "h": 18},
                        {"text": "My", "x": 96, "y": 90, "w": 26, "h": 18},
                        {"text": "bloody", "x": 126, "y": 90, "w": 58, "h": 18},
                        {"text": "nose", "x": 188, "y": 90, "w": 42, "h": 18},
                        {"text": "Sleepin", "x": 98, "y": 160, "w": 64, "h": 18},
                        {"text": "on", "x": 166, "y": 160, "w": 22, "h": 18},
                        {"text": "your", "x": 192, "y": 160, "w": 36, "h": 18},
                        {"text": "tippy", "x": 96, "y": 230, "w": 46, "h": 18},
                        {"text": "toes", "x": 146, "y": 230, "w": 40, "h": 18},
                    ],
                }
            )
        return frames

    out = _MODULE.collect_raw_frames_cached(
        video_path=video_path,
        duration=30.0,
        fps=2.0,
        roi_rect=(0, 0, 560, 332),
        cache_dir=cache_dir,
        cache_version="v1",
        collect_fn=fake_collect,
    )
    early_texts = [
        w["text"]
        for f in out
        if float(f.get("time", 0.0)) < 15.75
        for w in f.get("words", [])
    ]
    later_texts = [
        w["text"]
        for f in out
        if float(f.get("time", 0.0)) >= 16.0
        for w in f.get("words", [])
    ]
    assert not early_texts
    assert "White" in later_texts and "bloody" in later_texts


def test_suppress_transient_digit_heavy_frames_clears_single_glitch_frame():
    frames = [
        {
            "time": 205.76,
            "words": [
                {"text": "Although"},
                {"text": "my"},
                {"text": "heart"},
                {"text": "is"},
                {"text": "falling"},
                {"text": "too"},
                {"text": "I'm"},
            ],
        },
        {
            "time": 206.08,
            "words": [
                {"text": "116"},
                {"text": "010"},
                {"text": "10lai"},
                {"text": "as"},
                {"text": "mater"},
            ],
        },
        {
            "time": 206.40,
            "words": [
                {"text": "Although"},
                {"text": "my"},
                {"text": "heart"},
                {"text": "is"},
                {"text": "falling"},
                {"text": "too"},
                {"text": "I'm"},
                {"text": "in"},
            ],
        },
    ]
    out = _MODULE._suppress_transient_digit_heavy_frames(frames)
    assert out[0]["words"]
    assert out[2]["words"]
    assert out[1]["words"] == []


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


@pytest.mark.skipif(_MODULE.cv2 is None or _MODULE.np is None, reason="opencv required")
def test_append_predicted_words_rejects_blank_frame_hallucinations():
    np = _MODULE.np
    raw = []
    roi = np.zeros((120, 320, 3), dtype=np.uint8)
    pred = [
        {
            "rec_texts": ["116", "010", "10lai"],
            "rec_boxes": [
                {"word": [[50, 40], [95, 40], [95, 58], [50, 58]]},
                {"word": [[100, 40], [135, 40], [135, 58], [100, 58]]},
                {"word": [[140, 40], [200, 40], [200, 58], [140, 58]]},
            ],
            "rec_scores": [0.9, 0.8, 0.75],
        }
    ]

    _MODULE._append_predicted_words(
        raw,
        pred,
        [206.08],
        roi_frames=[roi],
        roi_shapes=[roi.shape[:2]],
    )

    assert raw == []


@pytest.mark.skipif(_MODULE.cv2 is None or _MODULE.np is None, reason="opencv required")
def test_append_predicted_words_keeps_boxes_with_visible_text_ink():
    np = _MODULE.np
    cv2 = _MODULE.cv2
    raw = []
    roi = np.zeros((120, 320, 3), dtype=np.uint8)
    # White glyph-like band in OCR box region.
    cv2.rectangle(roi, (60, 44), (190, 56), (255, 255, 255), thickness=-1)
    pred = [
        {
            "rec_texts": ["And", "trust", "me"],
            "rec_boxes": [
                {"word": [[60, 40], [92, 40], [92, 60], [60, 60]]},
                {"word": [[98, 40], [152, 40], [152, 60], [98, 60]]},
                {"word": [[158, 40], [190, 40], [190, 60], [158, 60]]},
            ],
            "rec_scores": [0.96, 0.95, 0.94],
        }
    ]

    _MODULE._append_predicted_words(
        raw,
        pred,
        [30.4],
        roi_frames=[roi],
        roi_shapes=[roi.shape[:2]],
    )

    assert len(raw) == 1
    assert [w["text"] for w in raw[0]["words"]] == ["And", "trust", "me"]
