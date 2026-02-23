"""Visual suitability analysis helpers for karaoke bootstrap workflows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency behavior
    cv2 = None  # type: ignore
    np = None  # type: ignore

from .ocr import get_ocr_engine, normalize_ocr_items
from .color import infer_lyric_colors, classify_word_state
from .roi import detect_lyric_roi
from ..exceptions import VisualRefinementError

_COLOR_CACHE_VERSION = "2"
_RAW_FRAMES_CACHE_VERSION = "4"


def _get_cache_key(video_path: Path, prefix: str, **kwargs: Any) -> str:
    """Generate a stable cache key for suitability artifacts."""
    stat = video_path.stat()
    digest = hashlib.md5(
        (
            f"{video_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:"
            f"{json.dumps(kwargs, sort_keys=True)}"
        ).encode()
    ).hexdigest()
    return f"{prefix}_{digest}.json"


def collect_raw_frames_with_confidence(
    video_path: Path,
    start: float,
    end: float,
    fps: float,
    c_un: Any,
    c_sel: Any,
    roi_rect: tuple[int, int, int, int],
) -> list[dict[str, Any]]:
    """Run OCR and color-state extraction over sampled frames."""
    if cv2 is None or np is None:
        raise ImportError("OpenCV and Numpy are required.")

    ocr = get_ocr_engine()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VisualRefinementError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, start - 0.1) * 1000.0)
    step = max(int(round(src_fps / max(fps, 0.01))), 1)
    frame_idx = max(int(round(max(0, start - 0.1) * src_fps)), 0)
    rx, ry, rw, rh = roi_rect
    raw: list[dict[str, Any]] = []

    while True:
        ok = cap.grab()
        if not ok:
            break
        t = frame_idx / src_fps
        if t > end + 0.2:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            frame_idx += 1
            continue

        roi = frame[ry : ry + rh, rx : rx + rw]
        res = ocr.predict(roi)
        if res and res[0]:
            items = normalize_ocr_items(res[0])
            rec_texts = items["rec_texts"]
            rec_boxes = items["rec_boxes"]
            rec_scores = items["rec_scores"]

            words = []
            for txt, box, conf in zip(rec_texts, rec_boxes, rec_scores):
                points = box["word"] if isinstance(box, dict) else box
                np_box = np.array(points).reshape(-1, 2)
                x, y = int(min(np_box[:, 0])), int(min(np_box[:, 1]))
                bw, bh = int(max(np_box[:, 0]) - x), int(max(np_box[:, 1]) - y)
                word_roi = roi[y : y + bh, x : x + bw]
                state, ratio, _ = classify_word_state(word_roi, c_un, c_sel)
                if state != "unknown":
                    words.append(
                        {
                            "text": txt,
                            "color": state,
                            "ratio": ratio,
                            "x": x,
                            "y": y,
                            "confidence": float(conf),
                        }
                    )
            if words:
                words.sort(key=lambda w: (w["y"] // 30, w["x"]))
                raw.append(
                    {
                        "time": t,
                        "timestamp": f"{int(t // 60):02d}:{t % 60:05.2f}",
                        "words": words,
                    }
                )
        frame_idx += 1
    cap.release()
    return raw


def calculate_visual_suitability(raw_frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate whether a video has usable word-level highlighting cues."""
    word_level_evidence_frames = 0
    total_active_frames = 0
    total_ocr_confidence = 0.0
    total_words_with_confidence = 0
    total_upper = 0
    total_lower = 0

    for frame in raw_frames:
        words = frame.get("words", [])
        if not words:
            continue

        lines: Dict[int, List[dict[str, Any]]] = {}
        for w in words:
            # Increase bin size to 40px to be more robust to vertical splitting/jitter
            y_bin = w["y"] // 40
            if y_bin not in lines:
                lines[y_bin] = []
            lines[y_bin].append(w)

            if "confidence" in w:
                total_ocr_confidence += w["confidence"]
                total_words_with_confidence += 1

            text = w.get("text", "")
            total_upper += sum(1 for ch in text if ch.isupper())
            total_lower += sum(1 for ch in text if ch.islower())

        has_any_highlight = False
        has_word_level_mix = False

        for line_words in lines.values():
            states = [w["color"] for w in line_words]
            has_sel = any(s in ("selected", "mixed") for s in states)
            has_unsel = any(s == "unselected" for s in states)
            has_mixed_word = any(s == "mixed" for s in states)

            if has_sel:
                has_any_highlight = True

            # Word-level evidence:
            # 1. A mix of selected and unselected words on the same line
            # 2. OR any word in a partially-highlighted 'mixed' state
            if (has_sel and has_unsel) or has_mixed_word:
                has_word_level_mix = True

        if has_any_highlight:
            total_active_frames += 1
            if has_word_level_mix:
                word_level_evidence_frames += 1

    word_level_score = (
        word_level_evidence_frames / total_active_frames
        if total_active_frames > 0
        else 0.0
    )
    avg_confidence = (
        total_ocr_confidence / total_words_with_confidence
        if total_words_with_confidence > 0
        else 0.0
    )

    is_all_caps = (total_upper + total_lower) >= 50 and total_lower < 0.1 * (
        total_upper + total_lower
    )

    return {
        "word_level_score": float(word_level_score),
        "avg_ocr_confidence": float(avg_confidence),
        "has_word_level_highlighting": word_level_score > 0.15,
        "is_all_caps": bool(is_all_caps),
        "detectability_score": float(
            avg_confidence * 0.7 + min(word_level_score * 2.0, 1.0) * 0.3
        ),
    }


def analyze_visual_suitability(
    video_path: Path,
    *,
    fps: float = 1.0,
    work_dir: Optional[Path] = None,
    roi_rect: Optional[tuple[int, int, int, int]] = None,
) -> Tuple[dict[str, Any], tuple[int, int, int, int]]:
    """Run full suitability analysis with optional caching."""
    if cv2 is None or np is None:
        raise ImportError("OpenCV and Numpy are required.")

    if not video_path.exists():
        raise VisualRefinementError(f"Video not found: {video_path}")

    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)

    if roi_rect is None:
        roi_rect = detect_lyric_roi(video_path, sample_fps=1.0)

    color_cache_path = (
        work_dir
        / _get_cache_key(
            video_path,
            "colors",
            roi=roi_rect,
            cache_version=_COLOR_CACHE_VERSION,
        )
        if work_dir is not None
        else None
    )
    if color_cache_path is not None and color_cache_path.exists():
        cached_colors = json.loads(color_cache_path.read_text())
        c_un = np.array(cached_colors["c_un"])
        c_sel = np.array(cached_colors["c_sel"])
    else:
        c_un, c_sel, _ = infer_lyric_colors(video_path, roi_rect)
        if color_cache_path is not None:
            color_cache_path.write_text(
                json.dumps({"c_un": c_un.tolist(), "c_sel": c_sel.tolist()})
            )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VisualRefinementError(f"Could not open video: {video_path}")
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()

    frames_cache_path = (
        work_dir
        / _get_cache_key(
            video_path,
            "raw_frames",
            fps=fps,
            roi=roi_rect,
            c_un=c_un.tolist(),
            c_sel=c_sel.tolist(),
            cache_version=_RAW_FRAMES_CACHE_VERSION,
        )
        if work_dir is not None
        else None
    )
    if frames_cache_path is not None and frames_cache_path.exists():
        raw_frames = json.loads(frames_cache_path.read_text())
    else:
        raw_frames = collect_raw_frames_with_confidence(
            video_path, 0, duration, fps, c_un, c_sel, roi_rect
        )
        if frames_cache_path is not None:
            frames_cache_path.write_text(json.dumps(raw_frames))

    metrics = calculate_visual_suitability(raw_frames)
    metrics["sampled_frames"] = len(raw_frames)
    metrics["duration_sec"] = float(duration)
    metrics["fps"] = float(fps)
    return metrics, roi_rect
