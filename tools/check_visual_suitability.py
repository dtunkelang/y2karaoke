#!/usr/bin/env python3
"""
Check if a karaoke video has suitable visual cues for bootstrapping.
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add src to path if running from tools/
src_path = Path(__file__).resolve().parents[1] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from y2karaoke.vision.ocr import get_ocr_engine  # noqa: E402
from y2karaoke.vision.color import infer_lyric_colors, classify_word_state  # noqa: E402
from y2karaoke.vision.roi import detect_lyric_roi  # noqa: E402
from y2karaoke.core.components.audio.downloader import YouTubeDownloader  # noqa: E402

try:
    import cv2
    import numpy as np
except ImportError:
    print("Error: OpenCV and Numpy are required.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_cache_key(video_path: Path, prefix: str, **kwargs) -> str:
    """Generate a unique filename for caching results based on parameters."""
    h = hashlib.md5(
        f"{video_path.name}_{json.dumps(kwargs, sort_keys=True)}".encode()
    ).hexdigest()
    return f"{prefix}_{h}.json"


def _collect_raw_frames_with_confidence(
    video_path: Path,
    start: float,
    end: float,
    fps: float,
    c_un: Any,
    c_sel: Any,
    roi_rect: tuple[int, int, int, int],
) -> list[dict[str, Any]]:
    """Perform OCR and extract word states including confidence."""
    ocr = get_ocr_engine()
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, start - 0.1) * 1000.0)
    step = max(int(round(src_fps / fps)), 1)
    rx, ry, rw, rh = roi_rect
    raw = []

    while True:
        ok, frame = cap.read()
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if not ok or t > end + 0.2:
            break
        if int(round(t * src_fps)) % step != 0:
            continue

        roi = frame[ry : ry + rh, rx : rx + rw]
        res = ocr.predict(roi)
        if res and res[0]:
            items = res[0]
            rec_texts = items.get("rec_texts", [])
            rec_boxes = items.get("rec_boxes", [])
            rec_scores = items.get("rec_scores", [1.0] * len(rec_texts))

            words = []
            for txt, box, conf in zip(rec_texts, rec_boxes, rec_scores):
                # box might be dict or points
                if isinstance(box, dict):
                    points = box["word"]
                else:
                    points = box

                np_box = np.array(points).reshape(-1, 2)
                x, y = int(min(np_box[:, 0])), int(min(np_box[:, 1]))
                bw, bh = int(max(np_box[:, 0]) - x), int(max(np_box[:, 1]) - y)

                word_roi = roi[y : y + bh, x : x + bw]
                state, ratio = classify_word_state(word_roi, c_un, c_sel)
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
    cap.release()
    return raw


def calculate_visual_suitability(raw_frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze OCR tokens to detect if the video has true word-level highlighting."""
    word_level_evidence_frames = 0
    total_active_frames = 0
    total_ocr_confidence = 0.0
    total_words_with_confidence = 0

    for frame in raw_frames:
        words = frame.get("words", [])
        if not words:
            continue

        # Group words by Y-coordinate (lines)
        lines: dict[int, list[dict[str, Any]]] = {}
        for w in words:
            y_bin = w["y"] // 20  # Group by approximate line
            if y_bin not in lines:
                lines[y_bin] = []
            lines[y_bin].append(w)

            if "confidence" in w:
                total_ocr_confidence += w["confidence"]
                total_words_with_confidence += 1

        has_any_highlight = False
        has_word_level_mix = False

        for y_bin, line_words in lines.items():
            states = [w["color"] for w in line_words]
            has_sel = any(s in ("selected", "mixed") for s in states)
            has_unsel = any(s == "unselected" for s in states)

            if has_sel:
                has_any_highlight = True
            # Word-level evidence: some words highlighted, some not in the same line
            if has_sel and has_unsel:
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

    return {
        "word_level_score": float(word_level_score),
        "avg_ocr_confidence": float(avg_confidence),
        "has_word_level_highlighting": word_level_score > 0.15,
        "detectability_score": float(
            avg_confidence * 0.7 + min(word_level_score * 2.0, 1.0) * 0.3
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Check karaoke visual suitability.")
    parser.add_argument("source", help="Path to karaoke video file or YouTube URL")
    parser.add_argument("--fps", type=float, default=1.0, help="Sampling FPS")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument(
        "--debug-lyrics",
        action="store_true",
        help="Print detected lyrics for debugging",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path(".cache/visual_suitability")
    )

    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    source = args.source
    if source.startswith("http"):
        print(f"Downloading video from: {source}")
        downloader = YouTubeDownloader(cache_dir=args.work_dir / "videos")
        try:
            info = downloader.download_video(source)
            video_path = Path(info["video_path"])
        except Exception as e:
            print(f"Error downloading video: {e}")
            return 1
    else:
        video_path = Path(source)

    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return 1

    print(f"Analyzing visual suitability for: {video_path.name}")

    # 1. Detect ROI
    roi_rect = detect_lyric_roi(video_path, sample_fps=1.0)

    # 2. Infer colors (with caching)
    color_cache_path = args.work_dir / _get_cache_key(
        video_path, "colors", roi=roi_rect
    )
    if color_cache_path.exists():
        print(f"Loading cached colors from {color_cache_path.name}...")
        cached_colors = json.loads(color_cache_path.read_text())
        c_un = np.array(cached_colors["c_un"])
        c_sel = np.array(cached_colors["c_sel"])
    else:
        c_un, c_sel, _ = infer_lyric_colors(video_path, roi_rect)
        color_cache_path.write_text(
            json.dumps({"c_un": c_un.tolist(), "c_sel": c_sel.tolist()})
        )

    # 3. Sample frames throughout the video (with caching)
    cap = cv2.VideoCapture(str(video_path))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()

    frames_cache_path = args.work_dir / _get_cache_key(
        video_path, "raw_frames", fps=args.fps, roi=roi_rect, c_un=c_un.tolist()
    )

    if frames_cache_path.exists():
        print(f"Loading cached frames from {frames_cache_path.name}...")
        raw_frames = json.loads(frames_cache_path.read_text())
    else:
        print(f"Sampling video ({duration:.1f}s) at {args.fps} FPS...")
        raw_frames = _collect_raw_frames_with_confidence(
            video_path, 0, duration, args.fps, c_un, c_sel, roi_rect
        )
        frames_cache_path.write_text(json.dumps(raw_frames))

    if args.debug_lyrics:
        print("\nDetected Lyrics per Frame:")
        for frame in raw_frames:
            txt = " ".join(
                [
                    f"[{w['text']}({w['color'][0].upper()}:{w.get('confidence', 0):.2f})]"
                    for w in frame["words"]
                ]
            )
            print(f"  {frame['timestamp']}: {txt}")

    # 4. Calculate metrics
    metrics = calculate_visual_suitability(raw_frames)

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print("\nVisual Suitability Results:")
        print(f"  Detectability Score: {metrics['detectability_score']:.4f}")
        print(f"  OCR Avg Confidence:  {metrics['avg_ocr_confidence']:.4f}")
        print(f"  Word-Level Score:    {metrics['word_level_score']:.4f}")
        print(f"  Has Word-Level Highlight: {metrics['has_word_level_highlighting']}")

        print("\nInterpretation:")
        score = metrics["detectability_score"]
        if score > 0.8:
            print(
                "  QUALITY: EXCELLENT - High contrast, clear word-level highlighting."
            )
        elif score > 0.5:
            print("  QUALITY: GOOD - Reliable for automated bootstrapping.")
        elif score > 0.3:
            print("  QUALITY: FAIR - Might only support line-level alignment.")
        else:
            print("  QUALITY: POOR - High noise or non-standard highlighting.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
