import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add current directory to path for sibling imports
sys.path.append(str(Path(__file__).resolve().parent))

from bootstrap_gold_from_karaoke import (  # noqa: E402
    detect_lyric_roi,
    _infer_lyric_colors,
    _collect_raw_frames,
    _cv2,
)


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
        lines = {}
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
    parser.add_argument("video", help="Path to karaoke video file")
    parser.add_argument("--fps", type=float, default=1.0, help="Sampling FPS")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument(
        "--work-dir", type=Path, default=Path(".cache/visual_suitability")
    )

    args = parser.parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return 1

    args.work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing visual suitability for: {video_path.name}")

    # 1. Detect ROI
    roi_rect = detect_lyric_roi(video_path, args.work_dir)

    # 2. Infer colors
    c_un, c_sel, _ = _infer_lyric_colors(video_path, roi_rect)

    # 3. Sample frames throughout the video
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(video_path))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()

    print(f"Sampling video ({duration:.1f}s) at {args.fps} FPS...")
    raw_frames = _collect_raw_frames(
        video_path, 0, duration, args.fps, c_un, c_sel, roi_rect
    )

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
