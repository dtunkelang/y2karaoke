#!/usr/bin/env python3
"""
Check if a karaoke video has suitable visual cues for bootstrapping.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path if running from tools/
src_path = Path(__file__).resolve().parents[1] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from y2karaoke.vision.suitability import (  # noqa: E402
    analyze_visual_suitability,
    calculate_visual_suitability as _calculate_visual_suitability,
)
from y2karaoke.core.components.audio.downloader import YouTubeDownloader  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_visual_suitability(raw_frames):
    """Backward-compatible export for tests and existing callers."""
    return _calculate_visual_suitability(raw_frames)


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

    metrics, _ = analyze_visual_suitability(
        video_path, fps=args.fps, work_dir=args.work_dir
    )

    if args.json:
        import json

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
