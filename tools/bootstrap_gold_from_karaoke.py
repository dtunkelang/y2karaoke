#!/usr/bin/env python3
"""
Bootstrap and refine word-level gold timings using computer vision.

This tool acts as a CLI wrapper around the `y2karaoke.vision` and `y2karaoke.core` libraries.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path if running from tools/
src_path = Path(__file__).resolve().parents[1] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from y2karaoke.utils.logging import setup_logging, get_logger  # noqa: E402
from y2karaoke.core.components.audio.downloader import YouTubeDownloader  # noqa: E402
from y2karaoke.core.visual.refinement import (  # noqa: E402
    refine_word_timings_at_high_fps,
)
from y2karaoke.core.visual.reconstruction import (  # noqa: E402
    reconstruct_lyrics_from_visuals,
)
from y2karaoke.core.visual.bootstrap_postprocess import (  # noqa: E402
    build_refined_lines_output as _build_refined_lines_output_impl,
    nearest_known_word_indices as _nearest_known_word_indices_impl,
)
from y2karaoke.core.visual.bootstrap_runtime import (  # noqa: E402
    build_run_report_payload as _build_run_report_payload_impl,
    ensure_selected_suitability as _ensure_selected_suitability_impl,
    is_suitability_good_enough as _is_suitability_good_enough_impl,
    write_run_report as _write_run_report_impl,
)
from y2karaoke.core.visual.bootstrap_candidates import (  # noqa: E402
    rank_candidates_by_suitability as _rank_candidates_by_suitability_impl,
    search_karaoke_candidates as _search_karaoke_candidates_impl,
)
from y2karaoke.core.visual.bootstrap_ocr import (  # noqa: E402
    collect_raw_frames as _collect_raw_frames_impl,
    collect_raw_frames_cached as _collect_raw_frames_cached_impl,
    raw_frames_cache_path as _raw_frames_cache_path_impl,
)
from y2karaoke.core.visual.bootstrap_media import (  # noqa: E402
    extract_audio_from_video as _extract_audio_from_video_impl,
    resolve_media_paths as _resolve_media_paths_impl,
)
from y2karaoke.core.visual.bootstrap_selection import (  # noqa: E402
    select_candidate_with_rankings as _select_candidate_with_rankings_impl,
)
from y2karaoke.core.text_utils import make_slug  # noqa: E402
from y2karaoke.vision.ocr import get_ocr_engine  # noqa: E402,F401
from y2karaoke.vision.roi import detect_lyric_roi  # noqa: E402
from y2karaoke.vision.suitability import analyze_visual_suitability  # noqa: E402

try:
    import cv2  # noqa: E402
    import numpy as np  # noqa: E402,F401
except ImportError:
    print("Error: OpenCV and Numpy are required. Please install them.")
    sys.exit(1)

logger = get_logger(__name__)
RAW_OCR_CACHE_VERSION = "3"
LINE_LEVEL_REFINE_SKIP_THRESHOLD = 0.05


def _search_karaoke_candidates(
    artist: str,
    title: str,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """Backward-compatible wrapper around shared candidate search helper."""
    return _search_karaoke_candidates_impl(
        artist, title, max_candidates, log_fn=logger.warning
    )


def _rank_candidates_by_suitability(
    candidates: list[dict[str, Any]],
    downloader: YouTubeDownloader,
    song_dir: Path,
    suitability_fps: float,
) -> list[dict[str, Any]]:
    """Backward-compatible wrapper around shared candidate ranking helper."""
    return _rank_candidates_by_suitability_impl(
        candidates,
        downloader=downloader,
        song_dir=song_dir,
        suitability_fps=suitability_fps,
        analyze_fn=analyze_visual_suitability,
        log_info_fn=logger.info,
        log_warning_fn=logger.warning,
    )


def _is_suitability_good_enough(
    metrics: dict[str, Any],
    min_detectability: float,
    min_word_level_score: float,
) -> bool:
    return _is_suitability_good_enough_impl(
        metrics, min_detectability, min_word_level_score
    )


def _select_candidate(
    args: argparse.Namespace,
    downloader: YouTubeDownloader,
    song_dir: Path,
) -> tuple[str, Optional[Path], dict[str, Any]]:
    """Resolve candidate URL and optional pre-downloaded video path."""
    candidate_url, cached_video_path, metrics, _ = _select_candidate_with_rankings(
        args, downloader, song_dir
    )
    return candidate_url, cached_video_path, metrics


def _select_candidate_with_rankings(
    args: argparse.Namespace,
    downloader: YouTubeDownloader,
    song_dir: Path,
) -> tuple[str, Optional[Path], dict[str, Any], list[dict[str, Any]]]:
    """Resolve candidate URL plus evaluated rankings for reporting."""
    return _select_candidate_with_rankings_impl(
        candidate_url=args.candidate_url,
        artist=args.artist,
        title=args.title,
        max_candidates=args.max_candidates,
        suitability_fps=args.suitability_fps,
        show_candidates=args.show_candidates,
        allow_low_suitability=args.allow_low_suitability,
        min_detectability=args.min_detectability,
        min_word_level_score=args.min_word_level_score,
        downloader=downloader,
        song_dir=song_dir,
        search_fn=_search_karaoke_candidates,
        rank_fn=_rank_candidates_by_suitability,
        suitability_check_fn=_is_suitability_good_enough,
        log_info_fn=logger.info,
    )


def _collect_raw_frames(
    video_path: Path,
    start: float,
    end: float,
    fps: float,
    roi_rect: tuple[int, int, int, int],
) -> list[dict]:
    """Backward-compatible wrapper around shared OCR collection helper."""
    return _collect_raw_frames_impl(
        video_path,
        start,
        end,
        fps,
        roi_rect,
        log_fn=logger.info,
        ocr_engine_fn=get_ocr_engine,
    )


def _raw_frames_cache_path(
    video_path: Path,
    cache_dir: Path,
    fps: float,
    roi_rect: tuple[int, int, int, int],
    cache_version: str = RAW_OCR_CACHE_VERSION,
) -> Path:
    """Backward-compatible wrapper around shared OCR cache-key helper."""
    return _raw_frames_cache_path_impl(
        video_path,
        cache_dir,
        fps,
        roi_rect,
        cache_version=cache_version,
    )


def _collect_raw_frames_cached(
    video_path: Path,
    duration: float,
    fps: float,
    roi_rect: tuple[int, int, int, int],
    cache_dir: Path,
    cache_version: str = RAW_OCR_CACHE_VERSION,
) -> list[dict]:
    """Backward-compatible wrapper around shared OCR cache helper."""
    return _collect_raw_frames_cached_impl(
        video_path,
        duration,
        fps,
        roi_rect,
        cache_dir,
        cache_version=cache_version,
        log_fn=logger.info,
        collect_fn=_collect_raw_frames,
    )


def _extract_audio_from_video(video_path: Path, output_dir: Path) -> Path:
    """Backward-compatible wrapper around shared media helper."""
    return _extract_audio_from_video_impl(
        video_path,
        output_dir,
        run_fn=subprocess.run,
    )


def _clamp_confidence(value: Optional[float], default: float = 0.0) -> float:
    """Backward-compatible helper retained for existing tests/callers."""
    if value is None:
        value = default
    return max(0.0, min(1.0, float(value)))


def _nearest_known_word_indices(
    known_indices: List[int], n_words: int
) -> tuple[List[int], List[int]]:
    """Backward-compatible wrapper around shared post-processing helper."""
    return _nearest_known_word_indices_impl(known_indices, n_words)


def _build_refined_lines_output(
    t_lines: list[Any], artist: Optional[str], title: Optional[str]
) -> list[dict[str, Any]]:
    """Backward-compatible wrapper around shared post-processing helper."""
    return _build_refined_lines_output_impl(t_lines, artist=artist, title=title)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--artist")
    p.add_argument("--title")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, default=Path(".cache/karaoke_bootstrap"))
    p.add_argument("--report-json", type=Path, default=None)
    p.add_argument("--candidate-url")
    p.add_argument("--visual-fps", type=float, default=2.0)
    p.add_argument("--max-candidates", type=int, default=5)
    p.add_argument("--show-candidates", action="store_true")
    p.add_argument("--suitability-fps", type=float, default=1.0)
    p.add_argument("--min-detectability", type=float, default=0.45)
    p.add_argument("--min-word-level-score", type=float, default=0.15)
    p.add_argument("--raw-ocr-cache-version", default=RAW_OCR_CACHE_VERSION)
    p.add_argument("--allow-low-suitability", action="store_true")
    p.add_argument(
        "--strict-sequential", action="store_true"
    )  # Kept for API compatibility
    return p.parse_args()


def _resolve_media_paths(
    downloader: YouTubeDownloader,
    candidate_url: str,
    cached_video_path: Optional[Path],
    song_dir: Path,
) -> tuple[Path, Path]:
    """Backward-compatible wrapper around shared media resolver helper."""

    def _log_message(message: str) -> None:
        if message.startswith("Could not extract audio"):
            logger.warning(message)
        else:
            logger.info(message)

    return _resolve_media_paths_impl(
        downloader,
        candidate_url,
        cached_video_path,
        song_dir,
        extract_audio_fn=_extract_audio_from_video,
        log_fn=_log_message,
    )


def _ensure_selected_suitability(
    selected_metrics: dict[str, Any],
    *,
    v_path: Path,
    song_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    try:
        return _ensure_selected_suitability_impl(
            selected_metrics,
            v_path=v_path,
            song_dir=song_dir,
            suitability_fps=args.suitability_fps,
            min_detectability=args.min_detectability,
            min_word_level_score=args.min_word_level_score,
            allow_low_suitability=args.allow_low_suitability,
            analyze_fn=analyze_visual_suitability,
        )
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        logger.warning(f"Suitability check failed for selected video: {e}")
        return {}


def _bootstrap_refined_lines(
    v_path: Path,
    args: argparse.Namespace,
    song_dir: Path,
    selected_metrics: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    roi = detect_lyric_roi(v_path, sample_fps=1.0)
    cap = cv2.VideoCapture(str(v_path))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()

    raw_frames = _collect_raw_frames_cached(
        v_path,
        duration,
        args.visual_fps,
        roi,
        song_dir / "cache",
        cache_version=args.raw_ocr_cache_version,
    )
    t_lines = reconstruct_lyrics_from_visuals(raw_frames, args.visual_fps)
    logger.info(f"Reconstructed {len(t_lines)} initial lines.")

    word_level_score = 0.0
    if selected_metrics:
        word_level_score = float(selected_metrics.get("word_level_score", 0.0))

    if word_level_score >= LINE_LEVEL_REFINE_SKIP_THRESHOLD:
        refine_word_timings_at_high_fps(v_path, t_lines, roi)
    else:
        logger.info(
            "Skipping high-FPS visual refinement due to low word-level suitability "
            f"(word_level_score={word_level_score:.3f} < "
            f"{LINE_LEVEL_REFINE_SKIP_THRESHOLD:.3f}); using line-level timing fallback."
        )

    return _build_refined_lines_output(t_lines, artist=args.artist, title=args.title)


def _write_run_report(
    args: argparse.Namespace,
    *,
    candidate_url: str,
    selected_metrics: dict[str, Any],
    ranked_candidates: list[dict[str, Any]],
) -> None:
    if not args.report_json:
        return
    report = _build_run_report_payload_impl(
        artist=args.artist,
        title=args.title,
        output_path=args.output,
        candidate_url=candidate_url,
        selected_metrics=selected_metrics,
        ranked_candidates=ranked_candidates,
        visual_fps=args.visual_fps,
        suitability_fps=args.suitability_fps,
        min_detectability=args.min_detectability,
        min_word_level_score=args.min_word_level_score,
        raw_ocr_cache_version=args.raw_ocr_cache_version,
        allow_low_suitability=args.allow_low_suitability,
    )
    _write_run_report_impl(args.report_json, report)
    logger.info(f"Wrote bootstrap report to {args.report_json}")


def main():
    setup_logging(verbose=True)
    args = _parse_args()

    slug_artist = make_slug(args.artist or "unk")
    slug_title = make_slug(args.title or "unk")
    song_dir = args.work_dir / slug_artist / slug_title

    downloader = YouTubeDownloader(cache_dir=song_dir.parent)
    try:
        (
            candidate_url,
            cached_video_path,
            selected_metrics,
            ranked_candidates,
        ) = _select_candidate_with_rankings(args, downloader, song_dir)
    except ValueError as e:
        logger.error(str(e))
        return 1

    logger.info(f"Selected candidate URL: {candidate_url}")

    try:
        v_path, a_path = _resolve_media_paths(
            downloader, candidate_url, cached_video_path, song_dir
        )
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return 1

    try:
        selected_metrics = _ensure_selected_suitability(
            selected_metrics, v_path=v_path, song_dir=song_dir, args=args
        )
    except ValueError as e:
        logger.error(str(e))
        return 1

    lines_out = _bootstrap_refined_lines(
        v_path, args, song_dir, selected_metrics=selected_metrics
    )

    res: Dict[str, Any] = {
        "schema_version": "1.2",
        "title": args.title,
        "artist": args.artist,
        "candidate_url": candidate_url,
        "visual_suitability": selected_metrics,
        "audio_path": str(a_path.resolve()) if a_path else "",
        "lines": lines_out,
    }

    args.output.write_text(json.dumps(res, indent=2))
    logger.info(f"Saved refined gold timings to {args.output}")
    _write_run_report(
        args,
        candidate_url=candidate_url,
        selected_metrics=selected_metrics,
        ranked_candidates=ranked_candidates,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
