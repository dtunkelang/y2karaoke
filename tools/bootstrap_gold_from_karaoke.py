#!/usr/bin/env python3
"""
Bootstrap and refine word-level gold timings using computer vision.

This tool acts as a CLI wrapper around the `y2karaoke.vision` and `y2karaoke.core` libraries.
"""

from __future__ import annotations

import argparse
import hashlib
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
from y2karaoke.core.text_utils import make_slug  # noqa: E402
from y2karaoke.vision.ocr import get_ocr_engine  # noqa: E402
from y2karaoke.vision.roi import detect_lyric_roi  # noqa: E402
from y2karaoke.vision.suitability import analyze_visual_suitability  # noqa: E402

try:
    import cv2  # noqa: E402
    import numpy as np  # noqa: E402
except ImportError:
    print("Error: OpenCV and Numpy are required. Please install them.")
    sys.exit(1)

logger = get_logger(__name__)
RAW_OCR_CACHE_VERSION = "2"


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
        log_fn=logger.info,
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
    if args.candidate_url:
        logger.info(f"Using explicit candidate URL: {args.candidate_url}")
        return args.candidate_url, None, {}, []

    if not args.artist or not args.title:
        raise ValueError(
            "Either --candidate-url or both --artist and --title are required."
        )

    candidates = _search_karaoke_candidates(
        args.artist, args.title, args.max_candidates
    )
    if not candidates:
        raise ValueError(
            "No candidate videos found. Provide --candidate-url to continue."
        )

    ranked = _rank_candidates_by_suitability(
        candidates,
        downloader,
        song_dir,
        args.suitability_fps,
    )
    if not ranked:
        raise ValueError(
            "Could not evaluate candidate videos. Provide --candidate-url to continue."
        )

    if args.show_candidates:
        logger.info("Candidate ranking by visual suitability:")
        for idx, cand in enumerate(ranked, start=1):
            m = cand["metrics"]
            logger.info(
                "  %d. %.3f (word=%.3f ocr=%.3f) %s | %s",
                idx,
                m["detectability_score"],
                m["word_level_score"],
                m["avg_ocr_confidence"],
                cand.get("title", ""),
                cand["url"],
            )

    best = ranked[0]
    best_metrics = best["metrics"]
    if not args.allow_low_suitability and not _is_suitability_good_enough(
        best_metrics,
        args.min_detectability,
        args.min_word_level_score,
    ):
        raise ValueError(
            "Best candidate did not meet suitability thresholds. "
            "Use --allow-low-suitability to override."
        )

    return best["url"], Path(best["video_path"]), best_metrics, ranked


def _collect_raw_frames(
    video_path: Path,
    start: float,
    end: float,
    fps: float,
    roi_rect: tuple[int, int, int, int],
) -> list[dict]:
    ocr = get_ocr_engine()
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(src_fps / fps)), 1)
    rx, ry, rw, rh = roi_rect
    raw = []
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    frame_idx = max(int(round(start * src_fps)), 0)

    logger.info(f"Sampling frames at {fps} FPS...")
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
            items = res[0]
            rec_texts = items.get("rec_texts", [])
            rec_boxes = items.get("rec_boxes", [])

            words = []
            for txt, box_data in zip(rec_texts, rec_boxes):
                points = box_data["word"] if isinstance(box_data, dict) else box_data

                nb = np.array(points).reshape(-1, 2)
                x, y = int(min(nb[:, 0])), int(min(nb[:, 1]))
                bw, bh = int(max(nb[:, 0]) - x), int(max(nb[:, 1]) - y)
                words.append({"text": txt, "x": x, "y": y, "w": bw, "h": bh})

            if words:
                raw.append({"time": t, "words": words})
        frame_idx += 1

    cap.release()
    return raw


def _raw_frames_cache_path(
    video_path: Path,
    cache_dir: Path,
    fps: float,
    roi_rect: tuple[int, int, int, int],
    cache_version: str = RAW_OCR_CACHE_VERSION,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    sig = (
        f"{cache_version}:{video_path.resolve()}:{video_path.stat().st_mtime_ns}:"
        f"{video_path.stat().st_size}:{fps}:{roi_rect}"
    )
    digest = hashlib.md5(sig.encode()).hexdigest()
    return cache_dir / f"raw_frames_{digest}.json"


def _collect_raw_frames_cached(
    video_path: Path,
    duration: float,
    fps: float,
    roi_rect: tuple[int, int, int, int],
    cache_dir: Path,
    cache_version: str = RAW_OCR_CACHE_VERSION,
) -> list[dict]:
    cache_path = _raw_frames_cache_path(
        video_path, cache_dir, fps, roi_rect, cache_version=cache_version
    )
    if cache_path.exists():
        logger.info(f"Loading cached OCR frames: {cache_path.name}")
        return json.loads(cache_path.read_text())

    raw = _collect_raw_frames(video_path, 0, duration, fps, roi_rect)
    cache_path.write_text(json.dumps(raw))
    return raw


def _extract_audio_from_video(video_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / f"{video_path.stem}.extracted.wav"
    if audio_path.exists():
        return audio_path
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "44100",
            str(audio_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not audio_path.exists():
        raise RuntimeError(f"Audio extraction failed for {video_path}")
    return audio_path


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
    if cached_video_path is None:
        vid_info = downloader.download_video(
            candidate_url, output_dir=song_dir / "video"
        )
        v_path = Path(vid_info["video_path"])
    else:
        v_path = cached_video_path

    a_path: Optional[Path] = None
    if cached_video_path is not None:
        try:
            a_path = _extract_audio_from_video(v_path, song_dir / "video")
            logger.info(f"Extracted audio from cached candidate video: {a_path}")
        except Exception as e:
            logger.warning(
                "Could not extract audio from cached video (%s); falling back to "
                "direct audio download.",
                e,
            )
    if a_path is None:
        aud_info = downloader.download_audio(
            candidate_url, output_dir=song_dir / "video"
        )
        a_path = Path(aud_info["audio_path"])
    return v_path, a_path


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
    v_path: Path, args: argparse.Namespace, song_dir: Path
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
    refine_word_timings_at_high_fps(v_path, t_lines, roi)
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

    lines_out = _bootstrap_refined_lines(v_path, args, song_dir)

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
