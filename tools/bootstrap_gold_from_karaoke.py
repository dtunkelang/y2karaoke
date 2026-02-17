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
    snap,
)
from y2karaoke.core.text_utils import make_slug, normalize_text_basic  # noqa: E402
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
    """Find candidate karaoke videos from YouTube search."""
    if not artist or not title:
        return []

    try:
        import yt_dlp
    except Exception:
        logger.warning("yt_dlp not available for candidate search")
        return []

    query = f"{artist} {title} karaoke"
    search_term = f"ytsearch{max_candidates}:{query}"
    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
    }

    candidates: list[dict[str, Any]] = []
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(search_term, download=False)
        entries = info.get("entries", []) if isinstance(info, dict) else []

        for ent in entries:
            if not isinstance(ent, dict):
                continue
            video_id = ent.get("id")
            if not video_id:
                continue
            url = f"https://www.youtube.com/watch?v={video_id}"
            candidates.append(
                {
                    "url": url,
                    "title": ent.get("title") or "",
                    "uploader": ent.get("uploader") or "",
                    "duration": ent.get("duration"),
                }
            )
    except Exception as exc:
        logger.warning(f"Candidate search failed: {exc}")

    return candidates


def _rank_candidates_by_suitability(
    candidates: list[dict[str, Any]],
    downloader: YouTubeDownloader,
    song_dir: Path,
    suitability_fps: float,
) -> list[dict[str, Any]]:
    """Download and score each candidate by visual suitability."""
    ranked: list[dict[str, Any]] = []

    for idx, cand in enumerate(candidates, start=1):
        url = cand["url"]
        eval_dir = song_dir / "candidates" / f"candidate_{idx:02d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        try:
            vid_info = downloader.download_video(url, output_dir=eval_dir)
            video_path = Path(vid_info["video_path"])
            metrics, _ = analyze_visual_suitability(
                video_path,
                fps=suitability_fps,
                work_dir=eval_dir / "suitability",
            )
            ranked.append(
                {
                    **cand,
                    "video_path": str(video_path),
                    "metrics": metrics,
                    "score": float(metrics["detectability_score"]),
                }
            )
            logger.info(
                "Candidate %d score=%.3f word_level=%.3f title=%s",
                idx,
                metrics["detectability_score"],
                metrics["word_level_score"],
                cand.get("title", ""),
            )
        except Exception as exc:
            logger.warning(f"Skipping candidate {url}: {exc}")

    ranked.sort(
        key=lambda c: (
            c["score"],
            c["metrics"].get("word_level_score", 0.0),
            c["metrics"].get("avg_ocr_confidence", 0.0),
        ),
        reverse=True,
    )
    return ranked


def _is_suitability_good_enough(
    metrics: dict[str, Any],
    min_detectability: float,
    min_word_level_score: float,
) -> bool:
    return (
        float(metrics.get("detectability_score", 0.0)) >= min_detectability
        and float(metrics.get("word_level_score", 0.0)) >= min_word_level_score
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


def _clamp_confidence(value: Optional[float], default: float = 0.0) -> float:
    if value is None:
        value = default
    return max(0.0, min(1.0, float(value)))


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


def _build_refined_lines_output(
    t_lines: list[Any], artist: Optional[str], title: Optional[str]
) -> list[dict[str, Any]]:
    lines_out: List[dict[str, Any]] = []
    prev_line_end = 5.0
    normalized_title = normalize_text_basic(title or "")
    normalized_artist = normalize_text_basic(artist or "")

    for idx, ln in enumerate(t_lines):
        if ln.start < 7.0 and (
            not ln.word_starts or all(s is None for s in ln.word_starts)
        ):
            continue

        norm_txt = normalize_text_basic(ln.text)
        if norm_txt in [normalized_title, normalized_artist]:
            continue

        w_out: List[dict[str, Any]] = []
        n_words = len(ln.words)
        l_start = max(ln.start, prev_line_end)

        if not ln.word_starts or all(s is None for s in ln.word_starts):
            line_duration = max((ln.end or (l_start + 1.0)) - l_start, 1.0)
            step = line_duration / max(n_words, 1)
            for j, txt in enumerate(ln.words):
                ws = l_start + j * step
                we = ws + step
                w_out.append(
                    {
                        "word_index": j + 1,
                        "text": txt,
                        "start": snap(ws),
                        "end": snap(we),
                        "confidence": 0.0,
                    }
                )
        else:
            word_starts = ln.word_starts
            word_ends = ln.word_ends or [None] * n_words
            word_confidences = ln.word_confidences or [None] * n_words

            vs = [j for j, s in enumerate(word_starts) if s is not None]
            out_s: List[float] = []
            out_e: List[float] = []
            out_c: List[float] = []

            for j in range(n_words):
                ws_val = word_starts[j]
                if ws_val is not None:
                    out_s.append(ws_val)
                    out_e.append(word_ends[j] or ws_val + 0.1)
                    out_c.append(_clamp_confidence(word_confidences[j], default=0.5))
                else:
                    prev_v = max([k for k in vs if k < j], default=-1)
                    next_v = min([k for k in vs if k > j], default=-1)

                    if prev_v == -1:
                        base = ln.start
                        first_vs_val = word_starts[vs[0]] if vs else base + 1.0
                        assert first_vs_val is not None
                        next_t = first_vs_val
                        step = max(0.1, (next_t - base) / (len(vs) + 1 if vs else 2))
                        out_s.append(
                            max(
                                base,
                                (
                                    next_t - (vs[0] - j + 1) * step
                                    if vs
                                    else base + j * 0.5
                                ),
                            )
                        )
                    elif next_v == -1:
                        base = out_e[prev_v]
                        out_s.append(base + (j - prev_v) * 0.3)
                    else:
                        frac = (j - prev_v) / (next_v - prev_v)
                        next_vs_val = word_starts[next_v]
                        assert next_vs_val is not None
                        out_s.append(
                            out_e[prev_v] + frac * (next_vs_val - out_e[prev_v])
                        )
                    out_e.append(out_s[-1] + 0.1)
                    out_c.append(0.25)

            for j in range(n_words):
                if j == 0:
                    out_s[j] = max(out_s[j], prev_line_end)
                else:
                    out_s[j] = max(out_s[j], out_e[j - 1] + 0.05)

                out_e[j] = min(max(out_e[j], out_s[j] + 0.1), out_s[j] + 0.8)

                w_out.append(
                    {
                        "word_index": j + 1,
                        "text": ln.words[j],
                        "start": snap(out_s[j]),
                        "end": snap(out_e[j]),
                        "confidence": round(out_c[j], 3),
                    }
                )

        if not w_out:
            continue

        line_start = w_out[0]["start"]
        line_end = w_out[-1]["end"]
        prev_line_end = line_end

        lines_out.append(
            {
                "line_index": idx + 1,
                "text": ln.text,
                "start": line_start,
                "end": line_end,
                "confidence": round(
                    sum(w["confidence"] for w in w_out) / max(len(w_out), 1), 3
                ),
                "words": w_out,
                "y": ln.y,
                "word_rois": ln.word_rois,
                "char_rois": [],
            }
        )

    for i, line_dict in enumerate(lines_out):
        line_dict["line_index"] = i + 1
    return lines_out


def main():  # noqa: C901
    setup_logging(verbose=True)
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
    args = p.parse_args()

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
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return 1

    if not selected_metrics:
        try:
            selected_metrics, _ = analyze_visual_suitability(
                v_path,
                fps=args.suitability_fps,
                work_dir=song_dir / "selected_suitability",
            )
        except Exception as e:
            logger.warning(f"Suitability check failed for selected video: {e}")
            selected_metrics = {}

    if (
        selected_metrics
        and not args.allow_low_suitability
        and not _is_suitability_good_enough(
            selected_metrics,
            args.min_detectability,
            args.min_word_level_score,
        )
    ):
        logger.error(
            "Selected candidate did not pass suitability thresholds: "
            f"detectability={selected_metrics.get('detectability_score', 0.0):.3f}, "
            f"word_level={selected_metrics.get('word_level_score', 0.0):.3f}"
        )
        return 1

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
    lines_out = _build_refined_lines_output(
        t_lines, artist=args.artist, title=args.title
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

    if args.report_json:
        report = {
            "schema_version": "1.0",
            "artist": args.artist,
            "title": args.title,
            "output_path": str(args.output.resolve()),
            "candidate_url": candidate_url,
            "selected_visual_suitability": selected_metrics,
            "candidate_rankings": [
                {
                    "rank": idx + 1,
                    "url": cand.get("url"),
                    "title": cand.get("title"),
                    "uploader": cand.get("uploader"),
                    "duration": cand.get("duration"),
                    "detectability_score": cand.get("metrics", {}).get(
                        "detectability_score"
                    ),
                    "word_level_score": cand.get("metrics", {}).get("word_level_score"),
                    "avg_ocr_confidence": cand.get("metrics", {}).get(
                        "avg_ocr_confidence"
                    ),
                }
                for idx, cand in enumerate(ranked_candidates)
            ],
            "settings": {
                "visual_fps": args.visual_fps,
                "suitability_fps": args.suitability_fps,
                "min_detectability": args.min_detectability,
                "min_word_level_score": args.min_word_level_score,
                "raw_ocr_cache_version": args.raw_ocr_cache_version,
                "allow_low_suitability": args.allow_low_suitability,
            },
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2))
        logger.info(f"Wrote bootstrap report to {args.report_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
