"""Candidate search/ranking helpers for karaoke visual bootstrap."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlparse


def search_karaoke_candidates(
    artist: str,
    title: str,
    max_candidates: int,
    *,
    yt_dlp_module: Any = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> list[dict[str, Any]]:
    if not artist or not title:
        return []

    if yt_dlp_module is None:
        try:
            import yt_dlp as yt_dlp_module  # type: ignore
        except Exception:
            if log_fn:
                log_fn("yt_dlp not available for candidate search")
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
        with yt_dlp_module.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(search_term, download=False)
        entries = info.get("entries", []) if isinstance(info, dict) else []
        for ent in entries:
            if not isinstance(ent, dict):
                continue
            video_id = ent.get("id")
            if not video_id:
                continue
            candidates.append(
                {
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "title": ent.get("title") or "",
                    "uploader": ent.get("uploader") or "",
                    "duration": ent.get("duration"),
                }
            )
    except Exception as exc:
        if log_fn:
            log_fn(f"Candidate search failed: {exc}")
    return candidates


def rank_candidates_by_suitability(
    candidates: list[dict[str, Any]],
    *,
    downloader: Any,
    song_dir: Path,
    suitability_fps: float,
    analyze_fn: Callable[..., tuple[dict[str, Any], tuple[int, int, int, int]]],
    log_info_fn: Optional[Callable[[str], None]] = None,
    log_warning_fn: Optional[Callable[[str], None]] = None,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []

    for idx, cand in enumerate(candidates, start=1):
        url = cand["url"]
        parsed = urlparse(url)
        video_id = parse_qs(parsed.query).get("v", [""])[0].strip()
        dir_suffix = video_id or f"{idx:02d}"
        eval_dir = song_dir / "candidates" / f"candidate_{dir_suffix}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        try:
            vid_info = downloader.download_video(url, output_dir=eval_dir)
            video_path = Path(vid_info["video_path"])
            metrics, _ = analyze_fn(
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
            if log_info_fn:
                log_info_fn(
                    "Candidate %d score=%.3f word_level=%.3f title=%s"
                    % (
                        idx,
                        metrics["detectability_score"],
                        metrics["word_level_score"],
                        cand.get("title", ""),
                    )
                )
        except Exception as exc:
            if log_warning_fn:
                log_warning_fn(f"Skipping candidate {url}: {exc}")

    ranked.sort(
        key=lambda c: (
            c["score"],
            c["metrics"].get("word_level_score", 0.0),
            c["metrics"].get("avg_ocr_confidence", 0.0),
        ),
        reverse=True,
    )
    return ranked
