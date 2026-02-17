"""Candidate search/ranking helpers for karaoke visual bootstrap."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlparse


def _metadata_prefilter_score(candidate: dict[str, Any]) -> float:
    """Cheap metadata score used to limit expensive video downloads."""
    title = str(candidate.get("title") or "").lower()
    uploader = str(candidate.get("uploader") or "").lower()
    duration_raw = candidate.get("duration")
    duration = float(duration_raw) if isinstance(duration_raw, (int, float)) else None

    score = 0.0
    if "karaoke" in title:
        score += 2.0
    if any(tag in title for tag in ("instrumental", "off vocal", "minus one")):
        score += 1.0
    if "lyrics" in title and "karaoke" not in title:
        score -= 0.8
    if "topic" in uploader:
        score -= 0.3
    if duration is not None:
        if 60 <= duration <= 900:
            score += 0.5
        else:
            score -= 0.5
    return score


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
    prefilter_limit: int = 3,
    log_info_fn: Optional[Callable[[str], None]] = None,
    log_warning_fn: Optional[Callable[[str], None]] = None,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []

    shortlist = list(candidates)
    if prefilter_limit > 0 and len(candidates) > prefilter_limit:
        shortlist = sorted(
            candidates,
            key=_metadata_prefilter_score,
            reverse=True,
        )[:prefilter_limit]
        if log_info_fn:
            log_info_fn(
                "Prefiltered candidates by metadata: evaluating %d/%d video(s)"
                % (len(shortlist), len(candidates))
            )

    for idx, cand in enumerate(shortlist, start=1):
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
