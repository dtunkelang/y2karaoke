"""Candidate search/ranking helpers for karaoke visual bootstrap."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlparse

_PREFERRED_BRANDS = {
    "sing king",
    "karafun",
    "the karaoke channel",
    "stingray karaoke",
    "cc karaoke",
    "mega karaoke",
}


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

    # Brand preference boost
    if any(brand in uploader for brand in _PREFERRED_BRANDS):
        score += 5.0

    if duration is not None:
        if 60 <= duration <= 900:
            score += 0.5
        else:
            score -= 0.5
    return score


def _resolve_yt_dlp(
    yt_dlp_module: Any,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Any:
    if yt_dlp_module is not None:
        return yt_dlp_module
    try:
        import yt_dlp as imported_yt_dlp  # type: ignore

        return imported_yt_dlp
    except Exception:
        if log_fn:
            log_fn("yt_dlp not available for candidate search")
        return None


def _extract_candidate_entries(
    yt_dlp_module: Any,
    *,
    search_term: str,
) -> list[dict[str, Any]]:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
    }
    with yt_dlp_module.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(search_term, download=False)
    entries = info.get("entries", []) if isinstance(info, dict) else []
    return [ent for ent in entries if isinstance(ent, dict)]


def _normalize_candidate_entry(entry: dict[str, Any]) -> Optional[dict[str, Any]]:
    video_id = entry.get("id")
    if not video_id:
        return None
    return {
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "title": entry.get("title") or "",
        "uploader": entry.get("uploader") or "",
        "duration": entry.get("duration"),
    }


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

    yt_dlp_module = _resolve_yt_dlp(yt_dlp_module, log_fn=log_fn)
    if yt_dlp_module is None:
        return []

    query = f"{artist} {title} karaoke"
    search_term = f"ytsearch{max_candidates}:{query}"
    try:
        entries = _extract_candidate_entries(yt_dlp_module, search_term=search_term)
    except Exception as exc:
        if log_fn:
            log_fn(f"Candidate search failed: {exc}")
        return []
    return [
        cand
        for entry in entries
        if (cand := _normalize_candidate_entry(entry)) is not None
    ]


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
        try:
            scored = _score_candidate(
                cand,
                idx=idx,
                song_dir=song_dir,
                suitability_fps=suitability_fps,
                downloader=downloader,
                analyze_fn=analyze_fn,
            )
            ranked.append(scored)
            if log_info_fn:
                log_info_fn(
                    "Candidate %d score=%.3f word_level=%.3f title=%s"
                    % (
                        idx,
                        scored["metrics"]["detectability_score"],
                        scored["metrics"]["word_level_score"],
                        cand.get("title", ""),
                    )
                )
        except Exception as exc:
            if log_warning_fn:
                log_warning_fn(f"Skipping candidate {cand['url']}: {exc}")

    ranked.sort(key=_candidate_sort_key, reverse=True)
    return ranked


def _candidate_eval_dir(song_dir: Path, *, url: str, idx: int) -> Path:
    parsed = urlparse(url)
    video_id = parse_qs(parsed.query).get("v", [""])[0].strip()
    dir_suffix = video_id or f"{idx:02d}"
    eval_dir = song_dir / "candidates" / f"candidate_{dir_suffix}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir


def _score_candidate(
    candidate: dict[str, Any],
    *,
    idx: int,
    song_dir: Path,
    suitability_fps: float,
    downloader: Any,
    analyze_fn: Callable[..., tuple[dict[str, Any], tuple[int, int, int, int]]],
) -> dict[str, Any]:
    url = candidate["url"]
    eval_dir = _candidate_eval_dir(song_dir, url=url, idx=idx)
    vid_info = downloader.download_video(url, output_dir=eval_dir)
    video_path = Path(vid_info["video_path"])
    metrics, _ = analyze_fn(
        video_path,
        fps=suitability_fps,
        work_dir=eval_dir / "suitability",
    )
    return {
        **candidate,
        "video_path": str(video_path),
        "metrics": metrics,
        "score": float(metrics["detectability_score"]),
    }


def _final_candidate_score(candidate: dict[str, Any]) -> float:
    base = float(candidate["score"])
    if candidate["metrics"].get("is_all_caps"):
        base -= 0.05
    uploader = str(candidate.get("uploader") or "").lower()
    if any(brand in uploader for brand in _PREFERRED_BRANDS):
        base += 0.1
    return base


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float]:
    return (
        _final_candidate_score(candidate),
        candidate["metrics"].get("word_level_score", 0.0),
        candidate["metrics"].get("avg_ocr_confidence", 0.0),
    )
