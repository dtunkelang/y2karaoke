"""Candidate selection orchestration for visual bootstrap."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional


def select_candidate_with_rankings(
    *,
    candidate_url: Optional[str],
    artist: Optional[str],
    title: Optional[str],
    max_candidates: int,
    suitability_fps: float,
    show_candidates: bool,
    allow_low_suitability: bool,
    min_detectability: float,
    min_word_level_score: float,
    downloader: Any,
    song_dir: Path,
    search_fn: Callable[[str, str, int], list[dict[str, Any]]],
    rank_fn: Callable[[list[dict[str, Any]], Any, Path, float], list[dict[str, Any]]],
    suitability_check_fn: Callable[[dict[str, Any], float, float], bool],
    log_info_fn: Optional[Callable[[str], None]] = None,
) -> tuple[str, Optional[Path], dict[str, Any], list[dict[str, Any]]]:
    if candidate_url:
        if log_info_fn:
            log_info_fn(f"Using explicit candidate URL: {candidate_url}")
        return candidate_url, None, {}, []

    if not artist or not title:
        raise ValueError(
            "Either --candidate-url or both --artist and --title are required."
        )

    candidates = search_fn(artist, title, max_candidates)
    if not candidates:
        raise ValueError(
            "No candidate videos found. Provide --candidate-url to continue."
        )

    ranked = rank_fn(candidates, downloader, song_dir, suitability_fps)
    if not ranked:
        raise ValueError(
            "Could not evaluate candidate videos. Provide --candidate-url to continue."
        )

    if show_candidates and log_info_fn:
        log_info_fn("Candidate ranking by visual suitability:")
        for idx, cand in enumerate(ranked, start=1):
            m = cand["metrics"]
            log_info_fn(
                "  %d. %.3f (word=%.3f ocr=%.3f) %s | %s"
                % (
                    idx,
                    m["detectability_score"],
                    m["word_level_score"],
                    m["avg_ocr_confidence"],
                    cand.get("title", ""),
                    cand["url"],
                )
            )

    best = ranked[0]
    best_metrics = best["metrics"]
    if not allow_low_suitability and not suitability_check_fn(
        best_metrics,
        min_detectability,
        min_word_level_score,
    ):
        raise ValueError(
            "Best candidate did not meet suitability thresholds. "
            "Use --allow-low-suitability to override."
        )

    return best["url"], Path(best["video_path"]), best_metrics, ranked
