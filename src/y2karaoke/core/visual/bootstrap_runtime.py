"""Runtime helpers for karaoke visual bootstrap orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def is_suitability_good_enough(
    metrics: dict[str, Any],
    min_detectability: float,
    min_word_level_score: float,
) -> bool:
    return (
        float(metrics.get("detectability_score", 0.0)) >= min_detectability
        and float(metrics.get("word_level_score", 0.0)) >= min_word_level_score
    )


def ensure_selected_suitability(
    selected_metrics: dict[str, Any],
    *,
    v_path: Path,
    song_dir: Path,
    suitability_fps: float,
    min_detectability: float,
    min_word_level_score: float,
    allow_low_suitability: bool,
    analyze_fn: Callable[..., tuple[dict[str, Any], tuple[int, int, int, int]]],
) -> dict[str, Any]:
    if not selected_metrics:
        selected_metrics, _ = analyze_fn(
            v_path,
            fps=suitability_fps,
            work_dir=song_dir / "selected_suitability",
        )

    if (
        selected_metrics
        and not allow_low_suitability
        and not is_suitability_good_enough(
            selected_metrics, min_detectability, min_word_level_score
        )
    ):
        raise ValueError(
            "Selected candidate did not pass suitability thresholds: "
            f"detectability={selected_metrics.get('detectability_score', 0.0):.3f}, "
            f"word_level={selected_metrics.get('word_level_score', 0.0):.3f}"
        )
    return selected_metrics


def build_run_report_payload(
    *,
    artist: str | None,
    title: str | None,
    output_path: Path,
    candidate_url: str,
    selected_metrics: dict[str, Any],
    ranked_candidates: list[dict[str, Any]],
    visual_fps: float,
    suitability_fps: float,
    min_detectability: float,
    min_word_level_score: float,
    raw_ocr_cache_version: str,
    allow_low_suitability: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "artist": artist,
        "title": title,
        "output_path": str(output_path.resolve()),
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
                "avg_ocr_confidence": cand.get("metrics", {}).get("avg_ocr_confidence"),
            }
            for idx, cand in enumerate(ranked_candidates)
        ],
        "settings": {
            "visual_fps": visual_fps,
            "suitability_fps": suitability_fps,
            "min_detectability": min_detectability,
            "min_word_level_score": min_word_level_score,
            "raw_ocr_cache_version": raw_ocr_cache_version,
            "allow_low_suitability": allow_low_suitability,
        },
    }


def write_run_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2))
