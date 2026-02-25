#!/usr/bin/env python3
"""
Bootstrap and refine word-level gold timings using computer vision.

This tool acts as a CLI wrapper around the `y2karaoke.vision` and `y2karaoke.core` libraries.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from time import perf_counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from difflib import SequenceMatcher

# Add src to path if running from tools/
src_path = Path(__file__).resolve().parents[1] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from y2karaoke.utils.logging import setup_logging, get_logger  # noqa: E402
from y2karaoke.core.components.audio.downloader import YouTubeDownloader  # noqa: E402
from y2karaoke.core.visual.reconstruction import (  # noqa: E402
    reconstruct_lyrics_from_visuals,
)
from y2karaoke.core.visual.extractor_mode import (  # noqa: E402
    resolve_visual_extractor_mode,
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
from y2karaoke.core.visual import bootstrap_ocr as _bootstrap_ocr_module  # noqa: E402
from y2karaoke.core.visual.bootstrap_media import (  # noqa: E402
    extract_audio_from_video as _extract_audio_from_video_impl,
    resolve_media_paths as _resolve_media_paths_impl,
)
from y2karaoke.core.visual.bootstrap_selection import (  # noqa: E402
    select_candidate_with_rankings as _select_candidate_with_rankings_impl,
)
from y2karaoke.core.text_utils import (
    make_slug,
    normalize_text_basic as normalize_text,
)  # noqa: E402
from y2karaoke.vision.ocr import get_ocr_engine  # noqa: E402,F401
from y2karaoke.vision.roi import detect_lyric_roi  # noqa: E402
from y2karaoke.vision.suitability import analyze_visual_suitability  # noqa: E402

try:
    import cv2  # noqa: E402
    import numpy as np  # noqa: E402,F401
except ImportError:
    cv2 = SimpleNamespace(  # type: ignore[assignment]
        VideoCapture=None,
        CAP_PROP_FRAME_COUNT=7,
        CAP_PROP_FPS=5,
    )
    np = None  # type: ignore[assignment]

logger = get_logger(__name__)
RAW_OCR_CACHE_VERSION = "4"
# Native-FPS word refinement is only reliable when mixed-state evidence is strong.
# Lower thresholds can route line-level videos into an expensive but lower-quality path.
LINE_LEVEL_REFINE_SKIP_THRESHOLD = 0.5
LOW_FPS_LINE_REFINE_FPS = 6.0


def _require_cv2() -> None:
    if not hasattr(cv2, "VideoCapture"):
        raise ImportError("OpenCV is required. Please install OpenCV.")


def _should_use_fade_fragment_text_guard(extractor_selection: Any) -> bool:
    """Keep stricter fade-fragment merging only for stable block-like layouts.

    This guard was introduced to prevent wrong same-lane merges on clean block
    videos (e.g. Counting Stars), but it regresses some noisier/non-stable
    line-first videos. We gate it using the same raw-frame layout diagnostics
    used by extractor auto-selection.
    """
    diag = getattr(extractor_selection, "diagnostics", None) or {}
    if not isinstance(diag, dict):
        return True
    modal_row_count = int(diag.get("modal_row_count", 0) or 0)
    modal_ratio = float(diag.get("modal_row_ratio", 0.0) or 0.0)
    eligible_ratio = float(diag.get("eligible_ratio", 0.0) or 0.0)
    long_run_ratio = float(diag.get("long_stable_run_ratio", 0.0) or 0.0)
    mean_row_y_std = diag.get("mean_row_y_std")
    if mean_row_y_std is None:
        return True
    y_std = float(mean_row_y_std)
    return (
        2 <= modal_row_count <= 5
        and modal_ratio >= 0.70
        and eligible_ratio >= 0.75
        and long_run_ratio >= 0.80
        and y_std <= 5.0
    )


def _looks_like_stable_block_layout(extractor_selection: Any) -> bool:
    return _should_use_fade_fragment_text_guard(extractor_selection)


def refine_word_timings_at_high_fps(*args: Any, **kwargs: Any) -> Any:
    from y2karaoke.core.visual.refinement import (  # noqa: E402
        refine_word_timings_at_high_fps as _impl,
    )

    return _impl(*args, **kwargs)


def refine_line_timings_at_low_fps(*args: Any, **kwargs: Any) -> Any:
    from y2karaoke.core.visual.refinement import (  # noqa: E402
        refine_line_timings_at_low_fps as _impl,
    )

    return _impl(*args, **kwargs)


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
    # Keep delegated module in sync with local monkeypatches used by tests/tools.
    if hasattr(_bootstrap_ocr_module, "cv2"):
        _bootstrap_ocr_module.cv2 = cv2
    if hasattr(_bootstrap_ocr_module, "np"):
        _bootstrap_ocr_module.np = np
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
    if os.environ.get("Y2K_VISUAL_BOOTSTRAP_TARGET_TRACE", "0") == "1":
        preview: list[dict[str, Any]] = []
        for ln in t_lines[:15]:
            word_starts = getattr(ln, "word_starts", None) or []
            word_ends = getattr(ln, "word_ends", None) or []
            s = (
                float(word_starts[0])
                if word_starts and word_starts[0] is not None
                else float(getattr(ln, "start", 0.0))
            )
            e = (
                float(word_ends[-1])
                if word_ends and word_ends[-1] is not None
                else float(getattr(ln, "end", s) or s)
            )
            preview.append(
                {
                    "i": int(getattr(ln, "line_index", 0)),
                    "s": round(s, 2),
                    "e": round(e, 2),
                    "y": round(float(getattr(ln, "y", 0.0)), 1),
                    "vs": (
                        round(float(getattr(ln, "visibility_start")), 2)
                        if getattr(ln, "visibility_start", None) is not None
                        else None
                    ),
                    "ve": (
                        round(float(getattr(ln, "visibility_end")), 2)
                        if getattr(ln, "visibility_end", None) is not None
                        else None
                    ),
                    "hint": getattr(ln, "block_order_hint", None),
                    "variants": (
                        (
                            (getattr(ln, "reconstruction_meta", {}) or {}).get(
                                "top_text_variants"
                            )
                        )[:3]
                        if isinstance(getattr(ln, "reconstruction_meta", None), dict)
                        and isinstance(
                            (getattr(ln, "reconstruction_meta", {}) or {}).get(
                                "top_text_variants"
                            ),
                            list,
                        )
                        else None
                    ),
                    "t": " ".join(list(getattr(ln, "words", []))[:6]),
                }
            )
        logger.info("BOOTSTRAP_TARGET_TRACE pre_build first15=%s", preview)
    return _build_refined_lines_output_impl(t_lines, artist=artist, title=title)


def _summarize_reconstruction_uncertainty(t_lines: list[Any]) -> dict[str, Any]:
    scores: list[float] = []
    weak_count = 0
    fallback_count = 0
    low_support_count = 0
    examples: list[dict[str, Any]] = []
    for ln in t_lines:
        meta = getattr(ln, "reconstruction_meta", None)
        if not isinstance(meta, dict):
            continue
        score = meta.get("uncertainty_score")
        if isinstance(score, (int, float)):
            score_f = float(score)
            scores.append(score_f)
            if score_f >= 0.55:
                weak_count += 1
        if bool(meta.get("used_observed_fallback")):
            fallback_count += 1
        support = meta.get("selected_text_support_ratio")
        if isinstance(support, (int, float)) and float(support) < 0.5:
            low_support_count += 1

        if isinstance(score, (int, float)):
            examples.append(
                {
                    "text": str(getattr(ln, "text", ""))[:120],
                    "uncertainty_score": round(float(score), 3),
                    "selected_text_support_ratio": meta.get(
                        "selected_text_support_ratio"
                    ),
                    "position_support_min": meta.get("position_support_min"),
                    "position_support_mean": meta.get("position_support_mean"),
                    "observations": meta.get("observations"),
                    "text_variant_count": meta.get("text_variant_count"),
                }
            )

    if not scores:
        return {"available": False}
    scores_sorted = sorted(scores)
    p95_idx = int(round((len(scores_sorted) - 1) * 0.95))
    return {
        "available": True,
        "line_count_with_meta": len(scores),
        "uncertainty_score_mean": round(sum(scores) / len(scores), 3),
        "uncertainty_score_p95": round(scores_sorted[p95_idx], 3),
        "high_uncertainty_line_count": weak_count,
        "fallback_selected_count": fallback_count,
        "low_text_support_line_count": low_support_count,
        "top_uncertain_examples": sorted(
            examples, key=lambda r: float(r.get("uncertainty_score", 0.0)), reverse=True
        )[:3],
    }


def _summarize_repeat_clusters(lines_out: list[dict[str, Any]]) -> dict[str, Any]:
    if not lines_out:
        return {"available": False}

    texts = [str(ln.get("text", "") or "") for ln in lines_out]
    norms = [normalize_text(text) for text in texts]
    token_lists = [[t for t in n.split() if t] for n in norms]
    n = len(lines_out)
    parents = list(range(n))

    def find(x: int) -> int:
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parents[rb] = ra

    for i in range(n):
        if len(token_lists[i]) < 2:
            continue
        for j in range(i + 1, n):
            if len(token_lists[j]) < 2:
                continue
            if abs(len(token_lists[i]) - len(token_lists[j])) > 2:
                continue
            ratio = SequenceMatcher(None, norms[i], norms[j]).ratio()
            if ratio < 0.84:
                continue
            shared = len(set(token_lists[i]) & set(token_lists[j]))
            if shared < 2:
                continue
            union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    repeat_groups = [sorted(g) for g in groups.values() if len(g) >= 2]
    if not repeat_groups:
        return {"available": True, "cluster_count": 0}

    cluster_summaries: list[dict[str, Any]] = []
    chronology_conflicts = 0
    variant_heavy = 0

    for group in repeat_groups:
        starts = [float(lines_out[i].get("start", 0.0) or 0.0) for i in group]
        group_norms = [norms[i] for i in group]
        unique_norms = sorted({g for g in group_norms if g})
        variant_count = len(unique_norms)
        if variant_count >= 3:
            variant_heavy += 1
        inversions = 0
        for a in range(len(group) - 1):
            for b in range(a + 1, len(group)):
                if starts[a] > starts[b]:
                    inversions += 1
        if inversions > 0:
            chronology_conflicts += 1
        span = max(starts) - min(starts) if starts else 0.0
        cluster_summaries.append(
            {
                "size": len(group),
                "variant_count": variant_count,
                "span_sec": round(span, 2),
                "chronology_inversions": inversions,
                "example_text": texts[group[0]][:120],
                "variant_examples": unique_norms[:3],
            }
        )

    cluster_summaries.sort(
        key=lambda c: (
            int(c.get("chronology_inversions", 0)),
            int(c.get("variant_count", 0)),
            int(c.get("size", 0)),
            float(c.get("span_sec", 0.0)),
        ),
        reverse=True,
    )

    return {
        "available": True,
        "cluster_count": len(repeat_groups),
        "clusters_size_ge_3": sum(1 for g in repeat_groups if len(g) >= 3),
        "variant_heavy_cluster_count": variant_heavy,
        "chronology_conflict_cluster_count": chronology_conflicts,
        "top_problem_clusters": cluster_summaries[:5],
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--artist")
    p.add_argument("--title")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, default=Path(".cache/karaoke_bootstrap"))
    p.add_argument("--report-json", type=Path, default=None)
    p.add_argument("--candidate-url")
    p.add_argument("--visual-fps", type=float, default=10.0)
    p.add_argument("--max-candidates", type=int, default=5)
    p.add_argument("--show-candidates", action="store_true")
    p.add_argument("--suitability-fps", type=float, default=1.0)
    p.add_argument("--min-detectability", type=float, default=0.45)
    p.add_argument("--min-word-level-score", type=float, default=0.15)
    p.add_argument("--raw-ocr-cache-version", default=RAW_OCR_CACHE_VERSION)
    p.add_argument(
        "--visual-spell-correction-mode",
        choices=["full", "no-guesses", "off", "auto"],
        default="off",
        help=(
            "OCR text cleanup mode during visual reconstruction: "
            "'full' uses OCR variants + system guesses, "
            "'no-guesses' disables generic OS spell guesses, "
            "'off' disables spell correction entirely, "
            "'auto' applies OCR-only correction to suspicious OCR tokens."
        ),
    )
    p.add_argument(
        "--visual-refinement-mode",
        choices=["auto", "high-fps-word", "low-fps-line"],
        default="auto",
        help=(
            "Visual timing refinement mode: "
            "'auto' picks based on suitability score, "
            "'high-fps-word' forces word-level refinement, "
            "'low-fps-line' forces line-level refinement."
        ),
    )
    p.add_argument(
        "--visual-extractor-mode",
        choices=["auto", "line-first", "block-first"],
        default="auto",
        help=(
            "Visual lyrics extraction backend: "
            "'line-first' (current default pipeline), "
            "'block-first' (new screen/block-oriented prototype), "
            "'auto' (currently resolves to line-first; future selector hook)."
        ),
    )
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
    roi_rect: Optional[tuple[int, int, int, int]] = None,
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
            analyze_fn=(
                (
                    lambda video_path, *, fps, work_dir: analyze_visual_suitability(
                        video_path, fps=fps, work_dir=work_dir, roi_rect=roi_rect
                    )
                )
                if roi_rect is not None
                else analyze_visual_suitability
            ),
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
    roi_rect: Optional[tuple[int, int, int, int]] = None,
    *,
    return_runtime_metrics: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
    _require_cv2()
    roi = roi_rect if roi_rect is not None else detect_lyric_roi(v_path, sample_fps=1.0)
    cap = cv2.VideoCapture(str(v_path))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()

    ocr_t0 = perf_counter()
    raw_frames = _collect_raw_frames_cached(
        v_path,
        duration,
        args.visual_fps,
        roi,
        song_dir / "cache",
        cache_version=args.raw_ocr_cache_version,
    )
    ocr_elapsed = perf_counter() - ocr_t0
    logger.info(
        "OCR sampling complete: "
        f"frames={len(raw_frames)} visual_fps={args.visual_fps:.2f} "
        f"elapsed={ocr_elapsed:.1f}s"
    )
    reconstruct_t0 = perf_counter()
    extractor_selection = resolve_visual_extractor_mode(
        str(getattr(args, "visual_extractor_mode", "auto")),
        raw_frames=raw_frames,
    )
    logger.info(
        "Visual extractor mode: requested=%s resolved=%s (%s)",
        extractor_selection.requested_mode,
        extractor_selection.resolved_mode,
        extractor_selection.reason,
    )
    if extractor_selection.diagnostics:
        logger.info(
            "Visual extractor auto diagnostics: %s", extractor_selection.diagnostics
        )
    prev_spell_mode = os.environ.get("Y2K_VISUAL_SPELL_CORRECTION_MODE")
    prev_disable_spell_correct = os.environ.get("Y2K_VISUAL_DISABLE_SPELL_CORRECT")
    prev_disable_spell_guesses = os.environ.get("Y2K_VISUAL_DISABLE_SPELL_GUESSES")
    prev_disable_fade_guard = os.environ.get(
        "Y2K_VISUAL_DISABLE_FADE_FRAGMENT_TEXT_GUARD"
    )
    prev_disable_ghost_guards = os.environ.get(
        "Y2K_VISUAL_DISABLE_GHOST_REENTRY_GUARDS"
    )
    prev_disable_visible_end_guard = os.environ.get(
        "Y2K_VISUAL_DISABLE_VISIBLE_END_GHOST_GUARD"
    )
    try:
        mode = str(getattr(args, "visual_spell_correction_mode", "full"))
        os.environ["Y2K_VISUAL_SPELL_CORRECTION_MODE"] = mode
        # Keep legacy toggles in sync for older codepaths/tests.
        if mode == "off":
            os.environ["Y2K_VISUAL_DISABLE_SPELL_CORRECT"] = "1"
            os.environ.pop("Y2K_VISUAL_DISABLE_SPELL_GUESSES", None)
        elif mode in {"no-guesses", "auto"}:
            os.environ.pop("Y2K_VISUAL_DISABLE_SPELL_CORRECT", None)
            os.environ["Y2K_VISUAL_DISABLE_SPELL_GUESSES"] = "1"
        else:
            os.environ.pop("Y2K_VISUAL_DISABLE_SPELL_CORRECT", None)
            os.environ.pop("Y2K_VISUAL_DISABLE_SPELL_GUESSES", None)
        use_fade_guard = _should_use_fade_fragment_text_guard(extractor_selection)
        is_line_first = extractor_selection.resolved_mode == "line-first"
        stable_block_layout = _looks_like_stable_block_layout(extractor_selection)
        if is_line_first and not use_fade_guard:
            os.environ["Y2K_VISUAL_DISABLE_FADE_FRAGMENT_TEXT_GUARD"] = "1"
            logger.info(
                "Disabling fade-fragment text guard for non-stable line-first layout"
            )
        else:
            os.environ.pop("Y2K_VISUAL_DISABLE_FADE_FRAGMENT_TEXT_GUARD", None)
        if is_line_first and not stable_block_layout:
            os.environ["Y2K_VISUAL_DISABLE_GHOST_REENTRY_GUARDS"] = "1"
            os.environ["Y2K_VISUAL_DISABLE_VISIBLE_END_GHOST_GUARD"] = "1"
            logger.info(
                "Using legacy ghost-tail handling for non-stable line-first layout"
            )
        else:
            os.environ.pop("Y2K_VISUAL_DISABLE_GHOST_REENTRY_GUARDS", None)
            os.environ.pop("Y2K_VISUAL_DISABLE_VISIBLE_END_GHOST_GUARD", None)
        t_lines = reconstruct_lyrics_from_visuals(
            raw_frames,
            args.visual_fps,
            artist=args.artist,
            extractor_mode=extractor_selection.resolved_mode,
        )
    finally:
        if prev_spell_mode is None:
            os.environ.pop("Y2K_VISUAL_SPELL_CORRECTION_MODE", None)
        else:
            os.environ["Y2K_VISUAL_SPELL_CORRECTION_MODE"] = prev_spell_mode
        if prev_disable_spell_correct is None:
            os.environ.pop("Y2K_VISUAL_DISABLE_SPELL_CORRECT", None)
        else:
            os.environ["Y2K_VISUAL_DISABLE_SPELL_CORRECT"] = prev_disable_spell_correct
        if prev_disable_spell_guesses is None:
            os.environ.pop("Y2K_VISUAL_DISABLE_SPELL_GUESSES", None)
        else:
            os.environ["Y2K_VISUAL_DISABLE_SPELL_GUESSES"] = prev_disable_spell_guesses
        if prev_disable_fade_guard is None:
            os.environ.pop("Y2K_VISUAL_DISABLE_FADE_FRAGMENT_TEXT_GUARD", None)
        else:
            os.environ["Y2K_VISUAL_DISABLE_FADE_FRAGMENT_TEXT_GUARD"] = (
                prev_disable_fade_guard
            )
        if prev_disable_ghost_guards is None:
            os.environ.pop("Y2K_VISUAL_DISABLE_GHOST_REENTRY_GUARDS", None)
        else:
            os.environ["Y2K_VISUAL_DISABLE_GHOST_REENTRY_GUARDS"] = (
                prev_disable_ghost_guards
            )
        if prev_disable_visible_end_guard is None:
            os.environ.pop("Y2K_VISUAL_DISABLE_VISIBLE_END_GHOST_GUARD", None)
        else:
            os.environ["Y2K_VISUAL_DISABLE_VISIBLE_END_GHOST_GUARD"] = (
                prev_disable_visible_end_guard
            )
    reconstruct_elapsed = perf_counter() - reconstruct_t0
    logger.info(
        f"Reconstructed {len(t_lines)} initial lines in {reconstruct_elapsed:.1f}s."
    )
    reconstruction_uncertainty = _summarize_reconstruction_uncertainty(t_lines)
    if reconstruction_uncertainty.get("available"):
        logger.info(
            "Reconstruction uncertainty: "
            "mean=%s p95=%s high=%s/%s fallback=%s low_support=%s"
            % (
                reconstruction_uncertainty.get("uncertainty_score_mean"),
                reconstruction_uncertainty.get("uncertainty_score_p95"),
                reconstruction_uncertainty.get("high_uncertainty_line_count"),
                reconstruction_uncertainty.get("line_count_with_meta"),
                reconstruction_uncertainty.get("fallback_selected_count"),
                reconstruction_uncertainty.get("low_text_support_line_count"),
            )
        )

    word_level_score = 0.0
    if selected_metrics:
        word_level_score = float(selected_metrics.get("word_level_score", 0.0))

    refine_t0 = perf_counter()
    requested_refine_mode = str(getattr(args, "visual_refinement_mode", "auto"))
    refine_mode = "high_fps_word"
    if requested_refine_mode == "high-fps-word":
        refine_word_timings_at_high_fps(v_path, t_lines, roi)
    elif requested_refine_mode == "low-fps-line":
        refine_mode = "low_fps_line"
        logger.info("Forcing low-FPS line-level timing refinement.")
        refine_line_timings_at_low_fps(
            v_path,
            t_lines,
            roi,
            sample_fps=LOW_FPS_LINE_REFINE_FPS,
            extractor_mode=extractor_selection.resolved_mode,
        )
    elif word_level_score >= LINE_LEVEL_REFINE_SKIP_THRESHOLD:
        refine_word_timings_at_high_fps(v_path, t_lines, roi)
    else:
        refine_mode = "low_fps_line"
        logger.info(
            "Skipping high-FPS visual refinement due to low word-level suitability "
            f"(word_level_score={word_level_score:.3f} < "
            f"{LINE_LEVEL_REFINE_SKIP_THRESHOLD:.3f}); running low-FPS line-level "
            "timing refinement."
        )
        refine_line_timings_at_low_fps(
            v_path,
            t_lines,
            roi,
            sample_fps=LOW_FPS_LINE_REFINE_FPS,
            extractor_mode=extractor_selection.resolved_mode,
        )
    refine_elapsed = perf_counter() - refine_t0
    logger.info(
        "Refinement complete: "
        f"mode={refine_mode} elapsed={refine_elapsed:.1f}s lines={len(t_lines)}"
    )
    lines_out = _build_refined_lines_output(
        t_lines, artist=args.artist, title=args.title
    )
    runtime_metrics = {
        "ocr": {
            "sampled_frames": len(raw_frames),
            "elapsed_sec": round(ocr_elapsed, 3),
            "visual_fps": float(args.visual_fps),
        },
        "reconstruction": {
            "initial_line_count": len(t_lines),
            "elapsed_sec": round(reconstruct_elapsed, 3),
            "spell_correction_mode": str(
                getattr(args, "visual_spell_correction_mode", "full")
            ),
            "extractor_mode": {
                "requested": extractor_selection.requested_mode,
                "resolved": extractor_selection.resolved_mode,
                "reason": extractor_selection.reason,
                "diagnostics": extractor_selection.diagnostics,
            },
            "uncertainty": reconstruction_uncertainty,
        },
        "refinement": {
            "mode": refine_mode,
            "requested_mode": requested_refine_mode,
            "elapsed_sec": round(refine_elapsed, 3),
            "output_line_count": len(lines_out),
            "word_level_score": round(word_level_score, 4),
            "repeat_structure": _summarize_repeat_clusters(lines_out),
        },
    }
    if return_runtime_metrics:
        return lines_out, runtime_metrics
    return lines_out


def _write_run_report(
    args: argparse.Namespace,
    *,
    candidate_url: str,
    selected_metrics: dict[str, Any],
    ranked_candidates: list[dict[str, Any]],
    runtime_metrics: Optional[dict[str, Any]] = None,
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
    if runtime_metrics:
        report["runtime_metrics"] = runtime_metrics
    _write_run_report_impl(args.report_json, report)
    logger.info(f"Wrote bootstrap report to {args.report_json}")


def main():
    setup_logging(verbose=True)
    total_t0 = perf_counter()
    args = _parse_args()

    slug_artist = make_slug(args.artist or "unk")
    slug_title = make_slug(args.title or "unk")
    song_dir = args.work_dir / slug_artist / slug_title

    downloader = YouTubeDownloader(cache_dir=song_dir.parent)
    select_t0 = perf_counter()
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
    select_elapsed = perf_counter() - select_t0

    logger.info(f"Selected candidate URL: {candidate_url}")

    media_t0 = perf_counter()
    try:
        v_path, a_path = _resolve_media_paths(
            downloader, candidate_url, cached_video_path, song_dir
        )
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return 1
    media_elapsed = perf_counter() - media_t0

    roi_rect = detect_lyric_roi(v_path, sample_fps=1.0)

    suit_t0 = perf_counter()
    try:
        selected_metrics = _ensure_selected_suitability(
            selected_metrics,
            v_path=v_path,
            song_dir=song_dir,
            args=args,
            roi_rect=roi_rect,
        )
    except ValueError as e:
        logger.error(str(e))
        return 1
    suit_elapsed = perf_counter() - suit_t0

    bootstrap_t0 = perf_counter()
    lines_out, bootstrap_runtime_metrics = _bootstrap_refined_lines(
        v_path,
        args,
        song_dir,
        selected_metrics=selected_metrics,
        roi_rect=roi_rect,
        return_runtime_metrics=True,
    )
    bootstrap_elapsed = perf_counter() - bootstrap_t0

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
    total_elapsed = perf_counter() - total_t0
    _write_run_report(
        args,
        candidate_url=candidate_url,
        selected_metrics=selected_metrics,
        ranked_candidates=ranked_candidates,
        runtime_metrics={
            **bootstrap_runtime_metrics,
            "stages": {
                "select_sec": round(select_elapsed, 3),
                "media_sec": round(media_elapsed, 3),
                "suitability_sec": round(suit_elapsed, 3),
                "bootstrap_sec": round(bootstrap_elapsed, 3),
                "total_sec": round(total_elapsed, 3),
            },
        },
    )
    logger.info(
        "Bootstrap stage timing (s): "
        f"select={select_elapsed:.1f} "
        f"media={media_elapsed:.1f} "
        f"suitability={suit_elapsed:.1f} "
        f"bootstrap={bootstrap_elapsed:.1f} "
        f"total={total_elapsed:.1f}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
