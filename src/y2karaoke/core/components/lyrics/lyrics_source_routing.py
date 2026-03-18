"""Routing policy for timed-lyrics source selection."""

from __future__ import annotations

from typing import List, Optional, Tuple

from .lrc import parse_lrc_with_timing
from .runtime_config import LyricsRuntimeConfig

_LYRIQ_PROVIDER_KEYS = {"lyriq", "lrclib", "lyriqlrclib"}
_DisagreementSourceMap = dict[str, tuple[Optional[str], Optional[int]]]


def _normalize_provider_key(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _filter_sources_for_preferred_provider(
    sources: _DisagreementSourceMap,
    *,
    preferred_provider: Optional[str] = None,
) -> _DisagreementSourceMap:
    preferred = (preferred_provider or "").strip().lower()
    if not preferred:
        return sources

    normalized_preferred = _normalize_provider_key(preferred)
    if normalized_preferred in _LYRIQ_PROVIDER_KEYS:
        filtered = {
            key: value
            for key, value in sources.items()
            if _normalize_provider_key(key) in _LYRIQ_PROVIDER_KEYS
        }
        return filtered or sources

    if normalized_preferred == "syncedlyrics":
        filtered = {
            key: value
            for key, value in sources.items()
            if _normalize_provider_key(key) not in _LYRIQ_PROVIDER_KEYS
        }
        return filtered or sources

    filtered = {
        key: value
        for key, value in sources.items()
        if _normalize_provider_key(key) == normalized_preferred
    }
    return filtered or sources


def _initialize_routing_diagnostics(
    routing_diagnostics: Optional[dict],
    *,
    target_duration: Optional[int],
    vocals_path: Optional[str],
    evaluate_sources: bool,
    offline: bool,
) -> None:
    if routing_diagnostics is None:
        return
    routing_diagnostics.setdefault("lyrics_source_audio_scoring_used", False)
    routing_diagnostics.setdefault("lyrics_source_disagreement_flagged", False)
    routing_diagnostics.setdefault("lyrics_source_disagreement_reasons", [])
    routing_diagnostics.setdefault("lyrics_source_candidate_count", 0)
    routing_diagnostics.setdefault("lyrics_source_comparable_candidate_count", 0)
    routing_diagnostics.setdefault("lyrics_source_selection_mode", "default")
    routing_diagnostics.setdefault("lyrics_source_routing_skip_reason", "none")
    if offline:
        routing_diagnostics["lyrics_source_routing_skip_reason"] = "offline"
    elif not target_duration:
        routing_diagnostics["lyrics_source_routing_skip_reason"] = "no_target_duration"
    elif not vocals_path:
        routing_diagnostics["lyrics_source_routing_skip_reason"] = "no_vocals_path"
    elif evaluate_sources:
        routing_diagnostics["lyrics_source_routing_skip_reason"] = (
            "explicit_audio_scoring"
        )


def _update_routing_diagnostics_from_disagreement(
    routing_diagnostics: Optional[dict],
    *,
    sources: dict,
    offline: bool,
    disagreement: dict,
) -> None:
    if routing_diagnostics is None:
        return
    if offline and not sources:
        routing_diagnostics["lyrics_source_routing_skip_reason"] = (
            "offline_no_cached_sources"
        )
    routing_diagnostics["lyrics_source_candidate_count"] = int(
        disagreement.get("source_count", 0) or 0
    )
    routing_diagnostics["lyrics_source_comparable_candidate_count"] = int(
        disagreement.get("comparable_source_count", 0) or 0
    )
    routing_diagnostics["lyrics_source_disagreement_flagged"] = bool(
        disagreement.get("flagged", False)
    )
    routing_diagnostics["lyrics_source_disagreement_reasons"] = list(
        disagreement.get("reasons", []) or []
    )
    if (
        not disagreement.get("flagged", False)
        and sources
        and routing_diagnostics["lyrics_source_routing_skip_reason"] == "offline"
    ):
        routing_diagnostics["lyrics_source_routing_skip_reason"] = (
            "no_material_disagreement"
        )


def _select_disagreement_source_if_needed(
    *,
    title: str,
    artist: str,
    target_duration: Optional[int],
    vocals_path: Optional[str],
    evaluate_sources: bool,
    filter_promos: bool,
    offline: bool,
    routing_diagnostics: Optional[dict],
    runtime_config: Optional[LyricsRuntimeConfig],
    logger,
) -> Optional[Tuple[Optional[str], Optional[List[Tuple[float, str]]], str]]:
    if not (target_duration and vocals_path and not evaluate_sources):
        return None
    from ..alignment.timing_evaluator import select_best_source
    from ..alignment.timing_evaluator_comparison import analyze_source_disagreement
    from .sync import fetch_from_all_sources

    sources = fetch_from_all_sources(title, artist, offline=offline)
    sources = _filter_sources_for_preferred_provider(
        sources,
        preferred_provider=(
            runtime_config.preferred_provider if runtime_config else None
        ),
    )
    disagreement = analyze_source_disagreement(title, artist, sources)
    _update_routing_diagnostics_from_disagreement(
        routing_diagnostics,
        sources=sources,
        offline=offline,
        disagreement=disagreement,
    )
    if not disagreement["flagged"]:
        return None

    reason_text = ", ".join(disagreement["reasons"])
    logger.info(
        "Lyrics source disagreement detected for %s - %s (%s); scoring candidates against audio",
        artist,
        title,
        reason_text,
    )
    lrc_text, source, report = select_best_source(
        title,
        artist,
        vocals_path,
        target_duration,
        sources=sources,
    )
    if not (lrc_text and source):
        return None
    if routing_diagnostics is not None:
        routing_diagnostics["lyrics_source_audio_scoring_used"] = True
        routing_diagnostics["lyrics_source_selection_mode"] = (
            "audio_scored_disagreement"
        )
        routing_diagnostics["lyrics_source_routing_skip_reason"] = "none"
    lines = parse_lrc_with_timing(lrc_text, title, artist, filter_promos=filter_promos)
    score_str = f" (score: {report.overall_score:.1f})" if report else ""
    logger.info(f"Selected best source after disagreement: {source}{score_str}")
    return lrc_text, lines, source
