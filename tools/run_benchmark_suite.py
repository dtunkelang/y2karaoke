#!/usr/bin/env python3
"""Run the benchmark song suite and emit an aggregated timing-quality report."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from urllib.parse import parse_qs, urlparse
from typing import Any, Iterable

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "benchmark_songs.yaml"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"
DEFAULT_GOLD_ROOT = REPO_ROOT / "benchmarks" / "gold_set"


@dataclass(frozen=True)
class BenchmarkSong:
    artist: str
    title: str
    youtube_id: str
    youtube_url: str

    @property
    def slug(self) -> str:
        safe = f"{self.artist}-{self.title}".lower()
        safe = re.sub(r"[^a-z0-9]+", "-", safe).strip("-")
        return safe or self.youtube_id


def _parse_manifest(path: Path) -> list[BenchmarkSong]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Manifest root must be a mapping")

    songs_raw = raw.get("songs", [])
    if not isinstance(songs_raw, list):
        raise ValueError("Manifest 'songs' must be a list")

    songs: list[BenchmarkSong] = []
    for idx, song in enumerate(songs_raw):
        if not isinstance(song, dict):
            raise ValueError(f"songs[{idx}] must be a mapping")
        try:
            songs.append(
                BenchmarkSong(
                    artist=str(song["artist"]),
                    title=str(song["title"]),
                    youtube_id=str(song["youtube_id"]),
                    youtube_url=str(song["youtube_url"]),
                )
            )
        except KeyError as exc:
            raise ValueError(f"songs[{idx}] missing required field: {exc}") from exc
    return songs


def _pctile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = pos - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _mean(values: Iterable[float]) -> float | None:
    data = list(values)
    if not data:
        return None
    return float(statistics.fmean(data))


def _normalize_agreement_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


def _agreement_text_similarity(left: Any, right: Any) -> float:
    a = _normalize_agreement_text(left)
    b = _normalize_agreement_text(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _normalize_word_text(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    cleaned = re.sub(r"[^a-z0-9'\s]", " ", raw.strip().lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _word_similarity(left: Any, right: Any) -> float:
    a = _normalize_word_text(left)
    b = _normalize_word_text(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _align_words_for_gold_comparison(
    generated_words: list[dict[str, Any]],
    gold_words: list[dict[str, Any]],
    *,
    lookahead: int = 12,
    min_similarity: float = 0.8,
) -> list[tuple[int, int, float]]:
    """Monotonic text-aware matching for generated-vs-gold words."""
    if not generated_words or not gold_words:
        return []

    matches: list[tuple[int, int, float]] = []
    gen_idx = 0
    gold_idx = 0
    gen_len = len(generated_words)
    gold_len = len(gold_words)

    while gen_idx < gen_len and gold_idx < gold_len:
        best: tuple[int, int, float, float] | None = None
        gen_hi = min(gen_len, gen_idx + lookahead + 1)
        gold_hi = min(gold_len, gold_idx + lookahead + 1)

        for i in range(gen_idx, gen_hi):
            for j in range(gold_idx, gold_hi):
                sim = _word_similarity(
                    generated_words[i].get("text", ""),
                    gold_words[j].get("text", ""),
                )
                if sim < min_similarity:
                    continue
                start_delta = abs(
                    float(generated_words[i].get("start", 0.0))
                    - float(gold_words[j].get("start", 0.0))
                )
                # Prefer higher similarity, then minimal skip distance, then timing proximity.
                score = (
                    (sim * 2.0)
                    - (0.05 * ((i - gen_idx) + (j - gold_idx)))
                    - (0.01 * min(start_delta, 10.0))
                )
                if best is None or score > best[3]:
                    best = (i, j, sim, score)

        if best is not None:
            i, j, sim, _score = best
            matches.append((i, j, sim))
            gen_idx = i + 1
            gold_idx = j + 1
            continue

        gen_start = float(generated_words[gen_idx].get("start", 0.0))
        gold_start = float(gold_words[gold_idx].get("start", 0.0))
        if gen_start <= gold_start:
            gen_idx += 1
        else:
            gold_idx += 1

    return matches


def _flatten_words_from_timing_doc(
    doc: dict[str, Any], *, mark_parenthetical_optional: bool = False
) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    lines = doc.get("lines", [])
    if not isinstance(lines, list):
        return words
    for line in lines:
        if not isinstance(line, dict):
            continue
        line_words = line.get("words", [])
        if not isinstance(line_words, list):
            continue
        paren_depth = 0
        for w in line_words:
            if not isinstance(w, dict):
                continue
            start = w.get("start")
            end = w.get("end")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            text = str(w.get("text", ""))
            optional = False
            if mark_parenthetical_optional:
                open_count = text.count("(")
                close_count = text.count(")")
                # Parenthesized tokens are treated as optional/backing vocals.
                optional = paren_depth > 0 or open_count > 0
                paren_depth = max(0, paren_depth + open_count - close_count)
            words.append(
                {
                    "text": text,
                    "start": float(start),
                    "end": float(end),
                    "optional": optional,
                }
            )
    return words


def _gold_path_for_song(
    index: int, song: BenchmarkSong, gold_root: Path
) -> Path | None:
    candidates = [
        gold_root / f"{index:02d}_{song.slug}.gold.json",
        gold_root / f"{song.slug}.gold.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _default_gold_path_for_song(
    index: int, song: BenchmarkSong, gold_root: Path
) -> Path:
    return gold_root / f"{index:02d}_{song.slug}.gold.json"


def _resolve_gold_rebaseline_path(
    index: int, song: BenchmarkSong, gold_root: Path
) -> Path:
    existing = _gold_path_for_song(index=index, song=song, gold_root=gold_root)
    if existing is not None:
        return existing
    return _default_gold_path_for_song(index=index, song=song, gold_root=gold_root)


def _write_rebaseline_gold(target_path: Path, report_doc: dict[str, Any]) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(report_doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _rebaseline_song_from_report(
    *,
    index: int,
    song: BenchmarkSong,
    report_path: Path,
    gold_root: Path,
) -> Path | None:
    if not report_path.exists():
        return None
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return None
    gold_path = _resolve_gold_rebaseline_path(
        index=index, song=song, gold_root=gold_root
    )
    _write_rebaseline_gold(gold_path, loaded)
    return gold_path


def _load_gold_doc(
    index: int, song: BenchmarkSong, gold_root: Path
) -> dict[str, Any] | None:
    gold_path = _gold_path_for_song(index=index, song=song, gold_root=gold_root)
    if gold_path is None:
        return None
    try:
        loaded = json.loads(gold_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


def _build_generate_command(
    *,
    python_bin: str,
    song: BenchmarkSong,
    report_path: Path,
    cache_dir: Path | None,
    offline: bool,
    force: bool,
    whisper_map_lrc_dtw: bool,
    strategy: str = "hybrid_dtw",
    drop_lrc_line_timings: bool = False,
) -> list[str]:
    cmd = [
        python_bin,
        "-m",
        "y2karaoke.cli",
        "generate",
        song.youtube_url,
        "--title",
        song.title,
        "--artist",
        song.artist,
        "--no-render",
        "--timing-report",
        str(report_path),
    ]
    if cache_dir is not None:
        cmd.extend(["--work-dir", str(cache_dir)])
    if offline:
        cmd.append("--offline")
    if force:
        cmd.append("--force")
    if drop_lrc_line_timings:
        cmd.append("--drop-lrc-line-timings")
    if strategy == "hybrid_dtw":
        if whisper_map_lrc_dtw:
            cmd.append("--whisper-map-lrc-dtw")
        else:
            cmd.append("--whisper")
    elif strategy == "hybrid_whisper":
        cmd.append("--whisper")
    elif strategy == "whisper_only":
        cmd.append("--whisper-only")
    elif strategy == "lrc_only":
        pass
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return cmd


def _extract_song_metrics(
    report: dict[str, Any], gold_doc: dict[str, Any] | None = None
) -> dict[str, Any]:
    lines = report.get("lines", [])
    line_count = len(lines)
    low_conf = report.get("low_confidence_lines", [])
    alignment_method = str(report.get("alignment_method") or "")
    dtw_line_coverage = report.get("dtw_line_coverage")
    has_independent_anchor = isinstance(dtw_line_coverage, (int, float))

    agreement_min_text_similarity = 0.7
    agreement_good_start_sec = 0.35
    agreement_warn_start_sec = 0.8
    whisper_anchor_start_abs_deltas: list[float] = []
    agreement_text_sims: list[float] = []
    for line in lines:
        line_start = line.get("start")
        whisper_anchor_start = line.get("nearest_segment_start")
        line_text = line.get("text")
        whisper_anchor_text = line.get("nearest_segment_start_text")
        if not isinstance(line_start, (int, float)):
            continue
        if not isinstance(whisper_anchor_start, (int, float)):
            continue
        sim = _agreement_text_similarity(line_text, whisper_anchor_text)
        if sim < agreement_min_text_similarity:
            continue
        agreement_text_sims.append(sim)
        whisper_anchor_start_abs_deltas.append(
            abs(float(line_start) - float(whisper_anchor_start))
        )

    # Independent agreement metric:
    # only available when we have DTW-based anchors (cross-strategy comparable path).
    agreement_start_abs_deltas = (
        whisper_anchor_start_abs_deltas if has_independent_anchor else []
    )

    def _bucket_counts(deltas: list[float]) -> tuple[int, int, int, int]:
        good = sum(1 for v in deltas if v <= agreement_good_start_sec)
        warn = sum(
            1
            for v in deltas
            if agreement_good_start_sec < v <= agreement_warn_start_sec
        )
        bad = sum(1 for v in deltas if v > agreement_warn_start_sec)
        severe = sum(1 for v in deltas if v > 1.5)
        return good, warn, bad, severe

    (
        agreement_good_count,
        agreement_warn_count,
        agreement_bad_count,
        agreement_severe_count,
    ) = _bucket_counts(agreement_start_abs_deltas)
    anchor_good_count, anchor_warn_count, anchor_bad_count, anchor_severe_count = (
        _bucket_counts(whisper_anchor_start_abs_deltas)
    )

    low_conf_ratio = (len(low_conf) / line_count) if line_count else 0.0
    metrics = {
        "alignment_method": alignment_method,
        "agreement_measurement_mode": (
            "independent_dtw_anchor"
            if has_independent_anchor
            else "unavailable_no_dtw_anchor"
        ),
        "line_count": line_count,
        "low_confidence_lines": len(low_conf),
        "low_confidence_ratio": round(low_conf_ratio, 4),
        "dtw_line_coverage": dtw_line_coverage,
        "dtw_word_coverage": report.get("dtw_word_coverage"),
        "dtw_phonetic_similarity_coverage": report.get(
            "dtw_phonetic_similarity_coverage"
        ),
        "agreement_min_text_similarity": agreement_min_text_similarity,
        "agreement_count": len(agreement_start_abs_deltas),
        "agreement_coverage_ratio": round(
            (len(agreement_start_abs_deltas) / line_count) if line_count else 0.0, 4
        ),
        "agreement_text_similarity_mean": round(_mean(agreement_text_sims) or 0.0, 4),
        "agreement_start_mean_abs_sec": round(
            _mean(agreement_start_abs_deltas) or 0.0, 4
        ),
        "agreement_start_p95_abs_sec": round(
            _pctile(agreement_start_abs_deltas, 0.95), 4
        ),
        "agreement_start_max_abs_sec": round(
            max(agreement_start_abs_deltas, default=0.0), 4
        ),
        "agreement_good_lines": agreement_good_count,
        "agreement_warn_lines": agreement_warn_count,
        "agreement_bad_lines": agreement_bad_count,
        "agreement_severe_lines": agreement_severe_count,
        "agreement_good_ratio": round(
            (agreement_good_count / line_count) if line_count else 0.0, 4
        ),
        "agreement_warn_ratio": round(
            (agreement_warn_count / line_count) if line_count else 0.0, 4
        ),
        "agreement_bad_ratio": round(
            (agreement_bad_count / line_count) if line_count else 0.0, 4
        ),
        "agreement_severe_ratio": round(
            (agreement_severe_count / line_count) if line_count else 0.0, 4
        ),
        # Diagnostic-only metric (self-referential to Whisper anchors) for debugging drift.
        "whisper_anchor_count": len(whisper_anchor_start_abs_deltas),
        "whisper_anchor_coverage_ratio": round(
            (len(whisper_anchor_start_abs_deltas) / line_count) if line_count else 0.0,
            4,
        ),
        "whisper_anchor_start_mean_abs_sec": round(
            _mean(whisper_anchor_start_abs_deltas) or 0.0, 4
        ),
        "whisper_anchor_start_p95_abs_sec": round(
            _pctile(whisper_anchor_start_abs_deltas, 0.95), 4
        ),
        "whisper_anchor_start_max_abs_sec": round(
            max(whisper_anchor_start_abs_deltas, default=0.0), 4
        ),
        "whisper_anchor_good_ratio": round(
            (anchor_good_count / line_count) if line_count else 0.0, 4
        ),
        "whisper_anchor_warn_ratio": round(
            (anchor_warn_count / line_count) if line_count else 0.0, 4
        ),
        "whisper_anchor_bad_ratio": round(
            (anchor_bad_count / line_count) if line_count else 0.0, 4
        ),
        "whisper_anchor_severe_ratio": round(
            (anchor_severe_count / line_count) if line_count else 0.0, 4
        ),
    }

    generated_words = _flatten_words_from_timing_doc(report)
    gold_words_all = _flatten_words_from_timing_doc(
        gold_doc or {}, mark_parenthetical_optional=True
    )
    gold_words = [w for w in gold_words_all if not bool(w.get("optional"))]
    aligned_pairs = _align_words_for_gold_comparison(generated_words, gold_words)
    comparable = len(aligned_pairs)
    start_abs_deltas: list[float] = []
    end_abs_deltas: list[float] = []
    text_matches = 0

    for gen_idx, gold_idx, sim in aligned_pairs:
        gen = generated_words[gen_idx]
        gold = gold_words[gold_idx]
        start_abs_deltas.append(abs(gen["start"] - gold["start"]))
        end_abs_deltas.append(abs(gen["end"] - gold["end"]))
        if sim >= 0.999:
            text_matches += 1

    metrics.update(
        {
            "gold_available": bool(gold_words_all),
            "generated_word_count": len(generated_words),
            "gold_word_count": len(gold_words),
            "gold_optional_word_count": len(gold_words_all) - len(gold_words),
            "gold_comparable_word_count": comparable,
            "gold_alignment_mode": "monotonic_text_window_parenthetical_optional",
            "gold_word_coverage_ratio": round(
                (comparable / len(gold_words)) if gold_words else 0.0, 4
            ),
            "gold_word_text_match_ratio": round(
                (text_matches / comparable) if comparable else 0.0, 4
            ),
            "avg_abs_word_start_delta_sec": round(_mean(start_abs_deltas) or 0.0, 4),
            "gold_start_mean_abs_sec": round(_mean(start_abs_deltas) or 0.0, 4),
            "gold_start_p95_abs_sec": round(_pctile(start_abs_deltas, 0.95), 4),
            "gold_start_max_abs_sec": round(max(start_abs_deltas, default=0.0), 4),
            "gold_end_mean_abs_sec": round(_mean(end_abs_deltas) or 0.0, 4),
            "gold_end_p95_abs_sec": round(_pctile(end_abs_deltas, 0.95), 4),
            "gold_end_max_abs_sec": round(max(end_abs_deltas, default=0.0), 4),
        }
    )

    return metrics


def _infer_reference_divergence_suspicion(metrics: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {"suspected": False, "score": 0.0, "confidence": "low", "evidence": []}
    if not bool(metrics.get("gold_available")):
        return {
            "suspected": False,
            "score": 0.0,
            "confidence": "low",
            "evidence": ["no_gold_reference"],
        }

    def _f(key: str) -> float:
        v = metrics.get(key)
        return float(v) if isinstance(v, (int, float)) else 0.0

    line_count = int(metrics.get("line_count", 0) or 0)
    score = 0.0
    evidence: list[str] = []

    gold_cov = _f("gold_word_coverage_ratio")
    gold_start_mean = _f("gold_start_mean_abs_sec")
    gold_start_p95 = _f("gold_start_p95_abs_sec")
    dtw_line_cov = _f("dtw_line_coverage")
    dtw_word_cov = _f("dtw_word_coverage")
    agree_cov = _f("agreement_coverage_ratio")
    agree_sim = _f("agreement_text_similarity_mean")
    agree_bad = _f("agreement_bad_ratio")
    low_conf = _f("low_confidence_ratio")
    comparable_words = int(metrics.get("gold_comparable_word_count", 0) or 0)

    if comparable_words < 20 or line_count < 10:
        return {
            "suspected": False,
            "score": 0.0,
            "confidence": "low",
            "evidence": ["insufficient_comparable_coverage"],
            "signals": {
                "gold_word_coverage_ratio": round(gold_cov, 4),
                "gold_start_mean_abs_sec": round(gold_start_mean, 4),
                "gold_start_p95_abs_sec": round(gold_start_p95, 4),
                "dtw_line_coverage": round(dtw_line_cov, 4),
                "agreement_coverage_ratio": round(agree_cov, 4),
                "agreement_text_similarity_mean": round(agree_sim, 4),
                "low_confidence_ratio": round(low_conf, 4),
                "gold_comparable_word_count": comparable_words,
            },
        }

    strong_dtw_internal = (
        dtw_line_cov >= 0.9 and dtw_word_cov >= 0.4 and low_conf <= 0.1
    )
    strong_agreement_subset = agree_cov >= 0.5 and agree_sim >= 0.9 and low_conf <= 0.08
    if strong_dtw_internal:
        score += 1.0
        evidence.append("strong_dtw_internal_consistency")
    if strong_agreement_subset:
        score += 1.0
        evidence.append("strong_lrc_whisper_agreement_subset")
    if low_conf <= 0.1:
        score += 0.25
        evidence.append("low_internal_uncertainty")

    severe_gold_timing_mismatch = gold_start_mean >= 20.0 or gold_start_p95 >= 35.0
    gold_coverage_timing_combo = gold_cov <= 0.68 and gold_start_mean >= 10.0
    if severe_gold_timing_mismatch:
        score += 1.5
        evidence.append("severe_gold_timing_mismatch")
    elif gold_start_mean >= 10.0 and gold_start_p95 >= 20.0:
        score += 0.5
        evidence.append("elevated_gold_timing_mismatch")
    if gold_coverage_timing_combo:
        score += 1.0
        evidence.append("gold_coverage_timing_combo_mismatch")

    # If internal signals are weak, treat this as likely pipeline quality rather than reference mismatch.
    if not (strong_dtw_internal or strong_agreement_subset):
        score -= 1.5
        evidence.append("insufficient_internal_consistency")
    if dtw_line_cov < 0.75 or low_conf > 0.2:
        score -= 0.5
        evidence.append("weak_internal_alignment_signals")
    if agree_cov > 0 and agree_bad > 0.2:
        score -= 0.5
        evidence.append("high_agreement_bad_ratio")

    suspected = bool(score >= 2.5)
    confidence = "high" if score >= 4.0 else "medium" if score >= 2.5 else "low"
    return {
        "suspected": suspected,
        "score": round(score, 3),
        "confidence": confidence,
        "evidence": sorted(set(evidence)),
        "signals": {
            "gold_word_coverage_ratio": round(gold_cov, 4),
            "gold_start_mean_abs_sec": round(gold_start_mean, 4),
            "gold_start_p95_abs_sec": round(gold_start_p95, 4),
            "dtw_line_coverage": round(dtw_line_cov, 4),
            "dtw_word_coverage": round(dtw_word_cov, 4),
            "agreement_coverage_ratio": round(agree_cov, 4),
            "agreement_text_similarity_mean": round(agree_sim, 4),
            "agreement_bad_ratio": round(agree_bad, 4),
            "low_confidence_ratio": round(low_conf, 4),
            "gold_comparable_word_count": comparable_words,
        },
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:  # noqa: C901
    succeeded = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] != "ok"]
    metrics = [r.get("metrics", {}) for r in succeeded]

    def metric_values(key: str) -> list[float]:
        vals: list[float] = []
        for m in metrics:
            value = m.get(key)
            if isinstance(value, (int, float)):
                vals.append(float(value))
        return vals

    def weighted_metric_mean(
        key: str, *, weight_key: str = "line_count"
    ) -> float | None:
        weighted_sum = 0.0
        total_weight = 0.0
        for m in metrics:
            value = m.get(key)
            weight = m.get(weight_key)
            if not isinstance(value, (int, float)):
                continue
            if not isinstance(weight, (int, float)) or float(weight) <= 0:
                continue
            weighted_sum += float(value) * float(weight)
            total_weight += float(weight)
        if total_weight <= 0:
            return None
        return weighted_sum / total_weight

    def metric_mean(key: str) -> float | None:
        result = _mean(metric_values(key))
        return float(result) if result is not None else None

    total_lines = int(sum(metric_values("line_count")))
    low_conf_total = int(sum(metric_values("low_confidence_lines")))
    agreement_count_total = int(sum(metric_values("agreement_count")))
    whisper_anchor_count_total = int(sum(metric_values("whisper_anchor_count")))
    agreement_good_total = int(sum(metric_values("agreement_good_lines")))
    agreement_warn_total = int(sum(metric_values("agreement_warn_lines")))
    agreement_bad_total = int(sum(metric_values("agreement_bad_lines")))
    agreement_severe_total = int(sum(metric_values("agreement_severe_lines")))
    low_conf_ratio = (low_conf_total / total_lines) if total_lines else 0.0
    agreement_coverage_ratio = (
        agreement_count_total / total_lines if total_lines else 0.0
    )
    whisper_anchor_coverage_ratio = (
        whisper_anchor_count_total / total_lines if total_lines else 0.0
    )
    agreement_good_ratio = (agreement_good_total / total_lines) if total_lines else 0.0
    agreement_warn_ratio = (agreement_warn_total / total_lines) if total_lines else 0.0
    agreement_bad_ratio = (agreement_bad_total / total_lines) if total_lines else 0.0
    agreement_severe_ratio = (
        agreement_severe_total / total_lines if total_lines else 0.0
    )
    measured_dtw_songs = [
        r
        for r in succeeded
        if isinstance(r.get("metrics", {}).get("dtw_line_coverage"), (int, float))
    ]
    measured_dtw_song_count = len(measured_dtw_songs)
    measured_dtw_line_count = int(
        sum(
            float(r.get("metrics", {}).get("line_count", 0))
            for r in measured_dtw_songs
            if isinstance(r.get("metrics", {}).get("line_count"), (int, float))
        )
    )
    gold_measured_songs = [
        r
        for r in succeeded
        if isinstance(
            r.get("metrics", {}).get("avg_abs_word_start_delta_sec"), (int, float)
        )
        and int(r.get("metrics", {}).get("gold_comparable_word_count", 0) or 0) > 0
    ]
    gold_metric_song_count = len(gold_measured_songs)
    gold_word_count_total = int(sum(metric_values("gold_word_count")))
    gold_comparable_word_count_total = int(
        sum(metric_values("gold_comparable_word_count"))
    )
    gold_word_coverage_ratio_total = (
        (gold_comparable_word_count_total / gold_word_count_total)
        if gold_word_count_total
        else 0.0
    )
    sum_song_elapsed = round(
        sum(
            float(r.get("elapsed_sec", 0.0))
            for r in results
            if isinstance(r.get("elapsed_sec"), (int, float))
        ),
        2,
    )
    phase_totals: dict[str, float] = {}
    for r in results:
        phase_map = r.get("phase_durations_sec")
        if not isinstance(phase_map, dict):
            continue
        for phase_name, raw_val in phase_map.items():
            if not isinstance(raw_val, (int, float)):
                continue
            phase_totals[phase_name] = phase_totals.get(phase_name, 0.0) + float(
                raw_val
            )
    phase_totals = {k: round(v, 2) for k, v in sorted(phase_totals.items())}
    phase_shares = (
        {k: round(v / sum_song_elapsed, 4) for k, v in phase_totals.items()}
        if sum_song_elapsed > 0
        else {}
    )

    def _hotspot_records(
        *,
        key: str,
        top_n: int = 3,
        reverse: bool = False,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for r in succeeded:
            m = r.get("metrics", {})
            if not isinstance(m, dict):
                continue
            value = m.get(key)
            if not isinstance(value, (int, float)):
                continue
            rows.append(
                {
                    "song": f"{r['artist']} - {r['title']}",
                    "value": round(float(value), 4),
                    "line_count": (
                        int(m.get("line_count", 0))
                        if isinstance(m.get("line_count"), (int, float))
                        else 0
                    ),
                }
            )
        rows.sort(key=lambda row: float(row["value"]), reverse=reverse)
        return rows[:top_n]

    def _cache_bucket(decision: Any) -> str:
        if not isinstance(decision, str):
            return "unknown"
        normalized = decision.strip().lower()
        if normalized.startswith("hit"):
            return "hit"
        if normalized.startswith("likely_hit"):
            return "likely_hit"
        if normalized.startswith("miss"):
            return "miss"
        if normalized.startswith("computed"):
            return "computed"
        return "unknown"

    cache_summary: dict[str, dict[str, Any]] = {}
    for phase_name in ("audio", "separation", "whisper", "alignment"):
        decisions: list[str] = []
        for r in succeeded:
            if bool(r.get("result_reused", False)):
                continue
            cache = r.get("cache_decisions", {})
            if not isinstance(cache, dict):
                continue
            value = cache.get(phase_name)
            if isinstance(value, str):
                decisions.append(value)
        counts = {
            "hit": 0,
            "likely_hit": 0,
            "miss": 0,
            "computed": 0,
            "unknown": 0,
        }
        for item in decisions:
            counts[_cache_bucket(item)] += 1
        total = len(decisions)
        cache_summary[phase_name] = {
            "total": total,
            "hit_count": counts["hit"],
            "likely_hit_count": counts["likely_hit"],
            "miss_count": counts["miss"],
            "computed_count": counts["computed"],
            "unknown_count": counts["unknown"],
            "cached_count": counts["hit"] + counts["likely_hit"],
            "cached_ratio": round(
                ((counts["hit"] + counts["likely_hit"]) / total) if total else 0.0,
                4,
            ),
            "miss_ratio": round((counts["miss"] / total) if total else 0.0, 4),
        }

    reference_divergence_suspects = [
        {
            "song": f"{r['artist']} - {r['title']}",
            "score": float(r.get("reference_divergence", {}).get("score", 0.0) or 0.0),
            "confidence": str(
                r.get("reference_divergence", {}).get("confidence", "low") or "low"
            ),
            "evidence": list(
                r.get("reference_divergence", {}).get("evidence", []) or []
            ),
        }
        for r in succeeded
        if isinstance(r.get("reference_divergence"), dict)
        and bool(r["reference_divergence"].get("suspected"))
    ]
    reference_divergence_suspects.sort(key=lambda row: row["score"], reverse=True)

    return {
        "songs_total": len(results),
        "songs_succeeded": len(succeeded),
        "songs_failed": len(failed),
        "success_rate": round((len(succeeded) / len(results)) if results else 0.0, 4),
        "line_count_total": total_lines,
        "low_confidence_lines_total": low_conf_total,
        "low_confidence_ratio_total": round(low_conf_ratio, 4),
        "agreement_count_total": agreement_count_total,
        "agreement_coverage_ratio_total": round(agreement_coverage_ratio, 4),
        "whisper_anchor_count_total": whisper_anchor_count_total,
        "whisper_anchor_coverage_ratio_total": round(whisper_anchor_coverage_ratio, 4),
        "agreement_good_lines_total": agreement_good_total,
        "agreement_warn_lines_total": agreement_warn_total,
        "agreement_bad_lines_total": agreement_bad_total,
        "agreement_severe_lines_total": agreement_severe_total,
        "agreement_good_ratio_total": round(agreement_good_ratio, 4),
        "agreement_warn_ratio_total": round(agreement_warn_ratio, 4),
        "agreement_bad_ratio_total": round(agreement_bad_ratio, 4),
        "agreement_severe_ratio_total": round(agreement_severe_ratio, 4),
        "dtw_line_coverage_mean": (
            round(float(metric_mean("dtw_line_coverage") or 0.0), 4)
            if metric_mean("dtw_line_coverage") is not None
            else None
        ),
        "dtw_word_coverage_mean": (
            round(float(metric_mean("dtw_word_coverage") or 0.0), 4)
            if metric_mean("dtw_word_coverage") is not None
            else None
        ),
        "dtw_phonetic_similarity_coverage_mean": (
            round(float(metric_mean("dtw_phonetic_similarity_coverage") or 0.0), 4)
            if metric_mean("dtw_phonetic_similarity_coverage") is not None
            else None
        ),
        "dtw_line_coverage_line_weighted_mean": (
            round(float(weighted_metric_mean("dtw_line_coverage") or 0.0), 4)
            if weighted_metric_mean("dtw_line_coverage") is not None
            else None
        ),
        "dtw_word_coverage_line_weighted_mean": (
            round(float(weighted_metric_mean("dtw_word_coverage") or 0.0), 4)
            if weighted_metric_mean("dtw_word_coverage") is not None
            else None
        ),
        "dtw_phonetic_similarity_coverage_line_weighted_mean": (
            round(
                float(weighted_metric_mean("dtw_phonetic_similarity_coverage") or 0.0),
                4,
            )
            if weighted_metric_mean("dtw_phonetic_similarity_coverage") is not None
            else None
        ),
        "agreement_coverage_ratio_mean": (
            round(float(metric_mean("agreement_coverage_ratio") or 0.0), 4)
            if metric_mean("agreement_coverage_ratio") is not None
            else None
        ),
        "agreement_text_similarity_mean": (
            round(float(metric_mean("agreement_text_similarity_mean") or 0.0), 4)
            if metric_mean("agreement_text_similarity_mean") is not None
            else None
        ),
        "agreement_start_mean_abs_sec_mean": (
            round(float(metric_mean("agreement_start_mean_abs_sec") or 0.0), 4)
            if metric_mean("agreement_start_mean_abs_sec") is not None
            else None
        ),
        "agreement_start_max_abs_sec_mean": (
            round(float(metric_mean("agreement_start_max_abs_sec") or 0.0), 4)
            if metric_mean("agreement_start_max_abs_sec") is not None
            else None
        ),
        "agreement_start_p95_abs_sec_mean": (
            round(float(metric_mean("agreement_start_p95_abs_sec") or 0.0), 4)
            if metric_mean("agreement_start_p95_abs_sec") is not None
            else None
        ),
        "agreement_bad_ratio_mean": (
            round(float(metric_mean("agreement_bad_ratio") or 0.0), 4)
            if metric_mean("agreement_bad_ratio") is not None
            else None
        ),
        "agreement_severe_ratio_mean": (
            round(float(metric_mean("agreement_severe_ratio") or 0.0), 4)
            if metric_mean("agreement_severe_ratio") is not None
            else None
        ),
        "whisper_anchor_start_mean_abs_sec_mean": (
            round(float(metric_mean("whisper_anchor_start_mean_abs_sec") or 0.0), 4)
            if metric_mean("whisper_anchor_start_mean_abs_sec") is not None
            else None
        ),
        "whisper_anchor_start_p95_abs_sec_mean": (
            round(float(metric_mean("whisper_anchor_start_p95_abs_sec") or 0.0), 4)
            if metric_mean("whisper_anchor_start_p95_abs_sec") is not None
            else None
        ),
        "whisper_anchor_bad_ratio_mean": (
            round(float(metric_mean("whisper_anchor_bad_ratio") or 0.0), 4)
            if metric_mean("whisper_anchor_bad_ratio") is not None
            else None
        ),
        "whisper_anchor_severe_ratio_mean": (
            round(float(metric_mean("whisper_anchor_severe_ratio") or 0.0), 4)
            if metric_mean("whisper_anchor_severe_ratio") is not None
            else None
        ),
        "dtw_metric_song_count": measured_dtw_song_count,
        "dtw_metric_song_coverage_ratio": round(
            (measured_dtw_song_count / len(succeeded)) if succeeded else 0.0, 4
        ),
        "dtw_metric_line_count": measured_dtw_line_count,
        "dtw_metric_line_coverage_ratio": round(
            (measured_dtw_line_count / total_lines) if total_lines else 0.0, 4
        ),
        "gold_metric_song_count": gold_metric_song_count,
        "gold_metric_song_coverage_ratio": round(
            (gold_metric_song_count / len(succeeded)) if succeeded else 0.0, 4
        ),
        "reference_divergence_suspected_count": len(reference_divergence_suspects),
        "reference_divergence_suspected_ratio": round(
            (len(reference_divergence_suspects) / len(succeeded)) if succeeded else 0.0,
            4,
        ),
        "gold_word_count_total": gold_word_count_total,
        "gold_comparable_word_count_total": gold_comparable_word_count_total,
        "gold_word_coverage_ratio_total": round(gold_word_coverage_ratio_total, 4),
        "avg_abs_word_start_delta_sec_mean": (
            round(float(metric_mean("avg_abs_word_start_delta_sec") or 0.0), 4)
            if metric_mean("avg_abs_word_start_delta_sec") is not None
            else None
        ),
        "avg_abs_word_start_delta_sec_word_weighted_mean": (
            round(
                float(
                    weighted_metric_mean(
                        "avg_abs_word_start_delta_sec",
                        weight_key="gold_comparable_word_count",
                    )
                    or 0.0
                ),
                4,
            )
            if weighted_metric_mean(
                "avg_abs_word_start_delta_sec",
                weight_key="gold_comparable_word_count",
            )
            is not None
            else None
        ),
        "gold_start_p95_abs_sec_mean": (
            round(float(metric_mean("gold_start_p95_abs_sec") or 0.0), 4)
            if metric_mean("gold_start_p95_abs_sec") is not None
            else None
        ),
        "gold_end_mean_abs_sec_mean": (
            round(float(metric_mean("gold_end_mean_abs_sec") or 0.0), 4)
            if metric_mean("gold_end_mean_abs_sec") is not None
            else None
        ),
        "gold_end_p95_abs_sec_mean": (
            round(float(metric_mean("gold_end_p95_abs_sec") or 0.0), 4)
            if metric_mean("gold_end_p95_abs_sec") is not None
            else None
        ),
        "songs_without_dtw_metrics": [
            f"{r['artist']} - {r['title']}"
            for r in succeeded
            if not isinstance(
                r.get("metrics", {}).get("dtw_line_coverage"), (int, float)
            )
        ],
        "songs_without_gold_metrics": [
            f"{r['artist']} - {r['title']}"
            for r in succeeded
            if not isinstance(
                r.get("metrics", {}).get("avg_abs_word_start_delta_sec"), (int, float)
            )
            or int(r.get("metrics", {}).get("gold_comparable_word_count", 0) or 0) <= 0
        ],
        "sum_song_elapsed_sec": sum_song_elapsed,
        "phase_totals_sec": phase_totals,
        "phase_shares_of_song_elapsed": phase_shares,
        "quality_hotspots": {
            "lowest_dtw_line_coverage": _hotspot_records(
                key="dtw_line_coverage", top_n=3, reverse=False
            ),
            "highest_low_confidence_ratio": _hotspot_records(
                key="low_confidence_ratio", top_n=3, reverse=True
            ),
            "lowest_agreement_coverage_ratio": _hotspot_records(
                key="agreement_coverage_ratio", top_n=3, reverse=False
            ),
            "highest_agreement_start_p95_abs_sec": _hotspot_records(
                key="agreement_start_p95_abs_sec", top_n=3, reverse=True
            ),
            "highest_agreement_bad_ratio": _hotspot_records(
                key="agreement_bad_ratio", top_n=3, reverse=True
            ),
            "highest_agreement_severe_ratio": _hotspot_records(
                key="agreement_severe_ratio", top_n=3, reverse=True
            ),
            "highest_whisper_anchor_start_p95_abs_sec": _hotspot_records(
                key="whisper_anchor_start_p95_abs_sec", top_n=3, reverse=True
            ),
            "highest_avg_abs_word_start_delta_sec": _hotspot_records(
                key="avg_abs_word_start_delta_sec", top_n=3, reverse=True
            ),
            "lowest_gold_word_coverage_ratio": _hotspot_records(
                key="gold_word_coverage_ratio", top_n=3, reverse=False
            ),
            "reference_divergence_suspects": reference_divergence_suspects[:5],
        },
        "cache_summary": cache_summary,
        "failed_songs": [f"{r['artist']} - {r['title']}" for r in failed],
    }


def _quality_coverage_warnings(
    *,
    aggregate: dict[str, Any],
    dtw_enabled: bool,
    min_song_coverage_ratio: float,
    min_line_coverage_ratio: float,
    suite_wall_elapsed_sec: float,
) -> list[str]:
    warnings: list[str] = []
    song_cov = float(aggregate.get("dtw_metric_song_coverage_ratio", 0.0) or 0.0)
    line_cov = float(aggregate.get("dtw_metric_line_coverage_ratio", 0.0) or 0.0)
    if not dtw_enabled:
        warnings.append(
            "DTW mapping is disabled (--no-whisper-map-lrc-dtw); quality metrics may be partially unmeasured."
        )
    if song_cov < min_song_coverage_ratio:
        warnings.append(
            "DTW song coverage below threshold: "
            f"{song_cov:.3f} < {min_song_coverage_ratio:.3f}"
        )
    if line_cov < min_line_coverage_ratio:
        warnings.append(
            "DTW line coverage below threshold: "
            f"{line_cov:.3f} < {min_line_coverage_ratio:.3f}"
        )
    if int(aggregate.get("agreement_count_total", 0) or 0) == 0:
        warnings.append(
            "Independent line-start agreement is unavailable for this strategy "
            "(no DTW anchor coverage); use Whisper-anchor diagnostics only for debugging."
        )
    else:
        agreement_cov = float(
            aggregate.get("agreement_coverage_ratio_mean", 0.0) or 0.0
        )
        if agreement_cov < 0.4:
            warnings.append(
                "LRC-Whisper agreement coverage is low: "
                f"{agreement_cov:.3f} < 0.400 (not enough lexically comparable lines)"
            )
        agreement_p95 = float(
            aggregate.get("agreement_start_p95_abs_sec_mean", 0.0) or 0.0
        )
        if agreement_p95 > 0.8:
            warnings.append(
                "Line-start agreement p95 is high on comparable lines: "
                f"{agreement_p95:.3f}s > 0.800s"
            )
        agreement_bad_ratio = float(
            aggregate.get("agreement_bad_ratio_total", 0.0) or 0.0
        )
        if agreement_bad_ratio > 0.1:
            warnings.append(
                "Too many comparable lines have poor start agreement (>0.8s): "
                f"{agreement_bad_ratio:.3f} > 0.100"
            )
        agreement_severe_ratio = float(
            aggregate.get("agreement_severe_ratio_total", 0.0) or 0.0
        )
        if agreement_severe_ratio > 0.03:
            warnings.append(
                "Comparable lines with severe agreement error (>1.5s) are high: "
                f"{agreement_severe_ratio:.3f} > 0.030"
            )
    gold_metric_song_count = int(aggregate.get("gold_metric_song_count", 0) or 0)
    if gold_metric_song_count > 0:
        gold_song_cov = float(
            aggregate.get("gold_metric_song_coverage_ratio", 0.0) or 0.0
        )
        if gold_song_cov < 0.5:
            warnings.append(
                "Gold-set metric song coverage is low: " f"{gold_song_cov:.3f} < 0.500"
            )
        gold_word_cov = float(
            aggregate.get("gold_word_coverage_ratio_total", 0.0) or 0.0
        )
        if gold_word_cov < 0.8:
            warnings.append(
                "Gold-set comparable word coverage is low: "
                f"{gold_word_cov:.3f} < 0.800"
            )
        gold_start_mean = float(
            aggregate.get("avg_abs_word_start_delta_sec_word_weighted_mean", 0.0) or 0.0
        )
        if gold_start_mean > 0.35:
            warnings.append(
                "Gold-set avg abs word-start delta is high: "
                f"{gold_start_mean:.3f}s > 0.350s"
            )

    sum_song_elapsed = float(aggregate.get("sum_song_elapsed_sec", 0.0) or 0.0)
    if suite_wall_elapsed_sec > 0 and sum_song_elapsed > suite_wall_elapsed_sec:
        warnings.append(
            "Per-song elapsed sum exceeds suite wall elapsed; compare runs using "
            "suite_wall_elapsed_sec and sum_song_elapsed_sec explicitly."
        )
    return warnings


def _cache_expectation_warnings(
    *,
    aggregate: dict[str, Any],
    expect_cached_separation: bool,
    expect_cached_whisper: bool,
) -> list[str]:
    warnings: list[str] = []
    cache_summary = aggregate.get("cache_summary", {})
    if not isinstance(cache_summary, dict):
        return warnings

    if expect_cached_separation:
        sep = cache_summary.get("separation", {})
        if isinstance(sep, dict):
            miss = int(sep.get("miss_count", 0) or 0)
            total = int(sep.get("total", 0) or 0)
            unknown = int(sep.get("unknown_count", 0) or 0)
            cached_ratio = float(sep.get("cached_ratio", 0.0) or 0.0)
            if total == 0:
                warnings.append(
                    "Expected cached separation but no executed-song cache data was available "
                    "(results likely reused). Re-run with --rerun-completed."
                )
            elif miss > 0:
                warnings.append(
                    "Expected cached separation but misses were observed: "
                    f"{miss}/{total} miss, cached_ratio={cached_ratio:.3f}"
                )
            elif unknown > 0:
                warnings.append(
                    "Expected cached separation but cache state was unknown for "
                    f"{unknown}/{total} song(s)."
                )
            elif cached_ratio < 1.0:
                warnings.append(
                    "Expected cached separation but cached_ratio was below 1.0: "
                    f"{cached_ratio:.3f}"
                )

    if expect_cached_whisper:
        whisper = cache_summary.get("whisper", {})
        if isinstance(whisper, dict):
            miss = int(whisper.get("miss_count", 0) or 0)
            total = int(whisper.get("total", 0) or 0)
            unknown = int(whisper.get("unknown_count", 0) or 0)
            cached_ratio = float(whisper.get("cached_ratio", 0.0) or 0.0)
            if total == 0:
                warnings.append(
                    "Expected cached whisper but no executed-song cache data was available "
                    "(results likely reused). Re-run with --rerun-completed."
                )
            elif miss > 0:
                warnings.append(
                    "Expected cached whisper but misses were observed: "
                    f"{miss}/{total} miss, cached_ratio={cached_ratio:.3f}"
                )
            elif unknown > 0:
                warnings.append(
                    "Expected cached whisper but cache state was unknown for "
                    f"{unknown}/{total} song(s)."
                )
            elif cached_ratio < 1.0:
                warnings.append(
                    "Expected cached whisper but cached_ratio was below 1.0: "
                    f"{cached_ratio:.3f}"
                )
    return warnings


def _write_markdown_summary(  # noqa: C901
    path: Path,
    *,
    run_id: str,
    manifest: Path,
    aggregate: dict[str, Any],
    songs: list[dict[str, Any]],
) -> None:
    def _fmt_num(value: Any, *, unit: str = "", digits: int = 3) -> str:
        if isinstance(value, (int, float)):
            return f"{float(value):.{digits}f}{unit}"
        return "-"

    lines: list[str] = []
    lines.append("# Benchmark Timing Quality Report")
    lines.append("")
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Manifest: `{manifest}`")
    lines.append(
        f"- Songs: `{aggregate['songs_succeeded']}/{aggregate['songs_total']}` succeeded"
    )
    lines.append(
        f"- Mean DTW line coverage: `{_fmt_num(aggregate.get('dtw_line_coverage_mean'))}`"
    )
    lines.append(
        "- Gold metric coverage: "
        f"`{aggregate.get('gold_metric_song_count', 0)}/{aggregate.get('songs_succeeded', 0)}` songs, "
        f"`{aggregate.get('gold_comparable_word_count_total', 0)}/{aggregate.get('gold_word_count_total', 0)}` words"
    )
    lines.append(
        "- Gold comparable word coverage ratio: "
        f"`{aggregate.get('gold_word_coverage_ratio_total', 0.0):.3f}`"
    )
    lines.append(
        "- Reference divergence suspects (first-pass heuristic): "
        f"`{aggregate.get('reference_divergence_suspected_count', 0)}` "
        f"({aggregate.get('reference_divergence_suspected_ratio', 0.0):.3f})"
    )
    lines.append(
        "- Primary metric (avg abs word-start delta): "
        f"`{_fmt_num(aggregate.get('avg_abs_word_start_delta_sec_word_weighted_mean'), unit='s')}` (word-weighted)"
    )
    lines.append(
        "- Secondary metric (avg abs word-end delta): "
        f"`{_fmt_num(aggregate.get('gold_end_mean_abs_sec_mean'), unit='s')}`"
    )
    lines.append(
        f"- Mean DTW word coverage: `{_fmt_num(aggregate.get('dtw_word_coverage_mean'))}`"
    )
    lines.append(
        "- Mean DTW phonetic similarity coverage: "
        f"`{_fmt_num(aggregate.get('dtw_phonetic_similarity_coverage_mean'))}`"
    )
    lines.append(
        f"- Low-confidence line ratio: `{aggregate['low_confidence_ratio_total']:.3f}`"
    )
    lines.append(
        "- Agreement coverage ratio: "
        f"`{_fmt_num(aggregate.get('agreement_coverage_ratio_total'))}`"
    )
    lines.append(
        "- Mean agreement text similarity: "
        f"`{_fmt_num(aggregate.get('agreement_text_similarity_mean'))}`"
    )
    lines.append(
        "- Mean abs line-start agreement error (comparable lines): "
        f"`{_fmt_num(aggregate.get('agreement_start_mean_abs_sec_mean'), unit='s')}`"
    )
    lines.append(
        "- p95 abs line-start agreement error (comparable lines): "
        f"`{_fmt_num(aggregate.get('agreement_start_p95_abs_sec_mean'), unit='s')}`"
    )
    lines.append(
        "- Poor agreement ratio (>0.8s): "
        f"`{aggregate.get('agreement_bad_ratio_total', 0.0):.3f}`"
    )
    lines.append(
        "- Severe agreement ratio (>1.5s): "
        f"`{aggregate.get('agreement_severe_ratio_total', 0.0):.3f}`"
    )
    lines.append(
        "- Whisper-anchor diagnostic p95 abs line-start delta: "
        f"`{_fmt_num(aggregate.get('whisper_anchor_start_p95_abs_sec_mean'), unit='s')}`"
    )
    lines.append(
        "- DTW metric coverage: "
        f"`{aggregate.get('dtw_metric_song_count', 0)}/{aggregate.get('songs_succeeded', 0)}` "
        f"songs, `{aggregate.get('dtw_metric_line_count', 0)}/{aggregate.get('line_count_total', 0)}` lines"
    )
    lines.append(
        "- Mean DTW line coverage (line-weighted): "
        f"`{_fmt_num(aggregate.get('dtw_line_coverage_line_weighted_mean'))}`"
    )
    cache_summary = aggregate.get("cache_summary", {})
    if isinstance(cache_summary, dict):
        sep = cache_summary.get("separation")
        if isinstance(sep, dict):
            lines.append(
                "- Separation cache ratio: "
                f"`{float(sep.get('cached_ratio', 0.0)):.3f}` "
                f"({int(sep.get('cached_count', 0) or 0)}/{int(sep.get('total', 0) or 0)})"
            )
    phase_totals = aggregate.get("phase_totals_sec", {})
    if isinstance(phase_totals, dict) and phase_totals:
        top_phase = max(phase_totals, key=lambda key: float(phase_totals.get(key, 0.0)))
        lines.append(
            "- Slowest phase by summed song time: "
            f"`{top_phase}` (`{float(phase_totals[top_phase]):.2f}s`)"
        )
    lines.append("")
    lines.append("## Per-song")
    lines.append("")
    lines.append(
        "| Song | Status | Gold words | Gold cov | Word start abs mean | Word start p95 | Word end abs mean | Alignment | DTW line | DTW word | Ref mismatch suspect | Elapsed |"  # noqa: E501
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for song in songs:
        metrics = song.get("metrics", {})
        ref_div = song.get("reference_divergence", {})
        ref_div_cell = "-"
        if isinstance(ref_div, dict) and ref_div:
            if bool(ref_div.get("suspected")):
                ref_div_cell = (
                    f"yes ({ref_div.get('confidence', 'low')}, "
                    f"{float(ref_div.get('score', 0.0) or 0.0):.2f})"
                )
            else:
                ref_div_cell = f"no ({float(ref_div.get('score', 0.0) or 0.0):.2f})"
        lines.append(
            "| "
            + f"{song['artist']} - {song['title']} | "
            + f"{song['status']} | "
            + f"{metrics.get('gold_comparable_word_count', '-')}/{metrics.get('gold_word_count', '-')}"
            + " | "
            + f"{metrics.get('gold_word_coverage_ratio', '-')}"
            + " | "
            + f"{metrics.get('avg_abs_word_start_delta_sec', '-')}"
            + " | "
            + f"{metrics.get('gold_start_p95_abs_sec', '-')}"
            + " | "
            + f"{metrics.get('gold_end_mean_abs_sec', '-')}"
            + " | "
            + f"{metrics.get('alignment_method', '-')}"
            + " | "
            + f"{metrics.get('dtw_line_coverage', '-')}"
            + " | "
            + f"{metrics.get('dtw_word_coverage', '-')}"
            + " | "
            + ref_div_cell
            + " | "
            + f"{song.get('elapsed_sec', '-')}"
            + "s |"
        )
    lines.append("")
    hotspots = aggregate.get("quality_hotspots", {})
    if isinstance(hotspots, dict):
        low_dtw = hotspots.get("lowest_dtw_line_coverage", [])
        high_low_conf = hotspots.get("highest_low_confidence_ratio", [])
        low_agree_cov = hotspots.get("lowest_agreement_coverage_ratio", [])
        high_agree_p95 = hotspots.get("highest_agreement_start_p95_abs_sec", [])
        high_agree_bad = hotspots.get("highest_agreement_bad_ratio", [])
        high_agree_severe = hotspots.get("highest_agreement_severe_ratio", [])
        high_gold_start = hotspots.get("highest_avg_abs_word_start_delta_sec", [])
        low_gold_cov = hotspots.get("lowest_gold_word_coverage_ratio", [])
        ref_div_suspects = hotspots.get("reference_divergence_suspects", [])
        if (
            low_dtw
            or high_low_conf
            or low_agree_cov
            or high_agree_p95
            or high_agree_bad
            or high_agree_severe
            or high_gold_start
            or low_gold_cov
            or ref_div_suspects
        ):
            lines.append("## Hotspots")
            lines.append("")
            if low_dtw:
                lines.append("- Lowest DTW line coverage:")
                for item in low_dtw:
                    if isinstance(item, dict):
                        lines.append(f"  - {item.get('song')}: {item.get('value')}")
            if high_low_conf:
                lines.append("- Highest low-confidence ratio:")
                for item in high_low_conf:
                    if isinstance(item, dict):
                        lines.append(f"  - {item.get('song')}: {item.get('value')}")
            if low_agree_cov:
                lines.append("- Lowest agreement coverage ratio:")
                for item in low_agree_cov:
                    if isinstance(item, dict):
                        lines.append(f"  - {item.get('song')}: {item.get('value')}")
            if high_agree_p95:
                lines.append("- Highest agreement start p95 error:")
                for item in high_agree_p95:
                    if isinstance(item, dict):
                        lines.append(f"  - {item.get('song')}: {item.get('value')}s")
            if high_agree_bad:
                lines.append("- Highest poor agreement ratio (>0.8s):")
                for item in high_agree_bad:
                    if isinstance(item, dict):
                        lines.append(f"  - {item.get('song')}: {item.get('value')}")
            if high_agree_severe:
                lines.append("- Highest severe agreement ratio (>1.5s):")
                for item in high_agree_severe:
                    if isinstance(item, dict):
                        lines.append(f"  - {item.get('song')}: {item.get('value')}")
            if high_gold_start:
                lines.append("- Highest avg abs word-start delta (gold):")
                for item in high_gold_start:
                    if isinstance(item, dict):
                        lines.append(f"  - {item.get('song')}: {item.get('value')}s")
            if low_gold_cov:
                lines.append("- Lowest gold comparable word coverage ratio:")
                for item in low_gold_cov:
                    if isinstance(item, dict):
                        lines.append(f"  - {item.get('song')}: {item.get('value')}")
            if ref_div_suspects:
                lines.append("- Reference divergence suspects (heuristic):")
                for item in ref_div_suspects:
                    if isinstance(item, dict):
                        evidence = item.get("evidence") or []
                        if isinstance(evidence, list):
                            evidence_text = ", ".join(str(v) for v in evidence[:3])
                        else:
                            evidence_text = str(evidence)
                        lines.append(
                            "  - "
                            f"{item.get('song')}: score={item.get('score')} "
                            f"({item.get('confidence')}); {evidence_text}"
                        )
            lines.append("")
    if aggregate["failed_songs"]:
        lines.append("## Failures")
        lines.append("")
        for item in aggregate["failed_songs"]:
            lines.append(f"- {item}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_run_dir(
    *,
    output_root: Path,
    run_id: str | None,
    resume_run_dir: Path | None,
    resume_latest: bool,
) -> tuple[Path, str]:
    if resume_run_dir is not None:
        return resume_run_dir.resolve(), resume_run_dir.resolve().name

    if resume_latest:
        candidates = sorted(
            [p for p in output_root.resolve().glob("20*") if p.is_dir()],
            key=lambda p: (p.stat().st_mtime, p.name),
            reverse=True,
        )
        if candidates:
            return candidates[0], candidates[0].name

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_root.resolve() / run_id, run_id


def _song_result_path(run_dir: Path, index: int, slug: str) -> Path:
    return run_dir / f"{index:02d}_{slug}_result.json"


def _load_song_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _refresh_cached_metrics(
    record: dict[str, Any],
    *,
    index: int | None = None,
    song: BenchmarkSong | None = None,
    gold_root: Path = DEFAULT_GOLD_ROOT,
) -> dict[str, Any]:
    """Backfill metrics from timing report for legacy cached result files."""
    if str(record.get("status", "")) != "ok":
        return record
    report_path_raw = record.get("report_path")
    if not isinstance(report_path_raw, str) or not report_path_raw:
        return record
    report_path = Path(report_path_raw)
    if not report_path.exists():
        return record
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return record
    if not isinstance(report, dict):
        return record
    gold_doc = None
    if index is not None and song is not None:
        gold_doc = _load_gold_doc(index=index, song=song, gold_root=gold_root)
    record["metrics"] = _extract_song_metrics(report, gold_doc=gold_doc)
    record["reference_divergence"] = _infer_reference_divergence_suspicion(
        record["metrics"]
    )
    return record


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_checkpoint(
    *,
    run_id: str,
    run_dir: Path,
    manifest_path: Path,
    args: argparse.Namespace,
    song_results: list[dict[str, Any]],
    suite_elapsed: float,
) -> None:
    aggregate = _aggregate(song_results)
    suite_wall_elapsed = round(suite_elapsed, 2)
    sum_song_elapsed = float(aggregate.get("sum_song_elapsed_sec", 0.0) or 0.0)
    report_json = {
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "repo_root": str(REPO_ROOT),
        "started_at_utc": run_id,
        "elapsed_sec": suite_wall_elapsed,
        "suite_wall_elapsed_sec": suite_wall_elapsed,
        "sum_song_elapsed_sec": round(sum_song_elapsed, 2),
        "scheduler_overhead_sec": round(suite_wall_elapsed - sum_song_elapsed, 2),
        "status": "running",
        "options": {
            "offline": args.offline,
            "force": args.force,
            "strategy": args.strategy,
            "scenario": args.scenario,
            "whisper_map_lrc_dtw": not args.no_whisper_map_lrc_dtw,
            "timeout_sec": args.timeout_sec,
            "heartbeat_sec": args.heartbeat_sec,
            "match": args.match,
            "max_songs": args.max_songs,
            "min_dtw_song_coverage_ratio": args.min_dtw_song_coverage_ratio,
            "min_dtw_line_coverage_ratio": args.min_dtw_line_coverage_ratio,
            "strict_quality_coverage": args.strict_quality_coverage,
            "expect_cached_separation": args.expect_cached_separation,
            "expect_cached_whisper": args.expect_cached_whisper,
            "strict_cache_expectations": args.strict_cache_expectations,
            "rebaseline": args.rebaseline,
            "gold_root": str(args.gold_root.resolve()),
        },
        "aggregate": aggregate,
        "songs": song_results,
    }
    _write_json(run_dir / "benchmark_progress.json", report_json)


def _build_run_signature(
    args: argparse.Namespace, manifest_path: Path
) -> dict[str, Any]:
    gold_root = getattr(args, "gold_root", None)
    return {
        "manifest_path": str(manifest_path),
        "offline": bool(args.offline),
        "force": bool(args.force),
        "strategy": str(args.strategy),
        "scenario": str(args.scenario),
        "whisper_map_lrc_dtw": not bool(args.no_whisper_map_lrc_dtw),
        "cache_dir": str(args.cache_dir.resolve()) if args.cache_dir else None,
        "gold_root": str(gold_root.resolve()) if gold_root else None,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Benchmark manifest YAML path",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory where benchmark run folders are created",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache dir passed to y2karaoke via --work-dir",
    )
    parser.add_argument(
        "--gold-root",
        type=Path,
        default=DEFAULT_GOLD_ROOT,
        help="Directory containing per-song *.gold.json files",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Use a stable run id (folder name under --output-root)",
    )
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        default=None,
        help="Resume a specific existing run directory",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume the most recently modified run directory in --output-root",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter to invoke for each song run",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=1800,
        help="Per-song timeout (seconds)",
    )
    parser.add_argument(
        "--max-songs",
        type=int,
        default=0,
        help="Run only the first N songs (0 means all)",
    )
    parser.add_argument(
        "--match",
        type=str,
        default="",
        help="Only run songs whose artist/title matches this case-insensitive regex",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Pass --offline to y2karaoke generate (requires cached data)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to y2karaoke generate",
    )
    parser.add_argument(
        "--no-whisper-map-lrc-dtw",
        action="store_true",
        help="Disable --whisper-map-lrc-dtw while running benchmark songs",
    )
    parser.add_argument(
        "--strategy",
        choices=["hybrid_dtw", "hybrid_whisper", "whisper_only", "lrc_only"],
        default="hybrid_dtw",
        help=(
            "Benchmark strategy: hybrid_dtw (default), hybrid_whisper, "
            "whisper_only, or lrc_only."
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=["default", "lyrics_no_timing"],
        default="default",
        help=(
            "Optional benchmark scenario variant. "
            "'lyrics_no_timing' ignores provider LRC timestamps to isolate "
            "lyrics-text + audio alignment behavior."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at first failing song",
    )
    parser.add_argument(
        "--heartbeat-sec",
        type=int,
        default=30,
        help="Progress heartbeat interval while a song is running",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="When resuming, rerun songs previously marked failed",
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="When resuming, rerun songs previously marked ok",
    )
    parser.add_argument(
        "--reuse-mismatched-results",
        action="store_true",
        help="Reuse cached per-song results even when run options differ",
    )
    parser.add_argument(
        "--min-dtw-song-coverage-ratio",
        type=float,
        default=0.9,
        help="Warn if successful-song DTW coverage ratio is below this threshold",
    )
    parser.add_argument(
        "--min-dtw-line-coverage-ratio",
        type=float,
        default=0.9,
        help="Warn if benchmark-line DTW coverage ratio is below this threshold",
    )
    parser.add_argument(
        "--strict-quality-coverage",
        action="store_true",
        help="Return non-zero if quality coverage warnings are present",
    )
    parser.add_argument(
        "--expect-cached-separation",
        action="store_true",
        help="Warn if separation phase is not fully cache-hit across successful songs",
    )
    parser.add_argument(
        "--expect-cached-whisper",
        action="store_true",
        help="Warn if whisper phase is not fully cache-hit across successful songs",
    )
    parser.add_argument(
        "--strict-cache-expectations",
        action="store_true",
        help="Return non-zero if cache expectation warnings are present",
    )
    parser.add_argument(
        "--rebaseline",
        action="store_true",
        help=(
            "Update per-song gold files from successful timing reports. "
            "Use --match/--max-songs to constrain scope."
        ),
    )
    return parser.parse_args()


def _tail_text(data: str | bytes | None, line_count: int = 30) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        text = data.decode("utf-8", errors="replace")
    else:
        text = data
    return "\n".join(text.splitlines()[-line_count:])


def _coerce_text(data: str | bytes | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


def _extract_stage_hint(stdout_text: str, stderr_text: str) -> str | None:
    """Infer a readable stage hint from accumulated subprocess output."""
    merged = f"{stdout_text}\n{stderr_text}"
    lines = [line.strip() for line in merged.splitlines()]
    candidates = [line for line in lines if line]
    if not candidates:
        return None

    # Prefer explicit y2karaoke log lines if present.
    log_lines = [
        line
        for line in candidates
        if "INFO:y2karaoke" in line
        or "WARNING:y2karaoke" in line
        or "ERROR:y2karaoke" in line
    ]
    target_lines = log_lines if log_lines else candidates

    stage_markers = [
        ("karaoke generation complete", "complete"),
        ("rendering karaoke video", "render"),
        ("skipping video rendering", "render_skip"),
        ("creating background segments", "backgrounds"),
        ("wrote timing report", "timing_report"),
        ("scaling lyrics timing", "timing_scale"),
        ("fetching lyrics", "lyrics_fetch"),
        ("whisper", "whisper_alignment"),
        ("transcrib", "whisper_alignment"),
        ("align", "whisper_alignment"),
        ("separat", "separation"),
        ("demucs", "separation"),
        ("stem", "separation"),
        ("using cached audio", "media_cached_audio"),
        ("downloading audio", "media_download_audio"),
        ("using cached video", "media_cached_video"),
        ("downloading video", "media_download_video"),
        ("identifying track", "identify_track"),
        ("video id:", "identify_track"),
    ]

    # Ignore tqdm/progress-noise lines and return the newest meaningful line.
    newest_line: str | None = None
    for raw in reversed(target_lines):
        line = raw.replace("\r", " ").strip()
        if not line:
            continue
        if line.startswith("%|") or "/it]" in line:
            continue
        if line.startswith("[") and line.endswith("]"):
            continue
        if len(line) > 180:
            line = line[-180:]
        newest_line = line
        break

    # Prefer stage inferred from the newest meaningful line so label and text match.
    if newest_line is not None:
        newest_lower = newest_line.lower()
        for marker, label in stage_markers:
            if marker in newest_lower:
                return f"[{label}] {newest_line}"
        return newest_line

    # Fallback: infer a coarse stage from all buffered output.
    merged_lower = merged.lower()
    for marker, label in stage_markers:
        if marker in merged_lower:
            return f"[{label}]"
    return None


def _stage_label_from_hint(stage_hint: str | None) -> str | None:
    if not stage_hint or not stage_hint.startswith("["):
        return None
    end = stage_hint.find("]")
    if end <= 1:
        return None
    return stage_hint[1:end]


def _read_process_cpu_percent(pid: int) -> float | None:
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "%cpu="],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            return None
        value = proc.stdout.strip()
        if not value:
            return None
        return float(value)
    except Exception:
        return None


def _find_flag_value(cmd: list[str], flag: str) -> str | None:
    for idx, token in enumerate(cmd):
        if token == flag and idx + 1 < len(cmd):
            return cmd[idx + 1]
    return None


def _extract_video_id_from_command(cmd: list[str]) -> str | None:
    for token in cmd:
        if token.startswith("http://") or token.startswith("https://"):
            parsed = urlparse(token)
            if parsed.netloc in {"www.youtube.com", "youtube.com", "m.youtube.com"}:
                values = parse_qs(parsed.query).get("v")
                if values and values[0]:
                    return values[0]
    return None


def _collect_process_tree_commands(root_pid: int) -> list[str]:
    try:
        proc = subprocess.run(
            ["ps", "-ax", "-o", "pid=,ppid=,command="],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            return []
        children_by_parent: dict[int, list[int]] = {}
        command_by_pid: dict[int, str] = {}
        for raw in proc.stdout.splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = line.split(maxsplit=2)
            if len(parts) < 3:
                continue
            pid_s, ppid_s, command = parts
            try:
                pid = int(pid_s)
                ppid = int(ppid_s)
            except ValueError:
                continue
            command_by_pid[pid] = command
            children_by_parent.setdefault(ppid, []).append(pid)

        commands: list[str] = []
        stack = [root_pid]
        seen: set[int] = set()
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            seen.add(pid)
            command_value = command_by_pid.get(pid)
            if command_value:
                commands.append(command_value)
            stack.extend(children_by_parent.get(pid, []))
        return commands
    except Exception:
        return []


def _infer_compute_substage(
    *,
    cmd: list[str],
    proc_pid: int,
    stage_hint: str | None,
    report_path: Path,
) -> str | None:
    commands = [c.lower() for c in _collect_process_tree_commands(proc_pid)]
    joined = "\n".join(commands)
    if any(key in joined for key in ["audio-separator", "demucs", "vocals", "stems"]):
        return "separation"
    if any(
        key in joined
        for key in ["whisperx", "whisper", "faster-whisper", "ctranslate2"]
    ):
        return "whisper"

    cache_dir_value = _find_flag_value(cmd, "--work-dir")
    video_id = _extract_video_id_from_command(cmd)
    if cache_dir_value and video_id:
        cache_dir = Path(cache_dir_value)
        video_cache = cache_dir / video_id
        whisper_files = list(video_cache.glob("*_whisper_*.json"))
        has_whisper_output = bool(whisper_files)
        stem_files = [
            p
            for p in video_cache.glob("*.wav")
            if any(
                key in p.name.lower()
                for key in ["vocals", "instrumental", "bass", "drums", "other"]
            )
        ]
        has_stems = bool(stem_files)
        if not has_stems and not report_path.exists():
            return "separation"
        if has_whisper_output and not report_path.exists():
            return "alignment"
        if has_stems and not has_whisper_output and not report_path.exists():
            return "whisper"

    hint_lower = (stage_hint or "").lower()
    if "lyrics_fetch" in hint_lower or "whisper_alignment" in hint_lower:
        return "alignment"
    return None


def _phase_from_stage_label(stage_label: str | None) -> str | None:
    if stage_label is None:
        return None
    mapping = {
        "identify_track": "identify",
        "media_cached_audio": "media_prepare",
        "media_download_audio": "media_prepare",
        "media_cached_video": "media_prepare",
        "media_download_video": "media_prepare",
        "separation": "separation",
        "lyrics_fetch": "lyrics_fetch",
        "whisper": "whisper",
        "whisper_alignment": "alignment",
        "alignment": "alignment",
        "timing_scale": "timing_finalize",
        "timing_report": "timing_finalize",
        "backgrounds": "render",
        "render": "render",
        "render_skip": "render_skip",
        "complete": "complete",
    }
    return mapping.get(stage_label, stage_label)


def _collect_cache_state(cmd: list[str], report_path: Path) -> dict[str, Any]:
    state: dict[str, Any] = {
        "audio_files": 0,
        "stem_files": 0,
        "whisper_files": 0,
        "report_exists": report_path.exists(),
    }
    cache_dir_value = _find_flag_value(cmd, "--work-dir")
    video_id = _extract_video_id_from_command(cmd)
    if not cache_dir_value or not video_id:
        return state

    cache_dir = Path(cache_dir_value)
    video_cache = cache_dir / video_id
    if not video_cache.exists():
        return state

    wav_files = list(video_cache.glob("*.wav"))
    stem_keys = ["vocals", "instrumental", "bass", "drums", "other"]
    stem_files = [p for p in wav_files if any(k in p.name.lower() for k in stem_keys)]
    audio_files = [p for p in wav_files if p not in stem_files]
    whisper_files = list(video_cache.glob("*_whisper_*.json"))

    state["audio_files"] = len(audio_files)
    state["stem_files"] = len(stem_files)
    state["whisper_files"] = len(whisper_files)
    return state


def _infer_cache_decisions(
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    combined_output: str,
    report_exists: bool,
) -> dict[str, str]:
    out_lower = combined_output.lower()
    decisions: dict[str, str] = {}

    if "using cached audio" in out_lower:
        decisions["audio"] = "hit (logged cached audio)"
    elif "downloading audio" in out_lower:
        decisions["audio"] = "miss (downloaded)"
    elif before.get("audio_files", 0) > 0:
        decisions["audio"] = "likely_hit (audio files already present)"
    else:
        decisions["audio"] = "unknown"

    if after.get("stem_files", 0) > before.get("stem_files", 0):
        decisions["separation"] = "miss (generated stems)"
    elif "using cached vocal separation" in out_lower:
        decisions["separation"] = "hit (logged cached vocal separation)"
    elif before.get("stem_files", 0) > 0:
        decisions["separation"] = "likely_hit (stems already present)"
    else:
        decisions["separation"] = "unknown"

    if after.get("whisper_files", 0) > before.get("whisper_files", 0):
        decisions["whisper"] = "miss (generated whisper output)"
    elif "loaded cached whisper transcription" in out_lower:
        decisions["whisper"] = "hit (logged cached whisper transcription)"
    elif before.get("whisper_files", 0) > 0:
        decisions["whisper"] = "likely_hit (whisper output already present)"
    else:
        decisions["whisper"] = "unknown"

    if report_exists:
        decisions["alignment"] = "computed (timing report written)"
    else:
        decisions["alignment"] = "unknown"
    return decisions


def _compose_heartbeat_stage_text(
    *,
    stage_hint: str | None,
    last_stage_hint: str | None,
    cpu_percent: float | None,
    compute_substage: str | None = None,
) -> str | None:
    base_hint = stage_hint or last_stage_hint
    if cpu_percent is None:
        return base_hint

    if base_hint:
        label = _stage_label_from_hint(base_hint)
        if (
            label in {"media_cached_audio", "media_download_audio", "identify_track"}
            and cpu_percent >= 120.0
        ):
            if compute_substage:
                return f"[{compute_substage}] cpu={cpu_percent:.1f}%"
            return f"[compute_active] {base_hint} (cpu={cpu_percent:.1f}%)"
        return f"{base_hint} (cpu={cpu_percent:.1f}%)"

    if cpu_percent >= 120.0:
        if compute_substage:
            return f"[{compute_substage}] cpu={cpu_percent:.1f}%"
        return f"[compute_active] cpu={cpu_percent:.1f}% (likely separation/whisper/alignment)"
    if cpu_percent >= 20.0:
        return f"[active] cpu={cpu_percent:.1f}%"
    return None


def _execute_song_process(
    *,
    cmd: list[str],
    env: dict[str, str],
    start: float,
    report_path: Path,
    timeout_sec: int,
    heartbeat_sec: int,
) -> dict[str, Any]:  # noqa: C901
    phase_started_at: dict[str, float] = {}
    phase_durations: dict[str, float] = {}
    current_phase: str | None = None

    def _begin_or_advance_phase(next_phase: str | None, elapsed_running: float) -> None:
        nonlocal current_phase
        if next_phase is None:
            return
        if current_phase is None:
            current_phase = next_phase
            phase_started_at[next_phase] = elapsed_running
            print(f"    >>> phase_start {next_phase} at {elapsed_running:.1f}s")
            return
        if next_phase == current_phase:
            return
        start_time = phase_started_at.get(current_phase, elapsed_running)
        phase_durations[current_phase] = phase_durations.get(current_phase, 0.0) + max(
            elapsed_running - start_time, 0.0
        )
        current_phase = next_phase
        phase_started_at[next_phase] = elapsed_running
        print(f"    >>> phase_start {next_phase} at {elapsed_running:.1f}s")

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out_accum = ""
    err_accum = ""
    last_stage_hint: str | None = None
    try:
        while True:
            try:
                stdout_text, stderr_text = proc.communicate(timeout=heartbeat_sec)
                out_accum += _coerce_text(stdout_text)
                err_accum += _coerce_text(stderr_text)
                break
            except subprocess.TimeoutExpired as exc:
                out_accum += _coerce_text(exc.stdout)
                err_accum += _coerce_text(exc.stderr)
                elapsed_running = round(time.monotonic() - start, 1)
                stage_hint = _extract_stage_hint(out_accum, err_accum)
                last_stage_hint = stage_hint or last_stage_hint
                cpu_percent = _read_process_cpu_percent(proc.pid)
                compute_substage = _infer_compute_substage(
                    cmd=cmd,
                    proc_pid=proc.pid,
                    stage_hint=stage_hint,
                    report_path=report_path,
                )
                heartbeat_stage_text = _compose_heartbeat_stage_text(
                    stage_hint=stage_hint,
                    last_stage_hint=last_stage_hint,
                    cpu_percent=cpu_percent,
                    compute_substage=compute_substage,
                )
                prefer_substage = (
                    compute_substage is not None
                    and cpu_percent is not None
                    and cpu_percent >= 120.0
                )
                stage_label = (
                    compute_substage
                    if prefer_substage
                    else (
                        _stage_label_from_hint(stage_hint)
                        or _stage_label_from_hint(last_stage_hint)
                    )
                )
                _begin_or_advance_phase(
                    _phase_from_stage_label(stage_label), elapsed_running
                )
                stage_suffix = (
                    f" stage: {heartbeat_stage_text}" if heartbeat_stage_text else ""
                )
                print(
                    f"    ... running {elapsed_running}s "
                    f"(timeout {timeout_sec}s){stage_suffix}"
                )
                if elapsed_running >= timeout_sec:
                    proc.kill()
                    stdout_text, stderr_text = proc.communicate()
                    out_accum += _coerce_text(stdout_text)
                    err_accum += _coerce_text(stderr_text)
                    raise subprocess.TimeoutExpired(
                        cmd=cmd,
                        timeout=timeout_sec,
                        output=out_accum,
                        stderr=err_accum,
                    )
    except subprocess.TimeoutExpired:
        elapsed = round(time.monotonic() - start, 2)
        if current_phase is not None:
            start_time = phase_started_at.get(current_phase, elapsed)
            phase_durations[current_phase] = phase_durations.get(
                current_phase, 0.0
            ) + max(elapsed - start_time, 0.0)
        if phase_durations:
            summary = ", ".join(
                f"{k}={v:.1f}s" for k, v in sorted(phase_durations.items())
            )
            print(f"    >>> phase_summary {summary}")
        raise

    elapsed = round(time.monotonic() - start, 2)
    if current_phase is not None:
        start_time = phase_started_at.get(current_phase, elapsed)
        phase_durations[current_phase] = phase_durations.get(current_phase, 0.0) + max(
            elapsed - start_time, 0.0
        )
    if phase_durations:
        summary = ", ".join(f"{k}={v:.1f}s" for k, v in sorted(phase_durations.items()))
        print(f"    >>> phase_summary {summary}")
    return {
        "out_accum": out_accum,
        "err_accum": err_accum,
        "elapsed": elapsed,
        "return_code": int(proc.returncode or 0),
        "last_stage_hint": last_stage_hint,
        "phase_durations_sec": (
            {key: round(value, 2) for key, value in phase_durations.items()}
            if phase_durations
            else {}
        ),
    }


def _run_song_command(
    *,
    cmd: list[str],
    env: dict[str, str],
    start: float,
    song: BenchmarkSong,
    report_path: Path,
    song_log_path: Path,
    timeout_sec: int,
    heartbeat_sec: int,
    run_signature: dict[str, Any],
    gold_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "artist": song.artist,
        "title": song.title,
        "youtube_id": song.youtube_id,
        "report_path": str(report_path),
        "command": cmd,
        "song_log_path": str(song_log_path),
        "run_signature": run_signature,
    }
    before_cache_state = _collect_cache_state(cmd, report_path)
    combined_output_for_cache = ""
    try:
        execution = _execute_song_process(
            cmd=cmd,
            env=env,
            start=start,
            report_path=report_path,
            timeout_sec=timeout_sec,
            heartbeat_sec=heartbeat_sec,
        )
        out_accum = str(execution["out_accum"])
        err_accum = str(execution["err_accum"])
        combined_output_for_cache = out_accum + "\n" + err_accum
        record["elapsed_sec"] = float(execution["elapsed"])
        phase_durations = execution.get("phase_durations_sec", {})
        if phase_durations:
            record["phase_durations_sec"] = phase_durations

        song_log_path.write_text(
            (
                f"$ {' '.join(cmd)}\n\n"
                + "=== STDOUT ===\n"
                + out_accum
                + "\n=== STDERR ===\n"
                + err_accum
            ),
            encoding="utf-8",
        )

        final_stage_hint = _extract_stage_hint(out_accum, err_accum) or execution.get(
            "last_stage_hint"
        )
        if final_stage_hint:
            record["last_stage_hint"] = str(final_stage_hint)

        record["return_code"] = int(execution["return_code"])
        if record["return_code"] != 0:
            record["status"] = "failed"
            record["error"] = f"command exited {record['return_code']}"
            record["stdout_tail"] = "\n".join(out_accum.splitlines()[-30:])
            record["stderr_tail"] = "\n".join(err_accum.splitlines()[-30:])
        elif not report_path.exists():
            record["status"] = "failed"
            record["error"] = "timing report was not produced"
        else:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            record["metrics"] = _extract_song_metrics(report, gold_doc=gold_doc)
            record["reference_divergence"] = _infer_reference_divergence_suspicion(
                record["metrics"]
            )
            record["status"] = "ok"
    except subprocess.TimeoutExpired as exc:
        elapsed = round(time.monotonic() - start, 2)
        record["elapsed_sec"] = elapsed
        record["status"] = "failed"
        record["error"] = f"timeout after {timeout_sec}s"
        out = _coerce_text(exc.stdout)
        err = _coerce_text(exc.stderr)
        combined_output_for_cache = out + "\n" + err
        stage_hint = _extract_stage_hint(out, err)
        if stage_hint:
            record["last_stage_hint"] = stage_hint
        song_log_path.write_text(
            (
                f"$ {' '.join(cmd)}\n\n"
                + "=== STDOUT (partial) ===\n"
                + out
                + "\n=== STDERR (partial) ===\n"
                + err
            ),
            encoding="utf-8",
        )
        record["stdout_tail"] = _tail_text(out)
        record["stderr_tail"] = _tail_text(err)

    after_cache_state = _collect_cache_state(cmd, report_path)
    cache_decisions = _infer_cache_decisions(
        before=before_cache_state,
        after=after_cache_state,
        combined_output=combined_output_for_cache,
        report_exists=bool(report_path.exists()),
    )
    record["cache_decisions"] = cache_decisions
    print(
        "    >>> cache_decisions "
        + ", ".join(f"{k}={v}" for k, v in cache_decisions.items())
    )
    return record


def main() -> int:  # noqa: C901
    args = _parse_args()
    if not 0.0 <= args.min_dtw_song_coverage_ratio <= 1.0:
        raise ValueError("--min-dtw-song-coverage-ratio must be between 0 and 1")
    if not 0.0 <= args.min_dtw_line_coverage_ratio <= 1.0:
        raise ValueError("--min-dtw-line-coverage-ratio must be between 0 and 1")

    manifest_path = args.manifest.resolve()
    songs = _parse_manifest(manifest_path)

    if args.match:
        regex = re.compile(args.match, re.IGNORECASE)
        songs = [s for s in songs if regex.search(f"{s.artist} {s.title}") is not None]

    if args.max_songs > 0:
        songs = songs[: args.max_songs]

    if not songs:
        print("benchmark_suite: no songs selected")
        return 1

    run_dir, run_id = _resolve_run_dir(
        output_root=args.output_root,
        run_id=args.run_id,
        resume_run_dir=args.resume_run_dir,
        resume_latest=args.resume_latest,
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'src'}:{existing_pythonpath}"
        if existing_pythonpath
        else str(REPO_ROOT / "src")
    )
    run_signature = _build_run_signature(args, manifest_path)
    gold_root = args.gold_root.resolve()

    song_results: list[dict[str, Any]] = []
    suite_start = time.monotonic()
    for index, song in enumerate(songs, start=1):
        report_path = run_dir / f"{index:02d}_{song.slug}_timing_report.json"
        result_path = _song_result_path(run_dir, index, song.slug)

        prior = _load_song_result(result_path)
        if prior:
            prior_signature = prior.get("run_signature")
            signature_matches = prior_signature == run_signature
            if not signature_matches and not args.reuse_mismatched_results:
                print(f"[{index}/{len(songs)}] {song.artist} - {song.title}")
                print("  -> cached result ignored (run options changed)")
            else:
                prior_status = str(prior.get("status", ""))
                if prior_status == "ok" and not args.rerun_completed:
                    prior = _refresh_cached_metrics(
                        prior, index=index, song=song, gold_root=gold_root
                    )
                    if args.rebaseline:
                        rebased_path = _rebaseline_song_from_report(
                            index=index,
                            song=song,
                            report_path=Path(
                                str(prior.get("report_path", report_path))
                            ),
                            gold_root=gold_root,
                        )
                        prior["gold_rebaselined"] = rebased_path is not None
                        if rebased_path is not None:
                            prior["gold_path"] = str(rebased_path)
                            print(f"  -> gold rebaselined: {rebased_path}")
                    prior["result_reused"] = True
                    song_results.append(prior)
                    print(f"[{index}/{len(songs)}] {song.artist} - {song.title}")
                    print("  -> ok (cached result)")
                    _write_checkpoint(
                        run_id=run_id,
                        run_dir=run_dir,
                        manifest_path=manifest_path,
                        args=args,
                        song_results=song_results,
                        suite_elapsed=time.monotonic() - suite_start,
                    )
                    continue
                if prior_status == "failed" and not args.rerun_failed:
                    prior["result_reused"] = True
                    song_results.append(prior)
                    print(f"[{index}/{len(songs)}] {song.artist} - {song.title}")
                    print("  -> failed (cached result)")
                    _write_checkpoint(
                        run_id=run_id,
                        run_dir=run_dir,
                        manifest_path=manifest_path,
                        args=args,
                        song_results=song_results,
                        suite_elapsed=time.monotonic() - suite_start,
                    )
                    continue

        cmd = _build_generate_command(
            python_bin=args.python_bin,
            song=song,
            report_path=report_path,
            cache_dir=args.cache_dir,
            offline=args.offline,
            force=args.force,
            whisper_map_lrc_dtw=not args.no_whisper_map_lrc_dtw,
            strategy=args.strategy,
            drop_lrc_line_timings=(args.scenario == "lyrics_no_timing"),
        )
        print(f"[{index}/{len(songs)}] {song.artist} - {song.title}")
        start = time.monotonic()
        song_log_path = run_dir / f"{index:02d}_{song.slug}_generate.log"
        gold_doc = _load_gold_doc(index=index, song=song, gold_root=gold_root)
        record = _run_song_command(
            cmd=cmd,
            env=env,
            start=start,
            song=song,
            report_path=report_path,
            song_log_path=song_log_path,
            timeout_sec=args.timeout_sec,
            heartbeat_sec=args.heartbeat_sec,
            run_signature=run_signature,
            gold_doc=gold_doc,
        )
        if gold_doc is not None:
            gold_path = _gold_path_for_song(index=index, song=song, gold_root=gold_root)
            if gold_path is not None:
                record["gold_path"] = str(gold_path)
        if args.rebaseline and record.get("status") == "ok":
            rebased_path = _rebaseline_song_from_report(
                index=index,
                song=song,
                report_path=report_path,
                gold_root=gold_root,
            )
            record["gold_rebaselined"] = rebased_path is not None
            if rebased_path is not None:
                record["gold_path"] = str(rebased_path)
                print(f"  -> gold rebaselined: {rebased_path}")

        song_results.append(record)
        _write_json(result_path, record)
        _write_checkpoint(
            run_id=run_id,
            run_dir=run_dir,
            manifest_path=manifest_path,
            args=args,
            song_results=song_results,
            suite_elapsed=time.monotonic() - suite_start,
        )
        print(f"  -> {record['status']} ({record.get('elapsed_sec', 0.0)}s)")
        if record["status"] != "ok" and args.fail_fast:
            break

    aggregate = _aggregate(song_results)
    suite_elapsed = round(time.monotonic() - suite_start, 2)
    quality_warnings = _quality_coverage_warnings(
        aggregate=aggregate,
        dtw_enabled=(args.strategy == "hybrid_dtw" and not args.no_whisper_map_lrc_dtw),
        min_song_coverage_ratio=args.min_dtw_song_coverage_ratio,
        min_line_coverage_ratio=args.min_dtw_line_coverage_ratio,
        suite_wall_elapsed_sec=suite_elapsed,
    )
    cache_warnings = _cache_expectation_warnings(
        aggregate=aggregate,
        expect_cached_separation=args.expect_cached_separation,
        expect_cached_whisper=args.expect_cached_whisper,
    )
    run_warnings = quality_warnings + cache_warnings
    sum_song_elapsed = float(aggregate.get("sum_song_elapsed_sec", 0.0) or 0.0)
    report_json = {
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "repo_root": str(REPO_ROOT),
        "started_at_utc": run_id,
        "elapsed_sec": suite_elapsed,
        "suite_wall_elapsed_sec": suite_elapsed,
        "sum_song_elapsed_sec": round(sum_song_elapsed, 2),
        "scheduler_overhead_sec": round(suite_elapsed - sum_song_elapsed, 2),
        "options": {
            "offline": args.offline,
            "force": args.force,
            "strategy": args.strategy,
            "scenario": args.scenario,
            "whisper_map_lrc_dtw": not args.no_whisper_map_lrc_dtw,
            "timeout_sec": args.timeout_sec,
            "heartbeat_sec": args.heartbeat_sec,
            "match": args.match,
            "max_songs": args.max_songs,
            "min_dtw_song_coverage_ratio": args.min_dtw_song_coverage_ratio,
            "min_dtw_line_coverage_ratio": args.min_dtw_line_coverage_ratio,
            "strict_quality_coverage": args.strict_quality_coverage,
            "expect_cached_separation": args.expect_cached_separation,
            "expect_cached_whisper": args.expect_cached_whisper,
            "strict_cache_expectations": args.strict_cache_expectations,
            "rebaseline": args.rebaseline,
            "gold_root": str(args.gold_root.resolve()),
        },
        "status": "finished_with_warnings" if run_warnings else "finished",
        "quality_warnings": quality_warnings,
        "cache_warnings": cache_warnings,
        "warnings": run_warnings,
        "aggregate": aggregate,
        "songs": song_results,
    }

    json_path = run_dir / "benchmark_report.json"
    md_path = run_dir / "benchmark_report.md"
    json_path.write_text(
        json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_json(run_dir / "benchmark_progress.json", report_json)
    _write_markdown_summary(
        md_path,
        run_id=run_id,
        manifest=manifest_path,
        aggregate=aggregate,
        songs=song_results,
    )

    latest = args.output_root.resolve() / "latest.json"
    latest.write_text(str(json_path) + "\n", encoding="utf-8")

    status = "OK" if aggregate["songs_failed"] == 0 else "FAIL"
    if status == "OK" and run_warnings:
        status = "WARN"
    print(f"benchmark_suite: {status}")
    print(f"- run_dir: {run_dir}")
    print(f"- json: {json_path}")
    print(f"- markdown: {md_path}")
    print(
        "- success: "
        f"{aggregate['songs_succeeded']}/{aggregate['songs_total']} "
        f"({aggregate['success_rate'] * 100:.1f}%)"
    )
    print(
        "- mean metrics: "
        f"gold_start_abs_mean_weighted={(aggregate.get('avg_abs_word_start_delta_sec_word_weighted_mean') or 0.0):.3f}s, "
        f"gold_start_p95_mean={(aggregate.get('gold_start_p95_abs_sec_mean') or 0.0):.3f}s, "
        f"gold_end_abs_mean={(aggregate.get('gold_end_mean_abs_sec_mean') or 0.0):.3f}s, "
        f"gold_cov={(aggregate.get('gold_word_coverage_ratio_total') or 0.0):.3f}, "
        f"dtw_line={(aggregate.get('dtw_line_coverage_mean') or 0.0):.3f}, "
        f"dtw_line_weighted={(aggregate.get('dtw_line_coverage_line_weighted_mean') or 0.0):.3f}, "
        f"dtw_word={(aggregate.get('dtw_word_coverage_mean') or 0.0):.3f}, "
        f"phonetic={(aggregate.get('dtw_phonetic_similarity_coverage_mean') or 0.0):.3f}, "
        f"low_conf_ratio={(aggregate.get('low_confidence_ratio_total') or 0.0):.3f}, "
        f"agreement_cov={(aggregate.get('agreement_coverage_ratio_total') or 0.0):.3f}, "
        f"agreement_text_sim={(aggregate.get('agreement_text_similarity_mean') or 0.0):.3f}, "
        f"agreement_start_abs_mean={(aggregate.get('agreement_start_mean_abs_sec_mean') or 0.0):.3f}s, "
        f"agreement_start_p95={(aggregate.get('agreement_start_p95_abs_sec_mean') or 0.0):.3f}s, "
        f"agreement_bad_ratio={(aggregate.get('agreement_bad_ratio_total') or 0.0):.3f}, "
        f"agreement_severe_ratio={(aggregate.get('agreement_severe_ratio_total') or 0.0):.3f}"
    )
    print(
        "- gold coverage: "
        f"songs={aggregate.get('gold_metric_song_coverage_ratio', 0.0):.3f}, "
        f"words={aggregate.get('gold_comparable_word_count_total', 0)}/{aggregate.get('gold_word_count_total', 0)}"
    )
    print(
        "- dtw coverage: "
        f"songs={aggregate['dtw_metric_song_coverage_ratio']:.3f}, "
        f"lines={aggregate['dtw_metric_line_coverage_ratio']:.3f}"
    )
    print(
        "- elapsed: "
        f"suite_wall={suite_elapsed:.2f}s, "
        f"sum_song={sum_song_elapsed:.2f}s, "
        f"overhead={suite_elapsed - sum_song_elapsed:.2f}s"
    )
    cache_summary = aggregate.get("cache_summary", {})
    if isinstance(cache_summary, dict):
        sep = cache_summary.get("separation")
        if isinstance(sep, dict):
            print(
                "- separation cache: "
                f"cached_ratio={float(sep.get('cached_ratio', 0.0) or 0.0):.3f}, "
                f"miss_count={int(sep.get('miss_count', 0) or 0)}"
            )
    if run_warnings:
        print("- warnings:")
        for item in run_warnings:
            print(f"  - {item}")

    if aggregate["songs_failed"] > 0:
        return 2
    if quality_warnings and args.strict_quality_coverage:
        return 3
    if cache_warnings and args.strict_cache_expectations:
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
