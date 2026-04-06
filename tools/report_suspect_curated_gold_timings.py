#!/usr/bin/env python3
"""Report benchmark rows that look like stale curated gold timing candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


def _fnum(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _entry_float(
    entry: dict[str, object], key: str, default: float | None = None
) -> float:
    value = entry.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    if default is not None:
        return default
    raise ValueError(f"Expected numeric {key}, got {value!r}")


def _extract_reasons(
    *,
    dtw_line_coverage: float,
    dtw_word_coverage: float,
    timing_quality_score: float,
    nearest_onset_start_mean: float,
    verdict: str | None,
) -> list[str]:
    reasons: list[str] = []
    if dtw_line_coverage >= 0.999 and dtw_word_coverage >= 0.999:
        reasons.append("perfect_dtw_coverage")
    if timing_quality_score >= 0.8:
        reasons.append("strong_internal_timing")
    if nearest_onset_start_mean <= 0.12:
        reasons.append("gold_far_from_audio_onsets")
    if verdict == "needs_manual_review":
        reasons.append("benchmark_triage_manual_review")
    return reasons


def _passes_suspect_thresholds(
    *,
    gold_start_mean: float,
    dtw_line_coverage: float,
    dtw_word_coverage: float,
    timing_quality_score: float,
    nearest_onset_start_mean: float,
    gold_word_coverage_ratio: float,
    min_gold_start_mean_abs_sec: float,
    min_dtw_line_coverage: float,
    min_dtw_word_coverage: float,
    min_timing_quality_score: float,
    max_nearest_onset_start_mean_abs_sec: float,
    min_gold_word_coverage_ratio: float,
) -> bool:
    return (
        gold_start_mean >= min_gold_start_mean_abs_sec
        and dtw_line_coverage >= min_dtw_line_coverage
        and dtw_word_coverage >= min_dtw_word_coverage
        and timing_quality_score >= min_timing_quality_score
        and nearest_onset_start_mean <= max_nearest_onset_start_mean_abs_sec
        and gold_word_coverage_ratio >= min_gold_word_coverage_ratio
    )


def _suspect_entry_from_row(
    row: dict[str, object],
    *,
    min_gold_start_mean_abs_sec: float,
    min_dtw_line_coverage: float,
    min_dtw_word_coverage: float,
    min_timing_quality_score: float,
    max_nearest_onset_start_mean_abs_sec: float,
    min_gold_word_coverage_ratio: float,
) -> dict[str, object] | None:
    if row.get("status") != "ok":
        return None
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return None

    gold_start_mean = _fnum(metrics.get("gold_start_mean_abs_sec"))
    dtw_line_coverage = _fnum(metrics.get("dtw_line_coverage"))
    dtw_word_coverage = _fnum(metrics.get("dtw_word_coverage"))
    timing_quality_score = _fnum(metrics.get("timing_quality_score"))
    nearest_onset_start_mean = _fnum(
        metrics.get("gold_nearest_onset_start_mean_abs_sec")
    )
    gold_word_coverage_ratio = _fnum(metrics.get("gold_word_coverage_ratio"))
    gold_end_mean = _fnum(metrics.get("gold_end_mean_abs_sec"))
    gold_start_p95 = _fnum(metrics.get("gold_start_p95_abs_sec"))
    if (
        gold_start_mean is None
        or dtw_line_coverage is None
        or dtw_word_coverage is None
        or timing_quality_score is None
        or nearest_onset_start_mean is None
        or gold_word_coverage_ratio is None
    ):
        return None
    if not _passes_suspect_thresholds(
        gold_start_mean=gold_start_mean,
        dtw_line_coverage=dtw_line_coverage,
        dtw_word_coverage=dtw_word_coverage,
        timing_quality_score=timing_quality_score,
        nearest_onset_start_mean=nearest_onset_start_mean,
        gold_word_coverage_ratio=gold_word_coverage_ratio,
        min_gold_start_mean_abs_sec=min_gold_start_mean_abs_sec,
        min_dtw_line_coverage=min_dtw_line_coverage,
        min_dtw_word_coverage=min_dtw_word_coverage,
        min_timing_quality_score=min_timing_quality_score,
        max_nearest_onset_start_mean_abs_sec=max_nearest_onset_start_mean_abs_sec,
        min_gold_word_coverage_ratio=min_gold_word_coverage_ratio,
    ):
        return None

    quality_diagnosis = row.get("quality_diagnosis")
    verdict = None
    if isinstance(quality_diagnosis, dict):
        verdict_value = quality_diagnosis.get("verdict")
        verdict = verdict_value if isinstance(verdict_value, str) else None

    return {
        "artist": row.get("artist"),
        "title": row.get("title"),
        "clip_id": _extract_clip_id(row),
        "gold_path": row.get("gold_path"),
        "report_path": row.get("report_path"),
        "gold_start_mean_abs_sec": gold_start_mean,
        "gold_end_mean_abs_sec": gold_end_mean,
        "gold_start_p95_abs_sec": gold_start_p95,
        "dtw_line_coverage": dtw_line_coverage,
        "dtw_word_coverage": dtw_word_coverage,
        "timing_quality_score": timing_quality_score,
        "gold_nearest_onset_start_mean_abs_sec": nearest_onset_start_mean,
        "gold_word_coverage_ratio": gold_word_coverage_ratio,
        "quality_diagnosis_verdict": verdict,
        "reasons": _extract_reasons(
            dtw_line_coverage=dtw_line_coverage,
            dtw_word_coverage=dtw_word_coverage,
            timing_quality_score=timing_quality_score,
            nearest_onset_start_mean=nearest_onset_start_mean,
            verdict=verdict,
        ),
    }


def _sort_suspect_key(entry: dict[str, object]) -> tuple[float, float, str, str, str]:
    return (
        -_entry_float(entry, "gold_start_mean_abs_sec"),
        -_entry_float(entry, "gold_start_p95_abs_sec", 0.0),
        str(entry.get("artist") or "").lower(),
        str(entry.get("title") or "").lower(),
        str(entry.get("clip_id") or "").lower(),
    )


def _format_suspect_entry(entry: dict[str, object], index: int) -> str:
    reasons_obj = entry.get("reasons")
    reasons = (
        ",".join(str(reason) for reason in reasons_obj)
        if isinstance(reasons_obj, list)
        else "none"
    )
    clip_suffix = (
        f" [{entry['clip_id']}]" if isinstance(entry.get("clip_id"), str) else ""
    )
    return (
        f"{index}. {entry['artist']} - {entry['title']}{clip_suffix} "
        f"gold_start={_entry_float(entry, 'gold_start_mean_abs_sec'):.3f}s "
        f"gold_end={_entry_float(entry, 'gold_end_mean_abs_sec', 0.0):.3f}s "
        f"p95={_entry_float(entry, 'gold_start_p95_abs_sec', 0.0):.3f}s "
        f"dtw_line={_entry_float(entry, 'dtw_line_coverage'):.3f} "
        f"dtw_word={_entry_float(entry, 'dtw_word_coverage'):.3f} "
        f"onset={_entry_float(entry, 'gold_nearest_onset_start_mean_abs_sec'):.3f}s "
        f"verdict={entry.get('quality_diagnosis_verdict') or 'unknown'} "
        f"reasons={reasons}"
    )


def collect_suspect_curated_gold_timing_entries(
    *,
    benchmark_report_path: Path,
    min_gold_start_mean_abs_sec: float,
    min_dtw_line_coverage: float,
    min_dtw_word_coverage: float,
    min_timing_quality_score: float,
    max_nearest_onset_start_mean_abs_sec: float,
    min_gold_word_coverage_ratio: float,
) -> list[dict[str, object]]:
    report = json.loads(benchmark_report_path.read_text(encoding="utf-8"))
    rows = report.get("songs")
    if not isinstance(rows, list):
        raise ValueError(f"Invalid benchmark report structure: {benchmark_report_path}")

    suspects: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        entry = _suspect_entry_from_row(
            row,
            min_gold_start_mean_abs_sec=min_gold_start_mean_abs_sec,
            min_dtw_line_coverage=min_dtw_line_coverage,
            min_dtw_word_coverage=min_dtw_word_coverage,
            min_timing_quality_score=min_timing_quality_score,
            max_nearest_onset_start_mean_abs_sec=max_nearest_onset_start_mean_abs_sec,
            min_gold_word_coverage_ratio=min_gold_word_coverage_ratio,
        )
        if entry is not None:
            suspects.append(entry)

    suspects.sort(key=_sort_suspect_key)
    return suspects


def _extract_clip_id(row: dict[str, object]) -> str | None:
    gold_path = row.get("gold_path")
    artist = row.get("artist")
    title = row.get("title")
    if not isinstance(gold_path, str):
        return None
    if not isinstance(artist, str) or not isinstance(title, str):
        return None
    stem = Path(gold_path).stem
    if stem.endswith(".gold"):
        stem = stem[:-5]
    stem = re.sub(r"^\d+_", "", stem)
    prefix = f"{_slugify(artist)}-{_slugify(title)}-"
    if not stem.startswith(prefix):
        return None
    clip_id = stem[len(prefix) :].strip("-")
    return clip_id or None


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-")


def _print_report(entries: list[dict[str, object]]) -> None:
    if not entries:
        print("suspect_curated_gold_timings: OK")
        return
    print(f"suspect_curated_gold_timings: FAIL ({len(entries)} candidates)")
    for index, entry in enumerate(entries, start=1):
        print(_format_suspect_entry(entry, index))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_report", type=Path)
    parser.add_argument("--min-gold-start-mean-abs-sec", type=float, default=0.75)
    parser.add_argument("--min-dtw-line-coverage", type=float, default=0.9)
    parser.add_argument("--min-dtw-word-coverage", type=float, default=0.9)
    parser.add_argument("--min-timing-quality-score", type=float, default=0.75)
    parser.add_argument(
        "--max-nearest-onset-start-mean-abs-sec", type=float, default=0.12
    )
    parser.add_argument("--min-gold-word-coverage-ratio", type=float, default=0.98)
    args = parser.parse_args()

    entries = collect_suspect_curated_gold_timing_entries(
        benchmark_report_path=args.benchmark_report.resolve(),
        min_gold_start_mean_abs_sec=args.min_gold_start_mean_abs_sec,
        min_dtw_line_coverage=args.min_dtw_line_coverage,
        min_dtw_word_coverage=args.min_dtw_word_coverage,
        min_timing_quality_score=args.min_timing_quality_score,
        max_nearest_onset_start_mean_abs_sec=args.max_nearest_onset_start_mean_abs_sec,
        min_gold_word_coverage_ratio=args.min_gold_word_coverage_ratio,
    )
    _print_report(entries)
    return 1 if entries else 0


if __name__ == "__main__":
    raise SystemExit(main())
