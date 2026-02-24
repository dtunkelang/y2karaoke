#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal CI envs
    yaml = None  # type: ignore[assignment]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run visual extraction eval across benchmark set"
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/benchmark_songs.yaml"),
    )
    p.add_argument(
        "--gold-dir",
        type=Path,
        default=Path("benchmarks/gold_set_karaoke_seed"),
        help="Directory containing *.visual.gold.json benchmark files",
    )
    p.add_argument(
        "--lrc-dir",
        type=Path,
        default=Path("benchmarks/reference_lrc"),
        help="Optional directory containing per-song local *.lrc references",
    )
    p.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("benchmarks/results/visual_eval"),
    )
    p.add_argument(
        "--summary-md",
        type=Path,
        default=Path("visual_metrics_summary.md"),
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=Path("benchmarks/results/visual_eval_summary.json"),
    )
    return p.parse_args()


def _f1_or_none(song_report: dict[str, Any], key: str) -> float | None:
    block = song_report.get(key)
    if isinstance(block, dict):
        val = block.get("f1")
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _song_slug(artist: str, title: str) -> str:
    return _slugify(f"{artist}-{title}")


def _parse_gold_filename(path: Path) -> tuple[int | None, str]:
    m = re.match(r"^(?:(\d+)_)?(.+)\.visual\.gold\.json$", path.name)
    if not m:
        return None, path.stem
    idx_raw = m.group(1)
    idx = int(idx_raw) if idx_raw is not None else None
    return idx, m.group(2)


def _list_visual_gold_files(gold_dir: Path) -> list[Path]:
    files = sorted(gold_dir.glob("*.visual.gold.json"))

    def _sort_key(path: Path) -> tuple[int, int, str]:
        idx, slug = _parse_gold_filename(path)
        return (0 if idx is not None else 1, idx if idx is not None else 9999, slug)

    return sorted(files, key=_sort_key)


def _load_gold_artist_title(gold_file: Path) -> tuple[str | None, str | None]:
    try:
        payload = json.loads(gold_file.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    if not isinstance(payload, dict):
        return None, None
    artist = payload.get("artist")
    title = payload.get("title")
    if isinstance(artist, str) and isinstance(title, str):
        return artist, title
    return None, None


def _resolve_lrc_reference(
    lrc_dir: Path,
    *,
    seed_idx: int | None,
    artist: str,
    title: str,
) -> Path | None:
    slug = _song_slug(artist, title)
    if seed_idx is not None:
        exact = lrc_dir / f"{seed_idx:02d}_{slug}.lrc"
        if exact.exists():
            return exact
        idx_candidates = sorted(lrc_dir.glob(f"{seed_idx:02d}_*.lrc"))
        for cand in idx_candidates:
            if cand.stem.endswith(slug):
                return cand
    slug_matches = sorted(lrc_dir.glob(f"*_{slug}.lrc"))
    if slug_matches:
        return slug_matches[0]
    return None


def _default_lrc_snapshot_path(
    lrc_dir: Path,
    *,
    seed_idx: int | None,
    artist: str,
    title: str,
) -> Path:
    slug = _song_slug(artist, title)
    if seed_idx is not None:
        indexed = lrc_dir / f"{seed_idx:02d}_{slug}.lrc"
        if not indexed.exists():
            # Avoid colliding with an existing different song that already uses this index.
            same_index = [p for p in lrc_dir.glob(f"{seed_idx:02d}_*.lrc")]
            if not same_index:
                return indexed
    return lrc_dir / f"{slug}.lrc"


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def _floor3(value: float) -> float:
    return math.floor(float(value) * 1000.0) / 1000.0


def _parse_yaml_scalar(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    return text


def _load_manifest_songs(path: Path) -> list[dict[str, str]]:
    if yaml is not None:
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw_songs = manifest.get("songs", []) if isinstance(manifest, dict) else []
        return [
            {"artist": str(song["artist"]), "title": str(song["title"])}
            for song in raw_songs
            if isinstance(song, dict) and "artist" in song and "title" in song
        ]

    # Minimal fallback parser for benchmarks/benchmark_songs.yaml in CI environments
    # that do not install PyYAML. We only need ordered artist/title pairs.
    songs: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    in_songs = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "songs:":
            in_songs = True
            continue
        if not in_songs:
            continue
        if stripped.startswith("- "):
            if current and "artist" in current and "title" in current:
                songs.append(current)
            current = {}
            stripped = stripped[2:].strip()
            if stripped.startswith("artist:"):
                current["artist"] = _parse_yaml_scalar(stripped.split(":", 1)[1])
            continue
        if current is None or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        if key in {"artist", "title"}:
            current[key] = _parse_yaml_scalar(value)
    if current and "artist" in current and "title" in current:
        songs.append(current)
    return songs


def main() -> int:
    args = _parse_args()
    manifest_songs = _load_manifest_songs(args.manifest)
    manifest_by_slug: dict[str, dict[str, Any]] = {}
    for idx, song in enumerate(manifest_songs, 1):
        artist = str(song["artist"])
        title = str(song["title"])
        manifest_by_slug[_song_slug(artist, title)] = {
            "index": idx,
            "artist": artist,
            "title": title,
        }
    gold_files = _list_visual_gold_files(args.gold_dir)

    results_md: list[str] = []
    summary_rows_md: list[str] = []
    aggregate_rows: list[dict[str, Any]] = []

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    if args.summary_json.parent:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.lrc_dir.mkdir(parents=True, exist_ok=True)

    seen_manifest_slugs: set[str] = set()

    for gold_file in gold_files:
        seed_idx, seed_slug = _parse_gold_filename(gold_file)
        artist, title = _load_gold_artist_title(gold_file)
        if not artist or not title:
            # Best-effort fallback to filename slug if metadata is absent.
            artist = seed_slug.split("-", 1)[0].replace("-", " ").title()
            title = (
                seed_slug.split("-", 1)[1].replace("-", " ").title()
                if "-" in seed_slug
                else seed_slug
            )
        row_slug = _song_slug(artist, title)
        manifest_match = manifest_by_slug.get(row_slug)
        if manifest_match:
            seen_manifest_slugs.add(row_slug)
        display_idx = (
            seed_idx if seed_idx is not None else (manifest_match or {}).get("index")
        )
        display_idx = int(display_idx) if isinstance(display_idx, int) else 0
        song_label = f"{artist} - {title}"

        report_stem = f"{display_idx:02d}" if display_idx else seed_slug
        report_path = args.reports_dir / f"{report_stem}.json"
        cmd = [
            sys.executable,
            "tools/evaluate_visual_lyrics_quality.py",
            "--gold-json",
            str(gold_file),
            "--title",
            title,
            "--artist",
            artist,
            "--output-json",
            str(report_path),
        ]
        lrc_ref = _resolve_lrc_reference(
            args.lrc_dir, seed_idx=seed_idx, artist=artist, title=title
        )
        if lrc_ref is not None:
            cmd.extend(["--lrc-file", str(lrc_ref)])
        else:
            cmd.extend(
                [
                    "--write-lrc-file",
                    str(
                        _default_lrc_snapshot_path(
                            args.lrc_dir,
                            seed_idx=seed_idx,
                            artist=artist,
                            title=title,
                        )
                    ),
                ]
            )

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            err = (res.stderr or res.stdout).strip()
            results_md.append(f"## {song_label}\nERROR: {err}\n")
            summary_rows_md.append(
                f"| {display_idx:02d} | {song_label} | ERROR | ERROR |"
            )
            aggregate_rows.append(
                {
                    "index": display_idx,
                    "seed_index": seed_idx,
                    "song_key": row_slug,
                    "artist": artist,
                    "title": title,
                    "status": "error",
                    "error": err,
                    "gold_json": str(gold_file),
                    "manifest_index": (
                        int(manifest_match["index"]) if manifest_match else None
                    ),
                    "manifest_match": bool(manifest_match),
                }
            )
            continue

        payload = json.loads(report_path.read_text(encoding="utf-8"))
        strict = payload["strict"]
        repeat = payload["repeat_capped"]
        strict_f1 = float(strict["f1"])
        repeat_f1 = float(repeat["f1"])

        strict_line = (
            f"strict: precision={strict['precision']:.4f} "
            f"recall={strict['recall']:.4f} "
            f"f1={strict_f1:.4f} "
            f"matched={strict['matched_token_count']}/"
            f"{strict['reference_token_count']} "
            f"ext={strict['extracted_token_count']}"
        )
        repeat_line = (
            f"repeat_capped: precision={repeat['precision']:.4f} "
            f"recall={repeat['recall']:.4f} "
            f"f1={repeat_f1:.4f} "
            f"matched={repeat['matched_token_count']}/"
            f"{repeat['reference_token_count']} "
            f"ext={repeat['extracted_token_count']}"
        )
        results_md.append(f"## {song_label}\n{strict_line}\n{repeat_line}\n")
        summary_rows_md.append(
            f"| {display_idx:02d} | {song_label} | {strict_f1:.4f} | {repeat_f1:.4f} |"
        )
        aggregate_rows.append(
            {
                "index": display_idx,
                "seed_index": seed_idx,
                "song_key": row_slug,
                "artist": artist,
                "title": title,
                "status": "ok",
                "gold_json": str(gold_file),
                "report_json": str(report_path),
                "reference_source": payload.get("reference_source", {}),
                "strict": strict,
                "repeat_capped": repeat,
                "manifest_index": (
                    int(manifest_match["index"]) if manifest_match else None
                ),
                "manifest_match": bool(manifest_match),
            }
        )

    manifest_missing = []
    for slug, meta in manifest_by_slug.items():
        if slug not in seen_manifest_slugs:
            manifest_missing.append(meta)
            song_label = f"{meta['artist']} - {meta['title']}"
            results_md.append(
                f"## {song_label}\nERROR: No visual gold file found for manifest song.\n"
            )
            summary_rows_md.append(
                f"| {int(meta['index']):02d} | {song_label} | NOT FOUND | NOT FOUND |"
            )
    gold_only_count = sum(1 for row in aggregate_rows if not row.get("manifest_match"))

    strict_values = [
        v
        for row in aggregate_rows
        if row.get("status") == "ok"
        for v in [_f1_or_none(row, "strict")]
        if v is not None
    ]
    repeat_values = [
        v
        for row in aggregate_rows
        if row.get("status") == "ok"
        for v in [_f1_or_none(row, "repeat_capped")]
        if v is not None
    ]
    aggregate_summary = {
        "song_count": len(gold_files),
        "gold_file_count": len(gold_files),
        "manifest_song_count": len(manifest_songs),
        "evaluated_count": sum(1 for r in aggregate_rows if r.get("status") == "ok"),
        "error_count": sum(1 for r in aggregate_rows if r.get("status") == "error"),
        "missing_count": len(manifest_missing),
        "manifest_missing_count": len(manifest_missing),
        "gold_only_count": gold_only_count,
        "strict_f1_min": min(strict_values) if strict_values else None,
        "strict_f1_p10": _pct(strict_values, 0.10),
        "strict_f1_p20": _pct(strict_values, 0.20),
        "strict_f1_mean": statistics.fmean(strict_values) if strict_values else None,
        "strict_f1_median": statistics.median(strict_values) if strict_values else None,
        "repeat_capped_f1_min": min(repeat_values) if repeat_values else None,
        "repeat_capped_f1_p10": _pct(repeat_values, 0.10),
        "repeat_capped_f1_p20": _pct(repeat_values, 0.20),
        "repeat_capped_f1_mean": (
            statistics.fmean(repeat_values) if repeat_values else None
        ),
        "repeat_capped_f1_median": (
            statistics.median(repeat_values) if repeat_values else None
        ),
    }
    recommendations = None
    if strict_values and repeat_values:
        # Conservative suite gates (aggregate) plus very permissive per-song floors.
        recommendations = {
            "guardrails": {
                "per_song": {
                    "min_visual_eval_strict_f1": _floor3(min(strict_values)),
                    "min_visual_eval_repeat_capped_f1": _floor3(min(repeat_values)),
                },
                "aggregate": {
                    "min_visual_eval_strict_f1_mean": _floor3(
                        float(aggregate_summary["strict_f1_mean"]) * 0.95
                    ),
                    "min_visual_eval_repeat_capped_f1_mean": _floor3(
                        float(aggregate_summary["repeat_capped_f1_mean"]) * 0.95
                    ),
                    "min_visual_eval_strict_f1_median": _floor3(
                        float(aggregate_summary["strict_f1_median"]) * 0.95
                    ),
                    "min_visual_eval_repeat_capped_f1_median": _floor3(
                        float(aggregate_summary["repeat_capped_f1_median"]) * 0.95
                    ),
                },
            }
        }

    args.summary_md.write_text(
        "# Visual Extraction Quality Summary\n\n"
        "| Index | Song | Strict F1 | Repeat-Capped F1 |\n"
        "|---|---|---|---|\n"
        + "\n".join(summary_rows_md)
        + "\n\n"
        + "\n".join(results_md),
        encoding="utf-8",
    )
    args.summary_json.write_text(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "gold_dir": str(args.gold_dir),
                "lrc_dir": str(args.lrc_dir),
                "reports_dir": str(args.reports_dir),
                "summary": aggregate_summary,
                "recommendations": recommendations,
                "songs": aggregate_rows,
                "manifest_missing": manifest_missing,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Results written to {args.summary_md}")
    print(f"Aggregate JSON written to {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
