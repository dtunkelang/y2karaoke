#!/usr/bin/env python3
"""Analyze curated gold start bias relative to nearby audio onsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any
import re

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from y2karaoke.core.audio_analysis import extract_audio_features
from y2karaoke.core.components.whisper.whisper_alignment_line_helpers import (
    first_onset_after,
)


def _iter_gold_files(gold_root: Path) -> list[Path]:
    return sorted(gold_root.glob("*.gold.json"))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_audio_paths(run_dir: Path) -> dict[str, Path]:
    return {}


def _slugify(artist: str, title: str) -> str:
    safe = f"{artist}-{title}".lower()
    safe = re.sub(r"[^a-z0-9]+", "-", safe).strip("-")
    return safe


def _resolve_audio_paths_from_manifest(manifest_path: Path) -> dict[str, Path]:
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    songs = raw.get("songs", []) if isinstance(raw, dict) else []
    out: dict[str, Path] = {}
    cache_root = REPO_ROOT / ".cache"
    for song in songs:
        if not isinstance(song, dict):
            continue
        artist = str(song.get("artist", "") or "")
        title = str(song.get("title", "") or "")
        youtube_id = str(song.get("youtube_id", "") or "")
        if not artist or not title or not youtube_id:
            continue
        cache_dir = cache_root / youtube_id
        wavs = sorted(cache_dir.glob("*.wav"))
        if not wavs:
            continue
        out[_slugify(artist, title)] = wavs[0]
    return out


def _resolve_audio_paths_from_existing_gold_sets() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for root_name in ["gold_set", "gold_set_candidate", "gold_set_karaoke_seed"]:
        root = REPO_ROOT / "benchmarks" / root_name
        if not root.exists():
            continue
        for gold_path in root.rglob("*.gold.json"):
            try:
                raw = _load_json(gold_path)
            except Exception:
                continue
            artist = str(raw.get("artist", "") or "")
            title = str(raw.get("title", "") or "")
            audio_path = raw.get("audio_path")
            if (
                not artist
                or not title
                or not isinstance(audio_path, str)
                or not audio_path.strip()
            ):
                continue
            resolved = Path(audio_path).expanduser()
            if not resolved.exists():
                continue
            out.setdefault(_slugify(artist, title), resolved.resolve())
    return out


def _resolve_audio_path_by_slug(slug: str) -> Path | None:
    parts = [part for part in slug.split("-") if len(part) >= 3]
    if not parts:
        return None
    candidates: list[tuple[int, Path]] = []
    for wav in (REPO_ROOT / ".cache").glob("*/*.wav"):
        name = wav.stem.lower()
        score = sum(1 for part in parts if part in name)
        if score > 0:
            candidates.append((score, wav))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], len(item[1].name)))
    best_score, best_path = candidates[0]
    if best_score < max(2, min(len(parts), 3)):
        return None
    return best_path


def _slug_from_gold_path(path: Path) -> str:
    stem = path.name.replace(".gold.json", "")
    parts = stem.split("_", 1)
    return parts[1] if len(parts) == 2 else stem


def _line_is_parenthetical_interjection(line: dict[str, Any]) -> bool:
    words = line.get("words")
    if not isinstance(words, list) or not words:
        return False
    tokens: list[str] = []
    for word in words:
        if not isinstance(word, dict):
            return False
        text = "".join(ch for ch in str(word.get("text", "")).lower() if ch.isalpha())
        if not text:
            continue
        tokens.append(text)
    if not tokens:
        return False
    filler = {
        "ah",
        "aah",
        "eh",
        "ha",
        "hey",
        "la",
        "na",
        "oh",
        "ooh",
        "uh",
        "um",
        "woo",
        "woah",
        "whoa",
        "ya",
        "yeah",
        "yo",
    }
    return all(tok in filler for tok in tokens)


def _analyze_song(gold_path: Path, audio_path: Path, window: float) -> dict[str, Any]:
    gold = _load_json(gold_path)
    lines = gold.get("lines", [])
    if not isinstance(lines, list):
        raise ValueError(f"{gold_path} missing lines")
    features = extract_audio_features(str(audio_path))
    if (
        features is None
        or features.onset_times is None
        or len(features.onset_times) == 0
    ):
        raise RuntimeError(f"Could not extract onset features for {audio_path}")

    rows: list[dict[str, Any]] = []
    deltas_all: list[float] = []
    deltas_non_interjection: list[float] = []
    deltas_interjection: list[float] = []

    for idx, line in enumerate(lines, start=1):
        if not isinstance(line, dict):
            continue
        start = line.get("start")
        end = line.get("end")
        text = str(line.get("text", ""))
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        onset = first_onset_after(
            features.onset_times, start=float(start), window=window
        )
        if onset is None:
            continue
        delta = float(onset) - float(start)
        is_interjection = _line_is_parenthetical_interjection(line)
        rows.append(
            {
                "line_index": idx,
                "text": text,
                "gold_start": float(start),
                "gold_end": float(end),
                "next_onset": float(onset),
                "onset_minus_gold_start_sec": delta,
                "parenthetical_interjection": is_interjection,
            }
        )
        deltas_all.append(delta)
        if is_interjection:
            deltas_interjection.append(delta)
        else:
            deltas_non_interjection.append(delta)

    return {
        "gold_path": str(gold_path),
        "audio_path": str(audio_path),
        "line_count_measured": len(rows),
        "onset_minus_gold_start_mean_sec": (
            round(mean(deltas_all), 4) if deltas_all else None
        ),
        "onset_minus_gold_start_p95_sec": (
            round(sorted(deltas_all)[int(0.95 * (len(deltas_all) - 1))], 4)
            if deltas_all
            else None
        ),
        "non_interjection_onset_minus_gold_start_mean_sec": (
            round(mean(deltas_non_interjection), 4) if deltas_non_interjection else None
        ),
        "interjection_onset_minus_gold_start_mean_sec": (
            round(mean(deltas_interjection), 4) if deltas_interjection else None
        ),
        "rows": rows,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Gold Onset Bias Report")
    lines.append("")
    lines.append(f"- Gold root: `{report['gold_root']}`")
    lines.append(f"- Run dir: `{report['run_dir']}`")
    lines.append(f"- Window: `{report['window_sec']:.2f}s`")
    lines.append("")
    agg = report["aggregate"]
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- Songs analyzed: `{agg['songs_analyzed']}`")
    lines.append(
        f"- Mean onset-minus-gold-start: `{_fmt(agg['onset_minus_gold_start_mean_sec'])}`"
    )
    lines.append(
        "- Mean non-interjection onset-minus-gold-start: "
        f"`{_fmt(agg['non_interjection_onset_minus_gold_start_mean_sec'])}`"
    )
    lines.append(
        "- Mean interjection onset-minus-gold-start: "
        f"`{_fmt(agg['interjection_onset_minus_gold_start_mean_sec'])}`"
    )
    if report["skipped"]:
        lines.append(f"- Songs skipped: `{len(report['skipped'])}`")
    lines.append("")
    lines.append("## Songs")
    lines.append("")
    lines.append(
        "| Song | Measured lines | Mean onset-gold start | Non-interjection mean | Interjection mean |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for song in report["songs"]:
        slug = Path(song["gold_path"]).name.replace(".gold.json", "")
        lines.append(
            f"| {slug} | {song['line_count_measured']} | "
            f"{_fmt(song['onset_minus_gold_start_mean_sec'])} | "
            f"{_fmt(song['non_interjection_onset_minus_gold_start_mean_sec'])} | "
            f"{_fmt(song['interjection_onset_minus_gold_start_mean_sec'])} |"
        )
    if report["skipped"]:
        lines.append("")
        lines.append("## Skipped")
        lines.append("")
        for item in report["skipped"]:
            lines.append(f"- `{item['slug']}`: {item['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}s"
    return "n/a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold-root",
        type=Path,
        required=True,
        help="Directory containing curated *.gold.json files",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Benchmark results run dir containing *_result.json with audio paths",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=1.8,
        help="Search window for next onset after gold start",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "benchmark_songs.yaml",
        help="Benchmark manifest for resolving cached audio paths",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional JSON output path (default: <run-dir>/gold_onset_bias.json)",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        help="Optional markdown output path (default: <run-dir>/gold_onset_bias.md)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gold_root = args.gold_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    audio_by_slug = _resolve_audio_paths(run_dir)
    audio_by_slug.update(
        {
            k: v
            for k, v in _resolve_audio_paths_from_manifest(
                args.manifest.expanduser().resolve()
            ).items()
            if k not in audio_by_slug
        }
    )
    audio_by_slug.update(
        {
            k: v
            for k, v in _resolve_audio_paths_from_existing_gold_sets().items()
            if k not in audio_by_slug
        }
    )
    songs: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    all_means: list[float] = []
    non_inter_means: list[float] = []
    inter_means: list[float] = []

    for gold_path in _iter_gold_files(gold_root):
        slug = _slug_from_gold_path(gold_path)
        audio_path = audio_by_slug.get(slug)
        if audio_path is None:
            audio_path = _resolve_audio_path_by_slug(slug)
        if audio_path is None:
            skipped.append({"slug": slug, "reason": "audio_not_found"})
            continue
        try:
            song = _analyze_song(gold_path, audio_path, args.window_sec)
        except Exception as exc:
            skipped.append({"slug": slug, "reason": str(exc)})
            continue
        songs.append(song)
        for key, bucket in [
            ("onset_minus_gold_start_mean_sec", all_means),
            ("non_interjection_onset_minus_gold_start_mean_sec", non_inter_means),
            ("interjection_onset_minus_gold_start_mean_sec", inter_means),
        ]:
            value = song.get(key)
            if isinstance(value, (int, float)):
                bucket.append(float(value))

    report = {
        "gold_root": str(gold_root),
        "run_dir": str(run_dir),
        "window_sec": float(args.window_sec),
        "aggregate": {
            "songs_analyzed": len(songs),
            "onset_minus_gold_start_mean_sec": (
                round(mean(all_means), 4) if all_means else None
            ),
            "non_interjection_onset_minus_gold_start_mean_sec": (
                round(mean(non_inter_means), 4) if non_inter_means else None
            ),
            "interjection_onset_minus_gold_start_mean_sec": (
                round(mean(inter_means), 4) if inter_means else None
            ),
        },
        "songs": songs,
        "skipped": skipped,
    }

    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json
        else run_dir / "gold_onset_bias.json"
    )
    output_md = (
        args.output_md.expanduser().resolve()
        if args.output_md
        else run_dir / "gold_onset_bias.md"
    )
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(output_md, report)
    print(f"gold_onset_bias: OK\n  output_json={output_json}\n  output_md={output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
