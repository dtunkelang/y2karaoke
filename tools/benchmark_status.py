#!/usr/bin/env python3
"""Show benchmark suite progress and active song information."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_latest_run_dir() -> Path:
    candidates = [
        path
        for path in RESULTS_ROOT.iterdir()
        if path.is_dir() and (path / "benchmark_progress.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError("No benchmark run directories with progress found")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _ps_active_generate_entry(run_dir: Path) -> dict[str, str] | None:
    try:
        out = subprocess.check_output(
            ["ps", "-ax", "-o", "pid=,etime=,%cpu=,command="],
            text=True,
        )
    except Exception:
        return None

    run_token = str(run_dir.resolve())
    for raw in out.splitlines():
        line = raw.strip()
        if "y2karaoke.cli generate" not in line:
            continue
        if run_token not in line:
            continue
        m = re.match(r"^(\d+)\s+(\S+)\s+(\S+)\s+(.*)$", line)
        if not m:
            continue
        pid, etime, cpu, command = m.groups()
        title = ""
        artist = ""
        title_m = re.search(r"--title\s+([^\-].*?)(?:\s+--|$)", command)
        if title_m:
            title = title_m.group(1).strip().strip('"')
        artist_m = re.search(r"--artist\s+([^\-].*?)(?:\s+--|$)", command)
        if artist_m:
            artist = artist_m.group(1).strip().strip('"')
        return {
            "pid": pid,
            "etime": etime,
            "cpu": cpu,
            "title": title,
            "artist": artist,
        }
    return None


def _fmt_elapsed(seconds: float | int | None) -> str:
    if seconds is None:
        return "n/a"
    sec = int(seconds)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _print_status(run_dir: Path) -> None:
    progress_path = run_dir / "benchmark_progress.json"
    report = _load_json(progress_path)
    run_id = report.get("run_id", run_dir.name)
    status = report.get("status", "unknown")
    elapsed = report.get("elapsed_sec")
    songs = report.get("songs", [])
    aggregate = report.get("aggregate", {})

    print(f"run: {run_id}")
    print(f"path: {run_dir}")
    print(f"status: {status}  elapsed: {_fmt_elapsed(elapsed)}")
    print(
        "songs: "
        f"{aggregate.get('songs_succeeded', 0)} ok / "
        f"{aggregate.get('songs_failed', 0)} failed / "
        f"{aggregate.get('songs_total', len(songs))} recorded"
    )

    if songs:
        last = songs[-1]
        print(
            "last completed: "
            f"{last.get('artist', '?')} - {last.get('title', '?')} "
            f"({last.get('status', 'unknown')}, {_fmt_elapsed(last.get('elapsed_sec'))})"
        )
        hint = last.get("last_stage_hint")
        if hint:
            print(f"last hint: {hint}")

    active = _ps_active_generate_entry(run_dir)
    if active:
        artist = active.get("artist") or "?"
        title = active.get("title") or "?"
        print(
            "active process: "
            f"pid={active['pid']} cpu={active['cpu']}% etime={active['etime']} "
            f"song={artist} - {title}"
        )
    else:
        print("active process: none detected for this run")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Benchmark run directory (default: latest run with benchmark_progress.json)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve() if args.run_dir else _find_latest_run_dir()
    if not (run_dir / "benchmark_progress.json").exists():
        raise FileNotFoundError(f"Missing benchmark_progress.json in {run_dir}")

    _print_status(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
