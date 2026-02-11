#!/usr/bin/env python3
"""Run the benchmark song suite and emit an aggregated timing-quality report."""

from __future__ import annotations

import argparse
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


def _build_generate_command(
    *,
    python_bin: str,
    song: BenchmarkSong,
    report_path: Path,
    cache_dir: Path | None,
    offline: bool,
    force: bool,
    whisper_map_lrc_dtw: bool,
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
    if whisper_map_lrc_dtw:
        cmd.append("--whisper-map-lrc-dtw")
    return cmd


def _extract_song_metrics(report: dict[str, Any]) -> dict[str, Any]:
    lines = report.get("lines", [])
    line_count = len(lines)
    low_conf = report.get("low_confidence_lines", [])

    start_deltas = [
        float(line["whisper_line_start_delta"])
        for line in lines
        if line.get("whisper_line_start_delta") is not None
    ]

    low_conf_ratio = (len(low_conf) / line_count) if line_count else 0.0
    return {
        "line_count": line_count,
        "low_confidence_lines": len(low_conf),
        "low_confidence_ratio": round(low_conf_ratio, 4),
        "dtw_line_coverage": report.get("dtw_line_coverage"),
        "dtw_word_coverage": report.get("dtw_word_coverage"),
        "dtw_phonetic_similarity_coverage": report.get(
            "dtw_phonetic_similarity_coverage"
        ),
        "start_delta_count": len(start_deltas),
        "start_delta_mean_sec": round(_mean(start_deltas) or 0.0, 4),
        "start_delta_mean_abs_sec": round(
            _mean(abs(v) for v in start_deltas) or 0.0, 4
        ),
        "start_delta_p95_abs_sec": round(
            _pctile([abs(v) for v in start_deltas], 0.95), 4
        ),
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
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

    total_lines = int(sum(metric_values("line_count")))
    low_conf_total = int(sum(metric_values("low_confidence_lines")))
    low_conf_ratio = (low_conf_total / total_lines) if total_lines else 0.0

    return {
        "songs_total": len(results),
        "songs_succeeded": len(succeeded),
        "songs_failed": len(failed),
        "success_rate": round((len(succeeded) / len(results)) if results else 0.0, 4),
        "line_count_total": total_lines,
        "low_confidence_lines_total": low_conf_total,
        "low_confidence_ratio_total": round(low_conf_ratio, 4),
        "dtw_line_coverage_mean": round(
            _mean(metric_values("dtw_line_coverage")) or 0.0, 4
        ),
        "dtw_word_coverage_mean": round(
            _mean(metric_values("dtw_word_coverage")) or 0.0, 4
        ),
        "dtw_phonetic_similarity_coverage_mean": round(
            _mean(metric_values("dtw_phonetic_similarity_coverage")) or 0.0, 4
        ),
        "start_delta_mean_abs_sec_mean": round(
            _mean(metric_values("start_delta_mean_abs_sec")) or 0.0, 4
        ),
        "start_delta_p95_abs_sec_mean": round(
            _mean(metric_values("start_delta_p95_abs_sec")) or 0.0, 4
        ),
        "failed_songs": [f"{r['artist']} - {r['title']}" for r in failed],
    }


def _write_markdown_summary(
    path: Path,
    *,
    run_id: str,
    manifest: Path,
    aggregate: dict[str, Any],
    songs: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# Benchmark Timing Quality Report")
    lines.append("")
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Manifest: `{manifest}`")
    lines.append(
        f"- Songs: `{aggregate['songs_succeeded']}/{aggregate['songs_total']}` succeeded"
    )
    lines.append(
        f"- Mean DTW line coverage: `{aggregate['dtw_line_coverage_mean']:.3f}`"
    )
    lines.append(
        f"- Mean DTW word coverage: `{aggregate['dtw_word_coverage_mean']:.3f}`"
    )
    lines.append(
        "- Mean DTW phonetic similarity coverage: "
        f"`{aggregate['dtw_phonetic_similarity_coverage_mean']:.3f}`"
    )
    lines.append(
        f"- Low-confidence line ratio: `{aggregate['low_confidence_ratio_total']:.3f}`"
    )
    lines.append(
        f"- Mean abs line-start delta: `{aggregate['start_delta_mean_abs_sec_mean']:.3f}s`"
    )
    lines.append(
        f"- Mean abs line-start p95 delta: `{aggregate['start_delta_p95_abs_sec_mean']:.3f}s`"
    )
    lines.append("")
    lines.append("## Per-song")
    lines.append("")
    lines.append(
        "| Song | Status | DTW line | DTW word | Phonetic cov | Low conf ratio | Start delta abs mean | Elapsed |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for song in songs:
        metrics = song.get("metrics", {})
        lines.append(
            "| "
            + f"{song['artist']} - {song['title']} | "
            + f"{song['status']} | "
            + f"{metrics.get('dtw_line_coverage', '-')}"
            + " | "
            + f"{metrics.get('dtw_word_coverage', '-')}"
            + " | "
            + f"{metrics.get('dtw_phonetic_similarity_coverage', '-')}"
            + " | "
            + f"{metrics.get('low_confidence_ratio', '-')}"
            + " | "
            + f"{metrics.get('start_delta_mean_abs_sec', '-')}"
            + " | "
            + f"{song.get('elapsed_sec', '-')}"
            + "s |"
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
            key=lambda p: p.stat().st_mtime,
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
    report_json = {
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "repo_root": str(REPO_ROOT),
        "started_at_utc": run_id,
        "elapsed_sec": round(suite_elapsed, 2),
        "status": "running",
        "options": {
            "offline": args.offline,
            "force": args.force,
            "whisper_map_lrc_dtw": not args.no_whisper_map_lrc_dtw,
            "timeout_sec": args.timeout_sec,
            "heartbeat_sec": args.heartbeat_sec,
            "match": args.match,
            "max_songs": args.max_songs,
        },
        "aggregate": aggregate,
        "songs": song_results,
    }
    _write_json(run_dir / "benchmark_progress.json", report_json)


def _build_run_signature(
    args: argparse.Namespace, manifest_path: Path
) -> dict[str, Any]:
    return {
        "manifest_path": str(manifest_path),
        "offline": bool(args.offline),
        "force": bool(args.force),
        "whisper_map_lrc_dtw": not bool(args.no_whisper_map_lrc_dtw),
        "cache_dir": str(args.cache_dir.resolve()) if args.cache_dir else None,
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
    elif before.get("stem_files", 0) > 0:
        decisions["separation"] = "likely_hit (stems already present)"
    else:
        decisions["separation"] = "unknown"

    if after.get("whisper_files", 0) > before.get("whisper_files", 0):
        decisions["whisper"] = "miss (generated whisper output)"
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
            record["metrics"] = _extract_song_metrics(report)
            record["status"] = "ok"
    except subprocess.TimeoutExpired as exc:
        elapsed = round(time.monotonic() - start, 2)
        record["elapsed_sec"] = elapsed
        record["status"] = "failed"
        record["error"] = f"timeout after {timeout_sec}s"
        out = _coerce_text(exc.stdout)
        err = _coerce_text(exc.stderr)
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
        combined_output=(
            record.get("stdout_tail", "") + "\n" + record.get("stderr_tail", "")
        ),
        report_exists=bool(report_path.exists()),
    )
    record["cache_decisions"] = cache_decisions
    print(
        "    >>> cache_decisions "
        + ", ".join(f"{k}={v}" for k, v in cache_decisions.items())
    )
    return record


def main() -> int:
    args = _parse_args()
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
        )
        print(f"[{index}/{len(songs)}] {song.artist} - {song.title}")
        start = time.monotonic()
        song_log_path = run_dir / f"{index:02d}_{song.slug}_generate.log"
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
        )

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
    report_json = {
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "repo_root": str(REPO_ROOT),
        "started_at_utc": run_id,
        "elapsed_sec": suite_elapsed,
        "options": {
            "offline": args.offline,
            "force": args.force,
            "whisper_map_lrc_dtw": not args.no_whisper_map_lrc_dtw,
            "timeout_sec": args.timeout_sec,
            "heartbeat_sec": args.heartbeat_sec,
            "match": args.match,
            "max_songs": args.max_songs,
        },
        "status": "finished",
        "aggregate": aggregate,
        "songs": song_results,
    }

    json_path = run_dir / "benchmark_report.json"
    md_path = run_dir / "benchmark_report.md"
    json_path.write_text(
        json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
        f"dtw_line={aggregate['dtw_line_coverage_mean']:.3f}, "
        f"dtw_word={aggregate['dtw_word_coverage_mean']:.3f}, "
        f"phonetic={aggregate['dtw_phonetic_similarity_coverage_mean']:.3f}, "
        f"low_conf_ratio={aggregate['low_confidence_ratio_total']:.3f}, "
        f"start_delta_abs_mean={aggregate['start_delta_mean_abs_sec_mean']:.3f}s"
    )
    return 0 if aggregate["songs_failed"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
