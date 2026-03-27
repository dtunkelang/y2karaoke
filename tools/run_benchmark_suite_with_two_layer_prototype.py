#!/usr/bin/env python3
"""Run the benchmark suite, then emit a guarded two-layer prototype sidecar."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LATEST_REPORT_PATH = REPO_ROOT / "benchmarks" / "results" / ".latest"


def _read_latest_report_path() -> Path | None:
    if not LATEST_REPORT_PATH.exists():
        return None
    raw = LATEST_REPORT_PATH.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _extract_resume_run_dir(argv: list[str]) -> Path | None:
    for index, token in enumerate(argv):
        if token == "--resume-run-dir" and index + 1 < len(argv):
            return Path(argv[index + 1]).expanduser().resolve()
    return None


def _resolve_report_path(argv: list[str], previous_latest: Path | None) -> Path | None:
    resume_run_dir = _extract_resume_run_dir(argv)
    if resume_run_dir is not None:
        candidate = resume_run_dir / "benchmark_report.json"
        return candidate if candidate.exists() else None

    latest = _read_latest_report_path()
    if latest is None:
        return None
    if (
        previous_latest is not None
        and latest == previous_latest
        and not latest.exists()
    ):
        return None
    return latest if latest.exists() else None


def _load_two_layer_benchmark_prototype_tool() -> Any:
    return importlib.import_module("tools.analyze_two_layer_benchmark_prototype")


def _write_two_layer_sidecar(report_path: Path) -> dict[str, Any]:
    tool = _load_two_layer_benchmark_prototype_tool()
    report_doc = json.loads(report_path.read_text(encoding="utf-8"))
    payload = tool.analyze(report_doc)
    output_json = report_path.parent / "two_layer_benchmark_prototype.json"
    output_md = report_path.parent / "two_layer_benchmark_prototype.md"
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tool._write_markdown(output_md, payload)
    payload["output_json"] = str(output_json)
    payload["output_md"] = str(output_md)
    return payload


def main(argv: list[str] | None = None) -> int:
    forwarded_args = list(argv if argv is not None else sys.argv[1:])
    previous_latest = _read_latest_report_path()
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_benchmark_suite.py"),
        *forwarded_args,
    ]
    completed = subprocess.run(cmd, check=False)
    report_path = _resolve_report_path(forwarded_args, previous_latest)
    if report_path is None:
        return completed.returncode

    payload = _write_two_layer_sidecar(report_path)
    prototype = payload.get("prototype", {}) or {}
    baseline = payload.get("baseline", {}) or {}
    prototype_hotspots = payload.get("prototype_hotspots", []) or []
    worst_hotspot = (
        prototype_hotspots[0].get("song", "-")
        if prototype_hotspots and isinstance(prototype_hotspots[0], dict)
        else "-"
    )
    print("two-layer prototype sidecar: OK")
    print(f"  report={report_path}")
    print(f"  output_json={payload['output_json']}")
    print(f"  output_md={payload['output_md']}")
    print(
        "  coverage="
        f"{baseline.get('agreement_coverage_ratio_total', 0.0)} -> "
        f"{prototype.get('agreement_coverage_ratio_total', 0.0)}"
    )
    print(
        "  bad_ratio="
        f"{baseline.get('agreement_bad_ratio_total', 0.0)} -> "
        f"{prototype.get('agreement_bad_ratio_total', 0.0)}"
    )
    print(f"  worst_hotspot={worst_hotspot}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
