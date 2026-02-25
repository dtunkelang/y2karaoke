#!/usr/bin/env python3
"""Run visual eval and enforce committed F1-based visual guardrails."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visual eval + guardrails wrapper")
    p.add_argument(
        "--guardrails-json",
        type=Path,
        default=Path("benchmarks/visual_eval_guardrails.json"),
        help="Committed guardrail threshold config JSON",
    )
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip running run_visual_eval.py and only enforce guardrails",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for child commands (default: current interpreter)",
    )
    return p.parse_args()


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.run(cmd).returncode


def _load_guardrails(path: Path) -> tuple[Path, dict[str, Any]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("Guardrails JSON root must be an object")
    summary_json = doc.get(
        "summary_json", "benchmarks/results/visual_eval_summary.json"
    )
    manifest = doc.get("manifest")
    manifest_only = bool(doc.get("manifest_only", False))
    thresholds = doc.get("thresholds", {})
    if not isinstance(summary_json, str):
        raise ValueError("'summary_json' must be a string")
    if manifest is not None and not isinstance(manifest, str):
        raise ValueError("'manifest' must be a string when provided")
    if not isinstance(thresholds, dict):
        raise ValueError("'thresholds' must be an object")
    opts: dict[str, Any] = {
        "summary_json": Path(summary_json),
        "thresholds": thresholds,
        "manifest": Path(manifest) if isinstance(manifest, str) else None,
        "manifest_only": manifest_only,
    }
    return Path(summary_json), opts


def main() -> int:
    args = _parse_args()
    _summary_json_path, config = _load_guardrails(args.guardrails_json)
    summary_json = config["summary_json"]
    thresholds = config["thresholds"]
    manifest = config.get("manifest")
    manifest_only = bool(config.get("manifest_only", False))

    if not args.skip_eval:
        eval_cmd = [args.python, "run_visual_eval.py"]
        if manifest is not None:
            eval_cmd.extend(["--manifest", str(manifest)])
        if manifest_only:
            eval_cmd.append("--manifest-only")
        rc = _run(eval_cmd)
        if rc != 0:
            return rc

    cmd = [
        args.python,
        "tools/bootstrap_quality_guardrails.py",
        "--visual-eval-summary-json",
        str(summary_json),
    ]
    for key, value in thresholds.items():
        if value is None:
            continue
        cmd.extend(["--" + str(key).replace("_", "-"), str(value)])
    return _run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
