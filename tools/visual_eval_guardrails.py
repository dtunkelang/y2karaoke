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
    summary_json = doc.get("summary_json", "benchmarks/results/visual_eval_summary.json")
    thresholds = doc.get("thresholds", {})
    if not isinstance(summary_json, str):
        raise ValueError("'summary_json' must be a string")
    if not isinstance(thresholds, dict):
        raise ValueError("'thresholds' must be an object")
    return Path(summary_json), thresholds


def main() -> int:
    args = _parse_args()
    summary_json, thresholds = _load_guardrails(args.guardrails_json)

    if not args.skip_eval:
        rc = _run([args.python, "run_visual_eval.py"])
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
