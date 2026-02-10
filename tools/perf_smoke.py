#!/usr/bin/env python3
"""Run lightweight performance smoke checks on representative unit tests."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PerfCase:
    test: str
    max_seconds: float


CASES = [
    PerfCase(
        "tests/unit/alignment/test_timing_evaluator_corrections.py::"
        "test_correct_line_timestamps_shifts_to_onset",
        3.0,
    ),
    PerfCase(
        "tests/unit/lyrics/test_sync_quality_unit.py::"
        "test_get_lyrics_quality_report_gap_penalty",
        3.0,
    ),
    PerfCase(
        "tests/unit/pipeline/test_karaoke_generate.py::"
        "test_generate_offsets_lines_and_uses_vocals_debug",
        4.0,
    ),
]


def run_case(case: PerfCase) -> tuple[bool, float, str]:
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(repo_root / "src")

    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", case.test],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started

    if result.returncode != 0:
        output = (result.stdout + "\n" + result.stderr).strip()
        return False, elapsed, f"failed to run:\n{output}"

    if elapsed > case.max_seconds:
        return (
            False,
            elapsed,
            f"exceeded budget ({elapsed:.2f}s > {case.max_seconds:.2f}s)",
        )
    return True, elapsed, ""


def main() -> int:
    failures: list[str] = []
    for case in CASES:
        ok, elapsed, reason = run_case(case)
        if ok:
            print(f"PASS {case.test} ({elapsed:.2f}s <= {case.max_seconds:.2f}s)")
            continue
        failures.append(f"{case.test}: {reason}")

    if failures:
        print("\nPerformance smoke checks failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("\nPerformance smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
