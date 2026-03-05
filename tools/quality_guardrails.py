#!/usr/bin/env python3
"""Repository quality guardrails."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TESTS_ROOT = REPO_ROOT / "tests"

MAX_SRC_FILE_LINES = 1400
MAX_TEST_FILE_LINES = 900
DISALLOWED_SRC_PATTERNS = [
    re.compile(r"monkeypatch", re.IGNORECASE),
]
STRICT_COMPLEXITY_MAX = 10


def _iter_python_files(root: Path):
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def _check_file_size(root: Path, max_lines: int, label: str) -> list[str]:
    violations: list[str] = []
    for path in _iter_python_files(root):
        line_count = sum(1 for _ in path.open("r", encoding="utf-8"))
        if line_count > max_lines:
            rel_path = path.relative_to(REPO_ROOT)
            violations.append(
                f"{label} file too large: {rel_path} has {line_count} lines (max {max_lines})"
            )
    return violations


def _check_src_patterns(root: Path) -> list[str]:
    violations: list[str] = []
    for path in _iter_python_files(root):
        text = path.read_text(encoding="utf-8")
        for pattern in DISALLOWED_SRC_PATTERNS:
            if pattern.search(text):
                rel_path = path.relative_to(REPO_ROOT)
                violations.append(
                    f"Disallowed pattern '{pattern.pattern}' found in {rel_path}"
                )
    return violations


def _check_strict_complexity() -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "flake8",
        str(SRC_ROOT),
        str(TESTS_ROOT),
        "--select",
        "C901",
        "--max-complexity",
        str(STRICT_COMPLEXITY_MAX),
        "--max-line-length",
        "127",
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return [f"Strict complexity guardrail could not run flake8: {exc}"]
    if proc.returncode == 0:
        return []

    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if not lines:
        stderr = (proc.stderr or "").strip()
        detail = stderr if stderr else "unknown flake8 failure"
        return [f"Strict complexity check failed unexpectedly: {detail}"]
    violations = [f"Strict complexity violation: {line}" for line in lines]
    violations.append(
        f"Strict complexity budget exceeded (max-complexity={STRICT_COMPLEXITY_MAX})"
    )
    return violations


def main() -> int:
    violations: list[str] = []
    violations.extend(_check_file_size(SRC_ROOT, MAX_SRC_FILE_LINES, "Source"))
    violations.extend(_check_file_size(TESTS_ROOT, MAX_TEST_FILE_LINES, "Test"))
    violations.extend(_check_src_patterns(SRC_ROOT))
    violations.extend(_check_strict_complexity())

    if not violations:
        print("quality_guardrails: OK")
        return 0

    print("quality_guardrails: FAIL")
    for violation in violations:
        print(f"- {violation}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
