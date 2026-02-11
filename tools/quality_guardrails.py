#!/usr/bin/env python3
"""Repository quality guardrails."""

from __future__ import annotations

from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TESTS_ROOT = REPO_ROOT / "tests"

MAX_SRC_FILE_LINES = 1400
MAX_TEST_FILE_LINES = 900
DISALLOWED_SRC_PATTERNS = [
    re.compile(r"monkeypatch", re.IGNORECASE),
]


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


def main() -> int:
    violations: list[str] = []
    violations.extend(_check_file_size(SRC_ROOT, MAX_SRC_FILE_LINES, "Source"))
    violations.extend(_check_file_size(TESTS_ROOT, MAX_TEST_FILE_LINES, "Test"))
    violations.extend(_check_src_patterns(SRC_ROOT))

    if not violations:
        print("quality_guardrails: OK")
        return 0

    print("quality_guardrails: FAIL")
    for violation in violations:
        print(f"- {violation}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
