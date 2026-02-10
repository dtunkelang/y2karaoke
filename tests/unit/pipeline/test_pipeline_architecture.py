"""Architecture guardrails for pipeline subsystem boundaries."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "y2karaoke"


def _module_name_for(path: Path) -> str:
    rel = path.relative_to(ROOT / "src")
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].removesuffix(".py")
    return ".".join(parts)


def _resolve_from_import(
    module_name: str, node: ast.ImportFrom, is_package: bool
) -> str:
    # Absolute import
    if node.level == 0:
        return node.module or ""

    # Relative import
    base_parts = module_name.split(".")
    parent = base_parts if is_package else base_parts[:-1]
    up = max(node.level - 1, 0)
    if up:
        parent = parent[:-up]
    if node.module:
        return ".".join(parent + node.module.split("."))
    return ".".join(parent)


def _iter_import_targets(path: Path) -> Iterable[str]:
    module_name = _module_name_for(path)
    is_package = path.name == "__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_from_import(module_name, node, is_package)
            if resolved:
                yield resolved


def test_core_modules_do_not_depend_on_pipeline_except_karaoke() -> None:
    violations: list[str] = []
    for path in sorted((SRC / "core").glob("*.py")):
        if path.name in {"karaoke.py", "__init__.py"}:
            continue
        for imported in _iter_import_targets(path):
            if imported.startswith("y2karaoke.pipeline"):
                violations.append(f"{path.relative_to(ROOT)} imports {imported}")

    assert not violations, "\n".join(violations)


def test_pipeline_modules_only_depend_on_core_or_pipeline_within_project() -> None:
    violations: list[str] = []
    for path in sorted((SRC / "pipeline").rglob("*.py")):
        for imported in _iter_import_targets(path):
            if not imported.startswith("y2karaoke"):
                continue
            if imported.startswith("y2karaoke.core"):
                continue
            if imported.startswith("y2karaoke.pipeline"):
                continue
            violations.append(f"{path.relative_to(ROOT)} imports {imported}")

    assert not violations, "\n".join(violations)
