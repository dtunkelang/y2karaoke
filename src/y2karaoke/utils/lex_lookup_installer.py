"""
Helpers that publish a temporary lex_lookup shim so Epitran always finds a
phonetic dictionary without requiring Flite.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from ..config import get_cache_dir

logger = logging.getLogger(__name__)

_LEX_LOOKUP_BIN_NAME = "lex_lookup"
_lex_lookup_added = False


def _build_script_content(python_executable: str) -> str:
    return (
        "#!/bin/sh\n" f'"{python_executable}" -m y2karaoke.utils.lex_lookup_stub "$@"\n'
    )


def _is_usable_lex_lookup(path: Path) -> bool:
    try:
        completed = subprocess.run(
            [str(path), "lexlookupprobe"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _resolve_shim_dir() -> Path:
    primary = Path(tempfile.gettempdir()) / "y2karaoke" / "lex_lookup"
    try:
        primary.mkdir(parents=True, exist_ok=True)
        return primary
    except OSError:
        pass
    cache_dir = get_cache_dir()
    fallback = cache_dir / "lex_lookup"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def ensure_local_lex_lookup() -> Optional[Path]:
    """Ensure a lex_lookup shim exists on PATH for this environment."""
    global _lex_lookup_added
    found = shutil.which(_LEX_LOOKUP_BIN_NAME)
    if found:
        found_path = Path(found)
        if _is_usable_lex_lookup(found_path):
            _lex_lookup_added = True
            return found_path
        logger.warning(
            "Ignoring unusable lex_lookup binary at %s; installing local shim.",
            found_path,
        )

    shim_dir = _resolve_shim_dir()
    stub_path = shim_dir / _LEX_LOOKUP_BIN_NAME
    content = _build_script_content(sys.executable)
    if not stub_path.exists() or stub_path.read_text(encoding="utf-8") != content:
        stub_path.write_text(content, encoding="utf-8")
        current_mode = stub_path.stat().st_mode
        stub_path.chmod(current_mode | 0o111)

    os.environ["PATH"] = os.pathsep.join([str(shim_dir), os.environ.get("PATH", "")])
    _lex_lookup_added = True
    logger.debug("Installed lex_lookup shim at %s", stub_path)
    return stub_path
