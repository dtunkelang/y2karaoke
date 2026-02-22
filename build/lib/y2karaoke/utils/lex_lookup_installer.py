"""
Helpers that publish a temporary lex_lookup shim so Epitran always finds a
phonetic dictionary without requiring Flite.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
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


def ensure_local_lex_lookup() -> Optional[Path]:
    """Ensure a lex_lookup shim exists on PATH for this environment."""
    global _lex_lookup_added
    found = shutil.which(_LEX_LOOKUP_BIN_NAME)
    if _lex_lookup_added and found:
        return Path(found)

    if found:
        _lex_lookup_added = True
        return Path(found)

    cache_dir = get_cache_dir()
    shim_dir = cache_dir / "lex_lookup"
    shim_dir.mkdir(parents=True, exist_ok=True)
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
