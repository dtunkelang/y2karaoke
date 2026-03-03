from __future__ import annotations

import os
from pathlib import Path

from y2karaoke.utils import lex_lookup_installer as installer


def test_ensure_local_lex_lookup_uses_usable_binary(monkeypatch):
    installer._lex_lookup_added = False
    monkeypatch.setattr(installer.shutil, "which", lambda _name: "/bin/echo")

    out = installer.ensure_local_lex_lookup()

    assert out == Path("/bin/echo")


def test_ensure_local_lex_lookup_installs_shim_when_found_binary_unusable(
    monkeypatch, tmp_path
):
    installer._lex_lookup_added = False
    fake_bin = tmp_path / "lex_lookup"
    fake_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_bin.chmod(0o644)

    monkeypatch.setattr(installer.shutil, "which", lambda _name: str(fake_bin))
    monkeypatch.setattr(installer, "get_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setenv("PATH", "/usr/bin")

    out = installer.ensure_local_lex_lookup()

    assert out is not None
    assert out != fake_bin
    assert out.exists()
    assert os.access(out, os.X_OK)
    assert str(out.parent) == os.environ["PATH"].split(os.pathsep)[0]
