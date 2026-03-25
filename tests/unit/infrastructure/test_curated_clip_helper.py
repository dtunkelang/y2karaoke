import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[3] / "tools" / "curated_clip_helper.py"
_SPEC = importlib.util.spec_from_file_location(
    "curated_clip_helper_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_ensure_editor_server_skips_spawn_when_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[bool] = []
    monkeypatch.setattr(_MODULE, "_is_editor_healthy", lambda: True)
    monkeypatch.setattr(_MODULE, "_start_editor_server", lambda: started.append(True))

    _MODULE._ensure_editor_server()

    assert started == []


def test_ensure_editor_server_starts_and_waits_until_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checks = iter([False, False, True])
    monkeypatch.setattr(
        _MODULE,
        "_is_editor_healthy",
        lambda: next(checks),
    )
    monkeypatch.setattr(_MODULE, "_is_editor_reachable", lambda host, port: False)

    class _Proc:
        def poll(self) -> None:
            return None

    started: list[bool] = []
    monkeypatch.setattr(
        _MODULE,
        "_start_editor_server",
        lambda: started.append(True) or _Proc(),
    )
    monkeypatch.setattr(_MODULE.time, "sleep", lambda _seconds: None)

    _MODULE._ensure_editor_server()

    assert started == [True]


def test_ensure_editor_server_raises_if_process_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_MODULE, "_is_editor_healthy", lambda: False)
    monkeypatch.setattr(_MODULE, "_is_editor_reachable", lambda host, port: False)

    class _Proc:
        def poll(self) -> int:
            return 1

    monkeypatch.setattr(_MODULE, "_start_editor_server", lambda: _Proc())

    with pytest.raises(RuntimeError, match="exited before becoming ready"):
        _MODULE._ensure_editor_server()


def test_ensure_editor_server_raises_for_wrong_service_on_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_MODULE, "_is_editor_healthy", lambda: False)
    monkeypatch.setattr(_MODULE, "_is_editor_reachable", lambda host, port: True)

    with pytest.raises(RuntimeError, match="already in use"):
        _MODULE._ensure_editor_server()


def test_gold_path_bootstraps_from_default_gold_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clip_root = tmp_path / "clip_gold"
    default_root = tmp_path / "default_gold"
    monkeypatch.setattr(_MODULE, "CLIP_GOLD_ROOT", clip_root)
    monkeypatch.setattr(_MODULE, "DEFAULT_GOLD_ROOT", default_root)

    song = {
        "artist": "Sabrina Carpenter",
        "title": "Please Please Please",
        "youtube_id": "zAgVtzhjfCA",
        "clip_id": "chorus-setup-tail",
    }
    slug = _MODULE._song_slug(song)
    source = default_root / f"45_{slug}.gold.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text('{"title": "seed"}\n', encoding="utf-8")

    resolved = _MODULE._gold_path(45, song)

    assert resolved == clip_root / f"45_{slug}.gold.json"
    assert resolved.exists()
    assert resolved.read_text(encoding="utf-8") == '{"title": "seed"}\n'
