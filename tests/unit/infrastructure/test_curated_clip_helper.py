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


def test_ensure_clip_audio_transcodes_non_wav_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    source = cache_dir / "clip.webm"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"fake")
    clip_path = cache_dir / "trimmed_from_79.00s_for_12.00s.wav"

    monkeypatch.setattr(
        _MODULE, "_canonical_trimmed_clip_path", lambda _song: clip_path
    )
    monkeypatch.setattr(_MODULE, "_source_audio_candidates", lambda _song: [source])

    captured: dict[str, object] = {}

    def _run(cmd: list[str], **kwargs: object) -> None:
        captured["cmd"] = cmd
        clip_path.with_suffix(".tmp.wav").write_bytes(b"x" * 128)

    monkeypatch.setattr(_MODULE.subprocess, "run", _run)

    song = {"audio_start_sec": 79, "clip_duration_sec": 12}
    resolved = _MODULE._ensure_clip_audio(song)

    assert resolved == clip_path
    assert captured["cmd"] == [
        "ffmpeg",
        "-y",
        "-ss",
        "79",
        "-t",
        "12",
        "-i",
        str(source),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-f",
        "wav",
        str(clip_path.with_suffix(".tmp.wav")),
    ]


def test_ensure_clip_audio_copies_wav_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    source = cache_dir / "clip.wav"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"fake")
    clip_path = cache_dir / "trimmed_from_79.00s_for_12.00s.wav"

    monkeypatch.setattr(
        _MODULE, "_canonical_trimmed_clip_path", lambda _song: clip_path
    )
    monkeypatch.setattr(_MODULE, "_source_audio_candidates", lambda _song: [source])

    captured: dict[str, object] = {}

    def _run(cmd: list[str], **kwargs: object) -> None:
        captured["cmd"] = cmd
        clip_path.with_suffix(".tmp.wav").write_bytes(b"x" * 128)

    monkeypatch.setattr(_MODULE.subprocess, "run", _run)

    song = {"audio_start_sec": 79, "clip_duration_sec": 12}
    resolved = _MODULE._ensure_clip_audio(song)

    assert resolved == clip_path
    assert captured["cmd"] == [
        "ffmpeg",
        "-y",
        "-ss",
        "79",
        "-t",
        "12",
        "-i",
        str(source),
        "-c",
        "copy",
        str(clip_path.with_suffix(".tmp.wav")),
    ]
