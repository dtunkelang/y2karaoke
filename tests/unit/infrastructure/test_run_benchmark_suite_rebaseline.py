"""Rebaseline-specific tests for benchmark suite helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[3] / "tools" / "run_benchmark_suite.py"
    )
    spec = importlib.util.spec_from_file_location("run_benchmark_suite", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rebaseline_song_from_report_writes_default_indexed_gold(tmp_path):
    module = _load_module()
    song = module.BenchmarkSong(
        manifest_index=2,
        artist="Billie Eilish",
        title="bad guy",
        youtube_id="ayxYgDgBD3g",
        youtube_url="https://www.youtube.com/watch?v=ayxYgDgBD3g",
    )
    report_doc = {
        "lines": [
            {"words": [{"text": "hello", "start": 1.0, "end": 1.2}]},
        ]
    }
    report_path = tmp_path / "song_report.json"
    report_path.write_text(json.dumps(report_doc), encoding="utf-8")

    out = module._rebaseline_song_from_report(
        index=2, song=song, report_path=report_path, gold_root=tmp_path
    )
    expected = tmp_path / f"02_{song.slug}.gold.json"
    assert out == expected
    assert expected.exists()
    loaded = json.loads(expected.read_text(encoding="utf-8"))
    assert loaded["lines"] == report_doc["lines"]
    assert loaded["candidate_url"] == song.youtube_url


def test_rebaseline_song_from_report_prefers_existing_gold_path(tmp_path):
    module = _load_module()
    song = module.BenchmarkSong(
        manifest_index=2,
        artist="Billie Eilish",
        title="bad guy",
        youtube_id="ayxYgDgBD3g",
        youtube_url="https://www.youtube.com/watch?v=ayxYgDgBD3g",
    )
    existing_gold = tmp_path / f"{song.slug}.gold.json"
    existing_gold.write_text("{}", encoding="utf-8")
    report_doc = {
        "lines": [
            {"words": [{"text": "world", "start": 2.0, "end": 2.4}]},
        ]
    }
    report_path = tmp_path / "song_report.json"
    report_path.write_text(json.dumps(report_doc), encoding="utf-8")

    out = module._rebaseline_song_from_report(
        index=2, song=song, report_path=report_path, gold_root=tmp_path
    )
    assert out == existing_gold
    loaded = json.loads(existing_gold.read_text(encoding="utf-8"))
    assert loaded["lines"] == report_doc["lines"]
    assert loaded["candidate_url"] == song.youtube_url


def test_rebaseline_song_from_report_preserves_prior_provenance(tmp_path):
    module = _load_module()
    song = module.BenchmarkSong(
        manifest_index=10,
        artist="J Balvin",
        title="Mi Gente",
        youtube_id="wnJ6LuUFpMo",
        youtube_url="https://www.youtube.com/watch?v=wnJ6LuUFpMo",
    )
    existing_gold = tmp_path / f"{song.manifest_index:02d}_{song.slug}.gold.json"
    existing_gold.write_text(
        json.dumps(
            {
                "candidate_url": "https://example.com/old",
                "audio_path": "/tmp/prior.wav",
                "lines": [],
            }
        ),
        encoding="utf-8",
    )
    report_doc = {
        "lines": [
            {"words": [{"text": "hello", "start": 1.0, "end": 1.2}]},
        ]
    }
    report_path = tmp_path / "song_report.json"
    report_path.write_text(json.dumps(report_doc), encoding="utf-8")

    out = module._rebaseline_song_from_report(
        index=10, song=song, report_path=report_path, gold_root=tmp_path
    )

    assert out == existing_gold
    loaded = json.loads(existing_gold.read_text(encoding="utf-8"))
    assert loaded["candidate_url"] == "https://example.com/old"
    assert loaded["audio_path"] == "/tmp/prior.wav"
    assert loaded["lines"] == report_doc["lines"]
