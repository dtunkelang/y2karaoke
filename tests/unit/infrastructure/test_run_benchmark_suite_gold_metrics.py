"""Unit tests for benchmark suite gold-metric helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


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


def test_extract_song_metrics_reports_gold_nearest_onset_start_deltas(tmp_path):
    module = _load_module()

    class _Features:
        def __init__(self):
            self.onset_times = np.array([10.55, 12.35, 15.3], dtype=float)

    module.extract_audio_features = lambda _path: _Features()
    module._AUDIO_FEATURES_CACHE.clear()

    report = {
        "lines": [
            {
                "words": [
                    {"text": "Hello", "start": 10.0, "end": 10.4},
                    {"text": "there", "start": 10.4, "end": 11.0},
                ]
            }
        ]
    }
    gold = {
        "lines": [
            {
                "text": "Hello there",
                "start": 10.0,
                "end": 11.0,
                "words": [
                    {"text": "Hello", "start": 10.0, "end": 10.4},
                    {"text": "there", "start": 10.4, "end": 11.0},
                ],
            },
            {
                "text": "(Hey, hey, hey)",
                "start": 12.0,
                "end": 12.8,
                "words": [
                    {"text": "(Hey,", "start": 12.0, "end": 12.2},
                    {"text": "hey,", "start": 12.2, "end": 12.5},
                    {"text": "hey)", "start": 12.5, "end": 12.8},
                ],
            },
        ]
    }

    audio_path = tmp_path / "fake.wav"
    audio_path.write_bytes(b"")

    metrics = module._extract_song_metrics(
        report,
        gold_doc=gold,
        audio_path=str(audio_path),
    )

    assert metrics["gold_nearest_onset_start_mean_abs_sec"] == 0.45
    assert metrics["gold_nearest_onset_start_p95_abs_sec"] == 0.54
    assert metrics["gold_nearest_onset_start_non_interjection_mean_abs_sec"] == 0.55


def test_extract_song_metrics_reports_later_onset_choice_opportunities(tmp_path):
    module = _load_module()

    class _Features:
        def __init__(self):
            self.onset_times = np.array([10.2, 10.95, 11.8], dtype=float)

    module.extract_audio_features = lambda _path: _Features()
    module._AUDIO_FEATURES_CACHE.clear()

    report = {
        "lines": [
            {
                "text": "No I can't sleep",
                "start": 10.0,
                "end": 11.0,
                "words": [
                    {"text": "No", "start": 10.0, "end": 10.2},
                    {"text": "I", "start": 10.2, "end": 10.5},
                    {"text": "can't", "start": 10.5, "end": 10.8},
                    {"text": "sleep", "start": 10.8, "end": 11.0},
                ],
            }
        ]
    }
    gold = {
        "lines": [
            {
                "text": "No I can't sleep",
                "start": 10.9,
                "end": 11.8,
                "words": [
                    {"text": "No", "start": 10.9, "end": 11.1},
                    {"text": "I", "start": 11.1, "end": 11.3},
                    {"text": "can't", "start": 11.3, "end": 11.5},
                    {"text": "sleep", "start": 11.5, "end": 11.8},
                ],
            }
        ]
    }
    audio_path = tmp_path / "fake.wav"
    audio_path.write_bytes(b"")

    metrics = module._extract_song_metrics(
        report,
        gold_doc=gold,
        audio_path=str(audio_path),
    )

    assert metrics["gold_later_onset_choice_line_count"] == 1
    assert metrics["gold_later_onset_choice_mean_improvement_sec"] == 0.85


def test_resolve_song_audio_path_falls_back_to_slug_cache_match(tmp_path):
    module = _load_module()
    module.REPO_ROOT = tmp_path
    cache_dir = tmp_path / ".cache" / "altid"
    cache_dir.mkdir(parents=True)
    primary = cache_dir / "The Weeknd - Blinding Lights (Official Audio).wav"
    stem = (
        cache_dir
        / "The Weeknd - Blinding Lights (Official Audio)_(Vocals)_htdemucs_ft.wav"
    )
    primary.write_bytes(b"")
    stem.write_bytes(b"")

    song = module.BenchmarkSong(
        manifest_index=1,
        artist="The Weeknd",
        title="Blinding Lights",
        youtube_id="manifestid",
        youtube_url="https://example.com",
    )

    resolved = module._resolve_song_audio_path(song, gold_doc={})

    assert resolved == str(primary.resolve())


def test_resolve_song_audio_path_uses_home_karaoke_cache(monkeypatch, tmp_path):
    module = _load_module()
    module.REPO_ROOT = tmp_path
    fake_home = tmp_path / "home"
    monkeypatch.setattr(module.Path, "home", lambda: fake_home)
    cache_dir = fake_home / ".cache" / "karaoke" / "wnJ6LuUFpMo"
    cache_dir.mkdir(parents=True)
    primary = cache_dir / "J Balvin, Willy William - Mi Gente (Official Video).wav"
    stem = (
        cache_dir
        / "J Balvin, Willy William - Mi Gente (Official Video)_(Vocals)_htdemucs_ft.wav"
    )
    primary.write_bytes(b"")
    stem.write_bytes(b"")

    song = module.BenchmarkSong(
        manifest_index=10,
        artist="J Balvin",
        title="Mi Gente",
        youtube_id="wnJ6LuUFpMo",
        youtube_url="https://www.youtube.com/watch?v=wnJ6LuUFpMo",
    )

    resolved = module._resolve_song_audio_path(song, gold_doc={})

    assert resolved == str(primary.resolve())


def test_extract_song_metrics_treats_parenthetical_gold_words_as_optional():
    module = _load_module()
    report = {
        "lines": [
            {
                "words": [
                    {"text": "Come", "start": 10.0, "end": 10.2},
                    {"text": "on", "start": 10.2, "end": 10.4},
                ]
            }
        ]
    }
    gold = {
        "lines": [
            {
                "words": [
                    {"text": "Come", "start": 10.0, "end": 10.2},
                    {"text": "on", "start": 10.2, "end": 10.4},
                    {"text": "(I'm", "start": 10.4, "end": 10.6},
                    {"text": "in", "start": 10.6, "end": 10.8},
                    {"text": "love", "start": 10.8, "end": 11.0},
                    {"text": "with", "start": 11.0, "end": 11.2},
                    {"text": "your", "start": 11.2, "end": 11.4},
                    {"text": "body)", "start": 11.4, "end": 11.6},
                ]
            }
        ]
    }
    metrics = module._extract_song_metrics(report, gold_doc=gold)
    assert metrics["gold_word_count"] == 2
    assert metrics["gold_optional_word_count"] == 6
    assert metrics["gold_comparable_word_count"] == 2
    assert metrics["gold_word_coverage_ratio"] == 1.0
    assert metrics["gold_trailing_parenthetical_softened_word_count"] == 1


def test_extract_song_metrics_softens_end_deltas_before_trailing_parenthetical_tail():
    module = _load_module()
    report = {
        "lines": [
            {
                "words": [
                    {"text": "Will", "start": 10.0, "end": 10.3},
                    {"text": "never", "start": 10.3, "end": 10.6},
                    {"text": "let", "start": 10.6, "end": 10.8},
                    {"text": "you", "start": 10.8, "end": 11.0},
                    {"text": "go", "start": 11.0, "end": 11.2},
                ]
            }
        ]
    }
    gold = {
        "lines": [
            {
                "words": [
                    {"text": "Will", "start": 10.0, "end": 10.3},
                    {"text": "never", "start": 10.3, "end": 10.6},
                    {"text": "let", "start": 10.6, "end": 10.8},
                    {"text": "you", "start": 10.8, "end": 11.0},
                    {"text": "go", "start": 11.0, "end": 11.7},
                    {"text": "(ooh)", "start": 11.7, "end": 12.1},
                ]
            }
        ]
    }

    metrics = module._extract_song_metrics(report, gold_doc=gold)

    assert metrics["gold_word_count"] == 5
    assert metrics["gold_optional_word_count"] == 1
    assert metrics["gold_trailing_parenthetical_softened_word_count"] == 1
    assert metrics["gold_end_mean_abs_sec"] == 0.0
    assert metrics["gold_end_mean_abs_sec_strict"] == 0.1


def test_extract_song_metrics_reports_parenthetical_interjection_line_deltas():
    module = _load_module()
    report = {
        "lines": [
            {
                "text": "(Hey, hey, hey)",
                "start": 169.91,
                "end": 170.82,
                "words": [
                    {"text": "(Hey,", "start": 169.91, "end": 170.19},
                    {"text": "hey,", "start": 170.22, "end": 170.49},
                    {"text": "hey)", "start": 170.52, "end": 170.82},
                ],
            }
        ]
    }
    gold = {
        "lines": [
            {
                "text": "(Hey, hey, hey)",
                "start": 171.94,
                "end": 173.10,
                "words": [
                    {"text": "(Hey,", "start": 171.94, "end": 172.33},
                    {"text": "hey,", "start": 172.34, "end": 172.72},
                    {"text": "hey)", "start": 172.72, "end": 173.10},
                ],
            }
        ]
    }

    metrics = module._extract_song_metrics(report, gold_doc=gold)

    assert metrics["gold_parenthetical_interjection_line_count"] == 1
    assert metrics["gold_parenthetical_interjection_comparable_line_count"] == 1
    assert metrics["gold_parenthetical_interjection_start_mean_abs_sec"] == 2.03
    assert metrics["gold_parenthetical_interjection_start_p95_abs_sec"] == 2.03
