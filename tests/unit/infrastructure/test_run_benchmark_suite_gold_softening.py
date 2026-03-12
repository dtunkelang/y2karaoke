"""Gold-softening tests for benchmark suite metrics."""

from __future__ import annotations

import importlib.util
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


def test_softened_gold_adlib_line_indexes_detects_low_confidence_tags() -> None:
    module = _load_module()
    report = {
        "low_confidence_lines": [{"index": 2, "text": "Chris Jedi"}],
        "lines": [
            {
                "text": "Baby, no me llame'",
                "whisper_window_word_count": 5,
                "whisper_window_avg_prob": 0.8,
                "whisper_window_words": [{"text": "Baby"}],
            },
            {
                "text": "Chris Jedi",
                "whisper_window_word_count": 3,
                "whisper_window_avg_prob": 0.2,
                "whisper_window_words": [
                    {"text": "Que"},
                    {"text": "Jedi"},
                    {"text": "Gaby"},
                ],
            },
            {
                "text": "Uh, uh, uh, uh",
                "whisper_window_word_count": 8,
                "whisper_window_avg_prob": 0.95,
                "whisper_window_words": [
                    {"text": "ja"},
                    {"text": "ja"},
                    {"text": "ja"},
                ],
            },
        ],
    }

    assert module._softened_gold_adlib_line_indexes(report) == {1, 2}


def test_extract_song_metrics_softens_low_confidence_adlib_tag_gold_lines() -> None:
    module = _load_module()
    report = {
        "low_confidence_lines": [{"index": 2, "text": "Chris Jedi"}],
        "lines": [
            {
                "text": "Que yo estoy ocupá' olvidando tus male'",
                "words": [
                    {"text": "Que", "start": 10.5, "end": 10.8},
                    {"text": "yo", "start": 10.8, "end": 11.0},
                ],
                "start": 10.5,
                "end": 11.0,
                "whisper_window_word_count": 5,
                "whisper_window_avg_prob": 0.9,
                "whisper_window_words": [{"text": "Que"}, {"text": "yo"}],
            },
            {
                "text": "Chris Jedi",
                "words": [
                    {"text": "Chris", "start": 20.0, "end": 20.3},
                    {"text": "Jedi", "start": 20.3, "end": 20.7},
                ],
                "start": 20.0,
                "end": 20.7,
                "whisper_window_word_count": 3,
                "whisper_window_avg_prob": 0.2,
                "whisper_window_words": [
                    {"text": "Que"},
                    {"text": "Jedi"},
                    {"text": "Gaby"},
                ],
            },
        ],
    }
    gold = {
        "lines": [
            {
                "words": [
                    {"text": "Que", "start": 10.2, "end": 10.6},
                    {"text": "yo", "start": 10.6, "end": 10.9},
                ]
            },
            {
                "words": [
                    {"text": "Chris", "start": 18.0, "end": 18.2},
                    {"text": "Jedi", "start": 18.2, "end": 18.5},
                ]
            },
        ]
    }

    metrics = module._extract_song_metrics(report, gold_doc=gold)

    assert metrics["gold_softened_adlib_line_count"] == 1
    assert metrics["gold_softened_adlib_word_count"] == 2
    assert metrics["gold_word_count"] == 2
    assert metrics["gold_comparable_word_count"] == 2
    assert metrics["gold_start_mean_abs_sec"] == 0.25
    assert metrics["gold_start_max_abs_sec"] == 0.3
