"""Agreement-anchor specific tests for benchmark suite metrics."""

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


def test_select_agreement_anchor_start_prefers_segment_end_for_suffix_line() -> None:
    module = _load_module()
    line = {
        "start": 73.08,
        "text": "Vient la douleur.",
        "nearest_segment_start": 71.14,
        "nearest_segment_end": 74.44,
        "nearest_segment_start_text": "Est-ce mon tour ? Vient la douleur",
    }
    assert module._select_agreement_anchor_start(line) == 74.44


def test_select_agreement_anchor_start_keeps_segment_start_for_full_line() -> None:
    module = _load_module()
    line = {
        "start": 179.26,
        "text": "Et je danse, danse, danse, danse, danse, danse, danse",
        "nearest_segment_start": 177.52,
        "nearest_segment_end": 183.78,
        "nearest_segment_start_text": "Et je danse, danse, danse, danse, danse, danse, danse",
    }
    assert module._select_agreement_anchor_start(line) == 177.52


def test_select_agreement_anchor_start_keeps_start_for_exact_text_even_if_end_closer() -> (
    None
):
    module = _load_module()
    line = {
        "start": 10.9,
        "text": "hello world",
        "nearest_segment_start": 10.0,
        "nearest_segment_end": 11.0,
        "nearest_segment_start_text": "hello world",
    }
    assert module._select_agreement_anchor_start(line) == 11.0


def test_select_agreement_anchor_start_keeps_start_for_long_exact_text_line() -> None:
    module = _load_module()
    line = {
        "start": 10.9,
        "text": "hello world this is long",
        "nearest_segment_start": 10.0,
        "nearest_segment_end": 11.0,
        "nearest_segment_start_text": "hello world this is long",
    }
    assert module._select_agreement_anchor_start(line) == 10.0


def test_select_agreement_anchor_start_prefers_second_token_for_lead_in_match() -> None:
    module = _load_module()
    line = {
        "start": 54.53,
        "text": "Je danse avec le vent la pluie",
        "nearest_segment_start": 53.62,
        "nearest_segment_end": 53.62,
        "nearest_segment_start_text": "Je danse avec le vent la pluie",
        "whisper_window_words": [
            {"text": "Je", "start": 53.62},
            {"text": "danse", "start": 54.68},
            {"text": "avec", "start": 55.22},
        ],
    }
    assert module._select_agreement_anchor_start(line) == 54.68


def test_select_agreement_anchor_start_prefers_split_token_after_lead_in() -> None:
    module = _load_module()
    line = {
        "start": 196.3,
        "text": "Et je m'envole, vole, vole",
        "nearest_segment_start": 195.58,
        "nearest_segment_end": 195.58,
        "nearest_segment_start_text": "Et je m'envole, vole, vole",
        "whisper_window_words": [
            {"text": "Et", "start": 195.58},
            {"text": "je", "start": 195.64},
            {"text": "m", "start": 195.84},
            {"text": "'envole,", "start": 196.08},
            {"text": "vole,", "start": 196.82},
        ],
    }
    assert module._select_agreement_anchor_start(line) == 195.84


def test_select_agreement_anchor_start_skips_missing_lead_in_when_window_starts_late() -> (
    None
):
    module = _load_module()
    line = {
        "start": 179.26,
        "text": "Et je danse, danse, danse",
        "nearest_segment_start": 177.52,
        "nearest_segment_end": 177.52,
        "nearest_segment_start_text": "Et je danse, danse, danse",
        "whisper_window_words": [
            {"text": "je", "start": 178.32},
            {"text": "danse,", "start": 179.1},
            {"text": "danse,", "start": 180.12},
            {"text": "danse", "start": 180.8},
        ],
    }
    assert module._select_agreement_anchor_start(line) == 179.1


def test_select_agreement_anchor_start_uses_lead_in_rescue_when_segment_end_precedes_start() -> (
    None
):
    module = _load_module()
    line = {
        "start": 161.05,
        "text": "Je suis une enfant du monde",
        "nearest_segment_start": 160.08,
        "nearest_segment_end": 158.9,
        "nearest_segment_start_text": "Je suis une enfant du monde",
        "whisper_window_words": [
            {"text": "Je", "start": 160.08},
            {"text": "suis", "start": 160.96},
            {"text": "une", "start": 161.44},
        ],
    }
    assert module._select_agreement_anchor_start(line) == 160.96


def test_select_agreement_anchor_start_prefers_window_sequence_for_truncated_segment_text() -> (
    None
):
    module = _load_module()
    line = {
        "start": 10.41,
        "text": "Que yo estoy ocupá' olvidando tus male'",
        "nearest_segment_start": 12.03,
        "nearest_segment_end": 12.03,
        "nearest_segment_start_text": "Olvidando tus males",
        "whisper_window_start": 9.41,
        "whisper_window_words": [
            {"text": "que", "start": 10.41},
            {"text": "yo", "start": 10.67},
            {"text": "estoy", "start": 10.83},
            {"text": "ocupada", "start": 11.15},
            {"text": "olvidando", "start": 12.03},
            {"text": "tus", "start": 13.13},
            {"text": "males", "start": 13.43},
        ],
    }
    assert module._select_agreement_anchor_start(line) == 10.41


def test_select_agreement_anchor_start_prefers_window_sequence_when_anchor_starts_before_window() -> (
    None
):
    module = _load_module()
    line = {
        "start": 115.37,
        "text": "Te distrae' y yo te adelanto por la derecha, uh",
        "nearest_segment_start": 113.79,
        "nearest_segment_end": 113.79,
        "nearest_segment_start_text": "Te distrae y yo te adelanto por la derecha",
        "whisper_window_start": 114.37,
        "whisper_window_words": [
            {"text": "distrae", "start": 114.73},
            {"text": "y", "start": 116.23},
            {"text": "yo", "start": 116.31},
            {"text": "te", "start": 116.43},
            {"text": "adelanto", "start": 116.45},
            {"text": "por", "start": 116.79},
            {"text": "la", "start": 116.95},
            {"text": "derecha", "start": 117.05},
        ],
    }
    assert module._select_agreement_anchor_start(line) == 114.73


def test_select_agreement_anchor_start_prefers_suffix_window_match_for_exact_text() -> (
    None
):
    module = _load_module()
    line = {
        "start": 149.3,
        "text": "Dans cette douce souffrance.",
        "nearest_segment_start": 147.54,
        "nearest_segment_end": 150.74,
        "nearest_segment_start_text": "Dans cette douce souffrance",
        "whisper_window_words": [
            {"text": "cette", "start": 149.16},
            {"text": "douce", "start": 149.46},
            {"text": "souffrance", "start": 150.08},
        ],
    }
    assert module._select_agreement_anchor_start(line) == 149.16


def test_select_agreement_anchor_start_prefers_prefix_window_match_for_prefix_line() -> (
    None
):
    module = _load_module()
    line = {
        "start": 14.37,
        "text": "Mi musica no discrimina a nadie",
        "nearest_segment_start": 13.12,
        "nearest_segment_end": 13.12,
        "nearest_segment_start_text": (
            "Mi musica no discrimina a nadie asi que vamos a romper"
        ),
        "whisper_window_words": [
            {"text": "musica", "start": 14.1},
            {"text": "no", "start": 14.6},
            {"text": "discrimina", "start": 14.94},
            {"text": "a", "start": 15.72},
            {"text": "nadie", "start": 15.9},
        ],
    }
    assert module._select_agreement_anchor_start(line) == 14.1


def test_select_agreement_anchor_start_prefers_token_sequence_match_with_split_token() -> (
    None
):
    module = _load_module()
    line = {
        "start": 192.66,
        "text": "Dans tout Paris, je m'abandonne",
        "nearest_segment_start": 191.34,
        "nearest_segment_end": 191.34,
        "nearest_segment_start_text": "Dans tout Paris, je m'abandonne",
        "whisper_window_words": [
            {"text": "tout", "start": 192.44},
            {"text": "Paris,", "start": 192.7},
            {"text": "je", "start": 193.46},
            {"text": "m", "start": 194.02},
            {"text": "'abandonne", "start": 194.28},
        ],
    }
    assert module._select_agreement_anchor_start(line) == 192.44


def test_select_agreement_anchor_start_prefers_closest_repeated_exact_window_match() -> (
    None
):
    module = _load_module()
    line = {
        "start": 2.17,
        "text": "Girls hit your hallelujah (whoo)",
        "nearest_segment_start": 4.05,
        "nearest_segment_end": 4.05,
        "nearest_segment_start_text": "Girls hit your hallelujah",
        "whisper_window_words": [
            {"text": "girls", "start": 1.65},
            {"text": "hit", "start": 2.63},
            {"text": "your", "start": 2.89},
            {"text": "hallelujah", "start": 3.05},
            {"text": "girls", "start": 4.05},
            {"text": "hit", "start": 4.77},
            {"text": "your", "start": 5.05},
            {"text": "hallelujah", "start": 5.23},
        ],
    }
    assert module._select_agreement_anchor_start(line) == 1.65


def test_select_agreement_anchor_start_keeps_base_when_no_lead_in_token() -> None:
    module = _load_module()
    line = {
        "start": 54.53,
        "text": "Danse avec le vent la pluie",
        "nearest_segment_start": 53.62,
        "nearest_segment_end": 53.62,
        "nearest_segment_start_text": "Danse avec le vent la pluie",
        "whisper_window_words": [
            {"text": "Danse", "start": 53.62},
            {"text": "danse", "start": 54.3},
        ],
    }
    assert module._select_agreement_anchor_start(line) == 53.62
