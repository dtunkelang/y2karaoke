"""Runtime signal and cache-inference tests for benchmark suite runner utilities."""

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


def test_extract_stage_hint_prefers_y2karaoke_line():
    module = _load_module()
    out = "random\n"
    err = "INFO:y2karaoke:📥 Downloading audio...\nnoise\n"
    hint = module._extract_stage_hint(out, err)
    assert hint is not None
    assert hint.startswith("[media_download_audio]")
    assert "Downloading audio" in hint


def test_extract_stage_hint_filters_progress_noise():
    module = _load_module()
    out = "%|█▉| 12/62 [00:19<01:20,  1.61s/it]\n"
    err = "\n"
    hint = module._extract_stage_hint(out, err)
    assert hint is None


def test_extract_stage_hint_classifies_separation_from_keyword():
    module = _load_module()
    out = "audio separator: demucs htdemucs processing stem 1/4\n"
    err = ""
    hint = module._extract_stage_hint(out, err)
    assert hint is not None
    assert hint.startswith("[separation]")


def test_compose_heartbeat_stage_text_promotes_compute_active():
    module = _load_module()
    text = module._compose_heartbeat_stage_text(
        stage_hint="[media_cached_audio] INFO:y2karaoke.core.karaoke:📁 Using cached audio",
        last_stage_hint=None,
        cpu_percent=240.0,
        compute_substage="separation",
    )
    assert text is not None
    assert text.startswith("[separation]")
    assert "cpu=240.0%" in text
    assert "stale_log_stage" not in text


def test_compose_heartbeat_stage_text_without_hint_uses_cpu_activity():
    module = _load_module()
    text = module._compose_heartbeat_stage_text(
        stage_hint=None,
        last_stage_hint=None,
        cpu_percent=130.5,
        compute_substage=None,
    )
    assert text == "[compute_active] cpu=130.5% (likely separation/whisper/alignment)"


def test_extract_video_id_from_command():
    module = _load_module()
    cmd = [
        "python",
        "-m",
        "y2karaoke.cli",
        "generate",
        "https://www.youtube.com/watch?v=abcdefghijk",
    ]
    assert module._extract_video_id_from_command(cmd) == "abcdefghijk"


def test_phase_from_stage_label():
    module = _load_module()
    assert module._phase_from_stage_label("media_cached_audio") == "media_prepare"
    assert module._phase_from_stage_label("whisper") == "whisper"
    assert module._phase_from_stage_label("unknown_label") == "unknown_label"


def test_infer_cache_decisions():
    module = _load_module()
    decisions = module._infer_cache_decisions(
        before={"audio_files": 1, "stem_files": 0, "whisper_files": 0},
        after={"audio_files": 1, "stem_files": 2, "whisper_files": 1},
        combined_output="INFO:y2karaoke.core.karaoke:📁 Using cached audio",
        report_exists=True,
    )
    assert decisions["audio"].startswith("hit")
    assert decisions["separation"].startswith("miss")
    assert decisions["whisper"].startswith("miss")
    assert decisions["alignment"].startswith("computed")


def test_infer_cache_decisions_uses_logged_cache_hits():
    module = _load_module()
    decisions = module._infer_cache_decisions(
        before={"audio_files": 0, "stem_files": 0, "whisper_files": 0},
        after={"audio_files": 0, "stem_files": 0, "whisper_files": 0},
        combined_output=(
            "INFO:y2karaoke: Using cached audio\n"
            "INFO:y2karaoke: Using cached vocal separation\n"
            "INFO:y2karaoke: Loaded cached Whisper transcription (large)"
        ),
        report_exists=True,
    )
    assert decisions["audio"].startswith("hit")
    assert decisions["separation"].startswith("hit")
    assert decisions["whisper"].startswith("hit")


def test_infer_reference_divergence_without_gold_uses_anchor_vs_dtw_signal():
    module = _load_module()
    result = module._infer_reference_divergence_suspicion(
        {
            "gold_available": False,
            "line_count": 76,
            "dtw_line_coverage": 0.447,
            "dtw_word_coverage": 0.353,
            "agreement_coverage_ratio": 0.0789,
            "agreement_text_similarity_mean": 0.9414,
            "agreement_start_p95_abs_sec": 0.9675,
            "low_confidence_ratio": 0.0132,
        }
    )
    assert result["suspected"] is True
    assert result["confidence"] == "medium"
    assert "low_dtw_with_strong_anchor_agreement" in result["evidence"]


def test_infer_reference_divergence_without_gold_stays_false_when_signals_weak():
    module = _load_module()
    result = module._infer_reference_divergence_suspicion(
        {
            "gold_available": False,
            "line_count": 76,
            "dtw_line_coverage": 0.447,
            "dtw_word_coverage": 0.353,
            "agreement_coverage_ratio": 0.02,
            "agreement_text_similarity_mean": 0.7,
            "agreement_start_p95_abs_sec": 3.2,
            "low_confidence_ratio": 0.22,
        }
    )
    assert result["suspected"] is False
    assert result["evidence"] == ["no_gold_reference"]
