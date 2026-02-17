import json

import pytest

from y2karaoke.core.visual.bootstrap_runtime import (
    build_run_report_payload,
    ensure_selected_suitability,
    is_suitability_good_enough,
    write_run_report,
)


def test_is_suitability_good_enough_thresholds():
    assert is_suitability_good_enough(
        {"detectability_score": 0.6, "word_level_score": 0.2}, 0.45, 0.15
    )
    assert not is_suitability_good_enough(
        {"detectability_score": 0.4, "word_level_score": 0.2}, 0.45, 0.15
    )


def test_ensure_selected_suitability_raises_when_low(tmp_path):
    with pytest.raises(ValueError):
        ensure_selected_suitability(
            {"detectability_score": 0.1, "word_level_score": 0.1},
            v_path=tmp_path / "v.mp4",
            song_dir=tmp_path,
            suitability_fps=1.0,
            min_detectability=0.45,
            min_word_level_score=0.15,
            allow_low_suitability=False,
            analyze_fn=lambda *a, **k: ({}, (0, 0, 1, 1)),
        )


def test_build_and_write_run_report_payload(tmp_path):
    payload = build_run_report_payload(
        artist="Artist",
        title="Title",
        output_path=tmp_path / "out.json",
        candidate_url="https://youtube.com/watch?v=abc",
        selected_metrics={"detectability_score": 0.8, "word_level_score": 0.4},
        ranked_candidates=[{"url": "u", "metrics": {"detectability_score": 0.8}}],
        visual_fps=2.0,
        suitability_fps=1.0,
        min_detectability=0.45,
        min_word_level_score=0.15,
        raw_ocr_cache_version="2",
        allow_low_suitability=False,
    )
    report_path = tmp_path / "report.json"
    write_run_report(report_path, payload)
    parsed = json.loads(report_path.read_text())
    assert parsed["artist"] == "Artist"
    assert parsed["settings"]["visual_fps"] == 2.0
    assert parsed["candidate_rankings"][0]["url"] == "u"
