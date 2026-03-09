import importlib.util
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "run_benchmark_suite.py"
    spec = importlib.util.spec_from_file_location("run_benchmark_suite", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aggregate_includes_lyrics_source_routing_summary():
    module = _load_module()
    results = [
        {
            "artist": "A",
            "title": "Song 1",
            "status": "ok",
            "metrics": {"line_count": 1},
            "alignment_diagnostics": {
                "alignment_method": "lrc_only",
                "lyrics_source_provider": "lyriq",
                "issue_tag_counts": {"lyrics_source_disagreement": 1},
                "lyrics_source_selection_mode": "audio_scored_disagreement",
                "lyrics_source_routing_skip_reason": "none",
                "lyrics_source_disagreement_flagged": True,
                "lyrics_source_audio_scoring_used": True,
                "fallback_map_attempted": False,
                "fallback_map_selected": False,
                "fallback_map_rejected": False,
                "fallback_map_decision_reason": "unknown",
                "fallback_map_score_gain": 0.0,
            },
        },
        {
            "artist": "B",
            "title": "Song 2",
            "status": "ok",
            "metrics": {"line_count": 1},
            "alignment_diagnostics": {
                "alignment_method": "lrc_only",
                "lyrics_source_provider": "lyriq",
                "issue_tag_counts": {},
                "lyrics_source_selection_mode": "default",
                "lyrics_source_routing_skip_reason": "offline",
                "lyrics_source_disagreement_flagged": False,
                "lyrics_source_audio_scoring_used": False,
                "fallback_map_attempted": False,
                "fallback_map_selected": False,
                "fallback_map_rejected": False,
                "fallback_map_decision_reason": "unknown",
                "fallback_map_score_gain": 0.0,
            },
        },
    ]
    agg = module._aggregate(results)
    summary = agg["alignment_diagnostics_summary"]
    assert (
        summary["lyrics_source_selection_mode_counts"]["audio_scored_disagreement"] == 1
    )
    assert summary["lyrics_source_selection_mode_counts"]["default"] == 1
    assert summary["lyrics_source_routing_skip_reason_counts"]["none"] == 1
    assert summary["lyrics_source_routing_skip_reason_counts"]["offline"] == 1
    assert summary["lyrics_source_disagreement_song_count"] == 1
    assert summary["lyrics_source_audio_scoring_song_count"] == 1
