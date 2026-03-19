"""Path and aggregation tests for benchmark suite metrics helpers."""

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


def test_gold_path_for_song_prefers_indexed_filename(tmp_path):
    module = _load_module()
    song = module.BenchmarkSong(
        manifest_index=2,
        artist="Billie Eilish",
        title="bad guy",
        youtube_id="ayxYgDgBD3g",
        youtube_url="https://www.youtube.com/watch?v=ayxYgDgBD3g",
    )
    indexed = tmp_path / f"02_{song.slug}.gold.json"
    fallback = tmp_path / f"{song.slug}.gold.json"
    fallback.write_text("{}", encoding="utf-8")
    indexed.write_text("{}", encoding="utf-8")
    found = module._gold_path_for_song(index=2, song=song, gold_root=tmp_path)
    assert found == indexed


def test_gold_path_for_song_uses_slug_match_when_index_prefix_changed(tmp_path):
    module = _load_module()
    song = module.BenchmarkSong(
        manifest_index=10,
        artist="J Balvin",
        title="Mi Gente",
        youtube_id="wnJ6LuUFpMo",
        youtube_url="https://www.youtube.com/watch?v=wnJ6LuUFpMo",
    )
    legacy_indexed = tmp_path / f"07_{song.slug}.gold.json"
    legacy_indexed.write_text("{}", encoding="utf-8")

    found = module._gold_path_for_song(index=3, song=song, gold_root=tmp_path)
    assert found == legacy_indexed


def test_gold_path_for_song_uses_clip_gold_root_for_clip_entries(tmp_path):
    module = _load_module()
    clip_root = tmp_path / "clip_gold_candidate"
    clip_root.mkdir(parents=True)
    module.DEFAULT_CLIP_GOLD_ROOT = clip_root
    song = module.BenchmarkSong(
        manifest_index=8,
        artist="The Weeknd",
        title="Blinding Lights",
        youtube_id="fHI8X4OXluQ",
        youtube_url="https://www.youtube.com/watch?v=fHI8X4OXluQ",
        clip_id="hook-repeat",
        audio_start_sec=112.0,
    )
    clip_gold = clip_root / "06_the-weeknd-blinding-lights-hook-repeat.gold.json"
    clip_gold.write_text("{}", encoding="utf-8")

    found = module._gold_path_for_song(index=1, song=song, gold_root=tmp_path)

    assert found == clip_gold


def test_shift_report_to_clip_window_shifts_whisper_window_words():
    module = _load_module()
    song = module.BenchmarkSong(
        manifest_index=1,
        artist="Bruno Mars",
        title="Uptown Funk",
        youtube_id="OPf0YbXqDm0",
        youtube_url="https://www.youtube.com/watch?v=OPf0YbXqDm0",
        clip_id="first-chorus",
        audio_start_sec=50.1,
    )
    report = {
        "lines": [
            {
                "text": "Girls hit your hallelujah (whoo)",
                "start": 52.27,
                "end": 54.48,
                "whisper_window_start": 51.27,
                "whisper_window_end": 54.53,
                "whisper_window_words": [
                    {"text": "girls", "start": 51.75, "end": 52.1},
                    {"text": "hit", "start": 52.73, "end": 52.95},
                ],
            }
        ]
    }
    gold_doc = {
        "lines": [
            {
                "text": "Girls hit your hallelujah (whoo)",
                "start": 0.2,
                "end": 1.8,
            }
        ]
    }

    shifted = module._shift_report_to_clip_window(report, song=song, gold_doc=gold_doc)

    assert shifted is not None
    line = shifted["lines"][0]
    assert line["whisper_window_start"] == 1.17
    assert line["whisper_window_end"] == 2.8
    assert line["whisper_window_words"][0]["start"] == 1.65
    assert line["whisper_window_words"][1]["start"] == 2.63


def test_aggregate_results():
    module = _load_module()
    results = [
        {
            "artist": "A",
            "title": "T1",
            "status": "ok",
            "metrics": {
                "line_count": 10,
                "low_confidence_lines": 1,
                "dtw_line_coverage": 0.9,
                "dtw_word_coverage": 0.8,
                "dtw_phonetic_similarity_coverage": 0.85,
                "agreement_count": 8,
                "agreement_good_lines": 5,
                "agreement_warn_lines": 2,
                "agreement_bad_lines": 1,
                "agreement_severe_lines": 0,
                "agreement_coverage_ratio": 0.8,
                "agreement_text_similarity_mean": 0.92,
                "agreement_start_mean_abs_sec": 0.31,
                "agreement_start_max_abs_sec": 0.72,
                "agreement_start_p95_abs_sec": 0.69,
                "agreement_bad_ratio": 0.1,
                "agreement_severe_ratio": 0.0,
            },
        },
        {"artist": "B", "title": "T2", "status": "failed"},
    ]
    agg = module._aggregate(results)
    assert agg["songs_total"] == 2
    assert agg["songs_succeeded"] == 1
    assert agg["songs_failed"] == 1
    assert agg["line_count_total"] == 10
    assert agg["low_confidence_lines_total"] == 1
    assert agg["dtw_line_coverage_mean"] == 0.9
    assert agg["dtw_line_coverage_line_weighted_mean"] == 0.9
    assert agg["dtw_metric_song_count"] == 1
    assert agg["dtw_metric_song_coverage_ratio"] == 1.0
    assert agg["dtw_metric_line_count"] == 10
    assert agg["dtw_metric_line_coverage_ratio"] == 1.0
    assert agg["agreement_start_max_abs_sec_mean"] == 0.72
    assert agg["agreement_coverage_ratio_total"] == 0.8
    assert agg["agreement_bad_ratio_total"] == 0.1
    assert agg["sum_song_elapsed_sec"] == 0.0
    assert agg["failed_songs"] == ["B - T2"]
