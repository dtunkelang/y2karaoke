"""Unit tests for benchmark suite runner utilities."""

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


def test_extract_song_metrics():
    module = _load_module()
    report = {
        "dtw_line_coverage": 0.8,
        "dtw_word_coverage": 0.7,
        "dtw_phonetic_similarity_coverage": 0.75,
        "low_confidence_lines": [{"index": 2}],
        "lines": [
            {
                "whisper_line_start_delta": -0.2,
                "pre_whisper_start": 9.0,
                "start": 10.0,
                "end": 12.0,
                "nearest_segment_start": 9.8,
                "text": "hello world",
                "nearest_segment_start_text": "hello world",
            },
            {
                "whisper_line_start_delta": 0.1,
                "pre_whisper_start": 19.5,
                "start": 20.0,
                "end": 21.0,
                "nearest_segment_start": 20.7,
                "text": "foo bar",
                "nearest_segment_start_text": "foo bar",
            },
            {
                "whisper_line_start_delta": None,
                "pre_whisper_start": 28.0,
                "start": 30.0,
                "end": 31.0,
                "nearest_segment_start": 33.6,
                "text": "third line",
                "nearest_segment_start_text": "mumble mumble",
            },
        ],
    }
    metrics = module._extract_song_metrics(report)
    assert metrics["line_count"] == 3
    assert metrics["low_confidence_lines"] == 1
    assert metrics["low_confidence_ratio"] == 0.3333
    assert metrics["dtw_line_coverage"] == 0.8
    assert metrics["agreement_count"] == 2
    assert metrics["agreement_eligible_lines"] == 3
    assert metrics["agreement_matched_lines"] == 2
    assert metrics["agreement_eligibility_ratio"] == 1.0
    assert metrics["agreement_match_ratio_within_eligible"] == 0.6667
    assert metrics["agreement_skip_reason_counts"]["low_text_similarity"] == 1
    assert metrics["agreement_coverage_ratio"] == 0.6667
    assert metrics["agreement_text_similarity_mean"] == 1.0
    assert metrics["agreement_start_mean_abs_sec"] == 0.45
    assert metrics["agreement_start_p95_abs_sec"] == 0.675
    assert metrics["agreement_start_max_abs_sec"] == 0.7
    assert metrics["agreement_good_lines"] == 1
    assert metrics["agreement_warn_lines"] == 1
    assert metrics["agreement_bad_lines"] == 0
    assert metrics["agreement_severe_lines"] == 0
    assert metrics["agreement_good_ratio"] == 0.3333
    assert metrics["agreement_warn_ratio"] == 0.3333
    assert metrics["agreement_bad_ratio"] == 0.0
    assert metrics["agreement_severe_ratio"] == 0.0
    assert metrics["pre_whisper_line_count"] == 0
    assert metrics["pre_whisper_start_shift_mean_abs_sec"] == 1.1667
    assert metrics["pre_whisper_late_shift_line_count"] == 3
    assert metrics["pre_whisper_late_shift_mean_sec"] == 1.1667
    assert isinstance(metrics["timing_quality_score"], float)
    assert metrics["timing_quality_band"] in {"poor", "fair", "good", "excellent"}
    assert metrics["timing_quality_score_mode"] in {"dtw_internal", "dtw_internal+gold"}


def test_extract_song_metrics_with_gold_word_deltas():
    module = _load_module()
    report = {
        "lines": [
            {
                "pre_whisper_start": 1.05,
                "words": [
                    {"text": "hello", "start": 1.0, "end": 1.5},
                    {"text": "world", "start": 2.0, "end": 2.7},
                ],
            }
        ]
    }
    gold = {
        "lines": [
            {
                "words": [
                    {"text": "hello", "start": 1.2, "end": 1.6},
                    {"text": "world", "start": 2.4, "end": 2.9},
                ]
            }
        ]
    }
    metrics = module._extract_song_metrics(report, gold_doc=gold)
    assert metrics["gold_available"] is True
    assert metrics["gold_word_count"] == 2
    assert metrics["gold_comparable_word_count"] == 2
    assert metrics["gold_word_coverage_ratio"] == 1.0
    assert metrics["avg_abs_word_start_delta_sec"] == 0.3
    assert metrics["gold_start_p95_abs_sec"] == 0.39
    assert metrics["gold_end_mean_abs_sec"] == 0.15
    assert metrics["gold_comparable_line_count"] == 1
    assert metrics["gold_line_duration_mean_abs_sec"] == 0.0
    assert metrics["gold_line_duration_p95_abs_sec"] == 0.0
    assert metrics["gold_pre_whisper_start_mean_abs_sec"] == 0.15
    assert metrics["gold_downstream_regression_line_count"] == 0
    assert metrics["gold_downstream_regression_mean_improvement_sec"] == 0.0
    assert metrics["timing_quality_score_mode"] == "anchor_fallback"
    assert (
        metrics["gold_alignment_mode"] == "monotonic_text_window_parenthetical_optional"
    )


def test_extract_song_metrics_separates_independent_and_anchor_agreement():
    module = _load_module()
    report = {
        "alignment_method": "whisper_only",
        "dtw_line_coverage": None,
        "lines": [
            {
                "start": 10.0,
                "nearest_segment_start": 9.7,
                "text": "hello world",
                "nearest_segment_start_text": "hello world",
            }
        ],
        "low_confidence_lines": [],
    }
    metrics = module._extract_song_metrics(report)
    assert metrics["agreement_measurement_mode"] == "unavailable_no_dtw_anchor"
    assert metrics["agreement_count"] == 0
    assert metrics["agreement_start_mean_abs_sec"] == 0.0
    assert metrics["whisper_anchor_count"] == 1
    assert metrics["whisper_anchor_start_mean_abs_sec"] == 0.3


def test_extract_song_metrics_skips_anchor_outside_window():
    module = _load_module()
    report = {
        "dtw_line_coverage": 1.0,
        "lines": [
            {
                "start": 174.53,
                "nearest_segment_start": 141.72,
                "text": "Well good for you I guess you moved on really easily",
                "nearest_segment_start_text": "I guess you moved on really easily",
                "whisper_window_start": 173.53,
                "whisper_window_end": 177.0,
                "whisper_window_word_count": 6,
                "words": [{"text": "Well"}, {"text": "good"}, {"text": "for"}],
            }
        ],
        "low_confidence_lines": [],
    }

    metrics = module._extract_song_metrics(report)

    assert metrics["agreement_count"] == 0
    assert metrics["agreement_eligible_lines"] == 0
    assert metrics["agreement_skip_reason_counts"]["anchor_outside_window"] == 1
    assert metrics["agreement_start_p95_abs_sec"] == 0.0


def test_extract_song_metrics_handles_long_noisy_anchor_text_without_matching() -> None:
    module = _load_module()
    report = {
        "dtw_line_coverage": 1.0,
        "lines": [
            {
                "start": 13.42,
                "nearest_segment_start": 30.64,
                "text": "I've been on my own for long enough",
                "nearest_segment_start_text": (
                    "I've been on my own for long, maybe you shouldn't have, maybe, "
                    "I know the drugs, you don't even have to do too much, you can tell me "
                    "I'm just a bitch, baby, maybe, you can tell me I'm wrong, but don't "
                    "want to bless me, I guess you clearly didn't get it, God"
                ),
                "whisper_window_start": 12.42,
                "whisper_window_end": 27.68,
                "whisper_window_word_count": 0,
                "whisper_window_avg_prob": None,
                "words": [{"text": token} for token in "i have been on my own".split()],
            }
        ],
        "low_confidence_lines": [],
    }

    metrics = module._extract_song_metrics(report)

    assert metrics["agreement_count"] == 0
    assert metrics["agreement_eligible_lines"] == 0
    assert (
        metrics["agreement_skip_reason_counts"][
            "insufficient_window_words_for_long_line"
        ]
        == 1
    )


def test_extract_song_metrics_skips_low_window_evidence_for_longer_line():
    module = _load_module()
    report = {
        "dtw_line_coverage": 1.0,
        "lines": [
            {
                "start": 100.0,
                "nearest_segment_start": 100.4,
                "text": "maybe i am too emotional",
                "nearest_segment_start_text": "maybe i am too emotional",
                "whisper_window_start": 99.0,
                "whisper_window_end": 101.0,
                "whisper_window_word_count": 1,
                "words": [
                    {"text": "maybe"},
                    {"text": "i"},
                    {"text": "am"},
                    {"text": "too"},
                    {"text": "emotional"},
                ],
            }
        ],
        "low_confidence_lines": [],
    }

    metrics = module._extract_song_metrics(report)

    assert metrics["agreement_count"] == 0
    assert metrics["agreement_eligible_lines"] == 0
    assert metrics["agreement_skip_reason_counts"]["explicit_window_too_sparse"] == 1
    assert metrics["agreement_start_mean_abs_sec"] == 0.0


def test_agreement_normalization_expands_contractions_for_overlap():
    module = _load_module()
    assert module._normalize_agreement_text("don't stop me now") == "do not stop me now"
    assert (
        module._agreement_token_overlap("don't stop me now", "do not stop me now")
        == 1.0
    )


def test_agreement_normalization_collapses_repeated_fillers():
    module = _load_module()
    norm = module._normalize_agreement_text("oh oh oh oh baby yeah yeah")
    assert norm == "oh baby yeah"
    sim = module._agreement_text_similarity("oh oh oh oh baby", "oh baby")
    assert sim >= 0.95


def test_agreement_normalization_strips_optional_hook_boundary_phrases():
    module = _load_module()
    assert module._normalize_agreement_text(
        "Don't believe me just watch (come on)"
    ) == ("do not believe me just watch come on")
    assert (
        module._normalize_agreement_text_hook_boundary(
            "Don't believe me just watch (come on)"
        )
        == "do not believe me just watch"
    )
    assert (
        module._agreement_token_overlap(
            "Girls hit your hallelujah (whoo)",
            "Girls hit your hallelujah",
            normalize_fn=module._normalize_agreement_text_hook_boundary,
        )
        == 1.0
    )


def test_agreement_normalization_expands_colloquialisms():
    module = _load_module()
    assert module._normalize_agreement_text("I'm gonna let 'em know") == (
        "i am going to let them know"
    )
    assert module._agreement_token_overlap("I wanna go", "I want to go") == 1.0


def test_agreement_normalization_converts_dropped_g_endings():
    module = _load_module()
    assert module._normalize_agreement_text("I'm lovin' it") == "i am loving it"
    assert (
        module._agreement_text_similarity("we singin' loud", "we singing loud") > 0.95
    )


def test_lexical_tokens_strip_optional_hook_boundary_fillers():
    module = _load_module()
    assert module._lexical_tokens_basic("Don't believe me just watch uh") == [
        "don't",
        "believe",
        "me",
        "just",
        "watch",
    ]
    assert module._lexical_tokens_compact("Uptown (woo) funk you up (come on)") == [
        "uptown",
        "funk",
        "you",
        "up",
    ]
    assert module._lexical_tokens_compact("Girls hit your hallelujah (whoo)") == [
        "girls",
        "hit",
        "your",
        "hallelujah",
    ]
    assert module._lexical_tokens_compact("Come on, dance, jump on it") == [
        "dance",
        "jump",
        "on",
        "it",
    ]
    assert module._lexical_tokens_compact("I'm too hot (hot damn)") == [
        "im",
        "too",
        "hot",
    ]


def test_agreement_window_skip_reason_variants() -> None:
    module = _load_module()
    line = {
        "whisper_window_word_count": 1,
        "whisper_window_avg_prob": 0.3,
    }
    assert (
        module._agreement_window_skip_reason(line, line_word_count=6)
        == "insufficient_window_words_for_long_line"
    )
    assert (
        module._agreement_window_skip_reason(line, line_word_count=5)
        == "low_window_confidence_and_sparse_words"
    )
    assert (
        module._agreement_window_skip_reason(line, line_word_count=4)
        == "explicit_window_too_sparse"
    )


def test_evaluate_agreement_line_low_token_overlap_marks_eligible() -> None:
    module = _load_module()
    line = {
        "start": 10.0,
        "nearest_segment_start": 10.1,
        "text": "we can make this right tonight",
        "nearest_segment_start_text": "radio static nonsense words here",
        "words": [{"text": token} for token in "a b c d e f".split()],
        "whisper_window_word_count": 6,
        "whisper_window_avg_prob": 0.9,
    }
    result = module._evaluate_agreement_line(
        line=line,
        min_text_similarity=0.0,
        min_token_overlap=0.55,
    )
    assert result["eligible"] is True
    assert result["skip_reason"] == "low_token_overlap"


def test_evaluate_agreement_line_skips_missing_window_line_start() -> None:
    module = _load_module()
    line = {
        "start": 50.72,
        "nearest_segment_start": 49.7,
        "text": "Sin City's cold and empty (oh)",
        "nearest_segment_start_text": "I look around and see the city's cold and empty",
        "whisper_window_word_count": 10,
        "whisper_window_avg_prob": 0.87,
        "whisper_window_words": [
            {"text": "look", "start": 50.22},
            {"text": "around", "start": 50.48},
            {"text": "and", "start": 50.78},
            {"text": "see", "start": 50.98},
            {"text": "the", "start": 51.48},
            {"text": "city's", "start": 51.78},
            {"text": "cold", "start": 52.22},
            {"text": "and", "start": 52.46},
            {"text": "empty", "start": 52.78},
        ],
        "words": [{"text": token} for token in "Sin City's cold and empty oh".split()],
    }
    result = module._evaluate_agreement_line(
        line=line,
        min_text_similarity=0.58,
        min_token_overlap=0.5,
    )
    assert result["eligible"] is True
    assert result["skip_reason"] == "missing_window_line_start"


def test_evaluate_agreement_line_keeps_exact_match_when_window_contains_start() -> None:
    module = _load_module()
    line = {
        "start": 20.0,
        "nearest_segment_start": 20.9,
        "text": "Maybe I can make this right",
        "nearest_segment_start_text": "Maybe I can make this right",
        "whisper_window_word_count": 6,
        "whisper_window_avg_prob": 0.92,
        "whisper_window_words": [
            {"text": "Maybe", "start": 20.9},
            {"text": "I", "start": 21.2},
            {"text": "can", "start": 21.4},
            {"text": "make", "start": 21.7},
        ],
        "words": [{"text": token} for token in "Maybe I can make this right".split()],
    }
    result = module._evaluate_agreement_line(
        line=line,
        min_text_similarity=0.58,
        min_token_overlap=0.5,
    )
    assert "skip_reason" not in result
    assert round(result["anchor_start_delta"], 3) == 0.9


def test_compute_timing_quality_score_anchor_with_gold_mode() -> None:
    module = _load_module()
    score, band, mode = module._compute_timing_quality_score(
        {
            "low_confidence_ratio": 0.08,
            "agreement_coverage_ratio": 0.35,
            "agreement_start_p95_abs_sec": 0.6,
            "agreement_bad_ratio": 0.03,
            "whisper_anchor_start_p95_abs_sec": 0.35,
            "gold_word_coverage_ratio": 0.92,
            "gold_start_mean_abs_sec": 0.12,
            "gold_comparable_word_count": 35,
        }
    )
    assert 0.0 <= score <= 1.0
    assert band in {"poor", "fair", "good", "excellent"}
    assert mode == "anchor_fallback+gold"


def test_extract_song_metrics_supports_env_agreement_threshold_overrides(
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setenv("Y2KARAOKE_BENCH_AGREEMENT_MIN_TEXT_SIM", "0.6")
    monkeypatch.setenv("Y2KARAOKE_BENCH_AGREEMENT_MIN_TOKEN_OVERLAP", "0.5")
    report = {
        "dtw_line_coverage": 1.0,
        "lines": [
            {
                "start": 10.0,
                "nearest_segment_start": 10.1,
                "text": "hello world",
                "nearest_segment_start_text": "hello world",
            }
        ],
        "low_confidence_lines": [],
    }
    metrics = module._extract_song_metrics(report)
    assert metrics["agreement_min_text_similarity"] == 0.6
    assert metrics["agreement_min_token_overlap"] == 0.5


def test_extract_song_metrics_contraction_lines_are_comparable():
    module = _load_module()
    report = {
        "dtw_line_coverage": 1.0,
        "lines": [
            {
                "start": 12.0,
                "nearest_segment_start": 12.2,
                "text": "don't stop me now",
                "nearest_segment_start_text": "do not stop me now",
            }
        ],
        "low_confidence_lines": [],
    }
    metrics = module._extract_song_metrics(report)
    assert metrics["agreement_eligible_lines"] == 1
    assert metrics["agreement_count"] == 1
    assert metrics["agreement_skip_reason_counts"] == {}


def test_extract_song_metrics_surfaces_local_transcribe_cache_counters() -> None:
    module = _load_module()
    report = {
        "dtw_line_coverage": 1.0,
        "dtw_metrics": {
            "local_transcribe_cache_hits": 2.0,
            "local_transcribe_cache_misses": 1.0,
        },
        "lines": [],
        "low_confidence_lines": [],
    }
    metrics = module._extract_song_metrics(report)
    assert metrics["local_transcribe_cache_hits"] == 2
    assert metrics["local_transcribe_cache_misses"] == 1


def test_extract_song_metrics_surfaces_tail_guardrail_fields() -> None:
    module = _load_module()
    report = {
        "dtw_line_coverage": 1.0,
        "dtw_metrics": {
            "tail_guardrail_flagged": 1.0,
            "tail_guardrail_fallback_attempted": 1.0,
            "tail_guardrail_fallback_applied": 0.0,
            "tail_guardrail_target_coverage_ratio": 0.81234,
            "tail_guardrail_target_shortfall_sec": 28.1234,
        },
        "lines": [],
        "low_confidence_lines": [],
    }
    metrics = module._extract_song_metrics(report)
    assert metrics["tail_guardrail_flagged"] == 1
    assert metrics["tail_guardrail_fallback_attempted"] == 1
    assert metrics["tail_guardrail_fallback_applied"] == 0
    assert metrics["tail_guardrail_target_coverage_ratio"] == 0.8123
    assert metrics["tail_guardrail_target_shortfall_sec"] == 28.1234


def test_extract_song_metrics_adaptive_rescue_accepts_good_timing_high_overlap() -> (
    None
):
    module = _load_module()
    report = {
        "dtw_line_coverage": 1.0,
        "lines": [
            {
                "start": 10.0,
                "nearest_segment_start": 10.1,
                "text": "you know i love it when the music starts",
                "nearest_segment_start_text": "when the music starts you know i love it",
                "words": [{"text": token} for token in "a b c d e f g h".split()],
                "whisper_window_word_count": 5,
                "whisper_window_avg_prob": 0.8,
            }
        ],
        "low_confidence_lines": [],
    }
    metrics = module._extract_song_metrics(report)
    assert metrics["agreement_eligible_lines"] == 1
    assert metrics["agreement_count"] == 1
    assert metrics["agreement_adaptive_rescue_count"] == 1


def test_extract_song_metrics_adaptive_rescue_accepts_medium_length_lines() -> None:
    module = _load_module()
    report = {
        "dtw_line_coverage": 1.0,
        "lines": [
            {
                "start": 15.0,
                "nearest_segment_start": 15.1,
                "text": "you know this rhythm feels right",
                "nearest_segment_start_text": "right right rhythm this you know",
                "words": [{"text": token} for token in "a b c d e f".split()],
                "whisper_window_word_count": 4,
                "whisper_window_avg_prob": 0.8,
            }
        ],
        "low_confidence_lines": [],
    }
    metrics = module._extract_song_metrics(report)
    assert metrics["agreement_eligible_lines"] == 1
    assert metrics["agreement_count"] == 1
    assert metrics["agreement_adaptive_rescue_count"] == 1


def test_extract_song_metrics_adaptive_rescue_accepts_five_word_lines() -> None:
    module = _load_module()
    report = {
        "dtw_line_coverage": 1.0,
        "lines": [
            {
                "start": 21.0,
                "nearest_segment_start": 21.1,
                "text": "you make this feel right",
                "nearest_segment_start_text": "right this make you right",
                "words": [{"text": token} for token in "a b c d e".split()],
                "whisper_window_word_count": 3,
                "whisper_window_avg_prob": 0.78,
            }
        ],
        "low_confidence_lines": [],
    }
    metrics = module._extract_song_metrics(report)
    assert metrics["agreement_eligible_lines"] == 1
    assert metrics["agreement_count"] == 1
    assert metrics["agreement_adaptive_rescue_count"] == 1


def test_extract_song_metrics_adaptive_rescue_accepts_short_high_confidence_lines() -> (
    None
):
    module = _load_module()
    report = {
        "dtw_line_coverage": 1.0,
        "lines": [
            {
                "start": 8.0,
                "nearest_segment_start": 8.08,
                "text": "feel the beat now",
                "nearest_segment_start_text": "beat now feel the",
                "words": [{"text": token} for token in "a b c d".split()],
                "whisper_window_word_count": 4,
                "whisper_window_avg_prob": 0.75,
            }
        ],
        "low_confidence_lines": [],
    }
    metrics = module._extract_song_metrics(report)
    assert metrics["agreement_eligible_lines"] == 1
    assert metrics["agreement_count"] == 1
    assert metrics["agreement_adaptive_rescue_count"] == 1


def test_extract_song_metrics_rescues_high_overlap_tight_delta_lines() -> None:
    module = _load_module()
    report = {
        "dtw_line_coverage": 1.0,
        "lines": [
            {
                "start": 30.0,
                "nearest_segment_start": 30.1,
                "text": "si el ritmo te lleva a mover la cabeza",
                "nearest_segment_start_text": "cabeza mover lleva ritmo te el si la",
                "words": [{"text": token} for token in "a b c d e f g h".split()],
                "whisper_window_word_count": 5,
                "whisper_window_avg_prob": 0.72,
            }
        ],
        "low_confidence_lines": [],
    }
    metrics = module._extract_song_metrics(report)
    assert metrics["agreement_eligible_lines"] == 1
    assert metrics["agreement_count"] == 1
    assert metrics["agreement_adaptive_rescue_count"] == 1


def test_extract_song_metrics_rescues_weak_lexical_but_tight_timing_lines() -> None:
    module = _load_module()
    report = {
        "dtw_line_coverage": 1.0,
        "lines": [
            {
                "start": 52.0,
                "nearest_segment_start": 52.17,
                "text": "if the rhythm takes you move your head",
                "nearest_segment_start_text": "move your head we are dancing all night",
                "words": [{"text": token} for token in "a b c d e f g".split()],
                "whisper_window_word_count": 4,
                "whisper_window_avg_prob": 0.7,
            }
        ],
        "low_confidence_lines": [],
    }
    metrics = module._extract_song_metrics(report)
    assert metrics["agreement_eligible_lines"] == 1
    assert metrics["agreement_count"] == 1
    assert metrics["agreement_adaptive_rescue_count"] == 1


def test_extract_song_metrics_adaptive_rescue_does_not_accept_large_timing_delta() -> (
    None
):
    module = _load_module()
    report = {
        "dtw_line_coverage": 1.0,
        "lines": [
            {
                "start": 10.0,
                "nearest_segment_start": 11.2,
                "text": "you know i love it when the music starts",
                "nearest_segment_start_text": "when the music starts you know i love it",
                "words": [{"text": token} for token in "a b c d e f g h".split()],
                "whisper_window_word_count": 5,
                "whisper_window_avg_prob": 0.8,
            }
        ],
        "low_confidence_lines": [],
    }
    metrics = module._extract_song_metrics(report)
    assert metrics["agreement_eligible_lines"] == 1
    assert metrics["agreement_count"] == 0
    assert metrics["agreement_skip_reason_counts"]["low_text_similarity"] == 1
    assert metrics["agreement_adaptive_rescue_count"] == 0


def test_extract_song_metrics_gold_matching_handles_insertions():
    module = _load_module()
    report = {
        "lines": [
            {
                "words": [
                    {"text": "hello", "start": 1.0, "end": 1.2},
                    {"text": "there", "start": 1.3, "end": 1.5},
                    {"text": "world", "start": 1.6, "end": 1.9},
                ]
            }
        ]
    }
    gold = {
        "lines": [
            {
                "words": [
                    {"text": "hello", "start": 1.05, "end": 1.25},
                    {"text": "world", "start": 1.65, "end": 1.95},
                ]
            }
        ]
    }
    metrics = module._extract_song_metrics(report, gold_doc=gold)
    assert metrics["gold_comparable_word_count"] == 2
    assert metrics["gold_word_coverage_ratio"] == 1.0
    assert metrics["avg_abs_word_start_delta_sec"] == 0.05


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
