from tools import analyze_alignment_candidate_starts as tool


def test_estimate_text_syllables_counts_basic_english_phrase():
    assert tool._estimate_text_syllables("I've been inclined") == 4


def test_build_repeated_family_stats_counts_normalized_repeats():
    stats = tool._build_repeated_family_stats(
        [
            {"text": "Turn around", "start": 1.0, "end": 2.0},
            {"text": "Turn around!", "start": 4.0, "end": 5.2},
            {"text": "Every now and then", "start": 6.0, "end": 8.0},
        ]
    )
    assert stats["turn around"]["count"] == 2
    assert stats["turn around"]["median_duration"] == 1.1


def test_analyze_line_flags_zero_support_and_compression():
    line = {
        "index": 3,
        "text": "I've been inclined",
        "start": 12.74,
        "end": 14.12,
        "pre_whisper_start": 10.824,
        "pre_whisper_end": 13.325,
        "words": [{}, {}, {}],
        "whisper_window_word_count": 0,
        "whisper_window_avg_prob": 0.0,
    }
    gold = {"start": 11.95, "end": 14.6}
    family_stats = {"i've been inclined": {"count": 1, "median_duration": 1.38}}

    analysis = tool._analyze_line(
        line,
        gold_line=gold,
        onset_times=[],
        family_stats=family_stats,
    )

    assert "zero_window_support" in analysis["flags"]
    assert "late_vs_source" in analysis["flags"]
