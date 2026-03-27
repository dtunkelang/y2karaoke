from tools import analyze_weak_evidence_restore_decisions as tool


def test_analyze_weak_evidence_restore_decisions_classifies_reason() -> None:
    trace = {
        "snapshots": [
            {
                "stage": "before",
                "count": 1,
                "lines": [
                    {"line_index": 9, "text": "Ya vi", "start": 25.34, "end": 27.92}
                ],
            },
            {
                "stage": "after_restore_weak_evidence_large_start_shifts",
                "count": 1,
                "lines": [
                    {"line_index": 9, "text": "Ya vi", "start": 22.458, "end": 24.403}
                ],
            },
        ]
    }
    report = {
        "lines": [
            {
                "index": 9,
                "text": "Ya vi",
                "start": 22.458,
                "end": 24.403,
                "words": [{"text": "Ya"}, {"text": "vi"}],
                "whisper_window_avg_prob": 0.95,
                "whisper_window_low_conf_count": 0,
                "whisper_window_word_count": 2,
                "whisper_window_words": [
                    {"text": "ya", "start": 21.5, "end": 22.08, "probability": 0.9},
                    {"text": "vi", "start": 22.08, "end": 22.22, "probability": 0.9},
                ],
            }
        ]
    }

    payload = tool.analyze(trace, report)

    assert payload["before_stage"] == "before"
    assert len(payload["rows"]) == 1
    row = payload["rows"][0]
    assert row["line_index"] == 9
    assert row["reason"] == "lexical_support_missing"
    assert row["has_lexical_support"] is False
    assert row["nearby_support_words"] == 0
