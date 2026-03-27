from tools import analyze_weak_evidence_restore_families as tool


def test_classify_family_detects_repeated_short_hook_suffix_support() -> None:
    payload = tool.analyze(
        {
            "snapshots": [
                {
                    "stage": "before",
                    "lines": [
                        {
                            "line_index": 2,
                            "text": "Take me on",
                            "start": 4.22,
                            "end": 8.22,
                        }
                    ],
                },
                {
                    "stage": "after_restore_weak_evidence_large_start_shifts",
                    "lines": [
                        {
                            "line_index": 2,
                            "text": "Take me on",
                            "start": 6.451,
                            "end": 10.033,
                        }
                    ],
                },
            ]
        },
        {
            "lines": [
                {
                    "index": 1,
                    "text": "Take on me",
                    "start": 0.64,
                    "end": 4.57,
                    "words": [{"text": "Take"}, {"text": "on"}, {"text": "me"}],
                },
                {
                    "index": 2,
                    "text": "Take me on",
                    "start": 6.451,
                    "end": 10.033,
                    "words": [{"text": "Take"}, {"text": "me"}, {"text": "on"}],
                    "whisper_window_words": [
                        {
                            "text": "me",
                            "start": 5.64,
                            "end": 8.16,
                            "probability": 0.986,
                        },
                        {
                            "text": "on,",
                            "start": 8.16,
                            "end": 9.64,
                            "probability": 0.724,
                        },
                        {
                            "text": "I'll",
                            "start": 9.84,
                            "end": 13.2,
                            "probability": 0.895,
                        },
                    ],
                    "whisper_window_avg_prob": 0.868,
                    "whisper_window_low_conf_count": 0,
                    "whisper_window_word_count": 3,
                },
                {
                    "index": 3,
                    "text": "I'll be gone",
                    "start": 11.912,
                    "end": 15.494,
                    "words": [{"text": "I'll"}, {"text": "be"}, {"text": "gone"}],
                },
            ]
        },
    )

    row = payload["rows"][0]
    assert row["family"] == "repeated_short_hook_suffix_support"
    assert row["prev_overlap_tokens"] == 3


def test_classify_family_detects_sparse_tail_suffix_support() -> None:
    payload = tool.analyze(
        {
            "snapshots": [
                {
                    "stage": "before",
                    "lines": [
                        {
                            "line_index": 3,
                            "text": "I'll be gone",
                            "start": 8.27,
                            "end": 11.43,
                        }
                    ],
                },
                {
                    "stage": "after_restore_weak_evidence_large_start_shifts",
                    "lines": [
                        {
                            "line_index": 3,
                            "text": "I'll be gone",
                            "start": 11.912,
                            "end": 15.494,
                        }
                    ],
                },
            ]
        },
        {
            "lines": [
                {
                    "index": 2,
                    "text": "Take me on",
                    "start": 6.451,
                    "end": 10.033,
                    "words": [{"text": "Take"}, {"text": "me"}, {"text": "on"}],
                },
                {
                    "index": 3,
                    "text": "I'll be gone",
                    "start": 11.912,
                    "end": 15.494,
                    "words": [{"text": "I'll"}, {"text": "be"}, {"text": "gone"}],
                    "whisper_window_words": [
                        {
                            "text": "be",
                            "start": 13.2,
                            "end": 13.82,
                            "probability": 0.947,
                        },
                        {
                            "text": "gone",
                            "start": 13.82,
                            "end": 15.44,
                            "probability": 0.98,
                        },
                        {
                            "text": "in",
                            "start": 15.44,
                            "end": 17.16,
                            "probability": 0.75,
                        },
                    ],
                    "whisper_window_avg_prob": 0.892,
                    "whisper_window_low_conf_count": 0,
                    "whisper_window_word_count": 3,
                },
                {
                    "index": 4,
                    "text": "In a day or two",
                    "start": 17.373,
                    "end": 21.417,
                    "words": [
                        {"text": "In"},
                        {"text": "a"},
                        {"text": "day"},
                        {"text": "or"},
                        {"text": "two"},
                    ],
                },
            ]
        },
    )

    row = payload["rows"][0]
    assert row["family"] == "sparse_tail_suffix_support"
    assert row["prev_overlap_tokens"] == 0
    assert row["next_overlap_tokens"] == 0
