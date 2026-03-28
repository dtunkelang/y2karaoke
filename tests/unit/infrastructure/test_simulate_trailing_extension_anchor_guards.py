from tools import simulate_trailing_extension_anchor_guards as tool


def test_anchor_guard_rejects_late_reanchor_candidate() -> None:
    payload = tool.analyze(
        stage_trace={
            "snapshots": [
                {
                    "stage": "postpass_shift_repeated",
                    "lines": [
                        {
                            "line_index": 2,
                            "text": "Take me on",
                            "start": 4.22,
                            "end": 8.22,
                            "words": [
                                {"text": "Take"},
                                {"text": "me"},
                                {"text": "on"},
                            ],
                        },
                        {
                            "line_index": 3,
                            "text": "I'll be gone",
                            "start": 12.34,
                            "end": 15.5,
                            "words": [
                                {"text": "I'll"},
                                {"text": "be"},
                                {"text": "gone"},
                            ],
                        },
                    ],
                }
            ]
        },
        timing_report={
            "lines": [
                {
                    "index": 2,
                    "text": "Take me on",
                    "words": [{"text": "Take"}, {"text": "me"}, {"text": "on"}],
                }
            ]
        },
        line_index=2,
        transcription_json={
            "segments": [
                {
                    "words": [
                        {"text": "take", "start": 4.22, "end": 5.48},
                        {"text": "me", "start": 5.48, "end": 8.08},
                        {"text": "on,", "start": 8.08, "end": 9.7},
                        {"text": "take", "start": 9.86, "end": 11.18},
                        {"text": "me,", "start": 11.84, "end": 12.34},
                        {"text": "gone.", "start": 13.84, "end": 15.5},
                    ]
                }
            ]
        },
    )

    assert payload["current_choice"]["matched_pairs"][0]["candidate_start"] == 9.86
    assert payload["anchor_guard_choice"]["matched_pairs"][0]["candidate_start"] == 4.22


def test_anchor_guard_keeps_candidate_when_shift_is_small() -> None:
    choice = tool._guarded_choice(
        [
            {
                "score": 3.0,
                "matched_count": 3,
                "matched_pairs": [{"candidate_start": 4.5}],
            }
        ],
        line_start=4.22,
        max_first_anchor_shift=1.0,
    )

    assert choice is not None
    assert choice["matched_pairs"][0]["candidate_start"] == 4.5
