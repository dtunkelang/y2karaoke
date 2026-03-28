from tools import simulate_trailing_extension_end_guards as tool


def test_guard_prefers_early_candidate_when_late_one_crosses_and_soft_matches() -> None:
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
                        {"text": "on", "start": 11.18, "end": 11.84},
                        {"text": "me,", "start": 11.84, "end": 12.34},
                        {"text": "I'll", "start": 12.34, "end": 13.18},
                        {"text": "be", "start": 13.18, "end": 13.84},
                        {"text": "gone.", "start": 13.84, "end": 15.5},
                    ]
                }
            ]
        },
    )

    assert payload["current_choice"]["matched_pairs"][0]["candidate_start"] == 9.86
    assert payload["guarded_choice"]["matched_pairs"][0]["candidate_start"] == 4.22


def test_guard_falls_back_to_current_when_only_current_candidate_allowed() -> None:
    candidates = [
        {
            "matched_count": 3,
            "score": 1.0,
            "last_end": 9.7,
            "end_distance": 1.48,
            "crosses_next_start": False,
            "short_soft_only_matches": 0,
        }
    ]

    guarded = tool._guarded_choice(candidates)

    assert guarded == candidates[0]
