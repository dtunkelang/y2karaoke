from tools import simulate_trailing_extension_candidate_scoring as tool


def test_score_candidate_penalizes_short_soft_only_matches() -> None:
    early = tool._score_candidate(
        {
            "matched_count": 3,
            "last_end": 9.7,
            "matched_pairs": [
                {"line_token": "take", "candidate_text": "take"},
                {"line_token": "me", "candidate_text": "me"},
                {"line_token": "on", "candidate_text": "on,"},
            ],
        },
        line_end=8.22,
        next_start=12.34,
    )
    late = tool._score_candidate(
        {
            "matched_count": 3,
            "last_end": 15.5,
            "matched_pairs": [
                {"line_token": "take", "candidate_text": "take"},
                {"line_token": "me", "candidate_text": "me,"},
                {"line_token": "on", "candidate_text": "gone."},
            ],
        },
        line_end=8.22,
        next_start=12.34,
    )

    assert early["short_soft_only_matches"] == 0
    assert late["short_soft_only_matches"] == 1
    assert early["score"] > late["score"]


def test_analyze_reorders_take_on_me_style_candidates() -> None:
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
                    "start": 0.0,
                    "end": 15.5,
                    "text": "take me on take on me i'll be gone",
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
                    ],
                }
            ]
        },
    )

    best = payload["candidates"][0]
    assert best["start_word"]["text"] == "take"
    assert best["start_word"]["start"] == 4.22
