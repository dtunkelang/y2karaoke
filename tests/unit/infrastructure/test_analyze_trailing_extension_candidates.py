from tools import analyze_trailing_extension_candidates as tool


def test_analyze_trailing_extension_candidates_prefers_earliest_report_window_match():
    trace = {
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
    }
    report = {
        "lines": [
            {
                "index": 1,
                "whisper_window_words": [
                    {"text": "Take", "start": 4.28, "end": 5.64, "probability": 0.8},
                    {"text": "me", "start": 5.64, "end": 8.16, "probability": 0.98},
                ],
            },
            {
                "index": 2,
                "whisper_window_words": [
                    {"text": "on,", "start": 8.16, "end": 9.64, "probability": 0.72},
                    {"text": "Take", "start": 9.86, "end": 10.4, "probability": 0.8},
                    {"text": "me", "start": 10.4, "end": 11.2, "probability": 0.9},
                    {"text": "on", "start": 11.2, "end": 12.34, "probability": 0.9},
                ],
            },
        ]
    }

    payload = tool.analyze(stage_trace=trace, timing_report=report, line_index=2)

    best = payload["candidates"][0]
    assert best["matched_count"] == 3
    assert best["start_word"]["text"] == "me"
    assert best["start_word"]["start"] == 5.64


def test_analyze_trailing_extension_candidates_uses_transcription_words_when_provided():
    trace = {
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
    }
    report = {
        "lines": [
            {
                "index": 2,
                "text": "Take me on",
                "words": [{"text": "Take"}, {"text": "me"}, {"text": "on"}],
            }
        ]
    }
    transcription = {
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
    }

    payload = tool.analyze(
        stage_trace=trace,
        timing_report=report,
        line_index=2,
        transcription_json=transcription,
    )

    best = payload["candidates"][0]
    assert best["matched_count"] == 3
    assert best["start_word"]["text"] == "me"
    assert best["last_end"] == 15.5
