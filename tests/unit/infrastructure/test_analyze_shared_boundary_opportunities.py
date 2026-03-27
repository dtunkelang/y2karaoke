from tools import analyze_shared_boundary_opportunities as tool


def test_analyze_detects_parenthetical_followup_boundary() -> None:
    report = {
        "artist": "Daddy Yankee & Snow",
        "title": "Con Calma",
        "lines": [
            {
                "index": 8,
                "text": "I like your poom-poom, girl (¡Hey!)",
                "start": 19.44,
                "end": 20.82,
                "words": [
                    {"text": "I"},
                    {"text": "like"},
                    {"text": "your"},
                    {"text": "poom-poom,"},
                    {"text": "girl"},
                    {"text": "(¡Hey!)"},
                ],
                "whisper_window_words": [
                    {"text": "Hey,", "start": 20.82, "end": 21.36},
                    {"text": "ya", "start": 21.5, "end": 22.08},
                    {"text": "vi", "start": 22.08, "end": 22.22},
                    {"text": "que", "start": 22.22, "end": 22.36},
                ],
            },
            {
                "index": 9,
                "text": "Ya vi que estás solita, acompáñame",
                "start": 22.458,
                "end": 24.403,
                "words": [
                    {"text": "Ya"},
                    {"text": "vi"},
                    {"text": "que"},
                    {"text": "estás"},
                    {"text": "solita,"},
                    {"text": "acompáñame"},
                ],
                "whisper_window_words": [
                    {"text": "ya", "start": 21.5, "end": 22.08},
                    {"text": "vi", "start": 22.08, "end": 22.22},
                    {"text": "que", "start": 22.22, "end": 22.36},
                    {"text": "estás", "start": 22.36, "end": 22.52},
                ],
            },
        ],
    }
    gold = {
        "lines": [
            {"start": 19.65, "end": 21.75},
            {"start": 21.8, "end": 24.15},
        ]
    }

    payload = tool.analyze(report, gold)

    assert len(payload["rows"]) == 1
    row = payload["rows"][0]
    assert row["family"] == "parenthetical_followup_boundary"
    assert row["suggested_next_start"] == 21.5
    assert row["suggested_prev_end"] == 21.45
