import pytest

from tools import simulate_shared_boundary_opportunities as tool


def test_simulate_applies_shared_boundary_suggestions() -> None:
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
                "start": 22.46,
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

    payload = tool.simulate(report, gold)

    assert payload["current_start_mean"] == pytest.approx(0.435)
    assert payload["simulated_start_mean"] == pytest.approx(0.255)
    assert payload["current_end_mean"] == pytest.approx(0.5915)
    assert payload["simulated_end_mean"] == pytest.approx(0.2765)
    assert payload["opportunities"][0]["prev_end"]["simulated"] == 21.45
    assert payload["opportunities"][0]["next_start"]["simulated"] == 21.5
