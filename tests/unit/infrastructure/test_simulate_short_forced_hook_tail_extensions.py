import pytest

from tools import simulate_short_forced_hook_tail_extensions as tool


def test_candidate_target_end_requires_repeated_tail_support() -> None:
    line = {
        "text": "Guess who's back",
        "end": 3.577,
        "pre_whisper_end": 4.062,
        "words": [
            {"text": "Guess"},
            {"text": "who's"},
            {"text": "back"},
        ],
        "whisper_window_words": [
            {"text": "back,", "start": 3.26, "end": 3.48},
            {"text": "back", "start": 3.62, "end": 4.92},
        ],
    }

    target_end = tool._candidate_target_end(
        line,
        max_word_count=3,
        min_extension_sec=0.2,
        max_extension_sec=1.6,
        min_repeat_gap_sec=0.03,
    )

    assert target_end == pytest.approx(4.062)


def test_simulate_song_extends_only_candidates() -> None:
    report = {
        "lines": [
            {
                "text": "Guess who's back",
                "end": 3.577,
                "pre_whisper_end": 4.062,
                "words": [
                    {"text": "Guess"},
                    {"text": "who's"},
                    {"text": "back"},
                ],
                "whisper_window_words": [
                    {"text": "back", "start": 3.62, "end": 4.92},
                ],
            },
            {
                "text": "Back again",
                "end": 5.781,
                "pre_whisper_end": 5.781,
                "words": [
                    {"text": "Back"},
                    {"text": "again"},
                ],
                "whisper_window_words": [
                    {"text": "again", "start": 4.92, "end": 5.44},
                ],
            },
        ]
    }
    gold = {
        "lines": [
            {"end": 3.85},
            {"end": 5.65},
        ]
    }

    result = tool._simulate_song(
        report,
        gold,
        max_word_count=3,
        min_extension_sec=0.2,
        max_extension_sec=1.6,
        min_repeat_gap_sec=0.03,
    )

    assert result["simulated_end_mean"] < result["current_end_mean"]
    assert result["lines"][0]["simulated_end"] == pytest.approx(4.062)
    assert result["lines"][1]["simulated_end"] == pytest.approx(5.781)
