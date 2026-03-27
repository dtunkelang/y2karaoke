import pytest

from tools import simulate_override_opportunity_nudges as tool


def test_simulate_song_uses_selected_families() -> None:
    rows = [
        {
            "song": "Test - Song",
            "index": 1,
            "text": "De guayarte, ma...",
            "opportunity": "fuzzy_span_candidate",
            "current_start": 28.89,
            "fuzzy_estimated_start": 28.826,
            "gold_start": 28.75,
            "current_start_error": 0.14,
        },
        {
            "song": "Test - Song",
            "index": 2,
            "text": "No change",
            "opportunity": "no_clear_override",
            "current_start": 30.0,
            "fuzzy_estimated_start": 29.5,
            "gold_start": 30.0,
            "current_start_error": 0.0,
        },
    ]

    result = tool._simulate_song(rows, families={"fuzzy_span_candidate"})

    assert result["song"] == "Test - Song"
    assert result["current_start_mean"] == pytest.approx(0.07)
    assert result["simulated_start_mean"] < result["current_start_mean"]
    assert result["lines"][0]["simulated_start"] == 28.826
    assert result["lines"][1]["simulated_start"] == 30.0
