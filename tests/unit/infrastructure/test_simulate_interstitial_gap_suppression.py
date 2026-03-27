from tools import simulate_interstitial_gap_suppression as tool


def test_simulate_uses_pre_whisper_start_after_contaminated_gap() -> None:
    timing_report = {
        "lines": [
            {"index": 1, "text": "a", "start": 1.0, "pre_whisper_start": 0.8},
            {"index": 2, "text": "b", "start": 6.84, "pre_whisper_start": 6.45},
        ]
    }
    gold = {"lines": [{"start": 1.0}, {"start": 6.85}]}
    contamination: dict[str, list[dict[str, object]]] = {
        "gaps": [
            {"gap_index": 1, "classification": "echo_fragment"},
        ]
    }

    result = tool._simulate(
        timing_report=timing_report,
        gold=gold,
        contamination=contamination,
        contaminated_classes={"echo_fragment"},
    )

    assert result["lines"][1]["contaminated_gap_predecessor"] is True
    assert result["lines"][1]["simulated_start"] == 6.45


def test_simulate_leaves_clean_lines_unchanged() -> None:
    timing_report = {
        "lines": [
            {"index": 1, "text": "a", "start": 1.0, "pre_whisper_start": 0.8},
        ]
    }
    gold = {"lines": [{"start": 1.0}]}
    contamination: dict[str, list[dict[str, object]]] = {"gaps": []}

    result = tool._simulate(
        timing_report=timing_report,
        gold=gold,
        contamination=contamination,
        contaminated_classes={"echo_fragment"},
    )

    assert result["lines"][0]["simulated_start"] == 1.0
