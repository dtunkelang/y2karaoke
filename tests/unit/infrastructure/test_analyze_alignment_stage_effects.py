from tools import analyze_alignment_stage_effects as tool


def test_analyze_stage_effects_finds_first_and_largest_changes() -> None:
    payload = {
        "snapshots": [
            {
                "stage": "loaded",
                "count": 2,
                "lines": [
                    {"line_index": 1, "text": "A", "start": 1.0, "end": 2.0},
                    {"line_index": 2, "text": "B", "start": 3.0, "end": 4.0},
                ],
            },
            {
                "stage": "middle",
                "count": 2,
                "lines": [
                    {"line_index": 1, "text": "A", "start": 1.4, "end": 2.0},
                    {"line_index": 2, "text": "B", "start": 3.0, "end": 4.6},
                ],
            },
            {
                "stage": "final",
                "count": 2,
                "lines": [
                    {"line_index": 1, "text": "A", "start": 1.2, "end": 2.2},
                    {"line_index": 2, "text": "B", "start": 3.0, "end": 4.5},
                ],
            },
        ]
    }

    result = tool.analyze_stage_effects(payload)

    assert result["snapshots"] == 3
    assert result["initial_stage"] == "loaded"
    assert result["final_stage"] == "final"

    line1 = result["lines"][1]
    assert line1["line_index"] == 1
    assert line1["first_changed_stage"] == "middle"
    assert line1["max_start_delta_stage"] == "middle"
    assert line1["total_start_shift"] == 0.2
    assert line1["total_end_shift"] == 0.2

    line2 = result["lines"][0]
    assert line2["line_index"] == 2
    assert line2["first_changed_stage"] == "middle"
    assert line2["max_end_delta_stage"] == "middle"
    assert line2["total_end_shift"] == 0.5
