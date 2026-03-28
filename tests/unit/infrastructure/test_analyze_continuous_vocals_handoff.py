from tools import analyze_continuous_vocals_handoff as tool


def test_analyze_continuous_vocals_handoff_reports_left_and_right_growth() -> None:
    payload = tool.analyze(
        stage_trace={
            "snapshots": [
                {
                    "stage": "postpass_extend_trailing",
                    "lines": [
                        {
                            "line_index": 2,
                            "text": "Take me on",
                            "start": 4.22,
                            "end": 9.72,
                        }
                    ],
                },
                {
                    "stage": "postpass_pull_continuous_vocals",
                    "lines": [
                        {
                            "line_index": 2,
                            "text": "Take me on",
                            "start": 4.22,
                            "end": 12.22,
                        }
                    ],
                },
            ]
        }
    )

    row = payload["lines"][0]
    assert row["start_shift"] == 0.0
    assert row["end_shift"] == 2.5
    assert row["grew_left"] is False
    assert row["grew_right"] is True
