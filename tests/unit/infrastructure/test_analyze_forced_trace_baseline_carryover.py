import json

from tools.analyze_forced_trace_baseline_carryover import analyze_trace


def test_analyze_trace_flags_repeated_short_baseline_carryover(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "transcription_segment_preview": [
                        {"start": 28.71, "end": 31.33, "text": "Confusion never stops"}
                    ]
                },
                "snapshots": [
                    {
                        "stage": "baseline_lines",
                        "lines": [
                            {
                                "line_index": 1,
                                "text": "You are",
                                "start": 0.859,
                                "end": 6.747,
                            },
                            {
                                "line_index": 2,
                                "text": "You are",
                                "start": 10.156,
                                "end": 16.045,
                            },
                        ],
                    },
                    {
                        "stage": "final_forced_lines",
                        "lines": [
                            {
                                "line_index": 1,
                                "text": "You are",
                                "start": 0.81,
                                "end": 6.304,
                            },
                            {
                                "line_index": 2,
                                "text": "You are",
                                "start": 10.077,
                                "end": 15.966,
                            },
                        ],
                    },
                ],
            }
        )
    )

    hits = analyze_trace(trace_path)

    assert len(hits) == 1
    assert hits[0]["line_index"] == 2
    assert hits[0]["text"] == "You are"


def test_analyze_trace_skips_supported_or_changed_lines(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "transcription_segment_preview": [
                        {"start": 8.3, "end": 13.95, "text": "You are"}
                    ]
                },
                "snapshots": [
                    {
                        "stage": "baseline_lines",
                        "lines": [
                            {
                                "line_index": 1,
                                "text": "You are",
                                "start": 0.859,
                                "end": 6.747,
                            },
                            {
                                "line_index": 2,
                                "text": "You are",
                                "start": 10.156,
                                "end": 16.045,
                            },
                        ],
                    },
                    {
                        "stage": "final_forced_lines",
                        "lines": [
                            {
                                "line_index": 1,
                                "text": "You are",
                                "start": 0.81,
                                "end": 6.304,
                            },
                            {
                                "line_index": 2,
                                "text": "You are",
                                "start": 8.3,
                                "end": 13.95,
                            },
                        ],
                    },
                ],
            }
        )
    )

    assert analyze_trace(trace_path) == []
