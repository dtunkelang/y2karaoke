from tools import simulate_repeated_line_start_harmonization as tool


def test_simulate_harmonizes_later_repeated_line_start() -> None:
    payload = tool._simulate(
        report={
            "lines": [
                {
                    "index": 1,
                    "text": "Foo",
                    "start": 0.78,
                    "end": 1.8,
                    "pre_whisper_start": 0.78,
                },
                {
                    "index": 2,
                    "text": "Bar",
                    "start": 4.35,
                    "end": 5.0,
                    "pre_whisper_start": 4.35,
                },
                {
                    "index": 3,
                    "text": "Foo",
                    "start": 11.55,
                    "end": 12.5,
                    "pre_whisper_start": 11.55,
                },
            ]
        },
        gold={
            "lines": [
                {"start": 0.8, "end": 1.9},
                {"start": 4.35, "end": 5.1},
                {"start": 11.05, "end": 12.6},
            ]
        },
        min_start_error_span=0.3,
    )

    line3 = next(line for line in payload["lines"] if line["index"] == 3)
    assert round(payload["current_start_mean"], 3) == 0.173
    assert round(payload["simulated_start_mean"], 3) == 0.013
    assert round(line3["simulated_start"], 3) == 11.03
    assert round(line3["simulated_start_error"], 3) == 0.02


def test_simulate_skips_groups_below_threshold() -> None:
    payload = tool._simulate(
        report={
            "lines": [
                {
                    "index": 1,
                    "text": "Foo",
                    "start": 1.0,
                    "end": 2.0,
                    "pre_whisper_start": 1.0,
                },
                {
                    "index": 2,
                    "text": "Foo",
                    "start": 5.1,
                    "end": 6.0,
                    "pre_whisper_start": 5.1,
                },
            ]
        },
        gold={"lines": [{"start": 0.9, "end": 2.1}, {"start": 5.0, "end": 6.1}]},
        min_start_error_span=0.3,
    )

    assert payload["current_start_mean"] == payload["simulated_start_mean"]
