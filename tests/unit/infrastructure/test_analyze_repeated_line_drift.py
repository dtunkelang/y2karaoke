from tools import analyze_repeated_line_drift as tool


def test_normalize_text_strips_parentheticals() -> None:
    assert (
        tool._normalize_text("I like your poom-poom, girl (Sube, sube)")
        == "i like your poom poom girl"
    )


def test_analyze_groups_repeated_lines_and_computes_error_span() -> None:
    result = tool._analyze(
        report={
            "lines": [
                {
                    "index": 1,
                    "text": "Foo (Hey)",
                    "start": 1.0,
                    "end": 2.0,
                    "pre_whisper_start": 1.0,
                },
                {
                    "index": 3,
                    "text": "Foo",
                    "start": 5.6,
                    "end": 6.9,
                    "pre_whisper_start": 5.2,
                },
            ]
        },
        gold={
            "lines": [
                {"start": 0.8, "end": 2.1},
                {"start": 3.0, "end": 4.0},
                {"start": 5.0, "end": 6.5},
            ]
        },
    )

    assert len(result) == 1
    group = result[0]
    assert group["normalized_text"] == "foo"
    assert round(group["start_error_span"], 3) == 0.4
    assert [row["index"] for row in group["occurrences"]] == [1, 3]
