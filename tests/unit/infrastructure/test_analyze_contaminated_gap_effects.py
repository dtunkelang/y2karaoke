from pathlib import Path
import tempfile

from tools import analyze_contaminated_gap_effects as tool


def test_classify_effect_flags_prev_line_truncation() -> None:
    result = tool._classify_effect(
        gap_classification="hallucinated_interstitial",
        prev_end_error=-1.15,
        next_start_error=-0.01,
    )
    assert result == "prev_line_truncated"


def test_classify_effect_flags_next_line_delay() -> None:
    result = tool._classify_effect(
        gap_classification="echo_fragment",
        prev_end_error=-0.05,
        next_start_error=0.52,
    )
    assert result == "next_line_delayed"


def test_analyze_merges_contamination_with_timing(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        gold_json = tmp / "gold.json"
        report_json = tmp / "report.json"
        gold_json.write_text(
            """
            {
              "audio_path": "/tmp/audio.wav",
              "lines": [
                {"text": "Take on me", "start": 1.0, "end": 5.3},
                {"text": "Take me on", "start": 6.85, "end": 10.65}
              ]
            }
            """.strip(),
            encoding="utf-8",
        )
        report_json.write_text(
            """
            {
              "lines": [
                {"index": 1, "text": "Take on me", "start": 1.12, "end": 4.15},
                {"index": 2, "text": "Take me on", "start": 6.84, "end": 10.42}
              ]
            }
            """.strip(),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            tool.contamination_tool,
            "analyze_gold_json",
            lambda _path: {
                "gaps": [
                    {
                        "gap_index": 1,
                        "classification": "hallucinated_interstitial",
                        "aggressive_text": "I only say",
                    }
                ]
            },
        )

        payload = tool._analyze(gold_json=gold_json, timing_report_json=report_json)

    assert len(payload["rows"]) == 1
    row = payload["rows"][0]
    assert row["prev_end_error"] == -1.15
    assert row["next_start_error"] == -0.01
    assert row["effect"] == "prev_line_truncated"
