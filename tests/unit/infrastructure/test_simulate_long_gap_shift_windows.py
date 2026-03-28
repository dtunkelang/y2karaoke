import importlib.util
from pathlib import Path


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "simulate_long_gap_shift_windows.py"
    )
    spec = importlib.util.spec_from_file_location(
        "simulate_long_gap_shift_windows_module", module_path
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_simulate_long_gap_shift_windows_reports_current_and_baseline_ends(tmp_path):
    module = _load_module()
    trace_path = tmp_path / "trace.json"
    gold_path = tmp_path / "gold.json"
    baseline_path = tmp_path / "baseline.json"
    trace_path.write_text(
        """{
  "rows": [
    {
      "call_index": 1,
      "line_index": 2,
      "text": "Take me on",
      "start": 9.86,
      "end": 12.34,
      "chosen_onset": 5.387,
      "candidate_onsets": [5.387, 6.803],
      "decision": "shift"
    }
  ]
}""",
        encoding="utf-8",
    )
    gold_path.write_text(
        """{
  "lines": [
    {"line_index": 2, "text": "Take me on", "start": 6.85, "end": 10.65},
    {"line_index": 3, "text": "I'll be gone", "start": 12.03, "end": 15.25}
  ]
}""",
        encoding="utf-8",
    )
    baseline_path.write_text(
        """{
  "lines": [
    {"start": 0.99, "end": 4.21},
    {"start": 6.51, "end": 9.73},
    {"start": 12.03, "end": 15.25}
  ]
}""",
        encoding="utf-8",
    )

    result = module.analyze(
        trace_path=trace_path,
        gold_path=gold_path,
        baseline_timing_path=baseline_path,
    )

    row = result["rows"][0]
    assert row["current_duration"] == 2.48
    assert row["baseline_duration"] == 3.22
    assert row["candidate_windows"][1]["candidate_onset"] == 6.803
    assert row["candidate_windows"][1]["current_duration_end"] == 9.283
    assert row["candidate_windows"][1]["baseline_duration_end"] == 10.023
    assert (
        row["candidate_windows"][1]["overlaps_next_gold_with_current_duration"] is False
    )
    assert (
        row["candidate_windows"][1]["overlaps_next_gold_with_baseline_duration"]
        is False
    )
