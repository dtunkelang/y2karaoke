import importlib.util
from pathlib import Path


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "simulate_long_gap_shift_sequences.py"
    )
    spec = importlib.util.spec_from_file_location(
        "simulate_long_gap_shift_sequences_module", module_path
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_simulate_long_gap_shift_sequences_prefers_valid_low_score_pair(tmp_path):
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
      "candidate_onsets": [5.387, 6.803],
      "decision": "shift"
    },
    {
      "call_index": 2,
      "line_index": 3,
      "text": "I'll be gone",
      "start": 12.34,
      "end": 15.5,
      "candidate_onsets": [8.011, 12.144],
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
    {"line_index": 3, "text": "I'll be gone", "start": 12.35, "end": 16.5},
    {"line_index": 4, "text": "In a day or two", "start": 17.05, "end": 22.05}
  ]
}""",
        encoding="utf-8",
    )
    baseline_path.write_text(
        """{
  "lines": [
    {"start": 0.99, "end": 4.21},
    {"start": 6.51, "end": 9.73},
    {"start": 12.03, "end": 15.25},
    {"start": 17.05, "end": 22.05}
  ]
}""",
        encoding="utf-8",
    )

    result = module.analyze(
        trace_path=trace_path,
        gold_path=gold_path,
        baseline_timing_path=baseline_path,
    )

    best = result["rows"][0]["best_pairs"][0]
    assert best["valid_current_pair"] is True
    assert best["first_onset"] == 6.803
    assert best["second_onset"] == 12.144
    assert best["current_pair_score"] == 2.816


def test_simulate_long_gap_shift_sequences_prefers_later_first_onset_when_short(
    tmp_path,
):
    module = _load_module()
    trace_path = tmp_path / "trace.json"
    gold_path = tmp_path / "gold.json"
    trace_path.write_text(
        """{
  "rows": [
    {
      "call_index": 1,
      "line_index": 2,
      "text": "Take me on",
      "start": 9.86,
      "end": 12.34,
      "candidate_onsets": [6.803, 7.268],
      "decision": "shift"
    },
    {
      "call_index": 2,
      "line_index": 3,
      "text": "I'll be gone",
      "start": 12.34,
      "end": 15.5,
      "candidate_onsets": [12.144],
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
    {"line_index": 3, "text": "I'll be gone", "start": 12.35, "end": 16.5},
    {"line_index": 4, "text": "In a day or two", "start": 17.05, "end": 22.05}
  ]
}""",
        encoding="utf-8",
    )

    result = module.analyze(trace_path=trace_path, gold_path=gold_path)

    best = result["rows"][0]["best_pairs"][0]
    assert best["first_onset"] == 7.268
    assert best["second_onset"] == 12.144
    assert (
        best["current_pair_score"]
        < result["rows"][0]["best_pairs"][1]["current_pair_score"]
    )
