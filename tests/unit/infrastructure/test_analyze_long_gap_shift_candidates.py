import importlib.util
from pathlib import Path


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "analyze_long_gap_shift_candidates.py"
    )
    spec = importlib.util.spec_from_file_location(
        "analyze_long_gap_shift_candidates_module", module_path
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_analyze_long_gap_shift_candidates_prefers_gold_nearest_onset(tmp_path):
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
      "chosen_onset": 5.387,
      "candidate_onsets": [5.387, 6.432, 6.687, 6.803, 9.033],
      "decision": "shift"
    }
  ]
}""",
        encoding="utf-8",
    )
    gold_path.write_text(
        """{
  "lines": [
    {"line_index": 2, "text": "Take me on", "start": 6.85, "end": 10.65}
  ]
}""",
        encoding="utf-8",
    )

    result = module.analyze(trace_path=trace_path, gold_path=gold_path)

    assert result["rows"][0]["chosen_onset"] == 5.387
    assert result["rows"][0]["best_onset"] == 6.803
    assert result["rows"][0]["best_abs_error"] < result["rows"][0]["chosen_abs_error"]
