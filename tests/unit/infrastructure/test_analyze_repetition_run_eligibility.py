import importlib.util
from pathlib import Path


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "analyze_repetition_run_eligibility.py"
    )
    spec = importlib.util.spec_from_file_location(
        "analyze_repetition_run_eligibility_module", module_path
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_analyze_repetition_run_eligibility_flags_non_exact_hook_pair(tmp_path):
    module = _load_module()
    timing_path = tmp_path / "timing.json"
    timing_path.write_text(
        """{
  "lines": [
    {
      "line_index": 1,
      "text": "Take on me",
      "words": [
        {"text": "Take", "start": 1.12, "end": 2.13},
        {"text": "on", "start": 2.13, "end": 3.14},
        {"text": "me", "start": 3.14, "end": 4.15}
      ]
    },
    {
      "line_index": 2,
      "text": "Take me on",
      "words": [
        {"text": "Take", "start": 9.86, "end": 10.6},
        {"text": "me", "start": 10.6, "end": 11.2},
        {"text": "on", "start": 11.2, "end": 12.34}
      ]
    }
  ]
}""",
        encoding="utf-8",
    )

    result = module.analyze(timing_path=timing_path)

    row = result["rows"][0]
    assert row["passes_pair_overlap_gate"] is True
    assert row["exact_duplicate"] is False
    assert row["would_still_fail_run_gate"] is True
