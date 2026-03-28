import importlib.util
from pathlib import Path


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "simulate_final_tail_last_word_extension.py"
    )
    spec = importlib.util.spec_from_file_location(
        "simulate_final_tail_last_word_extension_module", module_path
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_simulate_final_tail_last_word_extension_targets_final_tail_only(tmp_path):
    module = _load_module()
    timing_path = tmp_path / "timing.json"
    baseline_path = tmp_path / "baseline.json"
    segments_path = tmp_path / "segments.json"
    payload = """{
  "lines": [
    {
      "line_index": 4,
      "text": "In a day or two",
      "start": 17.373,
      "end": 20.873,
      "words": [
        {"text": "In", "start": 17.373, "end": 18.073},
        {"text": "a", "start": 18.073, "end": 18.773},
        {"text": "day", "start": 18.773, "end": 19.473},
        {"text": "or", "start": 19.473, "end": 20.173},
        {"text": "two", "start": 20.173, "end": 20.873}
      ]
    }
  ]
}"""
    timing_path.write_text(payload, encoding="utf-8")
    baseline_path.write_text(
        """{
  "lines": [
    {
      "line_index": 4,
      "text": "In a day or two",
      "start": 17.05,
      "end": 22.05,
      "words": [
        {"text": "In", "start": 17.05, "end": 17.35},
        {"text": "a", "start": 17.35, "end": 17.7},
        {"text": "day", "start": 17.7, "end": 17.95},
        {"text": "or", "start": 17.95, "end": 18.35},
        {"text": "two", "start": 18.35, "end": 22.05}
      ]
    }
  ]
}""",
        encoding="utf-8",
    )
    segments_path.write_text(
        """{
  "segments": [
    {
      "start": 0.0,
      "end": 22.05,
      "text": "Take on me, take me on, I'll be gone in a day or two",
      "words": [
        {"start": 15.44, "end": 17.16, "text": "in"},
        {"start": 17.16, "end": 17.36, "text": "a"},
        {"start": 17.36, "end": 17.74, "text": "day"},
        {"start": 17.74, "end": 18.18, "text": "or"},
        {"start": 18.18, "end": 22.05, "text": "two"}
      ]
    }
  ]
}""",
        encoding="utf-8",
    )

    result = module.analyze(
        timing_path=timing_path,
        segments_path=segments_path,
        baseline_timing_path=baseline_path,
    )

    row = result["rows"][0]
    assert row["line_index"] == 4
    assert row["simulated_start"] == 17.373
    assert row["simulated_end"] == 22.05
