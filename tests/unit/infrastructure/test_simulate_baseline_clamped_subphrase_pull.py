import importlib.util
from pathlib import Path


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "simulate_baseline_clamped_subphrase_pull.py"
    )
    spec = importlib.util.spec_from_file_location(
        "simulate_baseline_clamped_subphrase_pull_module", module_path
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_simulate_baseline_clamped_subphrase_pull_holds_start_near_baseline(tmp_path):
    module = _load_module()
    timing_path = tmp_path / "timing.json"
    segments_path = tmp_path / "segments.json"
    baseline_path = tmp_path / "baseline.json"
    timing_path.write_text(
        """{
  "lines": [
    {
      "line_index": 1,
      "text": "Take on me",
      "start": 1.0,
      "end": 5.3,
      "words": [
        {"text": "Take", "start": 1.0, "end": 2.5},
        {"text": "on", "start": 2.5, "end": 4.0},
        {"text": "me", "start": 4.0, "end": 5.3}
      ]
    },
    {
      "line_index": 2,
      "text": "Take me on",
      "start": 6.85,
      "end": 10.65,
      "words": [
        {"text": "Take", "start": 6.85, "end": 8.1},
        {"text": "me", "start": 8.1, "end": 9.1},
        {"text": "on", "start": 9.1, "end": 10.65}
      ]
    },
    {
      "line_index": 3,
      "text": "I'll be gone",
      "start": 12.35,
      "end": 16.5,
      "words": [
        {"text": "I'll", "start": 12.35, "end": 13.2},
        {"text": "be", "start": 13.2, "end": 14.0},
        {"text": "gone", "start": 14.0, "end": 16.5}
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
      "end": 15.5,
      "text": "Okay, take on me, take me on, I'll be gone.",
      "words": [
        {"start": 0.64, "end": 1.2, "text": "take"},
        {"start": 1.2, "end": 3.08, "text": "on"},
        {"start": 3.08, "end": 4.22, "text": "me,"},
        {"start": 4.22, "end": 5.48, "text": "take"},
        {"start": 5.48, "end": 8.08, "text": "me"},
        {"start": 8.08, "end": 9.7, "text": "on,"},
        {"start": 12.34, "end": 13.18, "text": "I'll"},
        {"start": 13.18, "end": 13.84, "text": "be"},
        {"start": 13.84, "end": 15.5, "text": "gone."}
      ]
    }
  ]
}""",
        encoding="utf-8",
    )
    baseline_path.write_text(timing_path.read_text(encoding="utf-8"), encoding="utf-8")

    result = module.analyze(
        timing_path=timing_path,
        segments_path=segments_path,
        baseline_timing_path=baseline_path,
    )

    row = result["rows"][0]
    assert row["clamped_start"] == 6.45
    assert row["clamped_end"] == 10.25
