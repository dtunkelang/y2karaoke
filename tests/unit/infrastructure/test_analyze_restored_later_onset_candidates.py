import importlib.util
from pathlib import Path


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "analyze_restored_later_onset_candidates.py"
    )
    spec = importlib.util.spec_from_file_location(
        "analyze_restored_later_onset_candidates_module",
        module_path,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_analyze_restored_later_onset_candidates_flags_non_baseline_anchored_line(
    tmp_path,
):
    module = _load_module()
    timing_path = tmp_path / "timing.json"
    baseline_path = tmp_path / "baseline.json"
    segments_path = tmp_path / "segments.json"
    timing_path.write_text(
        """{
  "lines": [
    {
      "index": 3,
      "text": "I'll be gone",
      "start": 11.912,
      "end": 15.494,
      "words": [
        {"text": "I'll", "start": 11.912, "end": 13.024},
        {"text": "be", "start": 13.147, "end": 14.259},
        {"text": "gone", "start": 14.382, "end": 15.494}
      ]
    }
  ]
}""",
        encoding="utf-8",
    )
    baseline_path.write_text(
        """{
  "lines": [
    {
      "line_index": 3,
      "text": "I'll be gone",
      "start": 12.35,
      "end": 16.5,
      "words": [
        {"text": "I'll", "start": 12.35, "end": 13.8},
        {"text": "be", "start": 13.8, "end": 15.1},
        {"text": "gone", "start": 15.1, "end": 16.5}
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
      "text": "Take on me, take me on, I'll be gone.",
      "words": [
        {"start": 0.64, "end": 1.2, "text": "take"},
        {"start": 1.2, "end": 3.08, "text": "on"},
        {"start": 3.08, "end": 4.22, "text": "me"},
        {"start": 4.22, "end": 5.48, "text": "take"},
        {"start": 5.48, "end": 8.08, "text": "me"},
        {"start": 8.08, "end": 9.7, "text": "on"},
        {"start": 12.34, "end": 13.18, "text": "I'll"},
        {"start": 13.18, "end": 13.84, "text": "be"},
        {"start": 13.84, "end": 15.5, "text": "gone"}
      ]
    }
  ]
}""",
        encoding="utf-8",
    )

    result = module.analyze(
        timing_path=timing_path,
        baseline_timing_path=baseline_path,
        segments_path=segments_path,
    )

    row = result["rows"][0]
    assert row["line_index"] == 3
    assert row["phrase_start"] == 12.34
    assert row["start_gain_sec"] == 0.428
    assert row["blocked_by_baseline_anchor_tolerance"] is True
