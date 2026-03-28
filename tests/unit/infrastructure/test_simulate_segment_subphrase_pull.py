import importlib.util
from pathlib import Path


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "simulate_segment_subphrase_pull.py"
    )
    spec = importlib.util.spec_from_file_location(
        "simulate_segment_subphrase_pull_module", module_path
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_simulate_segment_subphrase_pull_anchors_to_phrase_window(tmp_path):
    module = _load_module()
    timing_path = tmp_path / "timing.json"
    segments_path = tmp_path / "segments.json"
    timing_path.write_text(
        """{
  "lines": [
    {
      "line_index": 2,
      "text": "Take me on",
      "start": 9.86,
      "end": 12.34,
      "words": [
        {"text": "Take", "start": 9.86, "end": 10.6},
        {"text": "me", "start": 10.6, "end": 11.2},
        {"text": "on", "start": 11.2, "end": 12.34}
      ]
    },
    {
      "line_index": 3,
      "text": "I'll be gone",
      "start": 12.34,
      "end": 15.5,
      "words": [
        {"text": "I'll", "start": 12.34, "end": 13.2},
        {"text": "be", "start": 13.2, "end": 14.0},
        {"text": "gone", "start": 14.0, "end": 15.5}
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
        {"start": 0.0, "end": 0.6, "text": "Okay,"},
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

    result = module.analyze(timing_path=timing_path, segments_path=segments_path)

    assert result["rows"][0]["simulated_start"] == 4.22
    assert result["rows"][0]["simulated_end"] == 6.7
    assert result["rows"][1]["simulated_start"] == 12.34
