import importlib.util
from pathlib import Path
import sys

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "analyze_anchor_outside_window_recovery.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "analyze_anchor_outside_window_recovery_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_analyze_anchor_outside_window_recovery_finds_local_phrase() -> None:
    payload = {
        "artist": "Artist",
        "title": "Song",
        "lines": [
            {
                "index": 1,
                "text": "Mueve ese poom-poom, girl",
                "start": 4.0,
                "end": 5.0,
                "nearest_segment_start": 0.0,
                "nearest_segment_start_text": (
                    "Con calma yo quiero ver como ella lo menea mueve ese pum pum girl"
                ),
                "whisper_window_start": 3.0,
                "whisper_window_end": 6.0,
                "whisper_window_word_count": 5,
                "whisper_window_avg_prob": 0.8,
                "whisper_window_words": [
                    {"text": "mueve", "start": 3.2, "end": 3.4},
                    {"text": "ese", "start": 3.4, "end": 3.6},
                    {"text": "pum", "start": 3.6, "end": 3.8},
                    {"text": "pum", "start": 3.8, "end": 4.0},
                    {"text": "girl", "start": 4.0, "end": 4.2},
                ],
            }
        ],
    }

    result = _MODULE.analyze(payload)

    assert result["recoverable_anchor_outside_window_lines"] == 1
    row = result["rows"][0]
    assert row["candidate_text"] == "mueve ese pum pum girl"
    assert row["candidate_would_match"] is True
