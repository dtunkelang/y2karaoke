import importlib.util
from pathlib import Path
import sys

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "analyze_agreement_anchor_clipping.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "analyze_agreement_anchor_clipping_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_analyze_agreement_anchor_clipping_recovers_merged_prefix() -> None:
    payload = {
        "artist": "Artist",
        "title": "Song",
        "lines": [
            {
                "index": 1,
                "text": "Con calma yo quiero ver",
                "start": 1.0,
                "end": 3.0,
                "nearest_segment_start": 1.0,
                "nearest_segment_start_text": (
                    "Con calma yo quiero ver mueve ese pum pum girl "
                    "es una asesina cuando baila quiere que todo el mundo la vea"
                ),
                "whisper_window_start": 0.8,
                "whisper_window_end": 3.2,
                "whisper_window_word_count": 8,
                "whisper_window_avg_prob": 0.8,
                "whisper_window_words": [
                    {"text": "Con", "start": 1.0, "end": 1.1},
                    {"text": "calma", "start": 1.1, "end": 1.4},
                    {"text": "yo", "start": 1.4, "end": 1.5},
                    {"text": "quiero", "start": 1.5, "end": 1.8},
                    {"text": "ver", "start": 1.8, "end": 2.0},
                ],
            }
        ],
    }

    result = _MODULE.analyze(payload)

    row = result["rows"][0]
    assert row["baseline_text_similarity"] < 0.58
    assert row["clipped_anchor_text"] == "con calma yo quiero ver"
    assert row["clipped_would_match"] is True
