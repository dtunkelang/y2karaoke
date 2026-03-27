import importlib.util
from pathlib import Path
import sys

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "analyze_agreement_skip_reasons.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "analyze_agreement_skip_reasons_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_analyze_agreement_skip_reasons_reports_line_outcomes() -> None:
    payload = {
        "artist": "Artist",
        "title": "Song",
        "lines": [
            {
                "index": 1,
                "text": "Take me on",
                "start": 6.8,
                "end": 10.4,
                "nearest_segment_start": 6.8,
                "nearest_segment_start_text": "Take me on",
                "whisper_window_start": 6.6,
                "whisper_window_end": 10.6,
                "whisper_window_word_count": 3,
                "whisper_window_avg_prob": 0.8,
                "whisper_window_words": [
                    {"text": "Take", "start": 6.8, "end": 7.1},
                    {"text": "me", "start": 7.1, "end": 7.3},
                    {"text": "on", "start": 7.3, "end": 7.6},
                ],
            },
            {
                "index": 2,
                "text": "Que ganas me dan-dan-dan",
                "start": 27.3,
                "end": 28.7,
                "nearest_segment_start": 22.0,
                "nearest_segment_start_text": "random unmatched phrase",
                "whisper_window_start": 21.4,
                "whisper_window_end": 24.0,
                "whisper_window_word_count": 6,
                "whisper_window_avg_prob": 0.7,
                "whisper_window_words": [
                    {"text": "ya", "start": 21.5, "end": 21.8},
                    {"text": "vi", "start": 21.8, "end": 22.0},
                    {"text": "que", "start": 22.0, "end": 22.2},
                    {"text": "estas", "start": 22.2, "end": 22.5},
                ],
            },
        ],
    }

    result = _MODULE.analyze(payload)

    assert result["skip_reason_counts"]["low_text_similarity"] == 1
    assert len(result["rows"]) == 2
    assert result["rows"][0]["skip_reason"] is None
    assert result["rows"][0]["eligible"] is True
    assert result["rows"][1]["skip_reason"] == "low_text_similarity"
    assert result["rows"][1]["anchor_text"] == "random unmatched phrase"
