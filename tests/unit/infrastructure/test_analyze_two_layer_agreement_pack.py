import importlib.util
import json
from pathlib import Path
import sys

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "analyze_two_layer_agreement_pack.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "analyze_two_layer_agreement_pack_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_analyze_two_layer_agreement_pack_aggregates_adjusted_coverage(
    tmp_path: Path,
) -> None:
    timing_report = {
        "artist": "Artist",
        "title": "Song",
        "lines": [
            {
                "index": 1,
                "text": "Con calma, yo quiero ver cómo ella lo menea",
                "start": 1.4,
                "end": 3.4,
                "nearest_segment_start": 0.0,
                "nearest_segment_start_text": (
                    "Con calma yo quiero ver como ella lo menea mueve ese pum pum "
                    "girl es una asesina cuando baila quiere que todo el mundo la vea"
                ),
                "whisper_window_start": -0.2,
                "whisper_window_end": 3.2,
                "whisper_window_word_count": 9,
                "whisper_window_avg_prob": 0.8,
                "whisper_window_words": [
                    {"text": "Con", "start": 1.0, "end": 1.1},
                    {"text": "calma", "start": 1.1, "end": 1.4},
                    {"text": "yo", "start": 1.4, "end": 1.5},
                    {"text": "quiero", "start": 1.5, "end": 1.8},
                    {"text": "ver", "start": 1.8, "end": 2.0},
                    {"text": "como", "start": 2.0, "end": 2.2},
                    {"text": "ella", "start": 2.2, "end": 2.4},
                    {"text": "lo", "start": 2.4, "end": 2.5},
                    {"text": "menea", "start": 2.5, "end": 2.8},
                ],
            },
            {
                "index": 2,
                "text": "La noche es de nosotros, tú lo sabe' (You know)",
                "start": 6.0,
                "end": 8.0,
                "nearest_segment_start": 0.0,
                "nearest_segment_start_text": (
                    "Hey ya vi que estas solita acompaname la noche de nosotros "
                    "tu lo sabes que gana me dam dam dam"
                ),
                "whisper_window_start": 5.0,
                "whisper_window_end": 8.5,
                "whisper_window_word_count": 5,
                "whisper_window_avg_prob": 0.8,
                "whisper_window_words": [
                    {"text": "la", "start": 5.8, "end": 5.9},
                    {"text": "noche", "start": 5.9, "end": 6.1},
                    {"text": "de", "start": 6.1, "end": 6.2},
                    {"text": "nosotros", "start": 6.2, "end": 6.6},
                    {"text": "tu", "start": 6.6, "end": 6.7},
                    {"text": "lo", "start": 6.7, "end": 6.8},
                ],
            },
        ],
    }
    report_path = tmp_path / "song_timing_report.json"
    report_path.write_text(json.dumps(timing_report), encoding="utf-8")
    benchmark_report = {
        "songs": [
            {
                "artist": "Artist",
                "title": "Song",
                "report_path": str(report_path),
            }
        ]
    }

    result = _MODULE.analyze(benchmark_report)

    assert result["baseline_eligible_lines_total"] == 1
    assert result["baseline_matched_lines_total"] == 0
    assert result["adjusted_eligible_lines_total"] == 2
    assert result["adjusted_matched_lines_total"] == 2
    assert result["adjusted_coverage_ratio_total"] == 1.0
