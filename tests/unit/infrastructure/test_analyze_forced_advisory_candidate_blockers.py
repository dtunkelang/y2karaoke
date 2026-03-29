import json

from tools.analyze_forced_advisory_candidate_blockers import _collect_blockers


def test_collect_blockers_flags_exact_aggressive_support_shadowed_by_default() -> None:
    payload = {
        "candidates": [],
        "lines": [
            {
                "index": 3,
                "text": "I've been inclined",
                "aggressive_best_segment_text": "I've been inclined",
                "aggressive_best_overlap": 1.0,
                "default_best_overlap": 0.13,
                "default_window_word_count": 4,
                "current_window_word_count": 4,
            }
        ],
    }

    blockers = _collect_blockers(payload)

    assert json.loads(json.dumps(blockers))[0]["index"] == 3
    assert blockers[0]["blockers"] == [
        "default_overlap=0.130",
        "default_window_word_count=4",
        "current_window_word_count=4",
    ]
