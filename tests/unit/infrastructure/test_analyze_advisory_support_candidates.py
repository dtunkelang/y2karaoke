from tools import analyze_advisory_support_candidates as tool


def test_candidate_bucket_marks_high_confidence_exact_gain() -> None:
    bucket = tool._candidate_bucket(
        {
            "current_window_word_count": 2,
            "default_window_word_count": 0,
            "aggressive_window_word_count": 4,
            "aggressive_best_overlap": 1.0,
        }
    )

    assert bucket == "high_confidence"


def test_candidate_bucket_rejects_weak_merged_overlap() -> None:
    bucket = tool._candidate_bucket(
        {
            "current_window_word_count": 0,
            "default_window_word_count": 0,
            "aggressive_window_word_count": 4,
            "aggressive_best_overlap": 0.273,
        }
    )

    assert bucket is None


def test_candidate_score_prefers_strong_aggressive_overlap() -> None:
    strong = tool._candidate_score(
        {
            "aggressive_best_overlap": 1.0,
            "aggressive_window_word_count": 4,
            "current_window_word_count": 0,
        }
    )
    weak = tool._candidate_score(
        {
            "aggressive_best_overlap": 0.45,
            "aggressive_window_word_count": 4,
            "current_window_word_count": 0,
        }
    )

    assert strong > weak
