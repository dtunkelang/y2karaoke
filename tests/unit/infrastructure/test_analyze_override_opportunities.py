from tools import analyze_override_opportunities as tool


def test_classify_opportunity_prefers_advisory_bucket() -> None:
    assert (
        tool._classify_opportunity(
            current_window_word_count=0,
            advisory_bucket="high_confidence",
            fuzzy_joint_score=1.0,
            fuzzy_estimated_start=12.0,
            current_start=12.7,
        )
        == "advisory_exact_start"
    )


def test_classify_opportunity_detects_fuzzy_span_candidate() -> None:
    assert (
        tool._classify_opportunity(
            current_window_word_count=4,
            advisory_bucket=None,
            fuzzy_joint_score=0.69,
            fuzzy_estimated_start=28.64,
            current_start=28.89,
        )
        == "fuzzy_span_candidate"
    )


def test_classify_opportunity_marks_zero_support_without_fuzzy_path() -> None:
    assert (
        tool._classify_opportunity(
            current_window_word_count=0,
            advisory_bucket=None,
            fuzzy_joint_score=0.0,
            fuzzy_estimated_start=None,
            current_start=6.84,
        )
        == "zero_support"
    )
