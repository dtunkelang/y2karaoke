from y2karaoke.core.components.alignment.timing_models import TranscriptionWord
from y2karaoke.core.components.whisper.whisper_assignment_confidence import (
    build_assignment_confidence_profile,
)
from y2karaoke.core.components.whisper.whisper_mapping_pipeline_orchestration import (
    _line_override_segment_votes,
)
from y2karaoke.core.models import Line, Word


def test_assignment_confidence_flags_placeholder_drifted_profile() -> None:
    line = Line(
        words=[
            Word(text="Sin", start_time=50.72, end_time=51.08),
            Word(text="City's", start_time=51.10, end_time=51.46),
            Word(text="cold", start_time=51.48, end_time=51.84),
            Word(text="and", start_time=51.86, end_time=52.22),
            Word(text="empty", start_time=52.24, end_time=52.60),
        ]
    )
    all_words = [
        TranscriptionWord(text="[VOCAL]", start=61.66, end=61.8, probability=0.9),
        TranscriptionWord(text="[VOCAL]", start=62.66, end=62.8, probability=0.9),
        TranscriptionWord(text="[VOCAL]", start=64.16, end=64.3, probability=0.9),
        TranscriptionWord(text="[VOCAL]", start=65.16, end=65.3, probability=0.9),
        TranscriptionWord(text="[VOCAL]", start=66.66, end=66.8, probability=0.9),
    ]
    profile = build_assignment_confidence_profile(
        line_idx=0,
        line=line,
        lrc_index_by_loc={(0, i): i for i in range(5)},
        lrc_assignments={i: [i] for i in range(5)},
        all_words=all_words,
        word_segment_idx={i: 3 for i in range(5)},
    )

    assert profile.total_assigned == 5
    assert profile.placeholder_ratio == 1.0
    assert profile.lexical_overlap_ratio == 0.0
    assert profile.low_confidence is True


def test_assignment_confidence_keeps_reasonable_lexical_profile() -> None:
    line = Line(
        words=[
            Word(text="I", start_time=117.70, end_time=118.10),
            Word(text="said,", start_time=118.10, end_time=118.50),
            Word(text="lights", start_time=121.20, end_time=121.97),
        ]
    )
    all_words = [
        TranscriptionWord(text="I", start=117.73, end=118.0, probability=0.9),
        TranscriptionWord(text="said", start=118.05, end=118.4, probability=0.9),
        TranscriptionWord(text="lights", start=121.18, end=121.9, probability=0.9),
    ]
    profile = build_assignment_confidence_profile(
        line_idx=0,
        line=line,
        lrc_index_by_loc={(0, i): i for i in range(3)},
        lrc_assignments={i: [i] for i in range(3)},
        all_words=all_words,
        word_segment_idx={i: 7 for i in range(3)},
    )

    assert profile.lexical_overlap_ratio == 1.0
    assert profile.placeholder_ratio == 0.0
    assert profile.low_confidence is False


def test_line_override_votes_still_count_segments() -> None:
    line = Line(
        words=[
            Word(text="Sin", start_time=50.72, end_time=51.08),
            Word(text="City's", start_time=51.10, end_time=51.46),
            Word(text="cold", start_time=51.48, end_time=51.84),
            Word(text="and", start_time=51.86, end_time=52.22),
            Word(text="empty", start_time=52.24, end_time=52.60),
        ]
    )
    all_words = [
        TranscriptionWord(text="[VOCAL]", start=61.66, end=61.8, probability=0.9),
        TranscriptionWord(text="[VOCAL]", start=62.66, end=62.8, probability=0.9),
        TranscriptionWord(text="[VOCAL]", start=64.16, end=64.3, probability=0.9),
        TranscriptionWord(text="[VOCAL]", start=65.16, end=65.3, probability=0.9),
        TranscriptionWord(text="[VOCAL]", start=66.66, end=66.8, probability=0.9),
    ]
    profile = build_assignment_confidence_profile(
        line_idx=0,
        line=line,
        lrc_index_by_loc={(0, i): i for i in range(5)},
        lrc_assignments={i: [i] for i in range(5)},
        all_words=all_words,
        word_segment_idx={i: 3 for i in range(5)},
    )

    votes = _line_override_segment_votes(
        line_idx=0,
        line=line,
        lrc_index_by_loc={(0, i): i for i in range(5)},
        lrc_assignments={i: [i] for i in range(5)},
        word_segment_idx={i: 3 for i in range(5)},
    )

    assert profile.low_confidence is True
    assert votes == {3: 5}
