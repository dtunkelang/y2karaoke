import pytest

from y2karaoke.core.components.alignment.timing_models import TranscriptionWord
from y2karaoke.core.components.whisper import (
    whisper_integration_forced_fallback as _forced,
)
from y2karaoke.core.components.whisper.whisper_forced_prefix_repairs import (
    reanchor_medium_lines_to_earlier_exact_prefixes,
)
from y2karaoke.core.models import Line, Word


def _dur_multi_line(start: float, end: float, tokens: list[str]) -> Line:
    step = (end - start) / max(len(tokens), 1)
    words = [
        Word(
            text=token,
            start_time=start + step * idx,
            end_time=start + step * (idx + 1),
        )
        for idx, token in enumerate(tokens)
    ]
    return Line(words=words)


def test_reanchor_medium_lines_to_earlier_exact_prefixes_hits_taste_shape_only():
    forced_lines = [
        _dur_multi_line(
            6.772,
            10.663,
            [
                "You'll",
                "just",
                "have",
                "to",
                "taste",
                "me",
                "when",
                "he's",
                "kissin'",
                "you",
            ],
        ),
        _dur_multi_line(
            11.377,
            14.753,
            ["If", "you", "want", "forever,", "I", "bet", "you", "do"],
        ),
        _dur_multi_line(
            15.044,
            17.512,
            ["Just", "know", "you'll", "taste", "me", "too"],
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="you'll", start=6.42, end=6.92, probability=0.99),
        TranscriptionWord(text="just", start=6.92, end=7.38, probability=0.99),
        TranscriptionWord(text="have", start=7.38, end=7.66, probability=0.99),
        TranscriptionWord(text="you", start=10.16, end=10.56, probability=0.99),
        TranscriptionWord(text="if", start=10.74, end=11.2, probability=0.99),
        TranscriptionWord(text="you", start=11.2, end=11.64, probability=0.99),
        TranscriptionWord(text="want", start=11.64, end=11.9, probability=0.99),
        TranscriptionWord(text="just", start=14.8, end=15.06, probability=0.99),
        TranscriptionWord(text="know", start=15.06, end=15.34, probability=0.99),
        TranscriptionWord(text="you'll", start=15.34, end=15.88, probability=0.99),
    ]

    repaired_lines, restored_count = reanchor_medium_lines_to_earlier_exact_prefixes(
        forced_lines,
        whisper_words,
        normalize_token_fn=_forced._normalize_token,
        can_apply_reanchored_line_fn=_forced._can_apply_reanchored_line,
    )

    assert restored_count == 1
    assert repaired_lines[0].start_time == pytest.approx(6.772)
    assert repaired_lines[1].start_time == pytest.approx(10.877)
    assert repaired_lines[1].end_time == pytest.approx(14.753)
    assert repaired_lines[2].start_time == pytest.approx(15.044)


def test_reanchor_medium_lines_to_earlier_exact_prefixes_requires_boundary_carry_over():
    forced_lines = [
        _dur_multi_line(
            5.778,
            10.755,
            [
                "I",
                "cut",
                "my",
                "teeth",
                "on",
                "wedding",
                "rings",
                "in",
                "the",
                "movies",
            ],
        ),
        _dur_multi_line(
            11.489,
            14.823,
            ["And", "I'm", "not", "proud", "of", "my", "address"],
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="And", start=11.02, end=11.42, probability=0.67),
        TranscriptionWord(text="I'm", start=11.42, end=11.76, probability=0.98),
        TranscriptionWord(text="not", start=11.76, end=12.0, probability=0.99),
        TranscriptionWord(text="proud", start=12.0, end=12.38, probability=0.99),
    ]

    repaired_lines, restored_count = reanchor_medium_lines_to_earlier_exact_prefixes(
        forced_lines,
        whisper_words,
        normalize_token_fn=_forced._normalize_token,
        can_apply_reanchored_line_fn=_forced._can_apply_reanchored_line,
    )

    assert restored_count == 0
    assert repaired_lines[1].start_time == pytest.approx(11.489)


def test_reanchor_medium_lines_to_earlier_exact_prefixes_requires_tight_boundary_gap():
    forced_lines = [
        _dur_multi_line(
            5.778,
            10.755,
            [
                "I",
                "cut",
                "my",
                "teeth",
                "on",
                "wedding",
                "rings",
                "in",
                "the",
                "movies",
            ],
        ),
        _dur_multi_line(
            11.489,
            14.823,
            ["And", "I'm", "not", "proud", "of", "my", "address"],
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="movies", start=9.26, end=10.0, probability=0.99),
        TranscriptionWord(text="And", start=11.02, end=11.42, probability=0.67),
        TranscriptionWord(text="I'm", start=11.42, end=11.76, probability=0.98),
        TranscriptionWord(text="not", start=11.76, end=12.0, probability=0.99),
    ]

    repaired_lines, restored_count = reanchor_medium_lines_to_earlier_exact_prefixes(
        forced_lines,
        whisper_words,
        normalize_token_fn=_forced._normalize_token,
        can_apply_reanchored_line_fn=_forced._can_apply_reanchored_line,
    )

    assert restored_count == 0
    assert repaired_lines[1].start_time == pytest.approx(11.489)


def test_retime_three_word_lines_from_suffix_matches_rebuilds_late_suffix_window():
    forced_lines = [
        _dur_multi_line(7.01, 7.86, ["Shady's", "back"]),
        Line(
            words=[
                Word(text="Tell", start_time=7.76, end_time=9.487),
                Word(text="a", start_time=9.547, end_time=9.668),
                Word(text="friend", start_time=9.708, end_time=9.989),
            ]
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="tell", start=8.26, end=8.6, probability=0.99),
        TranscriptionWord(text="a", start=8.6, end=8.73, probability=0.99),
        TranscriptionWord(text="friend", start=8.73, end=9.21, probability=0.99),
    ]

    repaired_lines, restored_count = (
        _forced._retime_three_word_lines_from_suffix_matches(
            forced_lines,
            whisper_words,
        )
    )

    assert restored_count == 1
    assert repaired_lines[1].start_time == pytest.approx(8.3, abs=0.05)
    assert repaired_lines[1].words[1].start_time == pytest.approx(8.6, abs=0.05)
    assert repaired_lines[1].end_time == pytest.approx(9.989, abs=0.05)


def test_retime_three_word_lines_from_suffix_matches_skips_balanced_refrain_lines():
    forced_lines = [
        _dur_multi_line(0.5, 1.55, ["Guess", "who's", "back?"]),
    ]
    whisper_words = [
        TranscriptionWord(text="guess", start=0.5, end=0.82, probability=0.99),
        TranscriptionWord(text="who's", start=0.82, end=1.12, probability=0.99),
        TranscriptionWord(text="back", start=1.12, end=1.38, probability=0.99),
    ]

    repaired_lines, restored_count = (
        _forced._retime_three_word_lines_from_suffix_matches(
            forced_lines,
            whisper_words,
        )
    )

    assert restored_count == 0
    assert repaired_lines[0].start_time == pytest.approx(forced_lines[0].start_time)
    assert repaired_lines[0].end_time == pytest.approx(forced_lines[0].end_time)
