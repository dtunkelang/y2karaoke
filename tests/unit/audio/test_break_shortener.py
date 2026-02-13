"""Tests for instrumental break shortening module."""

import pytest
from unittest.mock import patch, MagicMock  # noqa: F401
import numpy as np  # noqa: F401

from y2karaoke.core.break_shortener import (
    InstrumentalBreak,
    BreakEdit,
    adjust_lyrics_timing,
    shorten_break,
)
from y2karaoke.core.models import Word, Line

# ------------------------------
# Dataclass Tests
# ------------------------------


class TestInstrumentalBreak:
    def test_creation(self):
        brk = InstrumentalBreak(start=10.0, end=30.0)
        assert brk.start == 10.0
        assert brk.end == 30.0

    def test_duration_property(self):
        brk = InstrumentalBreak(start=10.0, end=30.0)
        assert brk.duration == 20.0

    def test_zero_duration(self):
        brk = InstrumentalBreak(start=10.0, end=10.0)
        assert brk.duration == 0.0


class TestBreakEdit:
    def test_creation(self):
        edit = BreakEdit(
            original_start=10.0,
            original_end=50.0,
            new_end=20.0,
            time_removed=30.0,
        )
        assert edit.original_start == 10.0
        assert edit.original_end == 50.0
        assert edit.new_end == 20.0
        assert edit.time_removed == 30.0

    def test_original_duration_property(self):
        edit = BreakEdit(
            original_start=10.0,
            original_end=50.0,
            new_end=20.0,
            time_removed=30.0,
        )
        assert edit.original_duration == 40.0

    def test_new_duration_property(self):
        edit = BreakEdit(
            original_start=10.0,
            original_end=50.0,
            new_end=20.0,
            time_removed=30.0,
        )
        assert edit.new_duration == 10.0  # new_end - original_start

    def test_cut_start_default(self):
        edit = BreakEdit(
            original_start=10.0,
            original_end=50.0,
            new_end=20.0,
            time_removed=30.0,
        )
        assert edit.cut_start == 0.0  # Default value

    def test_cut_start_explicit(self):
        edit = BreakEdit(
            original_start=10.0,
            original_end=50.0,
            new_end=20.0,
            time_removed=30.0,
            cut_start=15.0,
        )
        assert edit.cut_start == 15.0


# ------------------------------
# Lyrics Timing Adjustment Tests
# ------------------------------


class TestAdjustLyricsTiming:
    def test_no_edits_returns_unchanged(self):
        words = [Word(text="hello", start_time=5.0, end_time=6.0)]
        lines = [Line(words=words)]
        result = adjust_lyrics_timing(lines, [])
        assert len(result) == 1
        assert result[0].words[0].start_time == 5.0
        assert result[0].words[0].end_time == 6.0

    def test_empty_lines_returns_empty(self):
        result = adjust_lyrics_timing([], [])
        assert result == []

    def test_single_edit_shifts_later_lyrics(self):
        # Lyrics before and after a break
        lines = [
            Line(words=[Word(text="before", start_time=5.0, end_time=6.0)]),
            Line(words=[Word(text="after", start_time=50.0, end_time=51.0)]),
        ]

        # Break from 10s to 45s, removing 20s
        edit = BreakEdit(
            original_start=10.0,
            original_end=45.0,
            new_end=25.0,
            time_removed=20.0,
            cut_start=15.0,  # Cut starts at 15s
        )

        result = adjust_lyrics_timing(lines, [edit])

        # "before" at 5s is before cut_start (15s), should be unchanged
        assert result[0].words[0].start_time == 5.0
        # "after" at 50s is after cut_start (15s), should shift by -20s
        assert result[1].words[0].start_time == 30.0  # 50 - 20 = 30

    def test_multiple_edits_cumulative_shift(self):
        # Lyrics after two breaks
        lines = [
            Line(words=[Word(text="after_both", start_time=100.0, end_time=101.0)]),
        ]

        edits = [
            BreakEdit(
                original_start=10.0,
                original_end=30.0,
                new_end=18.0,
                time_removed=10.0,
                cut_start=15.0,
            ),
            BreakEdit(
                original_start=50.0,
                original_end=80.0,
                new_end=58.0,
                time_removed=15.0,
                cut_start=55.0,
            ),
        ]

        result = adjust_lyrics_timing(lines, edits)

        # Word at 100s should shift by both edits (10 + 15 = 25s)
        assert result[0].words[0].start_time == 75.0  # 100 - 25 = 75

    def test_word_between_edits_only_shifted_by_earlier(self):
        # Word between two breaks
        lines = [
            Line(words=[Word(text="middle", start_time=40.0, end_time=41.0)]),
        ]

        edits = [
            BreakEdit(
                original_start=10.0,
                original_end=25.0,
                new_end=18.0,
                time_removed=5.0,
                cut_start=15.0,
            ),
            BreakEdit(
                original_start=50.0,
                original_end=70.0,
                new_end=58.0,
                time_removed=10.0,
                cut_start=55.0,
            ),
        ]

        result = adjust_lyrics_timing(lines, edits)

        # Word at 40s is after first cut (15s) but before second cut (55s)
        # Should only shift by first edit (5s)
        assert result[0].words[0].start_time == 35.0  # 40 - 5 = 35

    def test_preserves_word_text_and_singer(self):
        lines = [
            Line(
                words=[Word(text="hello", start_time=50.0, end_time=51.0, singer="s1")],
                singer="s1",
            )
        ]

        edit = BreakEdit(
            original_start=10.0,
            original_end=30.0,
            new_end=18.0,
            time_removed=10.0,
            cut_start=15.0,
        )

        result = adjust_lyrics_timing(lines, [edit])

        assert result[0].words[0].text == "hello"
        assert result[0].words[0].singer == "s1"
        assert result[0].singer == "s1"

    def test_preserves_end_time_offset(self):
        # End time should shift by same amount as start time
        lines = [Line(words=[Word(text="word", start_time=50.0, end_time=52.0)])]

        edit = BreakEdit(
            original_start=10.0,
            original_end=30.0,
            new_end=18.0,
            time_removed=10.0,
            cut_start=15.0,
        )

        result = adjust_lyrics_timing(lines, [edit])

        # Both start and end should shift by 10s
        assert result[0].words[0].start_time == 40.0  # 50 - 10
        assert result[0].words[0].end_time == 42.0  # 52 - 10
        # Duration preserved
        assert (
            result[0].words[0].end_time - result[0].words[0].start_time
        ) == pytest.approx(2.0)

    def test_empty_words_line_unchanged(self):
        lines = [Line(words=[])]
        edit = BreakEdit(
            original_start=10.0,
            original_end=30.0,
            new_end=18.0,
            time_removed=10.0,
            cut_start=15.0,
        )

        result = adjust_lyrics_timing(lines, [edit])

        assert len(result) == 1
        assert result[0].words == []

    def test_edits_sorted_by_start_time(self):
        """Edits should be processed in chronological order regardless of input order."""
        lines = [Line(words=[Word(text="word", start_time=100.0, end_time=101.0)])]

        # Provide edits in reverse order
        edits = [
            BreakEdit(
                original_start=50.0,
                original_end=70.0,
                new_end=58.0,
                time_removed=10.0,
                cut_start=55.0,
            ),
            BreakEdit(
                original_start=10.0,
                original_end=25.0,
                new_end=18.0,
                time_removed=5.0,
                cut_start=15.0,
            ),
        ]

        result = adjust_lyrics_timing(lines, edits)

        # Should shift by both (5 + 10 = 15)
        assert result[0].words[0].start_time == 85.0  # 100 - 15 = 85

    def test_uses_calculated_cut_start_when_cut_start_is_zero(self):
        """When cut_start is 0, use original_start + keep_start."""
        lines = [Line(words=[Word(text="after", start_time=50.0, end_time=51.0)])]

        # cut_start defaults to 0.0
        edit = BreakEdit(
            original_start=10.0,
            original_end=30.0,
            new_end=18.0,
            time_removed=10.0,
        )

        # With keep_start=5.0 (default), cut_start = 10.0 + 5.0 = 15.0
        result = adjust_lyrics_timing(lines, [edit], keep_start=5.0)

        # Word at 50s > calculated cut_start 15s, should shift
        assert result[0].words[0].start_time == 40.0  # 50 - 10 = 40


# ------------------------------
# Shorten Break Tests
# ------------------------------


class TestShortenBreak:
    @patch("y2karaoke.core.break_shortener.find_beat_near")
    def test_break_too_short_returns_none(self, mock_find_beat):
        """Break shorter than keep_start + keep_end + crossfade returns None."""
        brk = InstrumentalBreak(start=0.0, end=8.0)  # 8s duration

        # With keep_start=5, keep_end=3, crossfade=1, min is 9s
        result = shorten_break(
            "/fake/path.wav",
            brk,
            keep_start=5.0,
            keep_end=3.0,
            crossfade_duration=1.0,
        )

        # Should return None for cut points
        cut_start, cut_end, after_time, time_removed = result
        assert cut_start is None
        assert cut_end is None
        assert time_removed == 0.0

    @patch("y2karaoke.core.break_shortener.find_beat_near")
    def test_calculates_cut_points(self, mock_find_beat):
        """Test cut point calculation for a long break."""
        # Mock beat alignment to return same time
        mock_find_beat.side_effect = lambda path, time: time

        brk = InstrumentalBreak(start=10.0, end=60.0)  # 50s duration

        result = shorten_break(
            "/fake/path.wav",
            brk,
            keep_start=5.0,
            keep_end=3.0,
            crossfade_duration=1.0,
            align_to_beats=True,
        )

        cut_start, cut_end, after_time, time_removed = result

        # cut_out_start = 10 + 5 = 15
        # cut_out_end = 60 - 3 = 57
        # time_removed = (57 - 15) - 1 = 41
        assert cut_start == 15.0
        assert cut_end == 57.0
        assert time_removed == pytest.approx(41.0, abs=0.1)

    @patch("y2karaoke.core.break_shortener.find_beat_near")
    def test_no_beat_alignment(self, mock_find_beat):
        """Test cut points without beat alignment."""
        brk = InstrumentalBreak(start=10.0, end=30.0)  # 20s duration

        result = shorten_break(
            "/fake/path.wav",
            brk,
            keep_start=5.0,
            keep_end=3.0,
            crossfade_duration=1.0,
            align_to_beats=False,
        )

        cut_start, cut_end, after_time, time_removed = result

        # Without alignment: cut_out_start = 15, cut_out_end = 27
        assert cut_start == 15.0
        assert cut_end == 27.0
        # find_beat_near should not be called
        mock_find_beat.assert_not_called()


# ------------------------------
# Integration Tests
# ------------------------------


class TestBreakShortenerIntegration:
    def test_full_lyrics_adjustment_flow(self):
        """Test complete flow of detecting breaks and adjusting lyrics."""
        # Simulate a song with lyrics and an instrumental break

        # Lyrics: verse at 0-10s, break from 10-40s, chorus at 40-60s
        lines = [
            Line(
                words=[
                    Word(text="verse", start_time=0.0, end_time=2.0),
                    Word(text="line", start_time=2.0, end_time=4.0),
                ]
            ),
            Line(
                words=[
                    Word(text="chorus", start_time=40.0, end_time=42.0),
                    Word(text="starts", start_time=42.0, end_time=44.0),
                ]
            ),
        ]

        # Break edit: removed 20s from the 30s break
        edit = BreakEdit(
            original_start=10.0,
            original_end=40.0,
            new_end=20.0,
            time_removed=20.0,
            cut_start=15.0,  # Cut starts 5s into break
        )

        result = adjust_lyrics_timing(lines, [edit])

        # Verse should be unchanged (before cut_start at 15s)
        assert result[0].words[0].start_time == 0.0
        assert result[0].words[1].start_time == 2.0

        # Chorus should shift by 20s
        assert result[1].words[0].start_time == 20.0  # 40 - 20
        assert result[1].words[1].start_time == 22.0  # 42 - 20
