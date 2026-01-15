"""Tests for the lyrics module."""

import pytest
from dataclasses import dataclass

from y2karaoke.core.lyrics import (
    Word,
    Line,
    SongMetadata,
    split_long_lines,
    parse_lrc_timestamp,
    parse_lrc_with_timing,
    romanize_line,
)


class TestWord:
    """Tests for the Word dataclass."""

    def test_word_creation(self):
        """Word should be created with required fields."""
        word = Word(text="hello", start_time=0.0, end_time=0.5)
        assert word.text == "hello"
        assert word.start_time == 0.0
        assert word.end_time == 0.5

    def test_word_default_singer(self):
        """Word should have empty singer by default."""
        word = Word(text="hello", start_time=0.0, end_time=0.5)
        assert word.singer == ""

    def test_word_with_singer(self):
        """Word should accept singer parameter."""
        word = Word(text="hello", start_time=0.0, end_time=0.5, singer="singer1")
        assert word.singer == "singer1"


class TestLine:
    """Tests for the Line dataclass."""

    def test_line_creation(self):
        """Line should be created with words and timing."""
        words = [
            Word(text="hello", start_time=0.0, end_time=0.3),
            Word(text="world", start_time=0.3, end_time=0.6),
        ]
        line = Line(words=words, start_time=0.0, end_time=0.6)
        assert len(line.words) == 2
        assert line.start_time == 0.0
        assert line.end_time == 0.6

    def test_line_text_property(self):
        """Line should have text property that joins words."""
        words = [
            Word(text="hello", start_time=0.0, end_time=0.3),
            Word(text="world", start_time=0.3, end_time=0.6),
        ]
        line = Line(words=words, start_time=0.0, end_time=0.6)
        assert line.text == "hello world"


class TestSongMetadata:
    """Tests for the SongMetadata dataclass."""

    def test_creation_with_singers(self):
        """SongMetadata should be created with singers list."""
        metadata = SongMetadata(singers=["John"])
        assert metadata.singers == ["John"]
        assert metadata.is_duet is False

    def test_empty_singers_list(self):
        """SongMetadata should accept empty singers list."""
        metadata = SongMetadata(singers=[])
        assert metadata.singers == []

    def test_duet_with_singers(self):
        """SongMetadata should store duet singers."""
        metadata = SongMetadata(singers=["John", "Jane"], is_duet=True)
        assert metadata.is_duet is True
        assert len(metadata.singers) == 2
        assert "John" in metadata.singers

    def test_get_singer_id(self):
        """get_singer_id should return correct singer identifier."""
        metadata = SongMetadata(singers=["Alice", "Bob"], is_duet=True)
        assert metadata.get_singer_id("Alice") == "singer1"
        assert metadata.get_singer_id("Bob") == "singer2"

    def test_get_singer_id_unknown_defaults_to_first(self):
        """get_singer_id returns singer1 for unknown names when singers exist."""
        metadata = SongMetadata(singers=["Alice", "Bob"], is_duet=True)
        # Unknown singer defaults to singer1 when singers list is not empty
        assert metadata.get_singer_id("Unknown") == "singer1"

    def test_get_singer_id_empty_string_returns_empty(self):
        """get_singer_id should return empty string for empty input."""
        metadata = SongMetadata(singers=["Alice", "Bob"], is_duet=True)
        assert metadata.get_singer_id("") == ""

    def test_get_singer_id_both_indicator(self):
        """get_singer_id should detect 'both' indicators."""
        metadata = SongMetadata(singers=["Alice", "Bob"], is_duet=True)
        assert metadata.get_singer_id("Alice & Bob") == "both"
        assert metadata.get_singer_id("Both") == "both"


class TestSplitLongLines:
    """Tests for the split_long_lines function."""

    def test_short_line_unchanged(self):
        """Short lines should not be split."""
        word = Word(text="short", start_time=0.0, end_time=0.5)
        line = Line(words=[word], start_time=0.0, end_time=0.5)

        result = split_long_lines([line])
        assert len(result) == 1
        assert result[0].words[0].text == "short"

    def test_empty_list_returns_empty(self):
        """Empty list should return empty list."""
        result = split_long_lines([])
        assert result == []

    def test_long_line_split(self):
        """Long lines should be split into multiple lines."""
        # Create a very long line
        words = [
            Word(text=f"word{i}", start_time=i * 0.1, end_time=(i + 1) * 0.1)
            for i in range(30)
        ]
        line = Line(words=words, start_time=0.0, end_time=3.0)

        result = split_long_lines([line], max_width_ratio=0.5)
        # Should have split into at least 2 lines
        assert len(result) >= 2

    def test_timing_preserved_after_split(self):
        """Word timing should be preserved after splitting."""
        words = [
            Word(text="hello", start_time=0.0, end_time=0.3),
            Word(text="world", start_time=0.3, end_time=0.6),
            Word(text="testing", start_time=0.6, end_time=1.0),
        ]
        line = Line(words=words, start_time=0.0, end_time=1.0)

        result = split_long_lines([line])
        # All words should maintain their original timing
        all_words = []
        for l in result:
            all_words.extend(l.words)

        assert all_words[0].start_time == 0.0
        assert all_words[0].text == "hello"


class TestParseLrcTimestamp:
    """Tests for LRC timestamp parsing."""

    def test_parse_simple_timestamp(self):
        """Should parse simple LRC timestamp with brackets."""
        result = parse_lrc_timestamp("[00:30.50]")
        assert abs(result - 30.5) < 0.01

    def test_parse_minutes_and_seconds(self):
        """Should parse minutes and seconds correctly."""
        result = parse_lrc_timestamp("[02:15.00]")
        assert abs(result - 135.0) < 0.01  # 2*60 + 15 = 135

    def test_parse_zero_timestamp(self):
        """Should parse zero timestamp."""
        result = parse_lrc_timestamp("[00:00.00]")
        assert result == 0.0

    def test_invalid_format_returns_zero(self):
        """Invalid format should return 0.0."""
        result = parse_lrc_timestamp("invalid")
        assert result == 0.0


class TestParseLrcWithTiming:
    """Tests for LRC line parsing with timing."""

    def test_parse_lrc_text(self):
        """Should parse LRC text into time/text tuples."""
        lrc_text = "[00:30.50]Hello world\n[00:35.00]Test line"
        result = parse_lrc_with_timing(lrc_text)
        assert len(result) == 2
        assert abs(result[0][0] - 30.5) < 0.01
        assert result[0][1] == "Hello world"

    def test_empty_lrc_returns_empty(self):
        """Empty LRC should return empty list."""
        result = parse_lrc_with_timing("")
        assert result == []


class TestRomanization:
    """Tests for romanization functionality."""

    def test_english_unchanged(self):
        """English text should remain unchanged."""
        result = romanize_line("Hello world")
        assert result == "Hello world"

    def test_mixed_script_handling(self):
        """Mixed script text should be handled gracefully."""
        # This tests that the function doesn't crash on mixed input
        result = romanize_line("Hello 世界")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_string(self):
        """Empty string should return empty string."""
        result = romanize_line("")
        assert result == ""

    def test_numbers_unchanged(self):
        """Numbers should remain unchanged."""
        result = romanize_line("123")
        assert result == "123"
