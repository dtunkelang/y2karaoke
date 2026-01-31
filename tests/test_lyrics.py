"""Comprehensive tests for lyrics processing, including LRC parsing, line splitting, and romanization."""

import pytest
from y2karaoke.core.lyrics import (
    Word,
    Line,
    SongMetadata,
    split_long_lines,
    parse_lrc_timestamp,
    parse_lrc_with_timing,
    romanize_line,
)

# ------------------------------
# Dataclass Tests
# ------------------------------


class TestWord:
    def test_word_creation(self):
        word = Word(text="hello", start_time=0.0, end_time=0.5)
        assert word.text == "hello"
        assert word.start_time == 0.0
        assert word.end_time == 0.5

    def test_word_default_singer(self):
        word = Word(text="hello", start_time=0.0, end_time=0.5)
        assert word.singer == ""

    def test_word_with_singer(self):
        word = Word(text="hello", start_time=0.0, end_time=0.5, singer="singer1")
        assert word.singer == "singer1"


class TestLine:
    def test_line_creation(self):
        words = [
            Word(text="hello", start_time=0.0, end_time=0.3),
            Word(text="world", start_time=0.3, end_time=0.6),
        ]
        line = Line(words=words)
        assert len(line.words) == 2
        assert line.start_time == 0.0
        assert line.end_time == 0.6

    def test_line_text_property(self):
        words = [
            Word(text="hello", start_time=0.0, end_time=0.3),
            Word(text="world", start_time=0.3, end_time=0.6),
        ]
        line = Line(words=words)
        assert line.text == "hello world"


class TestSongMetadata:
    def test_creation_with_singers(self):
        metadata = SongMetadata(singers=["John"])
        assert metadata.singers == ["John"]
        assert metadata.is_duet is False

    def test_empty_singers_list(self):
        metadata = SongMetadata(singers=[])
        assert metadata.singers == []

    def test_duet_with_singers(self):
        metadata = SongMetadata(singers=["John", "Jane"], is_duet=True)
        assert metadata.is_duet is True
        assert len(metadata.singers) == 2

    def test_get_singer_id(self):
        metadata = SongMetadata(singers=["Alice", "Bob"], is_duet=True)
        assert metadata.get_singer_id("Alice") == "singer1"
        assert metadata.get_singer_id("Bob") == "singer2"

    def test_get_singer_id_unknown_defaults_to_first(self):
        metadata = SongMetadata(singers=["Alice", "Bob"], is_duet=True)
        assert metadata.get_singer_id("Unknown") == "singer1"

    def test_get_singer_id_empty_string_returns_empty(self):
        metadata = SongMetadata(singers=["Alice", "Bob"], is_duet=True)
        assert metadata.get_singer_id("") == ""

    def test_get_singer_id_both_indicator(self):
        metadata = SongMetadata(singers=["Alice", "Bob"], is_duet=True)
        assert metadata.get_singer_id("Alice & Bob") == "both"


# ------------------------------
# split_long_lines Tests
# ------------------------------


class TestSplitLongLines:
    def test_short_line_unchanged(self):
        word = Word(text="short", start_time=0.0, end_time=0.5)
        line = Line(words=[word])
        result = split_long_lines([line])
        assert len(result) == 1
        assert result[0].words[0].text == "short"

    def test_empty_list_returns_empty(self):
        result = split_long_lines([])
        assert result == []

    def test_long_line_split(self):
        words = [
            Word(text=f"word{i}", start_time=i * 0.1, end_time=(i + 1) * 0.1)
            for i in range(30)
        ]
        line = Line(words=words)
        result = split_long_lines([line], max_width_ratio=0.5)
        assert len(result) >= 2

    def test_timing_preserved_after_split(self):
        words = [
            Word(text="hello", start_time=0.0, end_time=0.3),
            Word(text="world", start_time=0.3, end_time=0.6),
            Word(text="testing", start_time=0.6, end_time=1.0),
        ]
        line = Line(words=words)
        result = split_long_lines([line])
        all_words = [w for l in result for w in l.words]
        assert all_words[0].start_time == 0.0
        assert all_words[0].text == "hello"


# ------------------------------
# LRC Parsing Tests
# ------------------------------


class TestParseLrcTimestamp:
    def test_parse_simple_timestamp(self):
        result = parse_lrc_timestamp("[00:30.50]")
        assert abs(result - 30.5) < 0.01

    def test_parse_minutes_and_seconds(self):
        result = parse_lrc_timestamp("[02:15.00]")
        assert abs(result - 135.0) < 0.01

    def test_parse_zero_timestamp(self):
        result = parse_lrc_timestamp("[00:00.00]")
        assert result == 0.0

    def test_invalid_format_returns_none(self):
        result = parse_lrc_timestamp("invalid")
        assert result is None


class TestParseLrcWithTiming:
    def test_parse_lrc_text(self):
        lrc_text = "[00:30.50]Hello world\n[00:35.00]Test line"
        result = parse_lrc_with_timing(lrc_text)
        assert len(result) == 2
        assert abs(result[0][0] - 30.5) < 0.01
        assert result[0][1] == "Hello world"

    def test_empty_lrc_returns_empty(self):
        # Parser skips empty lines, so expected result is []
        result = parse_lrc_with_timing("")
        assert result == []

    def test_lines_without_text_skipped(self):
        # LRC with timestamps but empty text
        lrc_text = "[01:00.00]\n[02:00.00]"
        result = parse_lrc_with_timing(lrc_text)
        assert result == []


# ------------------------------
# Romanization Tests
# ------------------------------


class TestRomanization:
    def test_english_unchanged(self):
        result = romanize_line("Hello world")
        assert result == "Hello world"

    def test_mixed_script_handling(self):
        result = romanize_line("Hello 世界")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_string(self):
        result = romanize_line("")
        assert result == ""

    def test_numbers_unchanged(self):
        result = romanize_line("123")
        assert result == "123"
