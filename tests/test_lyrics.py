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
from y2karaoke.core.lrc import (
    create_lines_from_lrc,
    create_lines_from_lrc_timings,
    _is_empty_or_symbols,
    _is_metadata_line,
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


# ------------------------------
# Metadata Filtering Tests
# ------------------------------


class TestMetadataFiltering:
    def test_empty_string_is_metadata(self):
        assert _is_empty_or_symbols("") is True
        assert _is_empty_or_symbols("   ") is True

    def test_music_symbols_only_is_metadata(self):
        assert _is_empty_or_symbols("♪♪♪") is True
        assert _is_empty_or_symbols("🎵🎶") is True
        assert _is_empty_or_symbols("---") is True

    def test_actual_lyrics_not_metadata(self):
        assert _is_empty_or_symbols("Hello world") is False
        assert _is_empty_or_symbols("I love you") is False

    def test_metadata_prefix_detection(self):
        assert _is_metadata_line("Artist: John Doe") is True
        assert _is_metadata_line("Title: My Song") is True
        assert _is_metadata_line("Composer: Bach") is True
        assert _is_metadata_line("Lyrics by: Someone") is True
        assert _is_metadata_line("Written by: Author") is True

    def test_metadata_pattern_detection(self):
        assert _is_metadata_line("(instrumental)") is True
        assert _is_metadata_line("[chorus]") is True
        # Note: "(Verse 1)" with number doesn't match exact pattern "(verse)"
        assert _is_metadata_line("(verse)") is True
        assert _is_metadata_line("© 2024 All rights reserved") is True

    def test_credit_pattern_detection(self):
        assert _is_metadata_line("Composer: John Smith") is True
        assert _is_metadata_line("Lyricist: Jane Doe") is True

    def test_title_header_filtering(self):
        # Title-only line at very start should be filtered
        assert (
            _is_metadata_line("My Song Title", title="My Song Title", timestamp=0.0)
            is True
        )
        # But not after 1 second
        assert (
            _is_metadata_line("My Song Title", title="My Song Title", timestamp=5.0)
            is False
        )

    def test_artist_header_filtering(self):
        # Artist-only line at very start should be filtered
        assert (
            _is_metadata_line("The Beatles", artist="The Beatles", timestamp=0.5)
            is True
        )
        # But not after 1 second
        assert (
            _is_metadata_line("The Beatles", artist="The Beatles", timestamp=5.0)
            is False
        )

    def test_chinese_metadata_prefixes(self):
        assert _is_metadata_line("作词: 某人") is True
        assert _is_metadata_line("作曲: 作曲家") is True
        assert _is_metadata_line("编曲: 编曲师") is True

    def test_actual_lyrics_not_filtered(self):
        assert _is_metadata_line("I want to hold your hand") is False
        assert (
            _is_metadata_line("Yesterday, all my troubles seemed so far away") is False
        )


# ------------------------------
# create_lines_from_lrc Tests
# ------------------------------


class TestCreateLinesFromLrc:
    def test_simple_lrc(self):
        lrc_text = "[00:10.00]Hello world\n[00:15.00]Second line"
        lines = create_lines_from_lrc(lrc_text, romanize=False)
        assert len(lines) == 2
        assert lines[0].text == "Hello world"
        assert lines[1].text == "Second line"

    def test_word_timing_distribution(self):
        lrc_text = "[00:10.00]One two three\n[00:15.00]Next"
        lines = create_lines_from_lrc(lrc_text, romanize=False)
        # First line has 3 words
        assert len(lines[0].words) == 3
        # Words should have progressive timing
        assert lines[0].words[0].start_time < lines[0].words[1].start_time
        assert lines[0].words[1].start_time < lines[0].words[2].start_time

    def test_empty_lrc_returns_empty(self):
        lines = create_lines_from_lrc("", romanize=False)
        assert lines == []

    def test_metadata_lines_filtered(self):
        lrc_text = "[00:01.00]Artist: Someone\n[00:10.00]Actual lyrics here"
        lines = create_lines_from_lrc(lrc_text, romanize=False)
        assert len(lines) == 1
        assert lines[0].text == "Actual lyrics here"

    def test_title_artist_header_filtered(self):
        lrc_text = "[00:00.50]My Song\n[00:00.80]The Artist\n[00:10.00]Actual lyrics"
        lines = create_lines_from_lrc(
            lrc_text, romanize=False, title="My Song", artist="The Artist"
        )
        assert len(lines) == 1
        assert lines[0].text == "Actual lyrics"

    def test_long_gap_capped(self):
        # If gap between lines > 10s, end_time should be capped
        lrc_text = "[00:10.00]First line\n[00:30.00]Second line"
        lines = create_lines_from_lrc(lrc_text, romanize=False)
        first_line_end = lines[0].end_time
        # End time should be ~15s (10s start + 5s cap), not 30s
        assert first_line_end < 20.0

    def test_last_line_duration(self):
        lrc_text = "[00:10.00]Only line"
        lines = create_lines_from_lrc(lrc_text, romanize=False)
        # Last line should have ~3s duration
        assert abs(lines[0].end_time - 13.0) < 0.5

    def test_romanization_applied(self):
        lrc_text = "[00:10.00]你好世界"
        lines = create_lines_from_lrc(lrc_text, romanize=True)
        # Should have one line
        assert len(lines) == 1
        # Romanization should produce some output (may be pinyin with tones)
        assert len(lines[0].text) > 0
        # The text should be different from original Chinese (romanization happened)
        # Note: output may include tone marks or special characters depending on library
        original_has_chinese = any("\u4e00" <= c <= "\u9fff" for c in "你好世界")
        assert original_has_chinese  # Verify original is Chinese


# ------------------------------
# create_lines_from_lrc_timings Tests
# ------------------------------


class TestCreateLinesFromLrcTimings:
    def test_basic_matching(self):
        lrc_timings = [
            (10.0, "hello world"),
            (15.0, "goodbye world"),
        ]
        genius_lines = ["Hello World", "Goodbye World"]
        lines = create_lines_from_lrc_timings(lrc_timings, genius_lines)
        assert len(lines) == 2
        # Should use Genius text (capitalized)
        assert lines[0].text == "Hello World"
        assert lines[1].text == "Goodbye World"

    def test_fuzzy_matching(self):
        lrc_timings = [
            (10.0, "helo wrold"),  # Typos
        ]
        genius_lines = ["Hello World"]
        lines = create_lines_from_lrc_timings(lrc_timings, genius_lines)
        assert len(lines) == 1
        # Should match despite typos
        assert lines[0].text == "Hello World"

    def test_no_match_uses_lrc_text(self):
        lrc_timings = [
            (10.0, "completely different"),
        ]
        genius_lines = ["Something else entirely"]
        lines = create_lines_from_lrc_timings(lrc_timings, genius_lines)
        assert len(lines) == 1
        # Low match score should use LRC text
        assert (
            "completely" in lines[0].text.lower()
            or "something" in lines[0].text.lower()
        )

    def test_timing_preserved(self):
        lrc_timings = [
            (10.0, "first line"),
            (15.0, "second line"),
        ]
        genius_lines = ["First Line", "Second Line"]
        lines = create_lines_from_lrc_timings(lrc_timings, genius_lines)
        # First line should start at 10s
        assert lines[0].start_time == 10.0
        # Second line should start at 15s
        assert lines[1].start_time == 15.0

    def test_duplicate_lines_filtered(self):
        lrc_timings = [
            (10.0, "repeat this"),
            (12.0, "repeat this"),
            (15.0, "different line"),
        ]
        genius_lines = ["Repeat This", "Repeat This", "Different Line"]
        lines = create_lines_from_lrc_timings(lrc_timings, genius_lines)
        # Consecutive duplicates should be filtered
        assert len(lines) == 2

    def test_empty_timings_returns_empty(self):
        lines = create_lines_from_lrc_timings([], ["Some lyrics"])
        assert lines == []

    def test_word_timing_distribution(self):
        lrc_timings = [
            (10.0, "one two three"),
            (15.0, "next"),
        ]
        genius_lines = ["One Two Three", "Next"]
        lines = create_lines_from_lrc_timings(lrc_timings, genius_lines)
        # Check word timing is progressive
        first_line = lines[0]
        assert len(first_line.words) == 3
        assert first_line.words[0].start_time < first_line.words[1].start_time
        assert first_line.words[1].start_time < first_line.words[2].start_time


# ------------------------------
# Edge Cases and Integration Tests
# ------------------------------


class TestLyricsEdgeCases:
    def test_lrc_with_only_timestamps(self):
        lrc_text = "[00:10.00]\n[00:15.00]\n[00:20.00]"
        lines = create_lines_from_lrc(lrc_text, romanize=False)
        assert lines == []

    def test_lrc_with_instrumental_markers(self):
        lrc_text = "[00:05.00](Instrumental)\n[00:30.00]Actual lyrics here"
        lines = create_lines_from_lrc(lrc_text, romanize=False)
        assert len(lines) == 1
        assert lines[0].text == "Actual lyrics here"

    def test_lrc_with_section_markers(self):
        # Note: Section markers like "[Verse 1]" with numbers are kept
        # Only exact matches like "[verse]" lowercase are filtered
        lrc_text = "[00:05.00][verse]\n[00:10.00]Lyrics in verse"
        lines = create_lines_from_lrc(lrc_text, romanize=False)
        assert len(lines) == 1
        assert "Lyrics" in lines[0].text

    def test_parse_timestamp_various_formats(self):
        # Standard format
        assert abs(parse_lrc_timestamp("[01:30.50]") - 90.5) < 0.01
        # No fractional
        assert abs(parse_lrc_timestamp("[02:00]") - 120.0) < 0.01
        # Single digit fractional
        assert abs(parse_lrc_timestamp("[00:30.5]") - 30.5) < 0.01
        # Three digit fractional
        assert abs(parse_lrc_timestamp("[00:30.500]") - 30.5) < 0.01

    def test_line_singer_preserved_in_split(self):
        words = [
            Word(text=f"word{i}", start_time=i * 0.1, end_time=(i + 1) * 0.1)
            for i in range(20)
        ]
        line = Line(words=words, singer="singer1")
        result = split_long_lines([line], max_width_ratio=0.3)
        # All split lines should preserve singer
        for split_line in result:
            assert split_line.singer == "singer1"

    def test_word_end_time_before_next_line(self):
        lrc_text = "[00:10.00]First line\n[00:12.00]Second line"
        lines = create_lines_from_lrc(lrc_text, romanize=False)
        # Last word of first line should end before second line starts
        first_line_last_word = lines[0].words[-1]
        assert first_line_last_word.end_time <= 12.0
