"""Tests for text utility functions."""

import pytest  # noqa: F401
from y2karaoke.core.text_utils import (
    make_slug,
    clean_title_for_search,
    strip_leading_artist_from_line,
    filter_singer_only_lines,
    normalize_ocr_line,
    normalize_ocr_tokens,
)


class TestMakeSlug:
    def test_basic_lowercase(self):
        assert make_slug("Hello World") == "hello-world"

    def test_removes_special_characters(self):
        assert make_slug("Hello! World?") == "hello-world"

    def test_handles_unicode(self):
        # Unicode normalization
        result = make_slug("café")
        assert "cafe" in result

    def test_multiple_spaces_become_single_dash(self):
        assert make_slug("hello   world") == "hello-world"

    def test_strips_leading_trailing_dashes(self):
        assert make_slug("--hello--") == "hello"

    def test_empty_string(self):
        assert make_slug("") == ""

    def test_preserves_numbers(self):
        assert make_slug("Song 123") == "song-123"

    def test_preserves_hyphens(self):
        assert make_slug("hello-world") == "hello-world"


class TestCleanTitleForSearch:
    def test_removes_patterns(self):
        patterns = [r"\(Official Video\)", r"\[HD\]"]
        result = clean_title_for_search(
            "Song Name (Official Video) [HD]",
            patterns,
            [],
        )
        assert result == "Song Name"

    def test_removes_youtube_suffixes(self):
        suffixes = [" - Topic", " VEVO"]
        result = clean_title_for_search(
            "Artist - Topic",
            [],
            suffixes,
        )
        assert result == "Artist"

    def test_case_insensitive_pattern_removal(self):
        patterns = [r"\(official\)"]
        result = clean_title_for_search(
            "Song (OFFICIAL)",
            patterns,
            [],
        )
        assert result == "Song"

    def test_preserves_unmatched_text(self):
        result = clean_title_for_search(
            "Normal Song Title",
            [r"\(Video\)"],
            [" - Topic"],
        )
        assert result == "Normal Song Title"

    def test_empty_patterns_and_suffixes(self):
        result = clean_title_for_search(
            "Song Title",
            [],
            [],
        )
        assert result == "Song Title"


class TestStripLeadingArtistFromLine:
    def test_strips_bracketed_artist(self):
        result = strip_leading_artist_from_line("[Artist] Hello world", "Artist")
        assert result == "Hello world"

    def test_strips_artist_with_dash(self):
        result = strip_leading_artist_from_line("Artist - Hello world", "Artist")
        assert result == "Hello world"

    def test_strips_artist_with_en_dash(self):
        result = strip_leading_artist_from_line("Artist – Hello world", "Artist")
        assert result == "Hello world"

    def test_case_insensitive(self):
        result = strip_leading_artist_from_line("[ARTIST] Hello world", "Artist")
        assert result == "Hello world"

    def test_empty_artist_returns_unchanged(self):
        result = strip_leading_artist_from_line("Hello world", "")
        assert result == "Hello world"

    def test_no_match_returns_unchanged(self):
        result = strip_leading_artist_from_line("Hello world", "Other")
        assert result == "Hello world"

    def test_artist_not_at_start_unchanged(self):
        result = strip_leading_artist_from_line("Hello Artist world", "Artist")
        assert result == "Hello Artist world"


class TestFilterSingerOnlyLines:
    def test_filters_singer_name_only_lines(self):
        lines = [
            ("Hello world", ""),
            ("Alice", ""),  # Singer name only
            ("Goodbye world", ""),
        ]
        result = filter_singer_only_lines(lines, ["Alice", "Bob"])
        # "Hello world" and "Goodbye world" should pass
        assert len(result) == 2
        assert result[0][0] == "Hello world"
        assert result[1][0] == "Goodbye world"

    def test_keeps_non_singer_lines(self):
        lines = [
            ("This is a lyric", "Singer1"),
            ("Another lyric", "Singer2"),
        ]
        result = filter_singer_only_lines(lines, ["Singer1", "Singer2"])
        assert len(result) == 2

    def test_handles_comma_separated_singers(self):
        lines = [
            ("Alice, Bob", ""),  # Both are singers
            ("Real lyrics here", ""),
        ]
        result = filter_singer_only_lines(lines, ["Alice", "Bob"])
        assert len(result) == 1
        assert result[0][0] == "Real lyrics here"

    def test_handles_slash_separated_singers(self):
        lines = [
            ("Alice/Bob", ""),  # Both are singers
            ("Real lyrics here", ""),
        ]
        result = filter_singer_only_lines(lines, ["Alice", "Bob"])
        assert len(result) == 1
        assert result[0][0] == "Real lyrics here"

    def test_empty_lines_list(self):
        result = filter_singer_only_lines([], ["Singer"])
        assert result == []

    def test_empty_singers_list(self):
        lines = [("Hello", ""), ("World", "")]
        # With no known singers, nothing is filtered
        result = filter_singer_only_lines(lines, [])
        assert len(result) == 2

    def test_case_insensitive_singer_match(self):
        lines = [
            ("ALICE", ""),  # Uppercase singer name
            ("Real lyrics", ""),
        ]
        result = filter_singer_only_lines(lines, ["Alice"])
        assert len(result) == 1
        assert result[0][0] == "Real lyrics"


class TestNormalizeOcrTokens:
    def test_merges_contraction_fragments(self):
        assert normalize_ocr_tokens(["you", "'", "re"]) == ["you're"]
        assert normalize_ocr_tokens(["don", "'", "t"]) == ["don't"]
        assert normalize_ocr_tokens(["I", "'", "m"]) == ["I'm"]
        assert normalize_ocr_tokens(["sleepin", "'"]) == ["sleepin'"]

    def test_preserves_regular_tokens(self):
        assert normalize_ocr_tokens(["White", "shirt", "now", "red"]) == [
            "White",
            "shirt",
            "now",
            "red",
        ]

    def test_repairs_you_start_artifacts(self):
        assert normalize_ocr_tokens(["'", "ou", "come", "over"]) == [
            "you",
            "come",
            "over",
        ]
        assert normalize_ocr_tokens(["ou", "know", "I", "want"]) == [
            "you",
            "know",
            "I",
            "want",
        ]
        assert normalize_ocr_tokens(["start", "tup", "a", "conversation"]) == [
            "start",
            "up",
            "a",
            "conversation",
        ]

    def test_repairs_common_fast_phrase_errors(self):
        assert normalize_ocr_tokens(["doing", "shots", "dinking", "fast"]) == [
            "doing",
            "shots",
            "drinking",
            "fast",
        ]
        assert normalize_ocr_tokens(["doing", "shots", "dilnking", "fast"]) == [
            "doing",
            "shots",
            "drinking",
            "fast",
        ]
        assert normalize_ocr_tokens(["come", "ony", "now"]) == ["come", "on", "now"]
        assert normalize_ocr_tokens(["come", "om", "baby"]) == ["come", "on", "baby"]

    def test_repairs_short_token_confusions_in_context(self):
        assert normalize_ocr_tokens(["girl", "you", "know", "l", "want"]) == [
            "girl",
            "you",
            "know",
            "I",
            "want",
        ]
        assert normalize_ocr_tokens(["in", "loh", "with", "your", "loh"]) == [
            "in",
            "love",
            "with",
            "your",
            "body",
        ]
        assert normalize_ocr_tokens(["Oh", "I", "loh", "l", "oh", "I"]) == [
            "Oh",
            "I",
            "oh",
            "I",
            "oh",
            "I",
        ]


class TestNormalizeOcrLine:
    def test_splits_confusable_chant_runs(self):
        assert normalize_ocr_line("Oh Ioh loh l") == "Oh I oh I oh I"
        assert normalize_ocr_line("oh lohlohlohl") in {
            "oh I oh I oh I",
            "oh I oh I oh I oh I",
        }

    def test_leaves_non_chant_tokens_unchanged(self):
        assert normalize_ocr_line("with your body") == "with your body"
