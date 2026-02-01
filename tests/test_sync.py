"""Tests for sync module - LRC validation and duration calculation."""

import pytest
from y2karaoke.core.sync import (
    _has_timestamps,
    get_lrc_duration,
    validate_lrc_quality,
    _count_large_gaps,
)


class TestHasTimestamps:
    def test_valid_lrc_with_timestamps(self):
        lrc = "[00:30.50]Hello world\n[00:35.00]Second line"
        assert _has_timestamps(lrc) is True

    def test_lrc_without_timestamps(self):
        lrc = "Hello world\nSecond line"
        assert _has_timestamps(lrc) is False

    def test_empty_string(self):
        assert _has_timestamps("") is False

    def test_none_value(self):
        assert _has_timestamps(None) is False

    def test_various_timestamp_formats(self):
        # Standard format mm:ss.xx
        assert _has_timestamps("[01:30.50]Text") is True
        # Format with colon instead of period
        assert _has_timestamps("[01:30:50]Text") is True
        # Single digit minutes
        assert _has_timestamps("[1:30.50]Text") is True
        # Three digit milliseconds
        assert _has_timestamps("[00:30.500]Text") is True

    def test_metadata_only_not_counted(self):
        # LRC metadata tags look like timestamps but aren't lyrics timestamps
        # This tests that real timestamps are detected
        lrc = "[ti:Title]\n[ar:Artist]\n[00:30.00]Actual lyrics"
        assert _has_timestamps(lrc) is True


class TestGetLrcDuration:
    def test_simple_lrc(self):
        lrc = "[00:10.00]First line\n[00:30.00]Second line\n[01:00.00]Third line"
        duration = get_lrc_duration(lrc)
        # Last timestamp is 60s, plus ~10% buffer (6s) = ~66s, but minimum 3s buffer
        assert duration is not None
        assert 60 <= duration <= 70

    def test_empty_lrc(self):
        assert get_lrc_duration("") is None

    def test_lrc_without_timestamps(self):
        lrc = "Plain text\nNo timestamps"
        assert get_lrc_duration(lrc) is None

    def test_single_timestamp(self):
        # Need at least 2 timestamps to calculate duration
        lrc = "[00:30.00]Only one line"
        assert get_lrc_duration(lrc) is None

    def test_long_song(self):
        # 4 minute song
        lrc = "[00:10.00]Start\n[02:00.00]Middle\n[04:00.00]End"
        duration = get_lrc_duration(lrc)
        assert duration is not None
        # Last ts is 240s, span is 230s, buffer ~23s, so ~263s
        assert 240 <= duration <= 280

    def test_short_song(self):
        lrc = "[00:05.00]Start\n[00:30.00]End"
        duration = get_lrc_duration(lrc)
        assert duration is not None
        # Short span uses minimum 3s buffer
        assert 30 <= duration <= 40


class TestValidateLrcQuality:
    def test_valid_lrc(self):
        # Build a valid LRC with good density
        lines = [f"[00:{i*5:02d}.00]Line {i}" for i in range(20)]
        lrc = "\n".join(lines)
        is_valid, reason = validate_lrc_quality(lrc)
        assert is_valid is True
        assert reason == ""

    def test_empty_lrc(self):
        is_valid, reason = validate_lrc_quality("")
        assert is_valid is False
        assert "No timestamps" in reason

    def test_too_few_lines(self):
        lrc = "[00:10.00]Line 1\n[00:20.00]Line 2\n[00:30.00]Line 3"
        is_valid, reason = validate_lrc_quality(lrc)
        assert is_valid is False
        assert "Too few" in reason

    def test_span_too_short(self):
        # All timestamps within 10 seconds
        lines = [f"[00:{i:02d}.00]Line {i}" for i in range(10)]
        lrc = "\n".join(lines)
        is_valid, reason = validate_lrc_quality(lrc)
        assert is_valid is False
        assert "too short" in reason.lower()

    def test_low_density(self):
        # Lines spread too far apart (more than 15s average gap)
        lrc = (
            "[00:00.00]Line 1\n"
            "[00:30.00]Line 2\n"
            "[01:00.00]Line 3\n"
            "[01:30.00]Line 4\n"
            "[02:00.00]Line 5\n"
            "[02:30.00]Line 6"
        )
        is_valid, reason = validate_lrc_quality(lrc)
        assert is_valid is False
        assert "density" in reason.lower()

    def test_duration_coverage_check(self):
        # LRC covers 100s but expected duration is 300s (33% coverage)
        lines = [f"[00:{i*5:02d}.00]Line {i}" for i in range(20)]
        lrc = "\n".join(lines)  # Spans 0-95s
        is_valid, reason = validate_lrc_quality(lrc, expected_duration=300)
        assert is_valid is False
        assert "covers only" in reason.lower()

    def test_good_coverage(self):
        # LRC covers most of expected duration
        lines = [f"[{i//60:02d}:{i%60:02d}.00]Line {i}" for i in range(0, 180, 5)]
        lrc = "\n".join(lines)  # Spans 0-175s
        is_valid, reason = validate_lrc_quality(lrc, expected_duration=200)
        assert is_valid is True


class TestCountLargeGaps:
    def test_no_gaps(self):
        timings = [(0.0, "a"), (5.0, "b"), (10.0, "c"), (15.0, "d")]
        assert _count_large_gaps(timings) == 0

    def test_one_large_gap(self):
        timings = [(0.0, "a"), (10.0, "b"), (50.0, "c"), (55.0, "d")]
        assert _count_large_gaps(timings) == 1  # 40s gap between b and c

    def test_multiple_large_gaps(self):
        timings = [(0.0, "a"), (50.0, "b"), (100.0, "c")]
        assert _count_large_gaps(timings) == 2

    def test_custom_threshold(self):
        timings = [(0.0, "a"), (20.0, "b"), (50.0, "c")]
        # Default 30s threshold: 1 gap (30s between b and c)
        assert _count_large_gaps(timings, threshold=30.0) == 1
        # 15s threshold: 2 gaps (20s and 30s)
        assert _count_large_gaps(timings, threshold=15.0) == 2

    def test_empty_list(self):
        assert _count_large_gaps([]) == 0

    def test_single_element(self):
        assert _count_large_gaps([(0.0, "a")]) == 0


class TestLrcDurationEdgeCases:
    def test_metadata_filtered_out(self):
        # LRC with metadata lines that should be filtered
        lrc = (
            "[ti:Song Title]\n"
            "[ar:Artist Name]\n"
            "[00:10.00]First lyric\n"
            "[00:20.00]Second lyric\n"
            "[00:30.00]Third lyric"
        )
        duration = get_lrc_duration(lrc)
        # Should calculate from actual lyric timestamps, not metadata
        assert duration is not None
        assert 30 <= duration <= 40

    def test_instrumental_breaks_handled(self):
        # Song with long instrumental break
        lrc = (
            "[00:10.00]Verse 1\n"
            "[00:20.00]Verse 1 end\n"
            "[01:30.00]Verse 2 start\n"  # 70s instrumental break
            "[01:40.00]Verse 2 end"
        )
        duration = get_lrc_duration(lrc)
        assert duration is not None
        # Duration should span the whole song including break
        assert 100 <= duration <= 120
