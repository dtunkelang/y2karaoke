import pytest

import y2karaoke.core.sync as sync

pytestmark = pytest.mark.usefixtures("isolated_sync_state")


def test_count_large_gaps():
    timings = [(0.0, ""), (10.0, ""), (45.0, ""), (80.0, "")]
    assert sync._count_large_gaps(timings, threshold=30.0) == 2


def test_calculate_quality_score_penalties():
    report = {
        "coverage": 0.5,
        "timestamp_density": 1.0,
        "duration_match": False,
        "issues": [],
    }
    score = sync._calculate_quality_score(report, num_timings=5)
    assert score < 50
    assert "Low coverage (50%)" in report["issues"]
    assert "Low timestamp density (1.0/10s)" in report["issues"]
    assert "Only 5 lines" in report["issues"]


def test_get_lrc_duration_returns_none_without_timestamps():
    assert sync.get_lrc_duration("plain lyrics") is None


def test_get_lrc_duration_from_timings(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.lrc.parse_lrc_with_timing",
        lambda _text, _title, _artist: [(0.0, ""), (10.0, ""), (40.0, "")],
    )
    duration = sync.get_lrc_duration("[00:00.00]a\n[00:10.00]b\n[00:40.00]c")
    assert duration == 44


def test_validate_lrc_quality_rejects_low_line_count(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.lrc.parse_lrc_with_timing",
        lambda _text, _title, _artist: [(0.0, ""), (5.0, ""), (10.0, ""), (15.0, "")],
    )
    ok, reason = sync.validate_lrc_quality("[00:00.00]a\n[00:05.00]b")
    assert ok is False
    assert "Too few timestamped lines" in reason


def test_validate_lrc_quality_rejects_short_span(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.lrc.parse_lrc_with_timing",
        lambda _text, _title, _artist: [
            (0.0, ""),
            (5.0, ""),
            (10.0, ""),
            (15.0, ""),
            (20.0, ""),
        ],
    )
    ok, reason = sync.validate_lrc_quality("[00:00.00]a\n[00:20.00]b")
    assert ok is False
    assert "Lyrics span too short" in reason


def test_validate_lrc_quality_rejects_low_density(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.lrc.parse_lrc_with_timing",
        lambda _text, _title, _artist: [
            (0.0, ""),
            (30.0, ""),
            (60.0, ""),
            (90.0, ""),
            (120.0, ""),
        ],
    )
    ok, reason = sync.validate_lrc_quality("[00:00.00]a\n[02:00.00]b")
    assert ok is False
    assert "Timestamp density too low" in reason


def test_validate_lrc_quality_rejects_low_coverage(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.lrc.parse_lrc_with_timing",
        lambda _text, _title, _artist: [
            (0.0, ""),
            (10.0, ""),
            (20.0, ""),
            (30.0, ""),
            (40.0, ""),
            (50.0, ""),
            (60.0, ""),
            (80.0, ""),
        ],
    )
    ok, reason = sync.validate_lrc_quality(
        "[00:00.00]a\n[01:20.00]b", expected_duration=200
    )
    assert ok is False
    assert "covers only 40%" in reason


def test_fetch_lyrics_for_duration_no_providers(monkeypatch):
    monkeypatch.setattr(sync, "SYNCEDLYRICS_AVAILABLE", False)
    monkeypatch.setattr(sync, "LYRIQ_AVAILABLE", False)
    result = sync.fetch_lyrics_for_duration("Title", "Artist", 200)
    assert result == (None, False, "", None)


def test_fetch_lyrics_for_duration_mismatch_returns_anyway(monkeypatch):
    monkeypatch.setattr(sync, "SYNCEDLYRICS_AVAILABLE", False)
    monkeypatch.setattr(sync, "LYRIQ_AVAILABLE", True)
    monkeypatch.setattr(
        sync,
        "fetch_lyrics_multi_source",
        lambda *_args, **_kwargs: ("[00:00.00]a", True, "source"),
    )
    monkeypatch.setattr(sync, "get_lrc_duration", lambda _text: 100)
    result = sync.fetch_lyrics_for_duration("Title", "Artist", 200, tolerance=10)
    assert result == ("[00:00.00]a", True, "source", 100)


def test_get_lyrics_quality_report_flags_mismatch_and_gaps(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.lrc.parse_lrc_with_timing",
        lambda _text, _title, _artist: [
            (0.0, ""),
            (40.0, ""),
            (80.0, ""),
            (120.0, ""),
            (160.0, ""),
        ],
    )
    monkeypatch.setattr(sync, "get_lrc_duration", lambda _text: 120)
    report = sync.get_lyrics_quality_report(
        "[00:00.00]a\n[02:40.00]b",
        source="test",
        target_duration=200,
        sources_tried=["a", "b"],
    )
    assert report["duration_match"] is False
    assert any("Duration mismatch" in issue for issue in report["issues"])
