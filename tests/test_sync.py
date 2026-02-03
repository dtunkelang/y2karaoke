import types

import pytest

from y2karaoke.core import sync


LRC_TEXT = "\n".join(
    [
        "[00:00.00]Line one",
        "[00:10.00]Line two",
        "[00:20.00]Line three",
        "[00:30.00]Line four",
        "[00:40.00]Line five",
    ]
)


@pytest.fixture(autouse=True)
def _clear_sync_caches():
    sync._failed_providers.clear()
    sync._search_cache.clear()
    sync._lyriq_cache.clear()
    sync._lrc_cache.clear()
    yield
    sync._failed_providers.clear()
    sync._search_cache.clear()
    sync._lyriq_cache.clear()
    sync._lrc_cache.clear()


def test_search_single_provider_skips_after_failures(monkeypatch):
    sync._failed_providers["Test"] = sync._FAILURE_THRESHOLD

    called = {"count": 0}

    def fake_search(*args, **kwargs):
        called["count"] += 1
        return "LRC"

    monkeypatch.setattr(sync, "syncedlyrics", types.SimpleNamespace(search=fake_search))
    result = sync._search_single_provider("query", "Test")
    assert result is None
    assert called["count"] == 0


def test_search_single_provider_retries_on_transient_error(monkeypatch):
    calls = {"count": 0}

    def fake_search(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("timeout")
        return "LRC"

    monkeypatch.setattr(sync, "syncedlyrics", types.SimpleNamespace(search=fake_search))
    monkeypatch.setattr(sync.time, "sleep", lambda *_: None)

    result = sync._search_single_provider(
        "query",
        "Test",
        max_retries=1,
        retry_delay=0.01,
    )

    assert result == "LRC"
    assert sync._failed_providers.get("Test", 0) == 0
    assert calls["count"] == 2


def test_search_with_fallback_caches_result(monkeypatch):
    calls = {"providers": []}

    def fake_search(search_term, provider, **kwargs):
        calls["providers"].append(provider)
        return "LRC" if provider == "A" else None

    monkeypatch.setattr(sync, "PROVIDER_ORDER", ["A", "B"])
    monkeypatch.setattr(sync, "_search_single_provider", fake_search)
    monkeypatch.setattr(sync.time, "sleep", lambda *_: None)

    first = sync._search_with_fallback("Song Artist")
    second = sync._search_with_fallback("Song Artist")

    assert first == ("LRC", "A")
    assert second == ("LRC", "A")
    assert calls["providers"] == ["A"]


def test_fetch_from_lyriq_returns_synced(monkeypatch):
    class Lyrics:
        synced_lyrics = LRC_TEXT
        plain_lyrics = None

    monkeypatch.setattr(sync, "LYRIQ_AVAILABLE", True)
    monkeypatch.setattr(sync, "lyriq_get_lyrics", lambda *a, **k: Lyrics())

    result = sync._fetch_from_lyriq("Title", "Artist")
    assert result == LRC_TEXT


def test_fetch_from_lyriq_returns_none_for_plain(monkeypatch):
    class Lyrics:
        synced_lyrics = None
        plain_lyrics = "plain lyrics"

    monkeypatch.setattr(sync, "LYRIQ_AVAILABLE", True)
    monkeypatch.setattr(sync, "lyriq_get_lyrics", lambda *a, **k: Lyrics())

    result = sync._fetch_from_lyriq("Title", "Artist")
    assert result is None


def test_fetch_lyrics_multi_source_uses_lyriq_first(monkeypatch):
    monkeypatch.setattr(sync, "LYRIQ_AVAILABLE", True)
    monkeypatch.setattr(sync, "SYNCEDLYRICS_AVAILABLE", False)
    monkeypatch.setattr(sync, "_fetch_from_lyriq", lambda *a, **k: LRC_TEXT)

    lrc, is_synced, source = sync.fetch_lyrics_multi_source("Title", "Artist")
    assert lrc == LRC_TEXT
    assert is_synced is True
    assert source == "lyriq (LRCLib)"


def test_validate_lrc_quality_reports_low_coverage():
    ok, reason = sync.validate_lrc_quality(LRC_TEXT, expected_duration=120)
    assert ok is False
    assert "covers only" in reason


def test_count_large_gaps_counts_only_big_gaps():
    timings = [(0.0, "a"), (10.0, "b"), (50.0, "c"), (90.0, "d")]
    assert sync._count_large_gaps(timings, threshold=30.0) == 2


def test_get_lyrics_quality_report_flags_mismatch(monkeypatch):
    monkeypatch.setattr(sync, "get_lrc_duration", lambda *_: 90)

    report = sync.get_lyrics_quality_report(
        LRC_TEXT,
        source="Test",
        target_duration=200,
        sources_tried=["A", "B"],
    )

    assert report["duration_match"] is False
    assert any("Duration mismatch" in issue for issue in report["issues"])


def test_fetch_lyrics_for_duration_uses_alternative_search(monkeypatch):
    monkeypatch.setattr(sync, "SYNCEDLYRICS_AVAILABLE", True)
    monkeypatch.setattr(sync, "LYRIQ_AVAILABLE", False)
    monkeypatch.setattr(
        sync,
        "fetch_lyrics_multi_source",
        lambda *a, **k: ("[00:00.00]Short", True, "primary"),
    )
    def fake_duration(lrc_text):
        return 120 if lrc_text == LRC_TEXT else 50

    monkeypatch.setattr(sync, "get_lrc_duration", fake_duration)

    def fake_search(term, **kwargs):
        return (LRC_TEXT, "Provider")

    monkeypatch.setattr(sync, "_search_with_fallback", fake_search)

    lrc, is_synced, source, duration = sync.fetch_lyrics_for_duration(
        "Title",
        "Artist",
        target_duration=120,
        tolerance=10,
    )

    assert lrc == LRC_TEXT
    assert is_synced is True
    assert "Provider" in source
    assert duration == 120


def test_fetch_from_all_sources_collects_results(monkeypatch):
    class Lyrics:
        synced_lyrics = LRC_TEXT
        plain_lyrics = None

    monkeypatch.setattr(sync, "LYRIQ_AVAILABLE", True)
    monkeypatch.setattr(sync, "SYNCEDLYRICS_AVAILABLE", True)
    monkeypatch.setattr(sync, "lyriq_get_lyrics", lambda *a, **k: Lyrics())
    monkeypatch.setattr(sync, "syncedlyrics", types.SimpleNamespace(search=lambda *a, **k: LRC_TEXT))

    results = sync.fetch_from_all_sources("Title", "Artist")

    assert "lyriq (LRCLib)" in results
    assert any(provider in results for provider in sync.PROVIDER_ORDER if provider != "Genius")
