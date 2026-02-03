import types

import pytest

from y2karaoke.core import sync


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


def test_fetch_lyrics_multi_source_cached_duration_mismatch_refetch(monkeypatch):
    cache_key = ("artist", "title")
    sync._lrc_cache[cache_key] = ("[00:00.00]A", True, "cached", 100)

    monkeypatch.setattr(sync, "LYRIQ_AVAILABLE", False)
    monkeypatch.setattr(sync, "SYNCEDLYRICS_AVAILABLE", True)

    called = {"count": 0}

    def fake_search(*_args, **_kwargs):
        called["count"] += 1
        return (None, "")

    monkeypatch.setattr(sync, "_search_with_fallback", fake_search)

    result = sync.fetch_lyrics_multi_source(
        "Title", "Artist", target_duration=200, duration_tolerance=10
    )

    assert result == (None, False, "")
    assert called["count"] == 1


def test_fetch_lyrics_multi_source_enhanced_plain_allowed(monkeypatch):
    monkeypatch.setattr(sync, "LYRIQ_AVAILABLE", False)
    monkeypatch.setattr(sync, "SYNCEDLYRICS_AVAILABLE", True)

    calls = {"count": 0}

    def fake_search(_term, synced_only=True, enhanced=False):
        calls["count"] += 1
        if enhanced:
            return ("plain lyrics", "Provider")
        return ("plain lyrics", "Provider")

    monkeypatch.setattr(sync, "_search_with_fallback", fake_search)

    lrc, is_synced, source = sync.fetch_lyrics_multi_source(
        "Title", "Artist", synced_only=False, enhanced=True
    )

    assert lrc == "plain lyrics"
    assert is_synced is False
    assert source == "Provider"
    assert calls["count"] == 2


def test_validate_lrc_quality_rejects_no_timestamps():
    ok, reason = sync.validate_lrc_quality("plain lyrics")
    assert ok is False
    assert "No timestamps" in reason


def test_validate_lrc_quality_accepts_good_lrc():
    lrc = "\n".join(
        [
            "[00:00.00]Line 1",
            "[00:10.00]Line 2",
            "[00:20.00]Line 3",
            "[00:30.00]Line 4",
            "[00:40.00]Line 5",
            "[00:50.00]Line 6",
            "[01:00.00]Line 7",
        ]
    )
    ok, reason = sync.validate_lrc_quality(lrc, expected_duration=90)
    assert ok is True
    assert reason == ""


def test_get_lyrics_quality_report_no_timestamps():
    report = sync.get_lyrics_quality_report("plain", source="Test")
    assert report["issues"]
    assert "No synced lyrics" in report["issues"][0]


def test_get_lyrics_quality_report_too_few_lines(monkeypatch):
    lrc = "[00:00.00]Line"
    monkeypatch.setattr(sync, "_has_timestamps", lambda *_a, **_k: True)
    monkeypatch.setattr("y2karaoke.core.lrc.parse_lrc_with_timing", lambda *_a, **_k: [])

    report = sync.get_lyrics_quality_report(lrc, source="Test")

    assert report["quality_score"] == 20.0
    assert any("Too few" in issue for issue in report["issues"])


def test_calculate_quality_score_mid_density_and_few_lines():
    report = {
        "coverage": 0.7,
        "timestamp_density": 1.7,
        "duration_match": False,
        "issues": [],
    }

    score = sync._calculate_quality_score(report, num_timings=9)

    assert score < 100.0
    assert any("Only 9 lines" in issue for issue in report["issues"])


def test_fetch_lyrics_for_duration_returns_match(monkeypatch):
    monkeypatch.setattr(sync, "SYNCEDLYRICS_AVAILABLE", True)
    monkeypatch.setattr(sync, "LYRIQ_AVAILABLE", False)

    lrc_text = "[00:00.00]Line\n[00:10.00]Next"

    monkeypatch.setattr(
        sync,
        "fetch_lyrics_multi_source",
        lambda *_a, **_k: (lrc_text, True, "source"),
    )
    monkeypatch.setattr(sync, "get_lrc_duration", lambda *_a, **_k: 200)

    result = sync.fetch_lyrics_for_duration("Title", "Artist", 200, tolerance=10)

    assert result == (lrc_text, True, "source", 200)


def test_fetch_from_all_sources_skips_genius_and_handles_errors(monkeypatch):
    monkeypatch.setattr(sync, "LYRIQ_AVAILABLE", True)
    monkeypatch.setattr(sync, "SYNCEDLYRICS_AVAILABLE", True)
    monkeypatch.setattr(sync, "lyriq_get_lyrics", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))

    providers = ["Genius", "A"]
    monkeypatch.setattr(sync, "PROVIDER_ORDER", providers)

    calls = {"providers": []}

    def fake_search(_term, providers=None, synced_only=True):
        calls["providers"].append(providers[0])
        raise RuntimeError("fail")

    monkeypatch.setattr(sync, "syncedlyrics", types.SimpleNamespace(search=fake_search))

    results = sync.fetch_from_all_sources("Title", "Artist")

    assert results == {}
    assert calls["providers"] == ["A"]
