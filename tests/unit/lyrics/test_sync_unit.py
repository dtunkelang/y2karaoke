import pytest

from y2karaoke.core import sync

pytestmark = pytest.mark.usefixtures("isolated_sync_state")


def test_has_timestamps():
    assert sync._has_timestamps("[00:01.00]Line")
    assert sync._has_timestamps("[0:01.00]Line")
    assert not sync._has_timestamps("No timestamps here")


def test_get_lrc_duration_from_timings():
    lrc_text = "[00:01.00]A\n[00:11.00]B\n"
    # span=10, buffer=max(3,1)=3, last=11 => 14
    assert sync.get_lrc_duration(lrc_text) == 14


def test_validate_lrc_quality_rejects_short():
    lrc_text = "[00:01.00]A\n[00:05.00]B\n"
    is_valid, reason = sync.validate_lrc_quality(lrc_text, expected_duration=120)
    assert is_valid is False
    assert "Too few timestamped lines" in reason or "Lyrics span too short" in reason


def test_get_lyrics_quality_report_includes_issues():
    lrc_text = "[00:01.00]A\n[00:50.00]B\n"
    report = sync.get_lyrics_quality_report(
        lrc_text, source="test", target_duration=200, sources_tried=["A", "B"]
    )
    assert report["source"] == "test"
    assert report["sources_tried"] == ["A", "B"]
    assert report["quality_score"] >= 0.0


def test_fetch_lyrics_multi_source_returns_cached(isolated_sync_state):
    cache_key = ("artist", "title")
    sync._lrc_cache[cache_key] = ("[00:01.00]A", True, "cached", 100)

    def fail_search(*args, **kwargs):
        raise AssertionError("_search_with_fallback should not be called")

    isolated_sync_state.search_with_fallback_fn = fail_search

    try:
        lrc_text, is_synced, source = sync.fetch_lyrics_multi_source("Title", "Artist")
        assert lrc_text == "[00:01.00]A"
        assert is_synced is True
        assert source == "cached"
    finally:
        sync._lrc_cache.pop(cache_key, None)


def test_fetch_lyrics_multi_source_handles_no_providers(monkeypatch):
    sync._DEFAULT_SYNC_STATE.syncedlyrics_available = False
    sync._DEFAULT_SYNC_STATE.lyriq_available = False

    lrc_text, is_synced, source = sync.fetch_lyrics_multi_source("Title", "Artist")
    assert lrc_text is None
    assert is_synced is False
    assert source == ""


def test_fetch_lyrics_multi_source_uses_enhanced(monkeypatch):
    sync._lrc_cache.clear()
    sync._DEFAULT_SYNC_STATE.lyriq_available = False
    sync._DEFAULT_SYNC_STATE.syncedlyrics_available = True

    def fake_search(term, synced_only=True, enhanced=False):
        if enhanced:
            return ("[00:01.00]Hi", "Provider")
        return (None, "")

    isolated_sync_state = sync._DEFAULT_SYNC_STATE
    isolated_sync_state.search_with_fallback_fn = fake_search

    lrc_text, is_synced, source = sync.fetch_lyrics_multi_source(
        "Title", "Artist", enhanced=True
    )
    assert lrc_text == "[00:01.00]Hi"
    assert is_synced is True
    assert "enhanced" in source


def test_fetch_lyrics_multi_source_plain_when_unsynced_allowed(monkeypatch):
    sync._lrc_cache.clear()
    sync._DEFAULT_SYNC_STATE.lyriq_available = False
    sync._DEFAULT_SYNC_STATE.syncedlyrics_available = True
    sync._DEFAULT_SYNC_STATE.search_with_fallback_fn = lambda *a, **k: (
        "plain",
        "Genius",
    )

    lrc_text, is_synced, source = sync.fetch_lyrics_multi_source(
        "Title", "Artist", synced_only=False
    )
    assert lrc_text == "plain"
    assert is_synced is False
    assert source == "Genius"


def test_fetch_lyrics_multi_source_returns_none_when_not_found(monkeypatch):
    sync._lrc_cache.clear()
    sync._DEFAULT_SYNC_STATE.lyriq_available = False
    sync._DEFAULT_SYNC_STATE.syncedlyrics_available = True
    sync._DEFAULT_SYNC_STATE.search_with_fallback_fn = lambda *a, **k: (None, "")

    lrc_text, is_synced, source = sync.fetch_lyrics_multi_source("Title", "Artist")
    assert lrc_text is None
    assert is_synced is False
    assert source == ""


def test_fetch_lyrics_multi_source_handles_exception(monkeypatch):
    sync._lrc_cache.clear()
    sync._DEFAULT_SYNC_STATE.lyriq_available = False
    sync._DEFAULT_SYNC_STATE.syncedlyrics_available = True

    def raise_error(*args, **kwargs):
        raise RuntimeError("boom")

    sync._DEFAULT_SYNC_STATE.search_with_fallback_fn = raise_error

    lrc_text, is_synced, source = sync.fetch_lyrics_multi_source("Title", "Artist")
    assert lrc_text is None
    assert is_synced is False
    assert source == ""
