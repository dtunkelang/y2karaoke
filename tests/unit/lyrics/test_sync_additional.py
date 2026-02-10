import pytest

from y2karaoke.core import sync

pytestmark = pytest.mark.usefixtures("isolated_sync_state")


def test_search_single_provider_skips_after_failures(monkeypatch):
    sync._failed_providers["Musixmatch"] = 3
    called = {"count": 0}

    class FakeSyncedLyrics:
        def search(self, *args, **kwargs):
            called["count"] += 1
            return None

    monkeypatch.setattr(sync, "syncedlyrics", FakeSyncedLyrics())

    try:
        assert sync._search_single_provider("term", "Musixmatch", max_retries=0) is None
        assert called["count"] == 0
    finally:
        sync._failed_providers.pop("Musixmatch", None)


def test_search_single_provider_transient_retry_succeeds(monkeypatch):
    calls = {"count": 0}

    class FakeSyncedLyrics:
        def search(self, *args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("timeout")
            return "[00:01.00]Hi"

    monkeypatch.setattr(sync, "syncedlyrics", FakeSyncedLyrics())
    monkeypatch.setattr(sync.time, "sleep", lambda *_: None)

    result = sync._search_single_provider("term", "NetEase", max_retries=1)

    assert result == "[00:01.00]Hi"
    assert sync._failed_providers.get("NetEase", 0) == 0


def test_search_single_provider_permanent_error_increments_failure(monkeypatch):
    class FakeSyncedLyrics:
        def search(self, *args, **kwargs):
            raise RuntimeError("invalid request")

    monkeypatch.setattr(sync, "syncedlyrics", FakeSyncedLyrics())

    result = sync._search_single_provider("term", "Megalobiz", max_retries=0)

    assert result is None
    assert sync._failed_providers.get("Megalobiz", 0) == 1
    sync._failed_providers.pop("Megalobiz", None)


def test_search_with_fallback_uses_cache(isolated_sync_state):
    cache_key = "term:True:False"
    sync._search_cache[cache_key] = ("[00:01.00]Hi", "Cached")

    def fail_search(*args, **kwargs):
        raise AssertionError("_search_single_provider should not be called")

    isolated_sync_state.search_single_provider_fn = fail_search

    try:
        lrc, provider = sync._search_with_fallback("term")
        assert lrc == "[00:01.00]Hi"
        assert provider == "Cached"
    finally:
        sync._search_cache.pop(cache_key, None)


def test_search_with_fallback_caches_first_success(monkeypatch, isolated_sync_state):
    calls = {"count": 0}

    def fake_search(term, provider, synced_only=True, enhanced=False):
        calls["count"] += 1
        if calls["count"] == 1:
            return None
        return "[00:01.00]Hi"

    isolated_sync_state.search_single_provider_fn = fake_search
    monkeypatch.setattr(sync.time, "sleep", lambda *_: None)

    lrc, provider = sync._search_with_fallback("term")

    assert lrc == "[00:01.00]Hi"
    assert provider in sync.PROVIDER_ORDER


def test_fetch_from_lyriq_returns_synced(monkeypatch):
    monkeypatch.setattr(sync, "LYRIQ_AVAILABLE", True)
    sync._lyriq_cache.clear()

    class LyricsObj:
        synced_lyrics = "[00:01.00]Hi"
        plain_lyrics = None

    sync._DEFAULT_SYNC_STATE.lyriq_get_lyrics_fn = lambda *a, **k: LyricsObj()

    result = sync._fetch_from_lyriq("Title", "Artist", max_retries=0)

    assert result == "[00:01.00]Hi"


def test_fetch_from_lyriq_plain_returns_none(monkeypatch):
    monkeypatch.setattr(sync, "LYRIQ_AVAILABLE", True)
    sync._lyriq_cache.clear()

    class LyricsObj:
        synced_lyrics = None
        plain_lyrics = "plain"

    sync._DEFAULT_SYNC_STATE.lyriq_get_lyrics_fn = lambda *a, **k: LyricsObj()

    result = sync._fetch_from_lyriq("Title", "Artist", max_retries=0)

    assert result is None


def test_fetch_from_lyriq_transient_retry(monkeypatch):
    monkeypatch.setattr(sync, "LYRIQ_AVAILABLE", True)
    sync._lyriq_cache.clear()
    calls = {"count": 0}

    class LyricsObj:
        synced_lyrics = "[00:01.00]Hi"
        plain_lyrics = None

    def fake_get(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("timeout")
        return LyricsObj()

    sync._DEFAULT_SYNC_STATE.lyriq_get_lyrics_fn = fake_get
    sync._DEFAULT_SYNC_STATE.sleep_fn = lambda *_: None

    result = sync._fetch_from_lyriq("Title", "Artist", max_retries=1)

    assert result == "[00:01.00]Hi"
