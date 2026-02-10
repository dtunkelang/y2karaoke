import types
import pytest

from y2karaoke.core import sync

pytestmark = pytest.mark.usefixtures("isolated_sync_state")


def test_search_with_fallback_uses_cache(monkeypatch):
    sync._search_cache.clear()
    cache_key = "artist - song:True:False"
    sync._search_cache[cache_key] = ("[00:01.00]Line", "Provider")

    def fail_search(*_args, **_kwargs):
        raise AssertionError("_search_single_provider should not be called")

    monkeypatch.setattr(sync, "_search_single_provider", fail_search)

    result = sync._search_with_fallback("Artist - Song")

    assert result == ("[00:01.00]Line", "Provider")


def test_search_with_fallback_caches_empty_result(monkeypatch):
    sync._search_cache.clear()

    monkeypatch.setattr(sync, "PROVIDER_ORDER", ["A", "B"])
    monkeypatch.setattr(sync, "_search_single_provider", lambda *_a, **_k: None)
    monkeypatch.setattr(sync.time, "sleep", lambda *_a, **_k: None)

    result = sync._search_with_fallback("Artist - Song")

    assert result == (None, "")
    cache_key = "artist - song:True:False"
    assert sync._search_cache[cache_key] == (None, "")


def test_search_single_provider_retries_transient_error(monkeypatch):
    sync._failed_providers.clear()

    calls = {"count": 0}

    def fake_search(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("timeout")
        return "[00:01.00]Line"

    monkeypatch.setattr(sync, "syncedlyrics", types.SimpleNamespace(search=fake_search))
    monkeypatch.setattr(sync.time, "sleep", lambda *_a, **_k: None)

    result = sync._search_single_provider("Artist - Song", "Provider")

    assert result == "[00:01.00]Line"
    assert sync._failed_providers.get("Provider", 0) == 0


def test_search_single_provider_counts_failure(monkeypatch):
    sync._failed_providers.clear()

    def fake_search(*_args, **_kwargs):
        raise RuntimeError("bad request")

    monkeypatch.setattr(sync, "syncedlyrics", types.SimpleNamespace(search=fake_search))

    result = sync._search_single_provider("Artist - Song", "Provider", max_retries=0)

    assert result is None
    assert sync._failed_providers.get("Provider") == 1


def test_fetch_from_lyriq_returns_synced_and_caches(monkeypatch):
    sync._lyriq_cache.clear()
    monkeypatch.setattr(sync, "LYRIQ_AVAILABLE", True)

    class Lyrics:
        synced_lyrics = "[00:01.00]Line"
        plain_lyrics = "Line"

    calls = {"count": 0}

    def fake_get_lyrics(*_args, **_kwargs):
        calls["count"] += 1
        return Lyrics()

    monkeypatch.setattr(sync, "lyriq_get_lyrics", fake_get_lyrics)

    result = sync._fetch_from_lyriq("Song", "Artist")
    assert result == "[00:01.00]Line"

    # Cache hit should avoid a second call
    result_again = sync._fetch_from_lyriq("Song", "Artist")
    assert result_again == "[00:01.00]Line"
    assert calls["count"] == 1


def test_fetch_from_lyriq_plain_only_returns_none(monkeypatch):
    sync._lyriq_cache.clear()
    monkeypatch.setattr(sync, "LYRIQ_AVAILABLE", True)

    class Lyrics:
        synced_lyrics = None
        plain_lyrics = "Line"

    monkeypatch.setattr(sync, "lyriq_get_lyrics", lambda *_a, **_k: Lyrics())

    result = sync._fetch_from_lyriq("Song", "Artist")

    assert result is None


def test_fetch_from_lyriq_retries_on_transient_error(monkeypatch):
    sync._lyriq_cache.clear()
    monkeypatch.setattr(sync, "LYRIQ_AVAILABLE", True)

    calls = {"count": 0}

    def fake_get_lyrics(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("timeout")
        return None

    monkeypatch.setattr(sync, "lyriq_get_lyrics", fake_get_lyrics)
    monkeypatch.setattr(sync.time, "sleep", lambda *_a, **_k: None)

    result = sync._fetch_from_lyriq("Song", "Artist", max_retries=1)

    assert result is None
    assert calls["count"] == 2
