import types
import pytest

from y2karaoke.core import sync

pytestmark = pytest.mark.usefixtures("isolated_sync_state")


def test_search_with_fallback_uses_cache(isolated_sync_state):
    isolated_sync_state.search_cache.clear()
    cache_key = "artist - song:True:False"
    isolated_sync_state.search_cache[cache_key] = ("[00:01.00]Line", "Provider")

    def fail_search(*_args, **_kwargs):
        raise AssertionError("_search_single_provider should not be called")

    isolated_sync_state.search_single_provider_fn = fail_search

    result = sync._search_with_fallback("Artist - Song", state=isolated_sync_state)

    assert result == ("[00:01.00]Line", "Provider")


def test_search_with_fallback_caches_empty_result(monkeypatch, isolated_sync_state):
    isolated_sync_state.search_cache.clear()

    monkeypatch.setattr(sync, "PROVIDER_ORDER", ["A", "B"])
    isolated_sync_state.search_single_provider_fn = lambda *_a, **_k: None
    isolated_sync_state.sleep_fn = lambda *_a, **_k: None

    result = sync._search_with_fallback("Artist - Song", state=isolated_sync_state)

    assert result == (None, "")
    cache_key = "artist - song:True:False"
    assert isolated_sync_state.search_cache[cache_key] == (None, "")


def test_search_single_provider_retries_transient_error(isolated_sync_state):
    isolated_sync_state.failed_providers.clear()

    calls = {"count": 0}

    def fake_search(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("timeout")
        return "[00:01.00]Line"

    isolated_sync_state.syncedlyrics_mod = types.SimpleNamespace(search=fake_search)
    isolated_sync_state.sleep_fn = lambda *_a, **_k: None

    result = sync._search_single_provider(
        "Artist - Song", "Provider", state=isolated_sync_state
    )

    assert result == "[00:01.00]Line"
    assert isolated_sync_state.failed_providers.get("Provider", 0) == 0


def test_search_single_provider_counts_failure(isolated_sync_state):
    isolated_sync_state.failed_providers.clear()

    def fake_search(*_args, **_kwargs):
        raise RuntimeError("bad request")

    isolated_sync_state.syncedlyrics_mod = types.SimpleNamespace(search=fake_search)

    result = sync._search_single_provider(
        "Artist - Song", "Provider", max_retries=0, state=isolated_sync_state
    )

    assert result is None
    assert isolated_sync_state.failed_providers.get("Provider") == 1


def test_fetch_from_lyriq_returns_synced_and_caches(isolated_sync_state):
    isolated_sync_state.lyriq_cache.clear()
    isolated_sync_state.lyriq_available = True

    class Lyrics:
        synced_lyrics = "[00:01.00]Line"
        plain_lyrics = "Line"

    calls = {"count": 0}

    def fake_get_lyrics(*_args, **_kwargs):
        calls["count"] += 1
        return Lyrics()

    isolated_sync_state.lyriq_get_lyrics_fn = fake_get_lyrics

    result = sync._fetch_from_lyriq("Song", "Artist", state=isolated_sync_state)
    assert result == "[00:01.00]Line"

    # Cache hit should avoid a second call
    result_again = sync._fetch_from_lyriq("Song", "Artist", state=isolated_sync_state)
    assert result_again == "[00:01.00]Line"
    assert calls["count"] == 1


def test_fetch_from_lyriq_plain_only_returns_none(isolated_sync_state):
    isolated_sync_state.lyriq_cache.clear()
    isolated_sync_state.lyriq_available = True

    class Lyrics:
        synced_lyrics = None
        plain_lyrics = "Line"

    isolated_sync_state.lyriq_get_lyrics_fn = lambda *_a, **_k: Lyrics()

    result = sync._fetch_from_lyriq("Song", "Artist", state=isolated_sync_state)

    assert result is None


def test_fetch_from_lyriq_retries_on_transient_error(isolated_sync_state):
    isolated_sync_state.lyriq_cache.clear()
    isolated_sync_state.lyriq_available = True

    calls = {"count": 0}

    def fake_get_lyrics(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("timeout")
        return None

    isolated_sync_state.lyriq_get_lyrics_fn = fake_get_lyrics
    isolated_sync_state.sleep_fn = lambda *_a, **_k: None

    result = sync._fetch_from_lyriq(
        "Song", "Artist", max_retries=1, state=isolated_sync_state
    )

    assert result is None
    assert calls["count"] == 2
