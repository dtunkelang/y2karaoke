import pytest

from y2karaoke.core import sync

pytestmark = pytest.mark.usefixtures("isolated_sync_state")


def test_search_single_provider_tracks_failures(isolated_sync_state):
    sync._failed_providers.clear()

    class FakeSync:
        def search(self, *args, **kwargs):
            raise RuntimeError("boom")

    isolated_sync_state.syncedlyrics_mod = FakeSync()
    isolated_sync_state.syncedlyrics_available = True
    isolated_sync_state.sleep_fn = lambda *_: None

    assert sync._search_single_provider("term", "Provider", max_retries=0) is None
    assert sync._failed_providers["Provider"] == 1


def test_search_single_provider_skips_after_threshold(isolated_sync_state):
    sync._failed_providers.clear()
    sync._failed_providers["Provider"] = 3
    assert sync._search_single_provider("term", "Provider") is None


def test_search_with_fallback_caches_result(monkeypatch, isolated_sync_state):
    sync._search_cache.clear()
    monkeypatch.setattr(sync, "PROVIDER_ORDER", ["Provider"])
    isolated_sync_state.search_single_provider_fn = lambda *a, **k: "[00:01.00]A"
    isolated_sync_state.sleep_fn = lambda *_: None

    lrc, provider = sync._search_with_fallback("term")
    assert lrc
    assert provider == "Provider"

    # cache hit should return without calling _search_single_provider again
    isolated_sync_state.search_single_provider_fn = lambda *a, **k: (
        _ for _ in ()
    ).throw(AssertionError("should not call"))
    lrc2, provider2 = sync._search_with_fallback("term")
    assert lrc2 == lrc
    assert provider2 == provider


def test_fetch_from_lyriq_caches_result(isolated_sync_state):
    sync._lyriq_cache.clear()
    isolated_sync_state.lyriq_available = True

    class Lyrics:
        synced_lyrics = "[00:01.00]A"

    isolated_sync_state.lyriq_get_lyrics_fn = lambda *a, **k: Lyrics()
    isolated_sync_state.sleep_fn = lambda *_: None

    lrc = sync._fetch_from_lyriq("Song", "Artist")
    assert lrc == "[00:01.00]A"
    assert sync._lyriq_cache[("artist", "song")] == "[00:01.00]A"


def test_fetch_lyrics_for_duration_alt_search(monkeypatch, isolated_sync_state):
    isolated_sync_state.syncedlyrics_available = True
    isolated_sync_state.lyriq_available = False

    monkeypatch.setattr(
        sync,
        "fetch_lyrics_multi_source",
        lambda *a, **k: (None, False, ""),
    )

    calls = {"count": 0}

    def fake_search(search_term, synced_only=True):
        calls["count"] += 1
        return ("[00:01.00]A\n[00:11.00]B\n", "Provider")

    isolated_sync_state.search_with_fallback_fn = fake_search
    isolated_sync_state.sleep_fn = lambda *_: None

    lrc, is_synced, source, duration = sync.fetch_lyrics_for_duration(
        "Title", "Artist", target_duration=14, tolerance=5
    )
    assert lrc is not None
    assert is_synced is True
    assert "Provider" in source
    assert duration == 14
    assert calls["count"] >= 1


def test_get_lyrics_quality_report_gap_penalty(isolated_sync_state):
    lrc_text = "[00:01.00]A\n[00:40.00]B\n[01:20.00]C\n[02:05.00]D"
    report = sync.get_lyrics_quality_report(
        lrc_text, source="test", target_duration=200
    )
    assert report["quality_score"] >= 0.0
    assert report["duration"] is not None
