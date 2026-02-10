import types
import pytest

from y2karaoke.core import sync

pytestmark = pytest.mark.usefixtures("isolated_sync_state")


class FakeLyrics:
    def __init__(self, synced=None, plain=None):
        self.synced_lyrics = synced
        self.plain_lyrics = plain


def test_fetch_from_lyriq_disabled(monkeypatch, isolated_sync_state):
    isolated_sync_state.lyriq_available = False
    assert sync._fetch_from_lyriq("Title", "Artist") is None


def test_fetch_from_lyriq_returns_synced(monkeypatch, isolated_sync_state):
    isolated_sync_state.lyriq_available = True
    isolated_sync_state.lyriq_get_lyrics_fn = lambda *_a, **_k: FakeLyrics(
        synced="[00:00.00]hi"
    )

    result = sync._fetch_from_lyriq("Title", "Artist")

    assert result == "[00:00.00]hi"


def test_fetch_from_lyriq_plain_only_returns_none(monkeypatch, isolated_sync_state):
    isolated_sync_state.lyriq_available = True
    isolated_sync_state.lyriq_get_lyrics_fn = lambda *_a, **_k: FakeLyrics(
        synced=None, plain="plain"
    )

    result = sync._fetch_from_lyriq("Title", "Artist")

    assert result is None


def test_fetch_from_lyriq_no_results(monkeypatch, isolated_sync_state):
    isolated_sync_state.lyriq_available = True
    isolated_sync_state.lyriq_get_lyrics_fn = lambda *_a, **_k: None

    result = sync._fetch_from_lyriq("Title", "Artist")

    assert result is None


def test_fetch_from_lyriq_retries_transient(monkeypatch, isolated_sync_state):
    isolated_sync_state.lyriq_available = True

    calls = {"count": 0}

    def fake_get(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("timeout")
        return FakeLyrics(synced="[00:00.00]hi")

    isolated_sync_state.lyriq_get_lyrics_fn = fake_get
    isolated_sync_state.sleep_fn = lambda *_a, **_k: None

    result = sync._fetch_from_lyriq("Title", "Artist", max_retries=1, retry_delay=0)

    assert result == "[00:00.00]hi"
    assert calls["count"] == 2


def test_search_single_provider_skips_failed(monkeypatch, isolated_sync_state):
    sync._failed_providers["Provider"] = sync._FAILURE_THRESHOLD
    isolated_sync_state.syncedlyrics_mod = types.SimpleNamespace(
        search=lambda *_a, **_k: "LRC"
    )

    result = sync._search_single_provider("term", "Provider")

    assert result is None


def test_search_single_provider_retries_transient(monkeypatch, isolated_sync_state):
    calls = {"count": 0}

    def fake_search(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("timeout")
        return "[00:00.00]hi"

    isolated_sync_state.syncedlyrics_mod = types.SimpleNamespace(search=fake_search)

    result = sync._search_single_provider(
        "term",
        "Provider",
        max_retries=1,
        retry_delay=0,
    )

    assert result == "[00:00.00]hi"
    assert calls["count"] == 2


def test_search_single_provider_permanent_error_tracks_failure(
    monkeypatch, isolated_sync_state
):
    def fake_search(*_args, **_kwargs):
        raise RuntimeError("boom")

    isolated_sync_state.syncedlyrics_mod = types.SimpleNamespace(search=fake_search)

    result = sync._search_single_provider("term", "Provider", max_retries=0)

    assert result is None
    assert sync._failed_providers["Provider"] == 1
