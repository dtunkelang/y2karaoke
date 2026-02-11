import builtins
import types
import pytest

from y2karaoke.core import sync

pytestmark = pytest.mark.usefixtures("isolated_sync_state")


def test_fetch_lyrics_multi_source_cached_duration_mismatch_refetch(
    monkeypatch, isolated_sync_state
):
    cache_key = ("artist", "title")
    sync._lrc_cache[cache_key] = ("[00:00.00]A", True, "cached", 100)

    isolated_sync_state.lyriq_available = False
    isolated_sync_state.syncedlyrics_available = True

    called = {"count": 0}

    def fake_search(*_args, **_kwargs):
        called["count"] += 1
        return (None, "")

    isolated_sync_state.search_with_fallback_fn = fake_search

    result = sync.fetch_lyrics_multi_source(
        "Title", "Artist", target_duration=200, duration_tolerance=10
    )

    # If refetch fails, reuse cached LRC even when duration is outside tolerance.
    assert result == ("[00:00.00]A", True, "cached")
    assert called["count"] == 1


def test_fetch_lyrics_multi_source_enhanced_plain_allowed(
    monkeypatch, isolated_sync_state
):
    isolated_sync_state.lyriq_available = False
    isolated_sync_state.syncedlyrics_available = True

    calls = {"count": 0}

    def fake_search(_term, synced_only=True, enhanced=False):
        calls["count"] += 1
        if enhanced:
            return ("plain lyrics", "Provider")
        return ("plain lyrics", "Provider")

    isolated_sync_state.search_with_fallback_fn = fake_search

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


def test_get_lyrics_quality_report_too_few_lines(monkeypatch, isolated_sync_state):
    lrc = "[00:00.00]Line"
    isolated_sync_state.has_timestamps_fn = lambda *_a, **_k: True
    monkeypatch.setattr(
        "y2karaoke.core.components.lyrics.lrc.parse_lrc_with_timing",
        lambda *_a, **_k: [],
    )

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


def test_sync_import_fallbacks_when_dependencies_missing(isolated_sync_state):
    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith(("syncedlyrics", "lyriq")):
            raise ImportError("missing")
        return builtins.__import__(name, globals, locals, fromlist, level)

    synced_mod, synced_ok, lyriq_get, lyriq_ok = sync._resolve_sync_dependencies(
        fake_import
    )

    assert synced_ok is False
    assert synced_mod is None
    assert lyriq_ok is False
    assert lyriq_get is None


def test_search_single_provider_returns_none_on_empty_result(
    monkeypatch, isolated_sync_state
):
    monkeypatch.setattr(
        sync, "syncedlyrics", types.SimpleNamespace(search=lambda *_a, **_k: None)
    )
    result = sync._search_single_provider("term", "Provider")
    assert result is None
    assert sync._failed_providers.get("Provider", 0) == 0


def test_search_single_provider_skips_after_failures(monkeypatch, isolated_sync_state):
    sync._failed_providers["Provider"] = sync._FAILURE_THRESHOLD
    monkeypatch.setattr(
        sync, "syncedlyrics", types.SimpleNamespace(search=lambda *_a, **_k: "hit")
    )
    result = sync._search_single_provider("term", "Provider")
    assert result is None


def test_search_single_provider_returns_none_when_no_attempts(
    monkeypatch, isolated_sync_state
):
    monkeypatch.setattr(
        sync, "syncedlyrics", types.SimpleNamespace(search=lambda *_a, **_k: "hit")
    )
    result = sync._search_single_provider("term", "Provider", max_retries=-1)
    assert result is None


def test_fetch_from_lyriq_no_synced_or_plain(monkeypatch, isolated_sync_state):
    class EmptyLyrics:
        synced_lyrics = None
        plain_lyrics = None

    isolated_sync_state.lyriq_available = True
    isolated_sync_state.lyriq_get_lyrics_fn = lambda *_a, **_k: EmptyLyrics()

    assert sync._fetch_from_lyriq("Title", "Artist") is None


def test_fetch_lyrics_multi_source_cache_hit_duration_match(
    monkeypatch, isolated_sync_state
):
    cache_key = ("artist", "title")
    sync._lrc_cache[cache_key] = ("[00:00.00]A", True, "cached", 120)

    isolated_sync_state.lyriq_available = False
    isolated_sync_state.syncedlyrics_available = True

    result = sync.fetch_lyrics_multi_source(
        "Title", "Artist", target_duration=125, duration_tolerance=10
    )

    assert result == ("[00:00.00]A", True, "cached")


def test_fetch_lyrics_multi_source_returns_synced_provider(
    monkeypatch, isolated_sync_state
):
    isolated_sync_state.lyriq_available = False
    isolated_sync_state.syncedlyrics_available = True
    isolated_sync_state.search_with_fallback_fn = lambda *_a, **_k: (
        "[00:00.00]A",
        "Provider",
    )

    result = sync.fetch_lyrics_multi_source("Title", "Artist")

    assert result == ("[00:00.00]A", True, "Provider")


def test_fetch_lyrics_multi_source_returns_plain_when_allowed(
    monkeypatch, isolated_sync_state
):
    isolated_sync_state.lyriq_available = False
    isolated_sync_state.syncedlyrics_available = True
    isolated_sync_state.search_with_fallback_fn = lambda *_a, **_k: (
        "Plain lyrics",
        "Provider",
    )

    result = sync.fetch_lyrics_multi_source("Title", "Artist", synced_only=False)

    assert result == ("Plain lyrics", False, "Provider")


def test_has_timestamps_empty_returns_false():
    assert sync._has_timestamps("") is False


def test_validate_lrc_quality_large_gap_and_low_coverage():
    lrc = "\n".join(
        [
            "[00:00.00]Line 1",
            "[00:10.00]Line 2",
            "[00:20.00]Line 3",
            "[01:00.00]Line 4",
            "[01:10.00]Line 5",
        ]
    )
    ok, reason = sync.validate_lrc_quality(lrc, expected_duration=200)
    assert ok is False
    assert "covers only" in reason


def test_fetch_lyrics_for_duration_returns_match(monkeypatch, isolated_sync_state):
    isolated_sync_state.syncedlyrics_available = True
    isolated_sync_state.lyriq_available = False

    lrc_text = "[00:00.00]Line\n[00:10.00]Next"

    monkeypatch.setattr(
        sync,
        "fetch_lyrics_multi_source",
        lambda *_a, **_k: (lrc_text, True, "source"),
    )
    isolated_sync_state.get_lrc_duration_fn = lambda *_a, **_k: 200

    result = sync.fetch_lyrics_for_duration("Title", "Artist", 200, tolerance=10)

    assert result == (lrc_text, True, "source", 200)


def test_fetch_lyrics_for_duration_alternative_search_match(
    monkeypatch, isolated_sync_state
):
    isolated_sync_state.syncedlyrics_available = True
    isolated_sync_state.lyriq_available = False

    lrc_text = "[00:00.00]Line\n[00:10.00]Next"
    alt_lrc = "[00:00.00]Alt\n[00:20.00]Next"

    monkeypatch.setattr(
        sync,
        "fetch_lyrics_multi_source",
        lambda *_a, **_k: (lrc_text, True, "source"),
    )
    monkeypatch.setattr(
        sync,
        "_search_with_fallback",
        lambda *_a, **_k: (alt_lrc, "Provider"),
    )
    monkeypatch.setattr(
        sync,
        "get_lrc_duration",
        lambda text: 100 if text == lrc_text else 205,
    )

    result = sync.fetch_lyrics_for_duration("Title", "Artist", 200, tolerance=10)

    assert result == (alt_lrc, True, "Provider (Title Artist)", 205)


def test_fetch_lyrics_for_duration_returns_mismatch_when_no_alternatives(
    monkeypatch, isolated_sync_state
):
    isolated_sync_state.syncedlyrics_available = True
    isolated_sync_state.lyriq_available = False

    lrc_text = "[00:00.00]Line\n[00:10.00]Next"

    monkeypatch.setattr(
        sync,
        "fetch_lyrics_multi_source",
        lambda *_a, **_k: (lrc_text, True, "source"),
    )
    isolated_sync_state.search_with_fallback_fn = lambda *_a, **_k: (None, "")
    isolated_sync_state.get_lrc_duration_fn = lambda *_a, **_k: 100

    result = sync.fetch_lyrics_for_duration("Title", "Artist", 200, tolerance=10)

    assert result == (lrc_text, True, "source", 100)


def test_fetch_lyrics_for_duration_returns_none_when_no_results(
    monkeypatch, isolated_sync_state
):
    isolated_sync_state.syncedlyrics_available = False
    isolated_sync_state.lyriq_available = False

    monkeypatch.setattr(
        sync,
        "fetch_lyrics_multi_source",
        lambda *_a, **_k: (None, False, ""),
    )

    result = sync.fetch_lyrics_for_duration("Title", "Artist", 200, tolerance=10)

    assert result == (None, False, "", None)


def test_fetch_from_all_sources_skips_genius_and_handles_errors(
    monkeypatch, isolated_sync_state
):
    isolated_sync_state.lyriq_available = True
    isolated_sync_state.syncedlyrics_available = True
    monkeypatch.setattr(
        sync,
        "lyriq_get_lyrics",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    providers = ["Genius", "A"]
    monkeypatch.setattr(sync, "PROVIDER_ORDER", providers)

    calls = {"providers": []}

    def fake_search(_term, providers=None, synced_only=True):
        calls["providers"].append(providers[0])
        raise RuntimeError("fail")

    isolated_sync_state.syncedlyrics_mod = types.SimpleNamespace(search=fake_search)

    results = sync.fetch_from_all_sources("Title", "Artist")

    assert results == {}
    assert calls["providers"] == ["A"]
