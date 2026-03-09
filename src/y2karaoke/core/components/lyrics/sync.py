"""Synced lyrics fetching using syncedlyrics and lyriq libraries."""

import logging
import json
import os
import time
import unicodedata
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, List

from ....config import get_cache_dir
from ....utils.logging import get_logger
from . import sync_quality
from . import sync_search
from . import sync_providers
from .sync_pipeline import (
    _normalize_for_provider_search,
    fetch_lyrics_multi_source_impl,
)

logger = get_logger(__name__)

_has_timestamps = sync_quality._has_timestamps
get_lrc_duration = sync_quality.get_lrc_duration
validate_lrc_quality = sync_quality.validate_lrc_quality
_count_large_gaps = sync_quality._count_large_gaps
_calculate_quality_score = sync_quality._calculate_quality_score
get_lyrics_quality_report = sync_quality.get_lyrics_quality_report


class _OncePerMessageFilter(logging.Filter):
    """Allow only the first occurrence of each unique log message."""

    def __init__(self, max_entries: int = 512):
        super().__init__()
        self.max_entries = max_entries
        self._seen: set[tuple[str, int, str]] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        signature = (record.name, record.levelno, record.getMessage())
        if signature in self._seen:
            return False
        self._seen.add(signature)
        if len(self._seen) > self.max_entries:
            self._seen.clear()
            self._seen.add(signature)
        return True


def _configure_third_party_lyrics_loggers() -> None:
    dedupe_filter = _OncePerMessageFilter()
    for logger_name in ("lyriq", "lyriq.lyriq", "syncedlyrics"):
        third_party_logger = logging.getLogger(logger_name)
        if not any(
            isinstance(existing_filter, _OncePerMessageFilter)
            for existing_filter in third_party_logger.filters
        ):
            third_party_logger.addFilter(dedupe_filter)


_configure_third_party_lyrics_loggers()


@dataclass
class SyncState:
    """In-memory state for sync provider retries and caches."""

    failed_providers: Dict[str, int] = field(default_factory=dict)
    search_cache: Dict[str, Tuple[Optional[str], str]] = field(default_factory=dict)
    lrc_cache: Dict[Tuple[str, str], Tuple[Optional[str], bool, str, Optional[int]]] = (
        field(default_factory=dict)
    )
    lyriq_cache: Dict[Tuple[str, str], Optional[str]] = field(default_factory=dict)
    all_sources_cache: Dict[
        Tuple[str, str], Dict[str, Tuple[Optional[str], Optional[int]]]
    ] = field(default_factory=dict)
    disk_cache: Dict[str, Any] = field(
        default_factory=lambda: {
            "search_cache": {},
            "lrc_cache": {},
            "lyriq_cache": {},
            "all_sources_cache": {},
        }
    )
    disk_cache_loaded: bool = False
    disk_cache_enabled: bool = True
    search_single_provider_fn: Optional[Callable[..., Optional[str]]] = None
    search_with_fallback_fn: Optional[Callable[..., Tuple[Optional[str], str]]] = None
    lyriq_get_lyrics_fn: Optional[Callable[..., Any]] = None
    sleep_fn: Callable[[float], None] = time.sleep
    syncedlyrics_mod: Any = None
    syncedlyrics_available: Optional[bool] = None
    lyriq_available: Optional[bool] = None
    has_timestamps_fn: Optional[Callable[[str], bool]] = None
    get_lrc_duration_fn: Optional[Callable[[str], Optional[int]]] = None
    warning_once_keys: set[str] = field(default_factory=set)


def create_sync_state(*, disk_cache_enabled: bool = True) -> SyncState:
    """Create a fresh sync state container for isolated calls/tests."""
    return SyncState(disk_cache_enabled=disk_cache_enabled)


def _resolve_sync_dependencies(import_fn=__import__):
    """Resolve optional providers via an injectable import function."""
    synced_mod = None
    lyriq_get = None
    synced_available = False
    lyriq_available = False

    try:
        synced_mod = import_fn("syncedlyrics")
        synced_available = True
    except ImportError:
        synced_mod = None

    try:
        lyriq_mod = import_fn("lyriq", fromlist=["get_lyrics"])
        lyriq_get = getattr(lyriq_mod, "get_lyrics", None)
        lyriq_available = lyriq_get is not None
    except ImportError:
        lyriq_get = None

    return synced_mod, synced_available, lyriq_get, lyriq_available


syncedlyrics, SYNCEDLYRICS_AVAILABLE, lyriq_get_lyrics, LYRIQ_AVAILABLE = (
    _resolve_sync_dependencies()
)

# Provider order for syncedlyrics: prioritize more reliable sources
# Note: lyriq (LRCLib) is tried first before syncedlyrics providers
# Musixmatch: Best quality but has rate limits
# NetEase: Good coverage, especially for Asian music
# Megalobiz: Less reliable but can have unique content
# Lrclib: Moved to end since lyriq already uses LRCLib API with potentially better search
# Genius: Plain text only (not useful for synced lyrics, kept as last resort)
PROVIDER_ORDER = ["Musixmatch", "NetEase", "Megalobiz", "Lrclib", "Genius"]


# Providers that have shown persistent failures (skip after repeated errors)
_DEFAULT_SYNC_STATE = create_sync_state()
_failed_providers = _DEFAULT_SYNC_STATE.failed_providers
_FAILURE_THRESHOLD = 3  # Skip provider after this many consecutive failures


def _suppress_stderr():
    return sync_search.suppress_stderr()


def _state_or_default(state: Optional[SyncState]) -> SyncState:
    return state or _DEFAULT_SYNC_STATE


def _get_syncedlyrics_module(state: Optional[SyncState] = None):
    runtime_state = _state_or_default(state)
    return (
        runtime_state.syncedlyrics_mod
        if runtime_state.syncedlyrics_mod is not None
        else syncedlyrics
    )


def _is_syncedlyrics_available(state: Optional[SyncState] = None) -> bool:
    runtime_state = _state_or_default(state)
    if runtime_state.syncedlyrics_available is not None:
        return runtime_state.syncedlyrics_available
    return SYNCEDLYRICS_AVAILABLE


def _is_lyriq_available(state: Optional[SyncState] = None) -> bool:
    runtime_state = _state_or_default(state)
    if runtime_state.lyriq_available is not None:
        return runtime_state.lyriq_available
    return LYRIQ_AVAILABLE


def _has_timestamps_for_state(lrc_text: str, state: Optional[SyncState] = None) -> bool:
    runtime_state = _state_or_default(state)
    fn = runtime_state.has_timestamps_fn or _has_timestamps
    return fn(lrc_text)


def _get_lrc_duration_for_state(
    lrc_text: str, state: Optional[SyncState] = None
) -> Optional[int]:
    runtime_state = _state_or_default(state)
    fn = runtime_state.get_lrc_duration_fn or get_lrc_duration
    return fn(lrc_text)


def _search_single_provider(
    search_term: str,
    provider: str,
    synced_only: bool = True,
    enhanced: bool = False,
    max_retries: int = 2,
    retry_delay: float = 1.0,
    state: Optional[SyncState] = None,
) -> Optional[str]:
    """Search a single provider with retry logic."""
    runtime_state = _state_or_default(state)
    if runtime_state.search_single_provider_fn is not None:
        attempts: List[Dict[str, Any]] = [
            {
                "synced_only": synced_only,
                "enhanced": enhanced,
                "max_retries": max_retries,
                "retry_delay": retry_delay,
                "state": runtime_state,
            },
            {
                "synced_only": synced_only,
                "enhanced": enhanced,
                "state": runtime_state,
            },
            {"synced_only": synced_only, "enhanced": enhanced},
        ]
        last_error: Optional[Exception] = None
        for kwargs in attempts:
            try:
                return runtime_state.search_single_provider_fn(
                    search_term,
                    provider,
                    **kwargs,
                )
            except TypeError as e:
                if "unexpected keyword argument" not in str(e):
                    raise
                last_error = e
                continue
        if last_error is not None:
            raise last_error
        return None
    syncedlyrics_mod = _get_syncedlyrics_module(runtime_state)
    if syncedlyrics_mod is None:
        return None
    return sync_search.search_single_provider(
        syncedlyrics_mod.search,
        search_term=search_term,
        provider=provider,
        failed_providers=runtime_state.failed_providers,
        failure_threshold=_FAILURE_THRESHOLD,
        logger=logger,
        synced_only=synced_only,
        enhanced=enhanced,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )


# Back-compat aliases of default in-memory state
_search_cache = _DEFAULT_SYNC_STATE.search_cache
_disk_cache = _DEFAULT_SYNC_STATE.disk_cache
_disk_cache_loaded = _DEFAULT_SYNC_STATE.disk_cache_loaded
_lrc_cache = _DEFAULT_SYNC_STATE.lrc_cache
_lyriq_cache = _DEFAULT_SYNC_STATE.lyriq_cache
_all_sources_cache = _DEFAULT_SYNC_STATE.all_sources_cache


def _bind_default_state(state: SyncState) -> None:
    """Bind module-level back-compat aliases to a new default state."""
    global _DEFAULT_SYNC_STATE
    global _failed_providers, _search_cache, _disk_cache, _disk_cache_loaded
    global _lrc_cache, _lyriq_cache, _all_sources_cache

    _DEFAULT_SYNC_STATE = state
    _failed_providers = state.failed_providers
    _search_cache = state.search_cache
    _disk_cache = state.disk_cache
    _disk_cache_loaded = state.disk_cache_loaded
    _lrc_cache = state.lrc_cache
    _lyriq_cache = state.lyriq_cache
    _all_sources_cache = state.all_sources_cache


@contextmanager
def use_sync_state(state: SyncState):
    """Temporarily use a sync state as the module default."""
    previous_state = _DEFAULT_SYNC_STATE
    _bind_default_state(state)
    try:
        yield state
    finally:
        _bind_default_state(previous_state)


def _search_with_fallback(
    search_term: str,
    synced_only: bool = True,
    enhanced: bool = False,
    state: Optional[SyncState] = None,
) -> Tuple[Optional[str], str]:
    """Search across providers with fallback."""
    runtime_state = _state_or_default(state)
    override_result = _search_with_runtime_override(
        search_term,
        runtime_state=runtime_state,
        synced_only=synced_only,
        enhanced=enhanced,
    )
    if override_result is not None:
        return override_result

    return sync_search.search_with_fallback(
        search_term=search_term,
        provider_order=PROVIDER_ORDER,
        search_single_provider_fn=lambda term, provider, synced_only=True, enhanced=False: _search_single_with_state_fallback(
            term,
            provider,
            synced_only=synced_only,
            enhanced=enhanced,
            runtime_state=runtime_state,
        ),
        search_cache=runtime_state.search_cache,
        disk_cache=runtime_state.disk_cache,
        disk_cache_enabled=_disk_cache_enabled(runtime_state),
        load_disk_cache_fn=lambda: _load_disk_cache(runtime_state),
        save_disk_cache_fn=lambda: _save_disk_cache(runtime_state),
        logger=logger,
        synced_only=synced_only,
        enhanced=enhanced,
    )


def _search_with_runtime_override(
    search_term: str,
    *,
    runtime_state: SyncState,
    synced_only: bool,
    enhanced: bool,
) -> Tuple[Optional[str], str] | None:
    if runtime_state.search_with_fallback_fn is None:
        return None
    attempts: List[Dict[str, Any]] = [
        {"synced_only": synced_only, "enhanced": enhanced, "state": runtime_state},
        {"synced_only": synced_only, "enhanced": enhanced},
        {"synced_only": synced_only},
    ]
    last_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            return runtime_state.search_with_fallback_fn(search_term, **kwargs)
        except TypeError as error:
            if "unexpected keyword argument" not in str(error):
                raise
            last_error = error
    if last_error is not None:
        raise last_error
    return None


def _search_single_with_state_fallback(
    search_term_inner: str,
    provider_inner: str,
    *,
    synced_only: bool,
    enhanced: bool,
    runtime_state: SyncState,
) -> Optional[str]:
    try:
        return _search_single_provider(
            search_term_inner,
            provider_inner,
            synced_only=synced_only,
            enhanced=enhanced,
            state=runtime_state,
        )
    except TypeError as error:
        if "unexpected keyword argument 'state'" not in str(error):
            raise
        return _search_single_provider(
            search_term_inner,
            provider_inner,
            synced_only=synced_only,
            enhanced=enhanced,
        )


def _search_with_state_fallback(
    search_term: str,
    *,
    synced_only: bool,
    enhanced: bool,
    state: SyncState,
) -> Tuple[Optional[str], str]:
    """Call _search_with_fallback with explicit sync state."""
    return _search_with_fallback(
        search_term,
        synced_only=synced_only,
        enhanced=enhanced,
        state=state,
    )


def _get_disk_cache_path() -> "Path":
    return get_cache_dir() / "lyrics_cache.json"


def _disk_cache_enabled(state: Optional[SyncState] = None) -> bool:
    # Disable caches for tests to keep results deterministic and isolated.
    runtime_state = _state_or_default(state)
    return runtime_state.disk_cache_enabled and "PYTEST_CURRENT_TEST" not in os.environ


def _empty_disk_cache() -> Dict[str, Any]:
    return {
        "search_cache": {},
        "lrc_cache": {},
        "lyriq_cache": {},
        "all_sources_cache": {},
    }


def _load_disk_cache(state: Optional[SyncState] = None) -> None:
    runtime_state = _state_or_default(state)
    if runtime_state.disk_cache_loaded:
        return
    runtime_state.disk_cache_loaded = True
    if not _disk_cache_enabled(runtime_state):
        runtime_state.disk_cache = _empty_disk_cache()
        return
    cache_path = _get_disk_cache_path()
    if not cache_path.exists():
        runtime_state.disk_cache = _empty_disk_cache()
        return
    try:
        runtime_state.disk_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        runtime_state.disk_cache = _empty_disk_cache()


def _save_disk_cache(state: Optional[SyncState] = None) -> None:
    runtime_state = _state_or_default(state)
    if not _disk_cache_enabled(runtime_state):
        return
    cache_path = _get_disk_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache_path.write_text(
            json.dumps(runtime_state.disk_cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except (OSError, ValueError, TypeError):
        logger.debug("Failed to write lyrics cache to disk")


def _set_lrc_cache(
    cache_key: Tuple[str, str],
    value: Tuple[Optional[str], bool, str, Optional[int]],
    state: Optional[SyncState] = None,
) -> None:
    runtime_state = _state_or_default(state)
    runtime_state.lrc_cache[cache_key] = value
    if _disk_cache_enabled(runtime_state):
        disk_key = f"{cache_key[0]}|{cache_key[1]}"
        runtime_state.disk_cache.setdefault("lrc_cache", {})[disk_key] = list(value)
        _save_disk_cache(runtime_state)


def _all_sources_cache_keys(
    title: str, artist: str
) -> list[tuple[tuple[str, str], str]]:
    def _fold(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value or "")
        return "".join(ch for ch in normalized if not unicodedata.combining(ch))

    raw_artist = artist.lower().strip()
    raw_title = title.lower().strip()
    raw_folded_artist = _fold(artist).lower().strip()
    raw_folded_title = _fold(title).lower().strip()
    normalized_artist = _normalize_for_provider_search(artist).lower().strip()
    normalized_title = _normalize_for_provider_search(title).lower().strip()
    normalized_folded_artist = (
        _fold(_normalize_for_provider_search(artist)).lower().strip()
    )
    normalized_folded_title = (
        _fold(_normalize_for_provider_search(title)).lower().strip()
    )
    keys: list[tuple[tuple[str, str], str]] = []
    for pair in [
        (raw_artist, raw_title),
        (raw_folded_artist, raw_folded_title),
        (normalized_artist or raw_artist, normalized_title or raw_title),
        (
            normalized_folded_artist or raw_folded_artist,
            normalized_folded_title or raw_folded_title,
        ),
    ]:
        disk_key = f"{pair[0]}|{pair[1]}"
        if pair not in [existing[0] for existing in keys]:
            keys.append((pair, disk_key))
    normalized_pair = keys[-1][0]
    primary_artist = (
        normalized_pair[0]
        .split(",", 1)[0]
        .split(" feat", 1)[0]
        .split(" featuring", 1)[0]
        .split(" & ", 1)[0]
        .strip()
    )
    for pair in [
        normalized_pair,
        (primary_artist or normalized_pair[0], normalized_pair[1]),
    ]:
        disk_key = f"{pair[0]}|{pair[1]}"
        if pair != keys[-1][0] and pair not in [existing[0] for existing in keys]:
            keys.append((pair, disk_key))
    return keys


def _serialize_all_sources_result(
    value: Dict[str, Tuple[Optional[str], Optional[int]]],
) -> Dict[str, list[Any]]:
    return {
        str(source): [payload[0], payload[1]]
        for source, payload in value.items()
        if isinstance(source, str) and isinstance(payload, tuple) and len(payload) == 2
    }


def _deserialize_all_sources_result(
    raw: Any,
) -> Dict[str, Tuple[Optional[str], Optional[int]]]:
    if not isinstance(raw, dict):
        return {}
    result: Dict[str, Tuple[Optional[str], Optional[int]]] = {}
    for source, payload in raw.items():
        if not isinstance(source, str):
            continue
        if not isinstance(payload, (list, tuple)) or len(payload) != 2:
            continue
        text = payload[0] if isinstance(payload[0], (str, type(None))) else None
        duration = (
            payload[1] if isinstance(payload[1], (int, float, type(None))) else None
        )
        result[source] = (
            text,
            int(duration) if isinstance(duration, (int, float)) else None,
        )
    return result


def _fold_cache_component(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return (
        "".join(ch for ch in normalized if not unicodedata.combining(ch))
        .lower()
        .strip()
    )


def _artist_aliases(value: str) -> set[str]:
    folded = _fold_cache_component(value)
    primary = (
        folded.split(",", 1)[0]
        .split(" feat", 1)[0]
        .split(" featuring", 1)[0]
        .split(" & ", 1)[0]
        .strip()
    )
    aliases = {folded}
    if primary:
        aliases.add(primary)
    return aliases


def _fetch_from_lyriq(
    title: str,
    artist: str,
    max_retries: int = 2,
    retry_delay: float = 1.0,
    state: Optional[SyncState] = None,
) -> Optional[str]:
    """Fetch lyrics from lyriq (LRCLib API)."""
    runtime_state = _state_or_default(state)
    lyriq_get = runtime_state.lyriq_get_lyrics_fn or lyriq_get_lyrics
    with _suppress_stderr():
        return sync_providers.fetch_from_lyriq(
            title,
            artist,
            max_retries=max_retries,
            retry_delay=retry_delay,
            state=runtime_state,
            lyriq_get_lyrics=lyriq_get,
            disk_cache_enabled_fn=_disk_cache_enabled,
            load_disk_cache_fn=_load_disk_cache,
            save_disk_cache_fn=_save_disk_cache,
            is_lyriq_available_fn=_is_lyriq_available,
            has_timestamps_fn=_has_timestamps_for_state,
            logger=logger,
        )


def fetch_lyrics_multi_source(
    title: str,
    artist: str,
    synced_only: bool = True,
    enhanced: bool = False,
    target_duration: Optional[int] = None,
    duration_tolerance: int = 20,
    offline: bool = False,
    state: Optional[SyncState] = None,
) -> Tuple[Optional[str], bool, str]:
    """Fetch lyrics from multiple sources using lyriq and syncedlyrics."""
    runtime_state = _state_or_default(state)
    return fetch_lyrics_multi_source_impl(
        title,
        artist,
        synced_only,
        enhanced,
        target_duration,
        duration_tolerance,
        offline,
        runtime_state,
        disk_cache_enabled_fn=_disk_cache_enabled,
        load_disk_cache_fn=_load_disk_cache,
        is_lyriq_available_fn=_is_lyriq_available,
        fetch_from_lyriq_fn=_fetch_from_lyriq,
        has_timestamps_fn=_has_timestamps_for_state,
        get_lrc_duration_fn=_get_lrc_duration_for_state,
        set_lrc_cache_fn=_set_lrc_cache,
        is_syncedlyrics_available_fn=_is_syncedlyrics_available,
        search_with_state_fallback_fn=_search_with_state_fallback,
        logger=logger,
    )


def fetch_lyrics_for_duration(
    title: str,
    artist: str,
    target_duration: int,
    tolerance: int = 8,
    offline: bool = False,
    state: Optional[SyncState] = None,
) -> Tuple[Optional[str], bool, str, Optional[int]]:
    """Fetch synced lyrics that match a target duration."""
    runtime_state = _state_or_default(state)
    return sync_providers.fetch_lyrics_for_duration(
        title,
        artist,
        target_duration=target_duration,
        tolerance=tolerance,
        offline=offline,
        state=runtime_state,
        is_syncedlyrics_available_fn=_is_syncedlyrics_available,
        is_lyriq_available_fn=_is_lyriq_available,
        fetch_lyrics_multi_source_fn=fetch_lyrics_multi_source,
        get_lrc_duration_fn=_get_lrc_duration_for_state,
        search_with_state_fallback_fn=_search_with_state_fallback,
        has_timestamps_fn=_has_timestamps_for_state,
        logger=logger,
    )


def fetch_from_all_sources(
    title: str,
    artist: str,
    offline: bool = False,
) -> Dict[str, Tuple[Optional[str], Optional[int]]]:
    """Fetch lyrics from all available sources for comparison."""
    runtime_state = _state_or_default(None)
    lyriq_get = runtime_state.lyriq_get_lyrics_fn or lyriq_get_lyrics
    cache_keys = _all_sources_cache_keys(title, artist)
    if _disk_cache_enabled(runtime_state):
        _load_disk_cache(runtime_state)
    cached = _get_cached_all_sources(runtime_state, cache_keys, offline=offline)
    if cached is not None:
        return cached
    if offline:
        return {}
    results = sync_providers.fetch_from_all_sources(
        title,
        artist,
        state=runtime_state,
        is_lyriq_available_fn=_is_lyriq_available,
        is_syncedlyrics_available_fn=_is_syncedlyrics_available,
        get_syncedlyrics_module_fn=_get_syncedlyrics_module,
        lyriq_get_lyrics=lyriq_get,
        has_timestamps_fn=_has_timestamps_for_state,
        get_lrc_duration_fn=_get_lrc_duration_for_state,
        provider_order=PROVIDER_ORDER,
        suppress_stderr=_suppress_stderr,
        logger=logger,
    )
    for cache_key, disk_key in cache_keys:
        runtime_state.all_sources_cache[cache_key] = results
        if _disk_cache_enabled(runtime_state):
            runtime_state.disk_cache.setdefault("all_sources_cache", {})[disk_key] = (
                _serialize_all_sources_result(results)
            )
    if _disk_cache_enabled(runtime_state):
        _save_disk_cache(runtime_state)
    return results


def _get_cached_all_sources(
    runtime_state: SyncState,
    cache_keys: list[tuple[tuple[str, str], str]],
    *,
    offline: bool,
) -> Optional[Dict[str, Tuple[Optional[str], Optional[int]]]]:
    cached = _get_runtime_cached_all_sources(runtime_state, cache_keys, offline=offline)
    if cached is not None:
        return cached
    if not _disk_cache_enabled(runtime_state):
        return None
    return _get_disk_cached_all_sources(runtime_state, cache_keys, offline=offline)


def _get_runtime_cached_all_sources(
    runtime_state: SyncState,
    cache_keys: list[tuple[tuple[str, str], str]],
    *,
    offline: bool,
) -> Optional[Dict[str, Tuple[Optional[str], Optional[int]]]]:
    for cache_key, _disk_key in cache_keys:
        cached = runtime_state.all_sources_cache.get(cache_key)
        if cached and (offline or cached):
            return cached
    folded_title, folded_artist_aliases = _cache_lookup_tokens(cache_keys)
    for existing_key, cached in runtime_state.all_sources_cache.items():
        if _is_cache_key_alias_match(existing_key, folded_title, folded_artist_aliases):
            if offline or cached:
                return cached
    return None


def _get_disk_cached_all_sources(
    runtime_state: SyncState,
    cache_keys: list[tuple[tuple[str, str], str]],
    *,
    offline: bool,
) -> Optional[Dict[str, Tuple[Optional[str], Optional[int]]]]:
    disk_cache = runtime_state.disk_cache.get("all_sources_cache", {})
    for cache_key, disk_key in cache_keys:
        disk_cached = _deserialize_all_sources_result(disk_cache.get(disk_key, {}))
        if disk_cached:
            runtime_state.all_sources_cache[cache_key] = disk_cached
            if offline:
                return disk_cached
    folded_title, folded_artist_aliases = _cache_lookup_tokens(cache_keys)
    for raw_key, payload in disk_cache.items():
        if not isinstance(raw_key, str):
            continue
        parts = raw_key.split("|", 1)
        if len(parts) != 2 or not _is_cache_key_alias_match(
            (parts[0], parts[1]), folded_title, folded_artist_aliases
        ):
            continue
        disk_cached = _deserialize_all_sources_result(payload)
        if not disk_cached:
            continue
        runtime_state.all_sources_cache[(parts[0], parts[1])] = disk_cached
        if offline:
            return disk_cached
    return None


def _cache_lookup_tokens(
    cache_keys: list[tuple[tuple[str, str], str]],
) -> tuple[str, set[str]]:
    folded_title = _fold_cache_component(cache_keys[0][0][1])
    folded_artist_aliases = {
        alias
        for cache_key, _disk_key in cache_keys
        for alias in _artist_aliases(cache_key[0])
    }
    return folded_title, folded_artist_aliases


def _is_cache_key_alias_match(
    cache_key: tuple[str, str],
    folded_title: str,
    folded_artist_aliases: set[str],
) -> bool:
    return _fold_cache_component(cache_key[1]) == folded_title and bool(
        _artist_aliases(cache_key[0]).intersection(folded_artist_aliases)
    )
