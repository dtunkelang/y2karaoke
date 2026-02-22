"""Synced lyrics fetching using syncedlyrics and lyriq libraries."""

import logging
import json
import os
import time
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, List

from ....config import get_cache_dir
from ....utils.logging import get_logger
from . import sync_quality
from . import sync_search
from . import sync_providers
from .sync_pipeline import fetch_lyrics_multi_source_impl

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
    disk_cache: Dict[str, Any] = field(
        default_factory=lambda: {"search_cache": {}, "lrc_cache": {}, "lyriq_cache": {}}
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


def _bind_default_state(state: SyncState) -> None:
    """Bind module-level back-compat aliases to a new default state."""
    global _DEFAULT_SYNC_STATE
    global _failed_providers, _search_cache, _disk_cache, _disk_cache_loaded
    global _lrc_cache, _lyriq_cache

    _DEFAULT_SYNC_STATE = state
    _failed_providers = state.failed_providers
    _search_cache = state.search_cache
    _disk_cache = state.disk_cache
    _disk_cache_loaded = state.disk_cache_loaded
    _lrc_cache = state.lrc_cache
    _lyriq_cache = state.lyriq_cache


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
    if runtime_state.search_with_fallback_fn is not None:
        attempts: List[Dict[str, Any]] = [
            {"synced_only": synced_only, "enhanced": enhanced, "state": runtime_state},
            {"synced_only": synced_only, "enhanced": enhanced},
            {"synced_only": synced_only},
        ]
        last_error: Optional[Exception] = None
        for kwargs in attempts:
            try:
                return runtime_state.search_with_fallback_fn(search_term, **kwargs)
            except TypeError as e:
                if "unexpected keyword argument" not in str(e):
                    raise
                last_error = e
                continue
        if last_error is not None:
            raise last_error
        return None, ""

    def _search_single(
        search_term_inner: str,
        provider_inner: str,
        synced_only: bool = True,
        enhanced: bool = False,
    ) -> Optional[str]:
        try:
            return _search_single_provider(
                search_term_inner,
                provider_inner,
                synced_only=synced_only,
                enhanced=enhanced,
                state=runtime_state,
            )
        except TypeError as e:
            if "unexpected keyword argument 'state'" not in str(e):
                raise
            return _search_single_provider(
                search_term_inner,
                provider_inner,
                synced_only=synced_only,
                enhanced=enhanced,
            )

    return sync_search.search_with_fallback(
        search_term=search_term,
        provider_order=PROVIDER_ORDER,
        search_single_provider_fn=_search_single,
        search_cache=runtime_state.search_cache,
        disk_cache=runtime_state.disk_cache,
        disk_cache_enabled=_disk_cache_enabled(runtime_state),
        load_disk_cache_fn=lambda: _load_disk_cache(runtime_state),
        save_disk_cache_fn=lambda: _save_disk_cache(runtime_state),
        logger=logger,
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
    return {"search_cache": {}, "lrc_cache": {}, "lyriq_cache": {}}


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


def fetch_lyrics_multi_source(  # noqa: C901
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
) -> Dict[str, Tuple[Optional[str], Optional[int]]]:
    """Fetch lyrics from all available sources for comparison."""
    runtime_state = _state_or_default(None)
    lyriq_get = runtime_state.lyriq_get_lyrics_fn or lyriq_get_lyrics
    return sync_providers.fetch_from_all_sources(
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
