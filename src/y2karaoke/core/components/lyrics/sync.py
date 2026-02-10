"""Synced lyrics fetching using syncedlyrics and lyriq libraries."""

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

logger = get_logger(__name__)

_has_timestamps = sync_quality._has_timestamps
get_lrc_duration = sync_quality.get_lrc_duration
validate_lrc_quality = sync_quality.validate_lrc_quality
_count_large_gaps = sync_quality._count_large_gaps
_calculate_quality_score = sync_quality._calculate_quality_score
get_lyrics_quality_report = sync_quality.get_lyrics_quality_report


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
    if syncedlyrics is None:
        return None
    return sync_search.search_single_provider(
        syncedlyrics.search,
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


def set_default_sync_state(state: SyncState) -> None:
    """Set the process-wide default sync state."""
    _bind_default_state(state)


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
    """Call _search_with_fallback with state, tolerating monkeypatched signatures."""
    attempts: List[Dict[str, Any]] = [
        {"synced_only": synced_only, "enhanced": enhanced, "state": state},
        {"synced_only": synced_only, "enhanced": enhanced},
        {"synced_only": synced_only, "state": state},
        {"synced_only": synced_only},
    ]
    last_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            return _search_with_fallback(search_term, **kwargs)
        except TypeError as e:
            if "unexpected keyword argument" not in str(e):
                raise
            last_error = e
            continue
    if last_error is not None:
        raise last_error
    return None, ""


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
    except Exception:
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
    except Exception:
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


def _fetch_from_lyriq(  # noqa: C901
    title: str,
    artist: str,
    max_retries: int = 2,
    retry_delay: float = 1.0,
    state: Optional[SyncState] = None,
) -> Optional[str]:
    """Fetch lyrics from lyriq (LRCLib API).

    lyriq provides a different search/matching algorithm than syncedlyrics'
    Lrclib provider, so it may find different results.
    """
    runtime_state = _state_or_default(state)
    lyriq_get = runtime_state.lyriq_get_lyrics_fn or lyriq_get_lyrics
    disk_cache_enabled = _disk_cache_enabled(runtime_state)
    if disk_cache_enabled:
        _load_disk_cache(runtime_state)
    if not LYRIQ_AVAILABLE:
        return None

    cache_key = (artist.lower().strip(), title.lower().strip())
    disk_key = f"{cache_key[0]}|{cache_key[1]}"
    if cache_key in runtime_state.lyriq_cache:
        return runtime_state.lyriq_cache[cache_key]
    if disk_cache_enabled:
        disk_lyriq = runtime_state.disk_cache.get("lyriq_cache", {})
        if disk_key in disk_lyriq:
            cached = disk_lyriq[disk_key]
            runtime_state.lyriq_cache[cache_key] = cached
            return cached

    for attempt in range(max_retries + 1):
        try:
            with _suppress_stderr():
                lyrics_obj = lyriq_get(title, artist) if lyriq_get else None

            if lyrics_obj is None:
                runtime_state.lyriq_cache[cache_key] = None
                if disk_cache_enabled:
                    runtime_state.disk_cache.setdefault("lyriq_cache", {})[
                        disk_key
                    ] = None
                    _save_disk_cache(runtime_state)
                return None

            synced = getattr(lyrics_obj, "synced_lyrics", None)
            if synced and _has_timestamps(synced):
                logger.debug("Found synced lyrics from lyriq (LRCLib)")
                runtime_state.lyriq_cache[cache_key] = synced
                if disk_cache_enabled:
                    runtime_state.disk_cache.setdefault("lyriq_cache", {})[
                        disk_key
                    ] = synced
                    _save_disk_cache(runtime_state)
                return synced

            plain = getattr(lyrics_obj, "plain_lyrics", None)
            if plain:
                logger.debug("Found plain lyrics from lyriq (no timestamps)")
                runtime_state.lyriq_cache[cache_key] = None
                if disk_cache_enabled:
                    runtime_state.disk_cache.setdefault("lyriq_cache", {})[
                        disk_key
                    ] = None
                    _save_disk_cache(runtime_state)
                return None

            runtime_state.lyriq_cache[cache_key] = None
            if disk_cache_enabled:
                runtime_state.disk_cache.setdefault("lyriq_cache", {})[disk_key] = None
                _save_disk_cache(runtime_state)
            return None

        except Exception as e:
            error_msg = str(e).lower()
            is_transient = any(
                x in error_msg
                for x in [
                    "connection",
                    "timeout",
                    "temporarily",
                    "rate limit",
                    "remote end closed",
                    "429",
                    "503",
                    "502",
                ]
            )

            if is_transient and attempt < max_retries:
                delay = retry_delay * (2**attempt)
                logger.debug(f"lyriq transient error, retrying in {delay:.1f}s: {e}")
                runtime_state.sleep_fn(delay)
                continue

            logger.debug(f"lyriq failed (attempt {attempt + 1}): {e}")
            runtime_state.lyriq_cache[cache_key] = None
            if disk_cache_enabled:
                runtime_state.disk_cache.setdefault("lyriq_cache", {})[disk_key] = None
                _save_disk_cache(runtime_state)
            return None

    runtime_state.lyriq_cache[cache_key] = None
    if disk_cache_enabled:
        runtime_state.disk_cache.setdefault("lyriq_cache", {})[disk_key] = None
        _save_disk_cache(runtime_state)
    return None


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
    disk_cache_enabled = _disk_cache_enabled(runtime_state)
    if disk_cache_enabled:
        _load_disk_cache(runtime_state)

    cache_key = (artist.lower().strip(), title.lower().strip())
    if cache_key in runtime_state.lrc_cache:
        cached = runtime_state.lrc_cache[cache_key]
        cached_duration = cached[3] if len(cached) > 3 else None
        if target_duration and cached_duration:
            if abs(cached_duration - target_duration) <= duration_tolerance:
                logger.debug(f"Using cached LRC result for {artist} - {title}")
                return (cached[0], cached[1], cached[2])
            logger.debug(
                "Cached LRC duration (%ss) doesn't match target (%ss), re-fetching",
                cached_duration,
                target_duration,
            )
        else:
            logger.debug(f"Using cached LRC result for {artist} - {title}")
            return (cached[0], cached[1], cached[2])

    if disk_cache_enabled:
        disk_key = f"{cache_key[0]}|{cache_key[1]}"
        disk_lrc = runtime_state.disk_cache.get("lrc_cache", {})
        if disk_key in disk_lrc:
            cached = disk_lrc[disk_key]
            cached_duration = cached[3] if len(cached) > 3 else None
            if target_duration and cached_duration:
                if abs(cached_duration - target_duration) <= duration_tolerance:
                    runtime_state.lrc_cache[cache_key] = tuple(cached)
                    logger.debug(f"Using cached LRC result for {artist} - {title}")
                    return (cached[0], cached[1], cached[2])
                logger.debug(
                    "Cached LRC duration (%ss) doesn't match target (%ss), re-fetching",
                    cached_duration,
                    target_duration,
                )
            else:
                runtime_state.lrc_cache[cache_key] = tuple(cached)
                logger.debug(f"Using cached LRC result for {artist} - {title}")
                return (cached[0], cached[1], cached[2])

    if offline:
        logger.info("Offline mode: skipping lyrics providers (cache only)")
        return (None, False, "")

    search_term = f"{artist} {title}"
    logger.debug(f"Searching for synced lyrics: {search_term}")

    try:
        if LYRIQ_AVAILABLE:
            logger.debug(f"Trying lyriq for: {title} - {artist}")
            lrc = _fetch_from_lyriq(title, artist, state=runtime_state)
            if lrc and _has_timestamps(lrc):
                logger.debug("Found synced lyrics from lyriq (LRCLib)")
                lrc_duration = get_lrc_duration(lrc)
                result = (lrc, True, "lyriq (LRCLib)", lrc_duration)
                _set_lrc_cache(cache_key, result, state=runtime_state)
                return (lrc, True, "lyriq (LRCLib)")

        if not SYNCEDLYRICS_AVAILABLE:
            logger.warning("syncedlyrics not installed")
            no_sync_result: Tuple[Optional[str], bool, str, Optional[int]] = (
                None,
                False,
                "",
                None,
            )
            _set_lrc_cache(cache_key, no_sync_result, state=runtime_state)
            return (None, False, "")

        if enhanced:
            lrc, provider = _search_with_state_fallback(
                search_term,
                synced_only=True,
                enhanced=True,
                state=runtime_state,
            )
            if lrc and _has_timestamps(lrc):
                logger.debug(
                    f"Found enhanced (word-level) synced lyrics from {provider}"
                )
                lrc_duration = get_lrc_duration(lrc)
                result = (lrc, True, f"{provider} (enhanced)", lrc_duration)
                _set_lrc_cache(cache_key, result, state=runtime_state)
                return (lrc, True, f"{provider} (enhanced)")

        lrc, provider = _search_with_state_fallback(
            search_term,
            synced_only=synced_only,
            enhanced=False,
            state=runtime_state,
        )

        if lrc:
            is_synced = _has_timestamps(lrc)
            if is_synced:
                logger.debug(f"Found synced lyrics from {provider}")
                lrc_duration = get_lrc_duration(lrc)
                result = (lrc, True, provider, lrc_duration)
                _set_lrc_cache(cache_key, result, state=runtime_state)
                return (lrc, True, provider)
            if not synced_only:
                logger.debug(f"Found plain lyrics from {provider}")
                result = (lrc, False, provider, None)
                _set_lrc_cache(cache_key, result, state=runtime_state)
                return (lrc, False, provider)

        logger.warning("No synced lyrics found from any provider")
        not_found: Tuple[Optional[str], bool, str, Optional[int]] = (
            None,
            False,
            "",
            None,
        )
        _set_lrc_cache(cache_key, not_found, state=runtime_state)
        return (None, False, "")

    except Exception as e:
        logger.error(f"Error fetching synced lyrics: {e}")
        error_result: Tuple[Optional[str], bool, str, Optional[int]] = (
            None,
            False,
            "",
            None,
        )
        _set_lrc_cache(cache_key, error_result, state=runtime_state)
        return (None, False, "")


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
    if offline:
        logger.info("Offline mode: skipping lyrics providers (cache only)")
        return None, False, "", None

    if not SYNCEDLYRICS_AVAILABLE and not LYRIQ_AVAILABLE:
        logger.warning("Neither syncedlyrics nor lyriq installed")
        return None, False, "", None

    lrc_text, is_synced, source = fetch_lyrics_multi_source(
        title,
        artist,
        synced_only=True,
        target_duration=target_duration,
        duration_tolerance=tolerance,
        offline=offline,
        state=runtime_state,
    )

    if is_synced and lrc_text:
        lrc_duration = get_lrc_duration(lrc_text)
        if lrc_duration:
            diff = abs(lrc_duration - target_duration)
            if diff <= tolerance:
                logger.info(
                    f"Found LRC with matching duration: {lrc_duration}s (target: {target_duration}s)"
                )
                return lrc_text, is_synced, source, lrc_duration
            logger.warning(
                f"LRC duration mismatch: LRC={lrc_duration}s, target={target_duration}s, diff={diff}s"
            )

    if SYNCEDLYRICS_AVAILABLE and not offline:
        alternative_searches = [
            f"{title} {artist}",
            f"{artist} {title} official",
            f"{artist} {title} album version",
        ]

        for search_term in alternative_searches:
            logger.debug(f"Trying alternative LRC search: {search_term}")
            lrc, provider = _search_with_state_fallback(
                search_term,
                synced_only=True,
                enhanced=False,
                state=runtime_state,
            )
            if lrc and _has_timestamps(lrc):
                alt_duration = get_lrc_duration(lrc)
                if alt_duration:
                    diff = abs(alt_duration - target_duration)
                    if diff <= tolerance:
                        logger.info(
                            f"Found LRC with alternative search '{search_term}' from {provider}: {alt_duration}s"
                        )
                        return lrc, True, f"{provider} ({search_term})", alt_duration

    if is_synced and lrc_text:
        lrc_duration = get_lrc_duration(lrc_text)
        logger.warning("Using LRC despite duration mismatch. LRC timing may be off.")
        return lrc_text, is_synced, source, lrc_duration

    return None, False, "", None


def fetch_from_all_sources(
    title: str,
    artist: str,
) -> Dict[str, Tuple[Optional[str], Optional[int]]]:
    """Fetch lyrics from all available sources for comparison."""
    results: Dict[str, Tuple[Optional[str], Optional[int]]] = {}

    if LYRIQ_AVAILABLE:
        try:
            with _suppress_stderr():
                lyrics_obj = lyriq_get_lyrics(title, artist)
            if lyrics_obj:
                synced = getattr(lyrics_obj, "synced_lyrics", None)
                if synced and _has_timestamps(synced):
                    duration = get_lrc_duration(synced)
                    results["lyriq (LRCLib)"] = (synced, duration)
        except Exception as e:
            logger.debug(f"lyriq fetch failed for comparison: {e}")

    if SYNCEDLYRICS_AVAILABLE:
        search_term = f"{artist} {title}"
        for provider in PROVIDER_ORDER:
            if provider == "Genius":
                continue
            try:
                with _suppress_stderr():
                    lrc = syncedlyrics.search(
                        search_term,
                        providers=[provider],
                        synced_only=True,
                    )
                if lrc and _has_timestamps(lrc):
                    duration = get_lrc_duration(lrc)
                    results[provider] = (lrc, duration)
            except Exception as e:
                logger.debug(f"{provider} fetch failed for comparison: {e}")

    return results
