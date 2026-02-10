"""Pipeline implementations for sync lyrics orchestration."""

from typing import Any, Callable, Optional, Tuple


def fetch_lyrics_multi_source_impl(  # noqa: C901
    title: str,
    artist: str,
    synced_only: bool,
    enhanced: bool,
    target_duration: Optional[int],
    duration_tolerance: int,
    offline: bool,
    runtime_state: Any,
    *,
    disk_cache_enabled_fn: Callable[..., bool],
    load_disk_cache_fn: Callable[..., None],
    is_lyriq_available_fn: Callable[..., bool],
    fetch_from_lyriq_fn: Callable[..., Optional[str]],
    has_timestamps_fn: Callable[..., bool],
    get_lrc_duration_fn: Callable[..., Optional[int]],
    set_lrc_cache_fn: Callable[..., None],
    is_syncedlyrics_available_fn: Callable[..., bool],
    search_with_state_fallback_fn: Callable[..., Tuple[Optional[str], str]],
    logger,
) -> Tuple[Optional[str], bool, str]:
    """Fetch lyrics from multiple sources using lyriq and syncedlyrics."""
    disk_cache_enabled = disk_cache_enabled_fn(runtime_state)
    if disk_cache_enabled:
        load_disk_cache_fn(runtime_state)

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
        if is_lyriq_available_fn(runtime_state):
            logger.debug(f"Trying lyriq for: {title} - {artist}")
            lrc = fetch_from_lyriq_fn(title, artist, state=runtime_state)
            if lrc and has_timestamps_fn(lrc, runtime_state):
                logger.debug("Found synced lyrics from lyriq (LRCLib)")
                lrc_duration = get_lrc_duration_fn(lrc, runtime_state)
                result = (lrc, True, "lyriq (LRCLib)", lrc_duration)
                set_lrc_cache_fn(cache_key, result, state=runtime_state)
                return (lrc, True, "lyriq (LRCLib)")

        if not is_syncedlyrics_available_fn(runtime_state):
            logger.warning("syncedlyrics not installed")
            no_sync_result = (None, False, "", None)
            set_lrc_cache_fn(cache_key, no_sync_result, state=runtime_state)
            return (None, False, "")

        if enhanced:
            lrc, provider = search_with_state_fallback_fn(
                search_term,
                synced_only=True,
                enhanced=True,
                state=runtime_state,
            )
            if lrc and has_timestamps_fn(lrc, runtime_state):
                logger.debug(
                    f"Found enhanced (word-level) synced lyrics from {provider}"
                )
                lrc_duration = get_lrc_duration_fn(lrc, runtime_state)
                result = (lrc, True, f"{provider} (enhanced)", lrc_duration)
                set_lrc_cache_fn(cache_key, result, state=runtime_state)
                return (lrc, True, f"{provider} (enhanced)")

        lrc, provider = search_with_state_fallback_fn(
            search_term,
            synced_only=synced_only,
            enhanced=False,
            state=runtime_state,
        )

        if lrc:
            is_synced = has_timestamps_fn(lrc, runtime_state)
            if is_synced:
                logger.debug(f"Found synced lyrics from {provider}")
                lrc_duration = get_lrc_duration_fn(lrc, runtime_state)
                result = (lrc, True, provider, lrc_duration)
                set_lrc_cache_fn(cache_key, result, state=runtime_state)
                return (lrc, True, provider)
            if not synced_only:
                logger.debug(f"Found plain lyrics from {provider}")
                result = (lrc, False, provider, None)
                set_lrc_cache_fn(cache_key, result, state=runtime_state)
                return (lrc, False, provider)

        logger.warning("No synced lyrics found from any provider")
        not_found = (None, False, "", None)
        set_lrc_cache_fn(cache_key, not_found, state=runtime_state)
        return (None, False, "")

    except Exception as e:
        logger.error(f"Error fetching synced lyrics: {e}")
        error_result = (None, False, "", None)
        set_lrc_cache_fn(cache_key, error_result, state=runtime_state)
        return (None, False, "")
