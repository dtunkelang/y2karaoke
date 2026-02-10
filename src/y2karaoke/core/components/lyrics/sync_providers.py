"""Provider-facing sync lyrics helpers extracted from sync.py."""

from typing import Any, Callable, Dict, Optional, Tuple


def fetch_from_lyriq(  # noqa: C901
    title: str,
    artist: str,
    *,
    max_retries: int,
    retry_delay: float,
    state: Any,
    lyriq_get_lyrics: Optional[Callable[..., Any]],
    disk_cache_enabled_fn: Callable[[Any], bool],
    load_disk_cache_fn: Callable[[Any], None],
    save_disk_cache_fn: Callable[[Any], None],
    is_lyriq_available_fn: Callable[[Any], bool],
    has_timestamps_fn: Callable[[str, Any], bool],
    logger: Any,
) -> Optional[str]:
    """Fetch lyrics from lyriq (LRCLib API)."""
    disk_cache_enabled = disk_cache_enabled_fn(state)
    if disk_cache_enabled:
        load_disk_cache_fn(state)
    if not is_lyriq_available_fn(state):
        return None

    cache_key = (artist.lower().strip(), title.lower().strip())
    disk_key = f"{cache_key[0]}|{cache_key[1]}"
    if cache_key in state.lyriq_cache:
        return state.lyriq_cache[cache_key]
    if disk_cache_enabled:
        disk_lyriq = state.disk_cache.get("lyriq_cache", {})
        if disk_key in disk_lyriq:
            cached = disk_lyriq[disk_key]
            state.lyriq_cache[cache_key] = cached
            return cached

    for attempt in range(max_retries + 1):
        try:
            lyrics_obj = lyriq_get_lyrics(title, artist) if lyriq_get_lyrics else None

            if lyrics_obj is None:
                state.lyriq_cache[cache_key] = None
                if disk_cache_enabled:
                    state.disk_cache.setdefault("lyriq_cache", {})[disk_key] = None
                    save_disk_cache_fn(state)
                return None

            synced = getattr(lyrics_obj, "synced_lyrics", None)
            if synced and has_timestamps_fn(synced, state):
                logger.debug("Found synced lyrics from lyriq (LRCLib)")
                state.lyriq_cache[cache_key] = synced
                if disk_cache_enabled:
                    state.disk_cache.setdefault("lyriq_cache", {})[disk_key] = synced
                    save_disk_cache_fn(state)
                return synced

            plain = getattr(lyrics_obj, "plain_lyrics", None)
            if plain:
                logger.debug("Found plain lyrics from lyriq (no timestamps)")
                state.lyriq_cache[cache_key] = None
                if disk_cache_enabled:
                    state.disk_cache.setdefault("lyriq_cache", {})[disk_key] = None
                    save_disk_cache_fn(state)
                return None

            state.lyriq_cache[cache_key] = None
            if disk_cache_enabled:
                state.disk_cache.setdefault("lyriq_cache", {})[disk_key] = None
                save_disk_cache_fn(state)
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
                state.sleep_fn(delay)
                continue

            logger.debug(f"lyriq failed (attempt {attempt + 1}): {e}")
            state.lyriq_cache[cache_key] = None
            if disk_cache_enabled:
                state.disk_cache.setdefault("lyriq_cache", {})[disk_key] = None
                save_disk_cache_fn(state)
            return None

    state.lyriq_cache[cache_key] = None
    if disk_cache_enabled:
        state.disk_cache.setdefault("lyriq_cache", {})[disk_key] = None
        save_disk_cache_fn(state)
    return None


def fetch_lyrics_for_duration(
    title: str,
    artist: str,
    *,
    target_duration: int,
    tolerance: int,
    offline: bool,
    state: Any,
    is_syncedlyrics_available_fn: Callable[[Any], bool],
    is_lyriq_available_fn: Callable[[Any], bool],
    fetch_lyrics_multi_source_fn: Callable[..., Tuple[Optional[str], bool, str]],
    get_lrc_duration_fn: Callable[[str, Any], Optional[int]],
    search_with_state_fallback_fn: Callable[..., Tuple[Optional[str], str]],
    has_timestamps_fn: Callable[[str, Any], bool],
    logger: Any,
) -> Tuple[Optional[str], bool, str, Optional[int]]:
    """Fetch synced lyrics that match a target duration."""
    if offline:
        logger.info("Offline mode: skipping lyrics providers (cache only)")
        return None, False, "", None

    if not is_syncedlyrics_available_fn(state) and not is_lyriq_available_fn(state):
        logger.warning("Neither syncedlyrics nor lyriq installed")
        return None, False, "", None

    lrc_text, is_synced, source = fetch_lyrics_multi_source_fn(
        title,
        artist,
        synced_only=True,
        target_duration=target_duration,
        duration_tolerance=tolerance,
        offline=offline,
        state=state,
    )

    if is_synced and lrc_text:
        lrc_duration = get_lrc_duration_fn(lrc_text, state)
        if lrc_duration:
            diff = abs(lrc_duration - target_duration)
            if diff <= tolerance:
                logger.info(
                    "Found LRC with matching duration: %ss (target: %ss)",
                    lrc_duration,
                    target_duration,
                )
                return lrc_text, is_synced, source, lrc_duration
            logger.warning(
                "LRC duration mismatch: LRC=%ss, target=%ss, diff=%ss",
                lrc_duration,
                target_duration,
                diff,
            )

    if is_syncedlyrics_available_fn(state) and not offline:
        alternative_searches = [
            f"{title} {artist}",
            f"{artist} {title} official",
            f"{artist} {title} album version",
        ]

        for search_term in alternative_searches:
            logger.debug(f"Trying alternative LRC search: {search_term}")
            lrc, provider = search_with_state_fallback_fn(
                search_term,
                synced_only=True,
                enhanced=False,
                state=state,
            )
            if lrc and has_timestamps_fn(lrc, state):
                alt_duration = get_lrc_duration_fn(lrc, state)
                if alt_duration:
                    diff = abs(alt_duration - target_duration)
                    if diff <= tolerance:
                        logger.info(
                            "Found LRC with alternative search '%s' from %s: %ss",
                            search_term,
                            provider,
                            alt_duration,
                        )
                        return lrc, True, f"{provider} ({search_term})", alt_duration

    if is_synced and lrc_text:
        lrc_duration = get_lrc_duration_fn(lrc_text, state)
        logger.warning("Using LRC despite duration mismatch. LRC timing may be off.")
        return lrc_text, is_synced, source, lrc_duration

    return None, False, "", None


def fetch_from_all_sources(
    title: str,
    artist: str,
    *,
    state: Any,
    is_lyriq_available_fn: Callable[[Any], bool],
    is_syncedlyrics_available_fn: Callable[[Any], bool],
    get_syncedlyrics_module_fn: Callable[[Any], Any],
    lyriq_get_lyrics: Optional[Callable[..., Any]],
    has_timestamps_fn: Callable[[str, Any], bool],
    get_lrc_duration_fn: Callable[[str, Any], Optional[int]],
    provider_order: list[str],
    suppress_stderr: Callable[[], Any],
    logger: Any,
) -> Dict[str, Tuple[Optional[str], Optional[int]]]:
    """Fetch lyrics from all available sources for comparison."""
    results: Dict[str, Tuple[Optional[str], Optional[int]]] = {}

    if is_lyriq_available_fn(state):
        try:
            with suppress_stderr():
                lyrics_obj = (
                    lyriq_get_lyrics(title, artist) if lyriq_get_lyrics else None
                )
            if lyrics_obj:
                synced = getattr(lyrics_obj, "synced_lyrics", None)
                if synced and has_timestamps_fn(synced, state):
                    duration = get_lrc_duration_fn(synced, state)
                    results["lyriq (LRCLib)"] = (synced, duration)
        except Exception as e:
            logger.debug(f"lyriq fetch failed for comparison: {e}")

    if is_syncedlyrics_available_fn(state):
        syncedlyrics_mod = get_syncedlyrics_module_fn(state)
        if syncedlyrics_mod is None:
            return results
        search_term = f"{artist} {title}"
        for provider in provider_order:
            if provider == "Genius":
                continue
            try:
                with suppress_stderr():
                    lrc = syncedlyrics_mod.search(
                        search_term,
                        providers=[provider],
                        synced_only=True,
                    )
                if lrc and has_timestamps_fn(lrc, state):
                    duration = get_lrc_duration_fn(lrc, state)
                    results[provider] = (lrc, duration)
            except Exception as e:
                logger.debug(f"{provider} fetch failed for comparison: {e}")

    return results
