"""Pipeline implementations for sync lyrics orchestration."""

import re
from typing import Any, Callable, Optional, Tuple


def _normalize_for_provider_search(value: str) -> str:
    """Normalize noisy metadata for synced-lyrics provider search."""
    text = (value or "").strip()
    if not text:
        return ""

    # Drop common metadata wrappers e.g. "(Official Video)", "[Lyrics]".
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)

    # Remove noisy suffix segments frequently appended from uploads.
    noisy_suffix = re.compile(
        r"\b(remix|version|live|lyrics?|audio|video|official|prod(?:uced)?|edit)\b",
        re.IGNORECASE,
    )
    segments = re.split(r"\s[-–—]\s", text)
    kept_segments = [seg for seg in segments if not noisy_suffix.search(seg)]
    if kept_segments:
        text = kept_segments[0]

    text = re.sub(r"\s+", " ", text).strip(" -")
    return text


def _build_provider_search_terms(title: str, artist: str) -> list[str]:
    normalized_title = _normalize_for_provider_search(title)
    normalized_artist = _normalize_for_provider_search(artist)
    original_term = f"{artist} {title}".strip()
    normalized_term = f"{normalized_artist} {normalized_title}".strip()

    terms: list[str] = []
    for term in [normalized_term, original_term]:
        if term and term not in terms:
            terms.append(term)
    return terms


def _warn_once(runtime_state: Any, key: str, logger, message: str, *args: Any) -> None:
    warning_keys = getattr(runtime_state, "warning_once_keys", None)
    if warning_keys is None:
        warning_keys = set()
        setattr(runtime_state, "warning_once_keys", warning_keys)
    if key in warning_keys:
        return
    warning_keys.add(key)
    logger.warning(message, *args)


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
    fallback_cached: Optional[Tuple[Optional[str], bool, str, Optional[int]]] = None
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
            fallback_cached = tuple(cached)
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
                fallback_cached = tuple(cached)
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

    search_terms = _build_provider_search_terms(title, artist)
    search_term = search_terms[0] if search_terms else f"{artist} {title}"
    logger.debug(
        "Searching for synced lyrics with terms: %s",
        " | ".join(search_terms) if search_terms else search_term,
    )

    try:
        if is_lyriq_available_fn(runtime_state):
            normalized_title = _normalize_for_provider_search(title)
            normalized_artist = _normalize_for_provider_search(artist)
            lyriq_attempts = [
                (normalized_title or title, normalized_artist or artist),
                (title, artist),
            ]
            seen_lyriq_attempts: set[tuple[str, str]] = set()
            for lyriq_title, lyriq_artist in lyriq_attempts:
                key = (lyriq_title, lyriq_artist)
                if key in seen_lyriq_attempts:
                    continue
                seen_lyriq_attempts.add(key)
                logger.debug("Trying lyriq for: %s - %s", lyriq_title, lyriq_artist)
                lrc = fetch_from_lyriq_fn(
                    lyriq_title, lyriq_artist, state=runtime_state
                )
                if lrc and has_timestamps_fn(lrc, runtime_state):
                    logger.debug("Found synced lyrics from lyriq (LRCLib)")
                    lrc_duration = get_lrc_duration_fn(lrc, runtime_state)
                    result = (lrc, True, "lyriq (LRCLib)", lrc_duration)
                    set_lrc_cache_fn(cache_key, result, state=runtime_state)
                    return (lrc, True, "lyriq (LRCLib)")

        if not is_syncedlyrics_available_fn(runtime_state):
            _warn_once(
                runtime_state,
                "syncedlyrics_missing",
                logger,
                "syncedlyrics not installed",
            )
            no_sync_result = (None, False, "", None)
            set_lrc_cache_fn(cache_key, no_sync_result, state=runtime_state)
            return (None, False, "")

        provider_terms = search_terms or [search_term]
        for term in provider_terms:
            if enhanced:
                lrc, provider = search_with_state_fallback_fn(
                    term,
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
                term,
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

        if fallback_cached and fallback_cached[0]:
            _warn_once(
                runtime_state,
                f"fallback_cached_duration:{search_term}:{fallback_cached[3]}:{target_duration}:{duration_tolerance}",
                logger,
                "No synced lyrics from providers for '%s'; reusing cached LRC "
                "despite duration mismatch (cached=%ss, target=%ss, tolerance=%ss)",
                search_term,
                fallback_cached[3],
                target_duration,
                duration_tolerance,
            )
            set_lrc_cache_fn(cache_key, fallback_cached, state=runtime_state)
            return (fallback_cached[0], bool(fallback_cached[1]), fallback_cached[2])

        _warn_once(
            runtime_state,
            f"no_synced_lyrics:{search_term}:{target_duration}:{duration_tolerance}",
            logger,
            "No synced lyrics found from any provider for '%s' (target=%ss, tolerance=%ss)",
            search_term,
            target_duration,
            duration_tolerance,
        )
        not_found = (None, False, "", None)
        set_lrc_cache_fn(cache_key, not_found, state=runtime_state)
        return (None, False, "")

    except Exception as e:
        logger.error(f"Error fetching synced lyrics: {e}")
        error_result = (None, False, "", None)
        set_lrc_cache_fn(cache_key, error_result, state=runtime_state)
        return (None, False, "")
