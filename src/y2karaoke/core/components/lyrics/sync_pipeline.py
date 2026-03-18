"""Pipeline implementations for sync lyrics orchestration."""

import re
from typing import Any, Callable, Optional, Tuple

from .runtime_config import LyricsRuntimeConfig, load_lyrics_runtime_config


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


def _source_matches_preference(source: str, preferred: str | None) -> bool:
    if not preferred:
        return True
    normalized_source = (source or "").strip().lower()
    if preferred == "lyriq":
        return "lyriq" in normalized_source or "lrclib" in normalized_source
    if preferred == "syncedlyrics":
        return "lyriq" not in normalized_source and "lrclib" not in normalized_source
    return preferred in normalized_source


def _warn_once(runtime_state: Any, key: str, logger, message: str, *args: Any) -> None:
    warning_keys = getattr(runtime_state, "warning_once_keys", None)
    if warning_keys is None:
        warning_keys = set()
        setattr(runtime_state, "warning_once_keys", warning_keys)
    if key in warning_keys:
        return
    warning_keys.add(key)
    logger.warning(message, *args)


def _resolve_cached_lrc_result(
    *,
    cached: tuple,
    preferred_provider: str | None,
    target_duration: Optional[int],
    duration_tolerance: int,
    logger,
) -> tuple[Optional[tuple], Optional[tuple]]:
    cached_source = cached[2] if len(cached) > 2 else ""
    if not _source_matches_preference(cached_source, preferred_provider):
        logger.debug(
            "Cached LRC source '%s' does not match preferred provider '%s'; re-fetching",
            cached_source,
            preferred_provider,
        )
        return None, tuple(cached)

    cached_duration = cached[3] if len(cached) > 3 else None
    if target_duration and cached_duration:
        if abs(cached_duration - target_duration) <= duration_tolerance:
            return tuple(cached), None
        logger.debug(
            "Cached LRC duration (%ss) doesn't match target (%ss), re-fetching",
            cached_duration,
            target_duration,
        )
        return None, tuple(cached)

    return tuple(cached), None


def _lookup_cached_lrc_result(
    *,
    cache_key: tuple[str, str],
    preferred_provider: str | None,
    target_duration: Optional[int],
    duration_tolerance: int,
    runtime_state: Any,
    logger,
) -> tuple[Optional[tuple], Optional[tuple]]:
    cached = runtime_state.lrc_cache.get(cache_key)
    if cached is None:
        return None, None
    return _resolve_cached_lrc_result(
        cached=tuple(cached),
        preferred_provider=preferred_provider,
        target_duration=target_duration,
        duration_tolerance=duration_tolerance,
        logger=logger,
    )


def _lookup_disk_cached_lrc_result(
    *,
    cache_key: tuple[str, str],
    preferred_provider: str | None,
    target_duration: Optional[int],
    duration_tolerance: int,
    runtime_state: Any,
    logger,
) -> tuple[Optional[tuple], Optional[tuple]]:
    disk_key = f"{cache_key[0]}|{cache_key[1]}"
    disk_lrc = runtime_state.disk_cache.get("lrc_cache", {})
    cached = disk_lrc.get(disk_key)
    if cached is None:
        return None, None

    selected, fallback = _resolve_cached_lrc_result(
        cached=tuple(cached),
        preferred_provider=preferred_provider,
        target_duration=target_duration,
        duration_tolerance=duration_tolerance,
        logger=logger,
    )
    if selected is not None:
        runtime_state.lrc_cache[cache_key] = selected
    return selected, fallback


def _build_lyriq_attempts(title: str, artist: str) -> list[tuple[str, str]]:
    normalized_title = _normalize_for_provider_search(title)
    normalized_artist = _normalize_for_provider_search(artist)
    attempts = [
        (normalized_title or title, normalized_artist or artist),
        (title, artist),
    ]
    unique_attempts: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for attempt in attempts:
        if attempt in seen:
            continue
        seen.add(attempt)
        unique_attempts.append(attempt)
    return unique_attempts


def _try_lyriq_provider(
    *,
    title: str,
    artist: str,
    runtime_state: Any,
    fetch_from_lyriq_fn: Callable[..., Optional[str]],
    has_timestamps_fn: Callable[..., bool],
    get_lrc_duration_fn: Callable[..., Optional[int]],
    set_lrc_cache_fn: Callable[..., None],
    cache_key: tuple[str, str],
    logger,
) -> tuple[Optional[str], bool, str]:
    for lyriq_title, lyriq_artist in _build_lyriq_attempts(title, artist):
        logger.debug("Trying lyriq for: %s - %s", lyriq_title, lyriq_artist)
        lrc = fetch_from_lyriq_fn(lyriq_title, lyriq_artist, state=runtime_state)
        if lrc and has_timestamps_fn(lrc, runtime_state):
            logger.debug("Found synced lyrics from lyriq (LRCLib)")
            lrc_duration = get_lrc_duration_fn(lrc, runtime_state)
            result = (lrc, True, "lyriq (LRCLib)", lrc_duration)
            set_lrc_cache_fn(cache_key, result, state=runtime_state)
            return (lrc, True, "lyriq (LRCLib)")
    return (None, False, "")


def fetch_lyrics_multi_source_impl(  # noqa: C901
    title: str,
    artist: str,
    synced_only: bool,
    enhanced: bool,
    target_duration: Optional[int],
    duration_tolerance: int,
    offline: bool,
    runtime_state: Any,
    runtime_config: Optional[LyricsRuntimeConfig] = None,
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
    runtime_config = runtime_config or load_lyrics_runtime_config(
        lrc_duration_tolerance_sec=duration_tolerance
    )
    preferred_provider = runtime_config.preferred_provider
    if disk_cache_enabled:
        load_disk_cache_fn(runtime_state)

    cache_key = (artist.lower().strip(), title.lower().strip())
    selected_cached, fallback_cached_candidate = _lookup_cached_lrc_result(
        cache_key=cache_key,
        preferred_provider=preferred_provider,
        target_duration=target_duration,
        duration_tolerance=duration_tolerance,
        runtime_state=runtime_state,
        logger=logger,
    )
    if selected_cached is not None:
        logger.debug(f"Using cached LRC result for {artist} - {title}")
        return (selected_cached[0], selected_cached[1], selected_cached[2])
    if fallback_cached_candidate is not None:
        fallback_cached = fallback_cached_candidate

    if disk_cache_enabled:
        selected_disk_cached, fallback_disk_cached = _lookup_disk_cached_lrc_result(
            cache_key=cache_key,
            preferred_provider=preferred_provider,
            target_duration=target_duration,
            duration_tolerance=duration_tolerance,
            runtime_state=runtime_state,
            logger=logger,
        )
        if selected_disk_cached is not None:
            logger.debug(f"Using cached LRC result for {artist} - {title}")
            return (
                selected_disk_cached[0],
                selected_disk_cached[1],
                selected_disk_cached[2],
            )
        if fallback_disk_cached is not None:
            fallback_cached = fallback_disk_cached

    if offline:
        logger.info("Offline mode: skipping lyrics providers (cache only)")
        if fallback_cached and fallback_cached[0]:
            warn_key = (
                "offline_fallback_cached_duration:"
                f"{cache_key[0]}:{cache_key[1]}:{fallback_cached[3]}:"
                f"{target_duration}:{duration_tolerance}"
            )
            _warn_once(
                runtime_state,
                warn_key,
                logger,
                "Offline mode: reusing cached LRC despite duration mismatch "
                "(cached=%ss, target=%ss, tolerance=%ss)",
                fallback_cached[3],
                target_duration,
                duration_tolerance,
            )
            set_lrc_cache_fn(cache_key, fallback_cached, state=runtime_state)
            return (fallback_cached[0], bool(fallback_cached[1]), fallback_cached[2])
        return (None, False, "")

    search_terms = _build_provider_search_terms(title, artist)
    search_term = search_terms[0] if search_terms else f"{artist} {title}"
    logger.debug(
        "Searching for synced lyrics with terms: %s",
        " | ".join(search_terms) if search_terms else search_term,
    )

    try:
        prefer_syncedlyrics = preferred_provider == "syncedlyrics"

        def _search_syncedlyrics_sources() -> Tuple[Optional[str], bool, str]:
            if not is_syncedlyrics_available_fn(runtime_state):
                _warn_once(
                    runtime_state,
                    "syncedlyrics_missing",
                    logger,
                    "syncedlyrics not installed",
                )
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
            return (None, False, "")

        if not prefer_syncedlyrics and is_lyriq_available_fn(runtime_state):
            lrc, is_synced, source = _try_lyriq_provider(
                title=title,
                artist=artist,
                runtime_state=runtime_state,
                fetch_from_lyriq_fn=fetch_from_lyriq_fn,
                has_timestamps_fn=has_timestamps_fn,
                get_lrc_duration_fn=get_lrc_duration_fn,
                set_lrc_cache_fn=set_lrc_cache_fn,
                cache_key=cache_key,
                logger=logger,
            )
            if lrc:
                return (lrc, is_synced, source)
        lrc, is_synced, source = _search_syncedlyrics_sources()
        if lrc:
            return (lrc, is_synced, source)

        if prefer_syncedlyrics and is_lyriq_available_fn(runtime_state):
            lrc, is_synced, source = _try_lyriq_provider(
                title=title,
                artist=artist,
                runtime_state=runtime_state,
                fetch_from_lyriq_fn=fetch_from_lyriq_fn,
                has_timestamps_fn=has_timestamps_fn,
                get_lrc_duration_fn=get_lrc_duration_fn,
                set_lrc_cache_fn=set_lrc_cache_fn,
                cache_key=cache_key,
                logger=logger,
            )
            if lrc:
                return (lrc, is_synced, source)

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
