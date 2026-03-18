"""Cache helpers for lyrics sync provider orchestration."""

import json
import os
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from ....config import get_cache_dir

if TYPE_CHECKING:
    from .sync import SyncState


def _get_disk_cache_path() -> Path:
    return get_cache_dir() / "lyrics_cache.json"


def _disk_cache_enabled(state: Optional["SyncState"] = None) -> bool:
    runtime_state = state
    assert runtime_state is not None
    return runtime_state.disk_cache_enabled and "PYTEST_CURRENT_TEST" not in os.environ


def _empty_disk_cache() -> Dict[str, Any]:
    return {
        "search_cache": {},
        "lrc_cache": {},
        "lyriq_cache": {},
        "all_sources_cache": {},
    }


def _load_disk_cache(state: Optional["SyncState"] = None) -> None:
    runtime_state = state
    assert runtime_state is not None
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


def _save_disk_cache(
    state: Optional["SyncState"] = None,
    *,
    logger,
) -> None:
    runtime_state = state
    assert runtime_state is not None
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
    cache_key,
    value,
    state: Optional["SyncState"] = None,
    *,
    save_disk_cache_fn,
) -> None:
    runtime_state = state
    assert runtime_state is not None
    runtime_state.lrc_cache[cache_key] = value
    if _disk_cache_enabled(runtime_state):
        disk_key = f"{cache_key[0]}|{cache_key[1]}"
        runtime_state.disk_cache.setdefault("lrc_cache", {})[disk_key] = list(value)
        save_disk_cache_fn(runtime_state)


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

    from .sync_pipeline import _normalize_for_provider_search

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
    value: Dict[str, tuple[Optional[str], Optional[int]]],
) -> Dict[str, list[Any]]:
    return {
        str(source): [payload[0], payload[1]]
        for source, payload in value.items()
        if isinstance(source, str) and isinstance(payload, tuple) and len(payload) == 2
    }


def _deserialize_all_sources_result(
    raw: Any,
) -> Dict[str, tuple[Optional[str], Optional[int]]]:
    if not isinstance(raw, dict):
        return {}
    result: Dict[str, tuple[Optional[str], Optional[int]]] = {}
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


def _get_cached_all_sources(
    runtime_state: "SyncState",
    cache_keys: list[tuple[tuple[str, str], str]],
    *,
    offline: bool,
) -> Optional[Dict[str, tuple[Optional[str], Optional[int]]]]:
    cached = _get_runtime_cached_all_sources(runtime_state, cache_keys, offline=offline)
    if cached is not None:
        return cached
    if not _disk_cache_enabled(runtime_state):
        return None
    return _get_disk_cached_all_sources(runtime_state, cache_keys, offline=offline)


def _get_runtime_cached_all_sources(
    runtime_state: "SyncState",
    cache_keys: list[tuple[tuple[str, str], str]],
    *,
    offline: bool,
) -> Optional[Dict[str, tuple[Optional[str], Optional[int]]]]:
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
    runtime_state: "SyncState",
    cache_keys: list[tuple[tuple[str, str], str]],
    *,
    offline: bool,
) -> Optional[Dict[str, tuple[Optional[str], Optional[int]]]]:
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
