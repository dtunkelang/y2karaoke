"""Helpers for matching cached audio assets to the requested song identity."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import re

_NOISE_TOKENS = {
    "official",
    "audio",
    "video",
    "music",
    "topic",
    "hd",
    "4k",
    "ft",
    "feat",
    "featuring",
}
_DISALLOWED_CACHE_TOKENS = {
    "karaoke",
    "karafun",
    "instrumental",
    "sing",
    "cover",
}


def _tokenize(value: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", (value or "").lower())
    normalized: list[str] = []
    for token in tokens:
        if not token or token in _NOISE_TOKENS:
            continue
        normalized.append(token)
        # Treat common dropped-g lyric/title spellings as equivalent for cache lookup.
        if token.endswith("in") and len(token) > 3:
            normalized.append(f"{token}g")
        elif token.endswith("ing") and len(token) > 4:
            normalized.append(token[:-1])
    return normalized


def _coerce_path(value: Path | str) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _path_sort_key(path: Path) -> tuple[float, str]:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (mtime, path.name)


def _required_title_tokens(title: Optional[str]) -> set[str]:
    return set(_tokenize(title or ""))


def _required_artist_tokens(artist: Optional[str]) -> set[str]:
    return set(_tokenize(artist or ""))


def _contains_disallowed_tokens(tokens: set[str]) -> bool:
    return bool(tokens & _DISALLOWED_CACHE_TOKENS)


def cache_asset_matches_request(
    path: Path,
    *,
    expected_title: Optional[str],
    expected_artist: Optional[str],
) -> bool:
    """Return True when a cached asset looks compatible with the requested song."""
    if not expected_title and not expected_artist:
        return True

    asset_tokens = set(_tokenize(path.stem))
    title_tokens = _required_title_tokens(expected_title)
    artist_tokens = _required_artist_tokens(expected_artist)

    if title_tokens and not title_tokens.issubset(asset_tokens):
        return False
    if artist_tokens and not artist_tokens.intersection(asset_tokens):
        return False
    if _contains_disallowed_tokens(asset_tokens):
        return False
    return True


def select_matching_cached_audio(
    paths: Iterable[Path | str],
    *,
    expected_title: Optional[str],
    expected_artist: Optional[str],
) -> Optional[Path]:
    """Pick the newest cached original audio compatible with the request."""
    candidates = sorted(
        (
            candidate
            for candidate in (_coerce_path(p) for p in paths)
            if cache_asset_matches_request(
                candidate,
                expected_title=expected_title,
                expected_artist=expected_artist,
            )
        ),
        key=_path_sort_key,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _stem_prefix_for_matching(audio_path: str) -> str:
    stem = Path(audio_path).stem.lower()
    for suffix in ("_vocals", " vocals", "_instrumental", " instrumental"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def select_matching_cached_stem(
    paths: Iterable[Path | str],
    *,
    audio_path: str,
) -> Optional[Path]:
    """Pick the newest cached stem that belongs to the active source audio."""
    source_stem = _stem_prefix_for_matching(audio_path)
    candidates = sorted(
        (
            candidate
            for candidate in (_coerce_path(p) for p in paths)
            if candidate.stem.lower().startswith(source_stem)
        ),
        key=_path_sort_key,
        reverse=True,
    )
    return candidates[0] if candidates else None
