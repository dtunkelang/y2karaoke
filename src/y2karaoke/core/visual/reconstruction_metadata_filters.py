"""Global metadata/credit noise filtering helpers for reconstruction."""

from __future__ import annotations

from typing import Any

from ..text_utils import normalize_text_basic

_GLOBAL_META_KEYWORDS = {
    "karaoke",
    "karafun",
    "entertainer",
    "digitop",
    "rights",
    "reserved",
    "copyright",
    "produced",
    "association",
    "global",
    "ltd",
    "www",
    "http",
}


def _compact_tokens(text: str) -> list[str]:
    tokens = [tok for tok in text.split() if tok]
    compact = ["".join(ch for ch in tok if ch.isalnum()) for tok in tokens]
    return [tok for tok in compact if tok]


def _classify_compact_tokens(compact_tokens: list[str]) -> tuple[int, int, int]:
    metadata_hits = 0
    providerish = 0
    urlish = 0
    for tok in compact_tokens:
        if any(key in tok for key in _GLOBAL_META_KEYWORDS):
            metadata_hits += 1
        if tok.startswith(("kara", "xara", "xora")) and len(tok) >= 5:
            providerish += 1
        if "www" in tok or tok.endswith(("com", "couk", "net")):
            urlish += 1
    return metadata_hits, providerish, urlish


def _is_metadata_pattern(
    compact_tokens: list[str],
    *,
    metadata_hits: int,
    providerish: int,
    urlish: int,
) -> bool:
    if urlish >= 1 and (metadata_hits >= 1 or providerish >= 1):
        return True
    if metadata_hits >= 3:
        return True
    if providerish >= 2 and len(compact_tokens) >= 2:
        return True
    if len(compact_tokens) >= 10 and (metadata_hits >= 2 or providerish >= 2):
        return True
    return False


def looks_global_metadata_noise(line: dict[str, Any]) -> bool:
    text = normalize_text_basic(str(line.get("text", ""))).strip().lower()
    if not text:
        return False
    compact_tokens = _compact_tokens(text)
    if not compact_tokens:
        return False

    metadata_hits, providerish, urlish = _classify_compact_tokens(compact_tokens)
    return _is_metadata_pattern(
        compact_tokens,
        metadata_hits=metadata_hits,
        providerish=providerish,
        urlish=urlish,
    )


def suppress_global_metadata_noise(
    lines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not lines:
        return []
    return [line for line in lines if not looks_global_metadata_noise(line)]
