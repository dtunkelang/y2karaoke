"""Runtime configuration for lyrics pipelines."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LyricsRuntimeConfig:
    preferred_provider: Optional[str] = None
    lrc_duration_tolerance_sec: int = 8


def load_lyrics_runtime_config(
    *,
    preferred_provider: Optional[str] = None,
    lrc_duration_tolerance_sec: Optional[int] = None,
) -> LyricsRuntimeConfig:
    resolved_provider = (
        preferred_provider.strip().lower() if preferred_provider else None
    ) or _preferred_provider_from_env()
    resolved_tolerance = (
        max(int(lrc_duration_tolerance_sec), 0)
        if lrc_duration_tolerance_sec is not None
        else _duration_tolerance_from_env(default=8)
    )
    return LyricsRuntimeConfig(
        preferred_provider=resolved_provider,
        lrc_duration_tolerance_sec=resolved_tolerance,
    )


def _preferred_provider_from_env() -> Optional[str]:
    value = os.getenv("Y2K_PREFERRED_LYRICS_PROVIDER", "").strip().lower()
    return value or None


def _duration_tolerance_from_env(*, default: int) -> int:
    raw_value = os.getenv("Y2K_LRC_DURATION_TOLERANCE_SEC", "").strip()
    if not raw_value:
        return default
    try:
        parsed = int(raw_value)
    except ValueError:
        logger.warning(
            "Ignoring invalid Y2K_LRC_DURATION_TOLERANCE_SEC=%r; using %ds",
            raw_value,
            default,
        )
        return default
    return max(parsed, 0)
