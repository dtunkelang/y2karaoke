"""Shared profile selection for Whisper heuristic presets."""

from __future__ import annotations

import os
from typing import Literal, cast

WhisperProfile = Literal["safe", "default", "aggressive"]


def get_whisper_profile() -> WhisperProfile:
    profile = os.getenv("Y2K_WHISPER_PROFILE", "default").strip().lower()
    if profile in {"safe", "aggressive"}:
        return cast(WhisperProfile, profile)
    return "default"
