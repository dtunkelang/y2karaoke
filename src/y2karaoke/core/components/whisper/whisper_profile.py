"""Shared profile selection for Whisper heuristic presets."""

from __future__ import annotations

from .whisper_runtime_config import (
    WhisperProfile,
    load_whisper_runtime_config,
    normalize_whisper_profile,
)


def get_whisper_profile() -> WhisperProfile:
    return load_whisper_runtime_config().profile


__all__ = ["WhisperProfile", "get_whisper_profile", "normalize_whisper_profile"]
