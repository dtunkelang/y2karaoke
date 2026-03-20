"""Runtime config loading for Whisper pipeline behavior."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal, Optional, cast

WhisperProfile = Literal["safe", "default", "aggressive"]


def normalize_whisper_profile(profile: Optional[str]) -> WhisperProfile:
    normalized = (profile or "default").strip().lower()
    if normalized in {"safe", "aggressive"}:
        return cast(WhisperProfile, normalized)
    return "default"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip() == "1"


@dataclass(frozen=True)
class WhisperRuntimeConfig:
    profile: WhisperProfile = "default"
    tail_shortfall_forced_fallback: bool = False
    low_support_onset_reanchor: bool = False
    repeat_cadence_reanchor: bool = False
    restored_run_onset_shift: bool = True
    repeat_duration_normalize: bool = False
    disable_repeat_shift: bool = False
    disable_monotonic_start_enforce: bool = False


def load_whisper_runtime_config(
    *,
    profile: Optional[str] = None,
    tail_shortfall_forced_fallback: Optional[bool] = None,
    low_support_onset_reanchor: Optional[bool] = None,
    repeat_cadence_reanchor: Optional[bool] = None,
    restored_run_onset_shift: Optional[bool] = None,
    repeat_duration_normalize: Optional[bool] = None,
    disable_repeat_shift: Optional[bool] = None,
    disable_monotonic_start_enforce: Optional[bool] = None,
) -> WhisperRuntimeConfig:
    return WhisperRuntimeConfig(
        profile=normalize_whisper_profile(
            os.getenv("Y2K_WHISPER_PROFILE") if profile is None else profile
        ),
        tail_shortfall_forced_fallback=(
            _env_flag("Y2K_WHISPER_ENABLE_TAIL_SHORTFALL_FORCED_FALLBACK")
            if tail_shortfall_forced_fallback is None
            else tail_shortfall_forced_fallback
        ),
        low_support_onset_reanchor=(
            _env_flag("Y2K_WHISPER_ENABLE_LOW_SUPPORT_ONSET_REANCHOR")
            if low_support_onset_reanchor is None
            else low_support_onset_reanchor
        ),
        repeat_cadence_reanchor=(
            _env_flag("Y2K_WHISPER_ENABLE_REPEAT_CADENCE_REANCHOR")
            if repeat_cadence_reanchor is None
            else repeat_cadence_reanchor
        ),
        restored_run_onset_shift=(
            _env_flag("Y2K_WHISPER_ENABLE_RESTORED_RUN_ONSET_SHIFT", default=True)
            if restored_run_onset_shift is None
            else restored_run_onset_shift
        ),
        repeat_duration_normalize=(
            _env_flag("Y2K_REPEAT_DURATION_NORMALIZE")
            if repeat_duration_normalize is None
            else repeat_duration_normalize
        ),
        disable_repeat_shift=(
            _env_flag("Y2K_WHISPER_DISABLE_REPEAT_SHIFT")
            if disable_repeat_shift is None
            else disable_repeat_shift
        ),
        disable_monotonic_start_enforce=(
            _env_flag("Y2K_WHISPER_DISABLE_MONOTONIC_START_ENFORCE")
            if disable_monotonic_start_enforce is None
            else disable_monotonic_start_enforce
        ),
    )
