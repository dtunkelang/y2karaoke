"""Whisper mapping postprocess compatibility facade."""

from __future__ import annotations

from importlib import import_module

__all__: list[str] = []

_LEGACY_EXPORTS = {
    "_default_postprocess_toggle_config": (
        "y2karaoke.core.components.whisper.whisper_mapping_post_core",
        "_default_postprocess_toggle_config",
    ),
    "_build_word_assignments_from_phoneme_path": (
        "y2karaoke.core.components.whisper.whisper_mapping_post_core",
        "_build_word_assignments_from_phoneme_path",
    ),
    "_enforce_monotonic_line_starts_whisper": (
        "y2karaoke.core.components.whisper.whisper_mapping_post_core",
        "_enforce_monotonic_line_starts_whisper",
    ),
    "_extend_line_to_trailing_whisper_matches": (
        "y2karaoke.core.components.whisper.whisper_mapping_post_repetition",
        "_extend_line_to_trailing_whisper_matches",
    ),
    "_pull_late_lines_to_matching_segments": (
        "y2karaoke.core.components.whisper.whisper_mapping_post_core",
        "_pull_late_lines_to_matching_segments",
    ),
    "_resolve_line_overlaps": (
        "y2karaoke.core.components.whisper.whisper_mapping_post_core",
        "_resolve_line_overlaps",
    ),
    "_retime_short_interjection_lines": (
        "y2karaoke.core.components.whisper.whisper_mapping_post_core",
        "_retime_short_interjection_lines",
    ),
    "_shift_repeated_lines_to_next_whisper": (
        "y2karaoke.core.components.whisper.whisper_mapping_post_core",
        "_shift_repeated_lines_to_next_whisper",
    ),
    "_snap_first_word_to_whisper_onset": (
        "y2karaoke.core.components.whisper.whisper_mapping_post_core",
        "_snap_first_word_to_whisper_onset",
    ),
}


def __getattr__(name: str):
    if name in _LEGACY_EXPORTS:
        module_name, attr_name = _LEGACY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LEGACY_EXPORTS.keys()))
