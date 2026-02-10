"""Compatibility facade for Whisper pull/merge timing heuristics."""

from .phonetic_utils import _phonetic_similarity
from . import whisper_alignment_pull_rules as _rules


def _sync_patchables() -> None:
    _rules._phonetic_similarity = _phonetic_similarity


def _merge_lines_to_whisper_segments(*args, **kwargs):
    _sync_patchables()
    return _rules._merge_lines_to_whisper_segments(*args, **kwargs)


def _pull_next_line_into_segment_window(*args, **kwargs):
    _sync_patchables()
    return _rules._pull_next_line_into_segment_window(*args, **kwargs)


def _pull_next_line_into_same_segment(*args, **kwargs):
    _sync_patchables()
    return _rules._pull_next_line_into_same_segment(*args, **kwargs)


def _merge_short_following_line_into_segment(*args, **kwargs):
    _sync_patchables()
    return _rules._merge_short_following_line_into_segment(*args, **kwargs)


def _pull_lines_near_segment_end(*args, **kwargs):
    _sync_patchables()
    return _rules._pull_lines_near_segment_end(*args, **kwargs)


def _pull_lines_to_best_segments(*args, **kwargs):
    _sync_patchables()
    return _rules._pull_lines_to_best_segments(*args, **kwargs)


__all__ = [
    "_phonetic_similarity",
    "_merge_lines_to_whisper_segments",
    "_pull_next_line_into_segment_window",
    "_pull_next_line_into_same_segment",
    "_merge_short_following_line_into_segment",
    "_pull_lines_near_segment_end",
    "_pull_lines_to_best_segments",
]
