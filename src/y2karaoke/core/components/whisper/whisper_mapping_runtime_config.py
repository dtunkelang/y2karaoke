"""Typed runtime config for Whisper mapping and segment assignment."""

from __future__ import annotations

from dataclasses import dataclass
import os


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip() != "0" if default else raw.strip() == "1"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _parse_line_range(raw: str | None) -> tuple[int, int] | None:
    if not raw:
        return None
    try:
        start_s, end_s = raw.strip().split("-", 1)
        start = int(start_s)
        end = int(end_s)
    except (AttributeError, TypeError, ValueError):
        return None
    if start <= 0 or end < start:
        return None
    return start, end


@dataclass(frozen=True)
class SegmentAssignmentRuntimeConfig:
    pipeline_mode: str = "default"
    selection_mode: str = "default"
    prefer_later_on_strong_merge: bool = True
    later_trace: bool = False
    zero_score_lookback_enabled: bool = True
    zero_score_lookback_segs: int = 36
    stalled_search_min_run: int = 1
    stalled_search_lookback_segs: int = 4
    terminal_stall_lookback_segs: int = 4
    terminal_stall_max_current_score: float = 0.2
    terminal_stall_min_rescue_score: float = 0.4
    terminal_stall_min_score_gain: float = 0.15


@dataclass(frozen=True)
class WhisperMappingTraceConfig:
    line_range: tuple[int, int] | None = None
    mapper_details_path: str = ""
    segment_selection_path: str = ""
    segment_assignments_path: str = ""
    syllable_assignments_path: str = ""
    mapper_candidates_path: str = ""


def load_segment_assignment_runtime_config(
    *,
    pipeline_mode: str | None = None,
    selection_mode: str | None = None,
    prefer_later_on_strong_merge: bool | None = None,
    later_trace: bool | None = None,
    zero_score_lookback_enabled: bool | None = None,
    zero_score_lookback_segs: int | None = None,
    stalled_search_min_run: int | None = None,
    stalled_search_lookback_segs: int | None = None,
    terminal_stall_lookback_segs: int | None = None,
    terminal_stall_max_current_score: float | None = None,
    terminal_stall_min_rescue_score: float | None = None,
    terminal_stall_min_score_gain: float | None = None,
) -> SegmentAssignmentRuntimeConfig:
    return SegmentAssignmentRuntimeConfig(
        pipeline_mode=(
            (
                os.getenv("Y2K_WHISPER_SEGMENT_ASSIGN_PIPELINE", "default").strip()
                or "default"
            )
            if pipeline_mode is None
            else pipeline_mode
        ),
        selection_mode=(
            (
                os.getenv(
                    "Y2K_WHISPER_SEGMENT_ASSIGN_SELECTION_MODE", "default"
                ).strip()
                or "default"
            )
            if selection_mode is None
            else selection_mode
        ),
        prefer_later_on_strong_merge=(
            _env_flag("Y2K_WHISPER_SEGMENT_ASSIGN_PREFER_LATER_ON_STRONG_MERGE", True)
            if prefer_later_on_strong_merge is None
            else prefer_later_on_strong_merge
        ),
        later_trace=(
            _env_flag("Y2K_WHISPER_SEGMENT_ASSIGN_LATER_TRACE", False)
            if later_trace is None
            else later_trace
        ),
        zero_score_lookback_enabled=(
            _env_flag("Y2K_WHISPER_SEGMENT_ASSIGN_ZERO_SCORE_LOOKBACK", True)
            if zero_score_lookback_enabled is None
            else zero_score_lookback_enabled
        ),
        zero_score_lookback_segs=(
            _env_int("Y2K_WHISPER_SEGMENT_ASSIGN_ZERO_SCORE_LOOKBACK_SEGS", 36)
            if zero_score_lookback_segs is None
            else zero_score_lookback_segs
        ),
        stalled_search_min_run=(
            _env_int("Y2K_WHISPER_SEGMENT_ASSIGN_STALLED_SEARCH_MIN_RUN", 1)
            if stalled_search_min_run is None
            else stalled_search_min_run
        ),
        stalled_search_lookback_segs=(
            _env_int("Y2K_WHISPER_SEGMENT_ASSIGN_STALLED_SEARCH_LOOKBACK_SEGS", 4)
            if stalled_search_lookback_segs is None
            else stalled_search_lookback_segs
        ),
        terminal_stall_lookback_segs=(
            _env_int("Y2K_WHISPER_SEGMENT_ASSIGN_TERMINAL_STALL_LOOKBACK_SEGS", 4)
            if terminal_stall_lookback_segs is None
            else terminal_stall_lookback_segs
        ),
        terminal_stall_max_current_score=(
            _env_float(
                "Y2K_WHISPER_SEGMENT_ASSIGN_TERMINAL_STALL_MAX_CURRENT_SCORE", 0.2
            )
            if terminal_stall_max_current_score is None
            else terminal_stall_max_current_score
        ),
        terminal_stall_min_rescue_score=(
            _env_float(
                "Y2K_WHISPER_SEGMENT_ASSIGN_TERMINAL_STALL_MIN_RESCUE_SCORE", 0.4
            )
            if terminal_stall_min_rescue_score is None
            else terminal_stall_min_rescue_score
        ),
        terminal_stall_min_score_gain=(
            _env_float("Y2K_WHISPER_SEGMENT_ASSIGN_TERMINAL_STALL_MIN_SCORE_GAIN", 0.15)
            if terminal_stall_min_score_gain is None
            else terminal_stall_min_score_gain
        ),
    )


def load_whisper_mapping_trace_config(
    *,
    line_range: tuple[int, int] | None = None,
    mapper_details_path: str | None = None,
    segment_selection_path: str | None = None,
    segment_assignments_path: str | None = None,
    syllable_assignments_path: str | None = None,
    mapper_candidates_path: str | None = None,
) -> WhisperMappingTraceConfig:
    return WhisperMappingTraceConfig(
        line_range=(
            _parse_line_range(os.environ.get("Y2K_TRACE_MAPPER_LINE_RANGE"))
            if line_range is None
            else line_range
        ),
        mapper_details_path=(
            os.environ.get("Y2K_TRACE_MAPPER_DETAILS_JSON", "").strip()
            if mapper_details_path is None
            else mapper_details_path
        ),
        segment_selection_path=(
            os.environ.get("Y2K_TRACE_SEGMENT_SELECTION_JSON", "").strip()
            if segment_selection_path is None
            else segment_selection_path
        ),
        segment_assignments_path=(
            os.environ.get("Y2K_TRACE_SEGMENT_ASSIGNMENTS_JSON", "").strip()
            if segment_assignments_path is None
            else segment_assignments_path
        ),
        syllable_assignments_path=(
            os.environ.get("Y2K_TRACE_SYLLABLE_ASSIGNMENTS_JSON", "").strip()
            if syllable_assignments_path is None
            else syllable_assignments_path
        ),
        mapper_candidates_path=(
            os.environ.get("Y2K_TRACE_MAPPER_CANDIDATES_JSON", "").strip()
            if mapper_candidates_path is None
            else mapper_candidates_path
        ),
    )
