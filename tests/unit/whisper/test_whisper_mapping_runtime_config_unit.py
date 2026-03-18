from y2karaoke.core.components.whisper.whisper_mapping_runtime_config import (
    SegmentAssignmentRuntimeConfig,
    WhisperMappingTraceConfig,
    load_segment_assignment_runtime_config,
    load_whisper_mapping_trace_config,
)


def test_load_segment_assignment_runtime_config_defaults(monkeypatch):
    monkeypatch.delenv("Y2K_WHISPER_SEGMENT_ASSIGN_SELECTION_MODE", raising=False)
    monkeypatch.delenv(
        "Y2K_WHISPER_SEGMENT_ASSIGN_PREFER_LATER_ON_STRONG_MERGE", raising=False
    )

    assert load_segment_assignment_runtime_config() == SegmentAssignmentRuntimeConfig()


def test_load_segment_assignment_runtime_config_reads_env(monkeypatch):
    monkeypatch.setenv("Y2K_WHISPER_SEGMENT_ASSIGN_PIPELINE", "parallel_experimental")
    monkeypatch.setenv(
        "Y2K_WHISPER_SEGMENT_ASSIGN_SELECTION_MODE",
        "experimental_terminal_stall_lookback",
    )
    monkeypatch.setenv("Y2K_WHISPER_SEGMENT_ASSIGN_PREFER_LATER_ON_STRONG_MERGE", "0")
    monkeypatch.setenv("Y2K_WHISPER_SEGMENT_ASSIGN_LATER_TRACE", "1")
    monkeypatch.setenv("Y2K_WHISPER_SEGMENT_ASSIGN_ZERO_SCORE_LOOKBACK", "0")
    monkeypatch.setenv("Y2K_WHISPER_SEGMENT_ASSIGN_ZERO_SCORE_LOOKBACK_SEGS", "12")

    cfg = load_segment_assignment_runtime_config()

    assert cfg.pipeline_mode == "parallel_experimental"
    assert cfg.selection_mode == "experimental_terminal_stall_lookback"
    assert not cfg.prefer_later_on_strong_merge
    assert cfg.later_trace
    assert not cfg.zero_score_lookback_enabled
    assert cfg.zero_score_lookback_segs == 12


def test_load_whisper_mapping_trace_config_reads_env(monkeypatch):
    monkeypatch.setenv("Y2K_TRACE_MAPPER_LINE_RANGE", "2-4")
    monkeypatch.setenv("Y2K_TRACE_MAPPER_DETAILS_JSON", "mapper.json")
    monkeypatch.setenv("Y2K_TRACE_SEGMENT_SELECTION_JSON", "segments.json")

    cfg = load_whisper_mapping_trace_config()

    assert cfg == WhisperMappingTraceConfig(
        line_range=(2, 4),
        mapper_details_path="mapper.json",
        segment_selection_path="segments.json",
        segment_assignments_path="",
        syllable_assignments_path="",
        mapper_candidates_path="",
    )
