# Whisper Pipeline

Primary compatibility facade:
- `src/y2karaoke/core/components/whisper/whisper_integration.py`

## Responsibilities

The Whisper stack is responsible for:
- cached transcription
- LRC-to-Whisper word mapping
- DTW / phonetic alignment
- line correction passes and rollback logic
- forced-alignment fallback
- alignment metrics and trace output

## Main Flow

1. `transcribe_vocals(...)`
   - cached Whisper transcription
2. `align_lrc_text_to_whisper_timings(...)`
   - align lyric words and lines to Whisper timing evidence
3. `correct_timing_with_whisper(...)`
   - choose and apply the hybrid correction path

## Current Module Layout

Compatibility / facade layer:
- `whisper_integration.py`
- `whisper_mapping.py`
- `whisper_mapping_post.py`

Major implementation modules:
- `whisper_integration_pipeline.py`
- `whisper_integration_pipeline_align.py`
- `whisper_integration_pipeline_correct.py`
- `whisper_integration_align.py`
- `whisper_integration_correct.py`
- `whisper_integration_retry.py`
- `whisper_integration_stages.py`

Supporting ownership modules:
- `whisper_runtime_config.py`
- `whisper_mapping_runtime_config.py`
- `whisper_segment_assignments.py`
- `whisper_block_dtw_assignments.py`
- `whisper_assignment_trace.py`
- `whisper_integration_align_heuristics.py`
- `whisper_integration_align_corrections.py`
- `whisper_mapping_post_core.py`

## Structural Notes

- The facade modules intentionally preserve backward-compatible import surfaces.
- Large orchestration files have been split by ownership, but compatibility aliases remain where tests and callers still depend on them.
- Remaining env-driven behavior is much narrower than before and should only be reduced further when it materially improves ownership.
