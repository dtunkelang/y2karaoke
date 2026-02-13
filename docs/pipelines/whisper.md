# Whisper Pipeline

Primary integration module:
- `src/y2karaoke/core/components/whisper/whisper_integration.py`

Execution flow:
1. `transcribe_vocals(...)` performs cached Whisper transcription.
2. `align_lrc_text_to_whisper_timings(...)` maps lyric words onto Whisper timestamps.
3. `correct_timing_with_whisper(...)` chooses hybrid/DTW strategy based on quality gates.

Refactor notes:
- Orchestration logic is split into:
  - `whisper_integration_pipeline.py`
- `whisper_integration.py` remains the stable compatibility layer:
  - alias wiring
  - test hook context manager
  - thin delegates into pipeline implementation

Related modules:
- `whisper_dtw.py`, `whisper_alignment.py`, `whisper_mapping.py`, and focused helper modules.

Recent Improvements:
- **Drift Allowance:** Increased `_MAX_LINE_FORWARD_SHIFT_FROM_LRC` to 8.0s to handle more varied intro lengths and sync drift.
- **Segment Overrides:** Refined `_should_override_line_segment` logic to trust DTW-based segment assignments when strong phonetic hits are present, even if they deviate significantly from initial estimates.
- **Anchor Logic:** Improved anchor time calculation to prevent runaway drift while maintaining local alignment flexibility.
