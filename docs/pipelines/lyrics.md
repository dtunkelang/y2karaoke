# Lyrics Pipeline

Primary facade:
- `src/y2karaoke/pipeline/lyrics/__init__.py`

Core components:
- `src/y2karaoke/core/components/lyrics/api.py`
- `src/y2karaoke/core/components/lyrics/lyrics_whisper.py`
- `src/y2karaoke/core/components/lyrics/lyrics_whisper_quality.py`
- `src/y2karaoke/core/components/lyrics/sync.py`
- `src/y2karaoke/core/components/lyrics/lyrics_whisper_pipeline.py`
- `src/y2karaoke/core/components/lyrics/sync_pipeline.py`

Execution flow (quality-aware path):
1. Fetch synced lyrics (multi-source) and/or fallback text.
2. Parse into timed lines and words.
3. Run quality evaluation and issue aggregation.
4. Optionally apply Whisper alignment refinement.
5. Return lines + metadata + quality report.

Refactor notes:
- Heavy orchestration is split into:
  - `lyrics_whisper_pipeline.py`
  - `sync_pipeline.py`
- Main modules preserve compatibility wrappers and hook seams used by tests.
- Benchmark reports now expose two agreement families:
  - Independent agreement metrics (`agreement_*`) for cross-strategy comparison.
  - Whisper-anchor diagnostics (`whisper_anchor_*`) for drift debugging.

Recent Improvements:
- **Auto-Offset Safety:** Reduced the default auto-offset clamp from 30s to 5s. This prevents applying dangerously large offsets automatically when vocal detection peaks are ambiguous.
- **Offline Reliability:** Enhanced offline mode to attempt resolution from local caches even when network providers are unavailable or disabled.
- **Quality Reporting:** Improved tracking of large timing deltas and duration mismatches in the quality reports.
