# Tech Debt Backlog

Last updated: 2026-03-04

## Priority Model

- **Impact:** expected quality, stability, or developer-velocity gain.
- **Effort:** implementation + test + rollout complexity.
- **Priority:** ordered by impact-to-effort and execution risk.

## Top Priorities

1. **Prevent complexity regression in CI (Completed)**
- Impact: high
- Effort: low
- Status: completed
- Notes: `tools/quality_guardrails.py` now enforces strict `C901` complexity budget (`max-complexity=10`) for `src/` and `tests/`.

2. **Consolidated debt map and execution queue (Completed)**
- Impact: medium
- Effort: low
- Status: completed
- Notes: this backlog defines actionable, ranked work instead of ad-hoc cleanup.

3. **Whisper pipeline observability: per-pass counters in metrics**
- Impact: high
- Effort: medium
- Status: completed
- Scope:
  - Emit stable pass-level counters for key finalize/postpass stages.
  - Emit per-pass durations (`*_sec`) for finalize stages and alignment-stage durations/counts.
  - Include these metrics in returned Whisper metrics for regression triage.

4. **Heuristic surface reduction via explicit config objects**
- Impact: high
- Effort: medium-high
- Status: in progress
- Scope:
  - Replace scattered thresholds/env checks with typed config structs.
  - Keep defaults unchanged; enable profile-based tuning (safe/aggressive).
  - Completed so far:
    - Shared `Y2K_WHISPER_PROFILE` parsing centralized in `whisper_profile.py`.
    - Alignment-pass dependency plumbing bundled via `_AlignmentPassHooks` in
      `whisper_integration_pipeline.py`.
    - Correct-timing dependency plumbing bundled via `_CorrectTimingHooks` in
      `whisper_integration_pipeline.py`.
    - Hook dataclasses and hook-kwargs expansion extracted into dedicated
      `whisper_integration_hooks.py` module.
    - Removed redundant pass-through wrappers from
      `whisper_integration_pipeline.py`; the pipeline now wires directly to stage/baseline/finalize helper implementations.
    - Split alignment orchestration into `whisper_integration_pipeline_align.py`,
      leaving `whisper_integration_pipeline.py` as a thinner facade for align/correct/transcribe entry wiring.
    - Split correct-timing orchestration into
      `whisper_integration_pipeline_correct.py`; `whisper_integration_pipeline.py`
      now serves primarily as a backward-compatible facade/export surface.
    - Segment assignment env heuristics centralized in `whisper_blocks.py`.
    - Correction decision thresholds centralized in `whisper_integration_correct.py`.
    - Mapping decision thresholds centralized in `whisper_integration_align.py`.
    - Retry-improvement thresholds centralized in `whisper_integration_retry.py`.
    - Silence-refinement env gating centralized in `whisper_alignment_refinement.py`.
    - Whisper transcription option selection centralized in `whisper_integration_transcribe.py`.
    - Added opt-in profile presets (`safe/default/aggressive`) via `Y2K_WHISPER_PROFILE` for mapping/correction threshold configs.

5. **Cross-pass integration tests for alignment edge cases**
- Impact: high
- Effort: medium
- Status: in progress
- Scope:
  - Add scenario tests for block transitions, repeated chorus resets, interjections, sparse Whisper output.
  - Focus on behavioral invariants (ordering, monotonic starts, no line loss).
  - Completed so far:
    - Added direct unit coverage for mapped-line postpass orchestration/invariants in
      `tests/unit/whisper/test_whisper_integration_stages_unit.py`.
    - Added pipeline invariant scenario for repeated-reset-style timing jitter without line loss
      in `tests/unit/whisper/test_whisper_integration_pipeline_unit.py`.

6. **Benchmark trust instrumentation for reference divergence**
- Impact: medium-high
- Effort: medium
- Status: in progress
- Scope:
  - Auto-flag likely LRC/video divergence with confidence tags.
  - Separate “pipeline error likely” vs “reference mismatch likely” in reports.
  - Completed so far:
    - Added per-song quality diagnosis classification with confidence/reason codes.
    - Surfaced diagnosis counts/ratios in aggregate output and markdown summaries.

7. **Performance profiling + budgets on Whisper alignment paths**
- Impact: medium
- Effort: medium
- Status: in progress
- Scope:
  - Add timing telemetry for major stages.
  - Define practical runtime budget checks in benchmark runs.
  - Completed so far:
    - Added benchmark runtime budget warnings for whisper/alignment phase-share and scheduler overhead.
    - Added CLI threshold controls and strict mode exit path for runtime budget enforcement.
