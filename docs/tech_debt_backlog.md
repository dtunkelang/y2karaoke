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
- Status: completed (phase 1)
- Scope:
  - Emit stable pass-level counters for key finalize/postpass stages.
  - Emit per-pass durations (`*_sec`) for the same stages.
  - Include these metrics in returned Whisper metrics for regression triage.

4. **Heuristic surface reduction via explicit config objects**
- Impact: high
- Effort: medium-high
- Status: in progress
- Scope:
  - Replace scattered thresholds/env checks with typed config structs.
  - Keep defaults unchanged; enable profile-based tuning (safe/aggressive).

5. **Cross-pass integration tests for alignment edge cases**
- Impact: high
- Effort: medium
- Status: queued
- Scope:
  - Add scenario tests for block transitions, repeated chorus resets, interjections, sparse Whisper output.
  - Focus on behavioral invariants (ordering, monotonic starts, no line loss).

6. **Benchmark trust instrumentation for reference divergence**
- Impact: medium-high
- Effort: medium
- Status: queued
- Scope:
  - Auto-flag likely LRC/video divergence with confidence tags.
  - Separate “pipeline error likely” vs “reference mismatch likely” in reports.

7. **Performance profiling + budgets on Whisper alignment paths**
- Impact: medium
- Effort: medium
- Status: queued
- Scope:
  - Add timing telemetry for major stages.
  - Define practical runtime budget checks in benchmark runs.
