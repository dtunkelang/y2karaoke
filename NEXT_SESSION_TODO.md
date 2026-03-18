# Next Session TODO

Last updated: 2026-03-18

Use this file as a session handoff, not as a second backlog.

## Current Position

- The major lyrics / Whisper / sync architecture cleanup is largely complete.
- The main reference doc is `docs/tech_debt_backlog.md`.
- The main structure doc is `ARCHITECTURE.md`.

## If Debt Work Resumes

Only continue if there is a concrete reason:
- feature work is getting blocked by ownership confusion
- a benchmark or CI regression points to a structural issue
- a test seam still depends on hidden state or unstable internals

Most likely next inspection targets:
- `src/y2karaoke/core/components/lyrics/helpers.py`
- `src/y2karaoke/core/components/lyrics/lrc.py`
- remaining isolated env-driven helper toggles in Whisper

## Guardrails

- Keep heuristic behavior unchanged.
- Do not add new ambient global state.
- Prefer typed config/state boundaries over new env reads.
- Preserve compatibility seams unless tests move in the same pass.
- Leave unrelated local artifacts alone:
  - `tools/run_benchmark_suite.py`
  - `1`
  - `benchmarks/gold_set_candidate/...`
