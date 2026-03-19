# Next Session TODO

Last updated: 2026-03-19

Use this file as a session handoff, not as a second backlog.

## Current Position

- The major lyrics / Whisper / sync architecture cleanup is largely complete.
- The main reference doc is `docs/tech_debt_backlog.md`.
- The main structure doc is `ARCHITECTURE.md`.
- The active work is quality iteration on curated short clips, not further broad architecture cleanup.

## Current Quality Position

- Recent short-clip improvements came from:
  - clip-bounded audio and clip-scoped gold lyrics
  - better clip scoring against gold
  - onset-aware seeding for untimed plain-text clip lyrics
- Current repeated-hook companion set:
  - `Houdini`
  - `Without Me`
  - `I Gotta Feeling`
- Current repeated-hook results:
  - `Houdini`: improved materially and is worth keeping
  - `I Gotta Feeling`: healthy control
  - `Without Me`: still the next likely quality target

## If Quality Work Resumes

- Start from the repeated-hook pack before adding more clips:
  - `PYTHONPATH=src ./.venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --match "Houdini|Without Me|Gotta Feeling" --offline`
- Treat hard clips as valid targets if they reflect real production problems.
- Do not optimize against a single hard clip in isolation when the failure mode is still ambiguous; add companion clips first.
- Prefer fixes that improve clip-scoped priors or shared alignment behavior over song-specific heuristics.

## Curation Process

- Open saved gold files through `tools/curated_clip_helper.py`.
- Do not guess filenames, editor URLs, or audio paths by hand.
- After any user curation:
  - verify the saved gold JSON on disk
  - verify clip audio duration/path on disk
  - reopen from the saved gold file
  - commit and push immediately

## If Debt Work Resumes

Only continue if there is a concrete reason:
- feature work is getting blocked by ownership confusion
- a benchmark or CI regression points to a structural issue
- a test seam still depends on hidden state or unstable internals

Most likely next inspection targets:
- `Without Me` repeated-hook drift in the clip-alignment path
- clip-scoped plain-text seeding behavior in `src/y2karaoke/core/components/lyrics/helpers.py`
- only then broader alignment heuristics if the clip-family signal stays consistent

## Guardrails

- Keep heuristic behavior unchanged.
- Do not add new ambient global state.
- Prefer typed config/state boundaries over new env reads.
- Preserve compatibility seams unless tests move in the same pass.
- Protect curated manual work before further experimentation.
