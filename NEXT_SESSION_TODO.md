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
- The repeated-hook pack is in a much healthier place:
  - `Houdini`: materially improved
  - `Without Me`: materially improved
  - `I Gotta Feeling`: healthy control
- The sparse/falsetto companion pack is now also useful:
  - `Take On Me`
  - `Time After Time`
  - `Total Eclipse of the Heart`
  - `Stayin' Alive`
- Key latest result:
  - `Stayin' Alive` improved from `2.5739s` to `0.1374s` start error in `benchmarks/results/20260319T223815Z`
  - driver: 2-line forced-alignment unlock plus better weak-onset seed fallback for two-line subset-refrain clips
  - `Take On Me` improved from `0.4731s` to `0.151s` start error in `benchmarks/results/20260320T023956Z`
  - driver: sparse-support forced fallback now redistributes words inside short sustained 5-word lines so held final words are not compressed into the tail
  - `Sweet Caroline` improved from `0.4788s` to `0.3104s` start error in `benchmarks/results/20260320T051849Z`
  - driver: short-title chorus layout now shortens the title line, widens the setup gap, and leaves more room for the tail line

## If Quality Work Resumes

- Recheck the broad curated pack before adding more clips:
  - `PYTHONPATH=src ./.venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --offline`
- If focusing on clip families, use:
  - `PYTHONPATH=src ./.venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --match "Houdini|Without Me|Gotta Feeling" --offline`
  - `PYTHONPATH=src ./.venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --match "Take On Me|Time After Time|Total Eclipse|Stayin' Alive" --offline`
- For mixed-density chorus / phrase-boundary checks, use:
  - `PYTHONPATH=src ./.venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --match "Con Calma|Sweet Caroline" --offline`
- Treat hard clips as valid targets if they reflect real production problems.
- Do not optimize against a single hard clip in isolation when the failure mode is still ambiguous; add companion clips first.
- Prefer fixes that improve clip-scoped priors or shared alignment behavior over song-specific heuristics.
- Broad offline tag runs can still be dominated by uncached clips; prefer known cached match-based subsets for clean quality comparisons.
- For stubborn clips, inspect in this order:
  - helper-generated seed on the real cached clip audio
  - accepted forced-alignment output
  - final timing report
- For sparse/falsetto clips, distinguish:
  - line-boundary problems
  - within-line word-distribution problems
  `Take On Me` turned out to be the second kind.

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
- rerun the broader cached canary subset after the `Sweet Caroline` improvement and identify the next true outlier
- latest mixed-density result:
  - `Con Calma` improved again after reducing mixed-density trailing pad and widening long-line -> short-response gaps slightly
  - representative run: `benchmarks/results/20260320T063430Z`
- likely remaining broad target: `Houdini`
- treat `Johnny Cash - Hurt` as a standalone hard canary unless a closer companion clip finally reproduces its line-end overextension

## Guardrails

- Keep heuristic behavior unchanged.
- Do not add new ambient global state.
- Prefer typed config/state boundaries over new env reads.
- Preserve compatibility seams unless tests move in the same pass.
- Protect curated manual work before further experimentation.
