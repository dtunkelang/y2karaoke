# Next Session TODO

Last updated: 2026-03-20

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
  - `Houdini` improved from `0.3772s / 0.3991s` to `0.2551s / 0.2771s` in `benchmarks/results/20260320T224703Z`
  - driver: default-on restored low-support run onset shifts now move repeated `Houdini` lines later inside the main `whisper_map_lrc_dtw` path without hurting repeated-hook controls
  - `Stayin' Alive` improved from `2.5739s` to `0.1374s` start error in `benchmarks/results/20260319T223815Z`
  - driver: 2-line forced-alignment unlock plus better weak-onset seed fallback for two-line subset-refrain clips
  - `Take On Me` improved from `0.4731s` to `0.151s` start error in `benchmarks/results/20260320T023956Z`
  - driver: sparse-support forced fallback now redistributes words inside short sustained 5-word lines so held final words are not compressed into the tail
  - `Sweet Caroline` improved from `0.4788s` to `0.3104s` start error in `benchmarks/results/20260320T051849Z`
  - driver: short-title chorus layout now shortens the title line, widens the setup gap, and leaves more room for the tail line
  - `Sweet Caroline` improved again from `0.3104s / 0.2650s` to `0.2154s / 0.2513s` in `benchmarks/results/20260321T000522Z`
  - driver: the short-title chorus helper was still over-allocating the line-1 setup gap, so tightening that narrow layout moved line 2 into the right window without disturbing the broad cached canary
  - `Con Calma` improved from `0.3426s / 0.3825s` to `0.2886s / 0.3020s` in `benchmarks/results/20260320T232439Z`
  - driver: mixed-density chorus clips now rebalance the late coda when the repeated response pair is followed by a denser four-line tail, which moved the seed later without hurting the broad cached canary
  - `Taste` improved from `0.3194s / 0.3125s` to `0.2240s / 0.2626s` in `benchmarks/results/20260320T234335Z`
  - driver: manual gold timing cleanup removed a misleading tail overhang, so `Taste` is no longer the top broad-return target
  - `Rap God` improved from `0.702s / 0.705s` to `0.233s / 0.213s` in `benchmarks/results/20260320T061849Z`
  - driver: dense short non-repeated rap verses need their own seed layout, distinct from both repeated-hook clips and the generic dense spread

## Current Cached Canary Baseline

- Best broad cached comparison set before the latest tiny `Con Calma` follow-up was:
  - `benchmarks/results/20260320T062234Z`
- Current broad cached ordering from that run:
  - `Houdini`: `0.377 / 0.399`
  - `Con Calma`: `0.350 / 0.390`
  - `Taste`: `0.319 / 0.312`
  - `Sweet Caroline`: `0.310 / 0.265`
  - `Without Me`: `0.279 / 0.224`
  - `Rap God`: `0.233 / 0.213`
  - `Time After Time`: `0.226 / 0.289`
  - `I Gotta Feeling`: `0.203 / 0.151`
  - `Total Eclipse of the Heart`: `0.196 / 0.298`
  - `Royals`: `0.188 / 0.156`
  - `Take On Me`: `0.151 / 0.335`
  - `Stayin' Alive`: `0.137 / 0.625`
- The latest focused chorus rerun is:
  - `benchmarks/results/20260320T063430Z`
  - `Con Calma`: `0.3426 / 0.3825`
  - `Sweet Caroline`: `0.3104 / 0.265`
- The latest broad rerun started clean in:
  - `benchmarks/results/20260320T063518Z`
  - confirmed early:
    - `Houdini`: flat at `0.3772 / 0.3991`
    - `Con Calma`: improved at `0.3426 / 0.3825`
- Latest broad cached canary subset after enabling default restored-run onset shifts and the `Con Calma` coda rebalance:
  - `benchmarks/results/20260320T232439Z/benchmark_report.json`
  - curated canary weighted start mean: `0.2478s`
  - key ordering:
    - `Taste`: `0.3194 / 0.3125`
    - `Sweet Caroline`: `0.3104 / 0.265`
    - `Con Calma`: `0.2886 / 0.3020`
    - `Without Me`: `0.2792 / 0.2243`
    - `Houdini`: `0.2551 / 0.2771`
    - `Rap God`: `0.2329 / 0.2128`
    - `Time After Time`: `0.2261 / 0.2892`
    - `I Gotta Feeling`: `0.2034 / 0.151`
    - `Total Eclipse of the Heart`: `0.1961 / 0.2979`
    - `Royals`: `0.1882 / 0.1555`
    - `Take On Me`: `0.1513 / 0.3345`
    - `Stayin' Alive`: `0.1374 / 0.6253`
  - only material movement from the prior broad cached subset was `Con Calma`
- Latest broad cached canary subset after the `Taste` gold refresh:
  - `benchmarks/results/20260320T235217Z/benchmark_report.json`
  - curated canary weighted start mean: `0.2398s`
  - key ordering:
    - `Sweet Caroline`: `0.3104 / 0.2650`
    - `Con Calma`: `0.2886 / 0.3020`
    - `Without Me`: `0.2792 / 0.2243`
    - `Houdini`: `0.2551 / 0.2771`
    - `Rap God`: `0.2329 / 0.2128`
    - `Time After Time`: `0.2261 / 0.2892`
    - `Taste`: `0.2240 / 0.2626`
    - `I Gotta Feeling`: `0.2034 / 0.1510`
    - `Total Eclipse of the Heart`: `0.1961 / 0.2979`
    - `Royals`: `0.1882 / 0.1555`
    - `Take On Me`: `0.1513 / 0.3345`
    - `Stayin' Alive`: `0.1374 / 0.6253`
  - only material movement from `20260320T232439Z` was the curated `Taste` improvement
- Latest broad cached canary subset after tightening the short-title chorus helper:
  - `benchmarks/results/20260321T000522Z/benchmark_report.json`
  - curated canary weighted start mean: `0.232s`
  - key ordering:
    - `Con Calma`: `0.2886 / 0.3020`
    - `Without Me`: `0.2792 / 0.2243`
    - `Houdini`: `0.2551 / 0.2771`
    - `Rap God`: `0.2329 / 0.2128`
    - `Time After Time`: `0.2261 / 0.2892`
    - `Taste`: `0.2240 / 0.2626`
    - `Sweet Caroline`: `0.2154 / 0.2513`
    - `I Gotta Feeling`: `0.2034 / 0.1510`
    - `Total Eclipse of the Heart`: `0.1961 / 0.2979`
    - `Royals`: `0.1882 / 0.1555`
    - `Take On Me`: `0.1513 / 0.3345`
    - `Stayin' Alive`: `0.1374 / 0.6253`
  - only material movement from `20260320T235217Z` was the `Sweet Caroline` improvement
  - likely remaining broad target: `Con Calma`

## New Curated Clips Added This Session

- `44_eminem-rap-god-dense-verse.gold.json`
  - dense short non-repeated rap verse
- `45_lorde-royals-opening-verse-conversational.gold.json`
  - low-energy conversational phrasing
- `43_nine-inch-nails-hurt-opening-verse-paired-lines.gold.json`
  - useful as a protected comparison clip, but it did not reproduce the Johnny Cash `Hurt` failure shape
- `42_gary-jules-mad-world-opening-verse-paired-lines.gold.json`
  - useful comparison clip, but not a close enough `Hurt` companion
- `41_r-e-m-everybody-hurts-paired-lines-verse.gold.json`
  - much closer to `Hurt` than `Creep`, but still not the same live failure shape
- `40_radiohead-creep-paired-lines-chorus.gold.json`
  - protected and useful, but not a true `Hurt` companion

## Current Diagnosis

- `Houdini` is no longer the clearest broad-return start-time outlier after the default restored-run onset shift change.
- `Con Calma` is materially healthier after the mixed-density coda rebalance and is no longer the top broad-return target.
- `Taste` is materially healthier after the recent gold refresh and is no longer the top broad-return target.
- `Sweet Caroline` is materially healthier after the latest short-title chorus rebalance and is no longer the top broad-return target.
- `Con Calma` is back to being the clearest remaining broad-return clip in the cached canary.
- `Royals` is healthy and does not currently expose a new failure family.
- `Johnny Cash - Hurt` remains a real hard canary, but companion attempts (`Creep`, `Everybody Hurts`, `Mad World`, `NIN Hurt`) did not reproduce its exact line-end overextension shape.
- Treat `Johnny Cash - Hurt` as a guardrail, not the current main optimization driver.

## If Quality Work Resumes

- Recheck the broad curated pack before adding more clips:
  - `PYTHONPATH=src ./.venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --offline`
- If focusing on clip families, use:
  - `PYTHONPATH=src ./.venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --match "Houdini|Without Me|Gotta Feeling" --offline`
  - `PYTHONPATH=src ./.venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --match "Take On Me|Time After Time|Total Eclipse|Stayin' Alive" --offline`
- For mixed-density chorus / phrase-boundary checks, use:
  - `PYTHONPATH=src ./.venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --match "Con Calma|Sweet Caroline" --offline`
- For the refreshed `Taste` comparison, use:
  - `PYTHONPATH=src ./.venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --match "Sweet Caroline|Taste|Con Calma" --offline`
- For the refreshed short-title chorus comparison, use:
  - `PYTHONPATH=src ./.venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --match "Sweet Caroline|Con Calma|Taste" --offline`
- For the current broad cached canary subset, use:
  - `PYTHONPATH=src ./.venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --match "Houdini|Con Calma|Sweet Caroline|Take On Me|Taste|Without Me|I Gotta Feeling|Time After Time|Total Eclipse|Stayin' Alive|Rap God|Royals" --offline`
- For the new dense/conversational shapes, use:
  - `PYTHONPATH=src ./.venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --match "Rap God|Royals" --offline`
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
- The editor requires non-empty `words` on every line.
- `tools/curated_clip_helper.py` was hardened this session to:
  - repair empty-word gold files before opening
  - resolve real cached source audio filenames instead of assuming `Title.wav`
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
- likely remaining broad target: `Con Calma`
- next broad-return fallback after `Con Calma`: `Without Me`
- latest mixed-density result:
  - `Con Calma` improved again after enabling a guarded mixed-density coda rebalance for the repeated-response-plus-tail shape
  - representative broad canary run: `benchmarks/results/20260320T232439Z`
  - focused confirmation run: `benchmarks/results/20260320T232236Z`
- latest curated refresh:
  - `Taste` gold timing cleanup improved the clip substantially without code changes
  - representative rerun: `benchmarks/results/20260320T234335Z`
- latest short-title chorus result:
  - `Sweet Caroline` improved again after tightening the setup-gap allocation inside the narrow short-title chorus helper
  - representative broad canary run: `benchmarks/results/20260321T000522Z`
  - focused confirmation run: `benchmarks/results/20260321T000412Z`
- treat `Johnny Cash - Hurt` as a standalone hard canary unless a closer companion clip finally reproduces its line-end overextension

## Guardrails

- Keep heuristic behavior unchanged.
- Do not add new ambient global state.
- Prefer typed config/state boundaries over new env reads.
- Preserve compatibility seams unless tests move in the same pass.
- Protect curated manual work before further experimentation.
