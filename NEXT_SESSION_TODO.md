# Next Session TODO

Last updated: 2026-03-20

Use this file as a session handoff, not as a second backlog.

## Current Position

- The major lyrics / Whisper / sync architecture cleanup is largely complete.
- The main reference doc is `docs/tech_debt_backlog.md`.
- The main structure doc is `ARCHITECTURE.md`.
- The active work is quality iteration on curated short clips, not further broad architecture cleanup.

## Next Week Plan

1. Rerank the broad cached canary before choosing the next target:
   `PYTHONPATH=src ./.venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --match "Houdini|Con Calma|Sweet Caroline|Take On Me|Taste|Without Me|I Gotta Feeling|Time After Time|Total Eclipse|Stayin' Alive|Rap God|Royals" --offline`
2. If `Con Calma` still leads after the rerank, inspect whether lines 9-12 can be improved with the same correction-pass evidence model without reopening seed helpers.
3. If `Con Calma` drops out of first place, pivot to the new top broad-return clip rather than extending the `Con Calma` heuristics.
4. Ask for manual curation verification when the remaining miss looks plausibly like gold drift rather than a clean pipeline failure.

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
- `Con Calma` improved again from `0.2886s / 0.3020s` to `0.2430s / 0.2662s` in `benchmarks/results/20260321T004151Z`
  - driver: a narrow late-start reanchor now keeps lines when the first two lyric tokens are supported in-order by earlier Whisper words, instead of leaving those starts pinned to the later baseline timing
  - `Taste` improved from `0.3194s / 0.3125s` to `0.2240s / 0.2626s` in `benchmarks/results/20260320T234335Z`
  - driver: manual gold timing cleanup removed a misleading tail overhang, so `Taste` is no longer the top broad-return target
  - `Rap God` improved from `0.702s / 0.705s` to `0.233s / 0.213s` in `benchmarks/results/20260320T061849Z`
  - driver: dense short non-repeated rap verses need their own seed layout, distinct from both repeated-hook clips and the generic dense spread

## Current Cached Canary Baseline

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
  - broad target at that point: `Con Calma`
- Latest focused 4-song canary after the earlier-Whisper late-start reanchor:
  - `benchmarks/results/20260321T004151Z/benchmark_report.json`
  - curated canary weighted start mean: `0.239s`
  - key results:
    - `Con Calma`: `0.2430 / 0.2662`
    - `Houdini`: `0.2463 / 0.2562`
    - `Without Me`: `0.2792 / 0.2243`
    - `I Gotta Feeling`: `0.2034 / 0.1510`
  - this is a kept focused win, not a full rerank of the 12-song cached canary
- Latest broad cached canary rerank after the earlier-Whisper late-start reanchor:
  - `benchmarks/results/20260321T004829Z/benchmark_report.json`
  - curated canary weighted start mean: `0.221s`
  - updated ordering:
    - `Without Me`: `0.2792 / 0.2243`
    - `Houdini`: `0.2463 / 0.2562`
    - `Con Calma`: `0.2430 / 0.2662`
    - `Rap God`: `0.2329 / 0.2128`
    - `Time After Time`: `0.2261 / 0.2892`
    - `Taste`: `0.2240 / 0.2626`
    - `Sweet Caroline`: `0.2154 / 0.2513`
    - `I Gotta Feeling`: `0.2034 / 0.1510`
    - `Total Eclipse of the Heart`: `0.1961 / 0.2979`
    - `Royals`: `0.1882 / 0.1555`
    - `Take On Me`: `0.1513 / 0.3345`
    - `Stayin' Alive`: `0.1374 / 0.6253`
  - next broad-return target is now `Without Me`, not `Con Calma`

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
- `Con Calma` is materially healthier after the mixed-density coda rebalance and is no longer the obvious top target on seed layout alone.
- `Taste` is materially healthier after the recent gold refresh and is no longer the top broad-return target.
- `Sweet Caroline` is materially healthier after the latest short-title chorus rebalance and is no longer the top broad-return target.
- `Con Calma` is materially healthier again after the late-start Whisper reanchor and is no longer the top broad-return target after the full rerank.
- `Without Me` is now the clearest broad-return start-error target in the cached canary.
- `Without Me` is the next target, but the failure shape is narrow:
  - line 2 (`Back again`) is the one clear downstream regression
  - line 4 (`Tell a friend`) is mostly still a weak-seed / weak-onset miss
- `Without Me` line 2 now has a direct mapped-line postpass repro:
  - `_pull_late_lines_to_matching_segments()` can pull a `Guess who's back` / `Back again` toy pair from `4.692s` to about `3.63s`
  - a characterization test for that shape now lives in `tests/unit/whisper/test_whisper_mapping_post.py`
  - live-path ordering note: in `whisper_integration_stages.py`, `_pull_late_lines_to_matching_segments()` runs twice in the mapped-line postpasses (`postpass_pull_late_segments`, then `postpass_pull_late_segments_second`) before the onset snap stages
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
- For the current saved 4-song canary around the latest Whisper-start reanchor, use:
  - `PYTHONPATH=src ./.venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --match "Con Calma|Houdini|Without Me|I Gotta Feeling" --offline`
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
- next broad-return target: `Without Me`
- fallback broad-return targets after `Without Me`: `Houdini`, then `Con Calma`
- current `Without Me` diagnostic from `benchmarks/results/20260321T004829Z/06_eminem-without-me-repeated-hook_timing_report.json`:
  - line 2 `Back again` regressed from a decent pre-Whisper start `4.692` to final `4.400`, while gold is `4.950`
  - line 4 `Tell a friend` stayed near its pre-Whisper timing (`8.131 -> 8.120`) while gold is later at `9.050`
  - this suggests the first pass should inspect downstream compaction/rollback on line 2 before changing the repeated-hook seed again
  - code read note: the current finalize restore for repeated compact runs only protects late-shifted or end-collapsed runs in `whisper_integration_finalize.py`; it does not obviously catch this early-compaction shape on line 2
  - likely first code reads for that pass:
    - `whisper_integration_finalize._apply_force_dtw_finalize_passes()`
    - `whisper_alignment_pull_rules._merge_short_following_line_into_segment()`
    - `whisper_alignment_utils._clamp_repeated_line_duration()`
    - `whisper_alignment_pull_rules._pull_lines_near_segment_end()`
  - follow-up code read note: `_merge_short_following_line_into_segment()` is now the most plausible first culprit for line 2, because it can explicitly reflow lines 1 and 2 together into one segment window when the second line is very short; `_pull_lines_near_segment_end()` still looks secondary, and `_clamp_repeated_line_duration()` looks less likely for this exact shape
  - important failed probe: the helper above can reproduce the `Guess who's back` / `Back again` collapse in isolation (see unit test added in commit `22c6b22`), but a narrow live guard there did not move the `Without Me|Houdini|I Gotta Feeling` canary, so `Without Me` is probably not hitting that helper on the active path
  - result-payload note: the live `Without Me` benchmark artifact only exposes `whisper_corrections: 1`, no lexical mismatch diagnostics, and no stage metrics, so the next pass will need direct code-path inspection rather than relying on the saved JSON to identify the active helper
  - code-path note: the live `Without Me` run reports `whisper_force_dtw: false`, so the first active path to inspect is the direct alignment flow in `whisper_integration_align.py` and its baseline/audio correction passes, not the force-DTW finalize passes
  - direct-path narrowing: in the non-force-DTW flow, the first stage family worth instrumenting is `_apply_baseline_restore_corrections()` / `_apply_audio_alignment_corrections()`, especially `snap_first_word_to_whisper_onset()` and `reanchor_late_supported_lines_to_earlier_whisper()`, because those are the obvious early-start movers on short hook lines
  - failed direct-path probe: blocking `reanchor_late_supported_lines_to_earlier_whisper()` for 2-word lines produced no change at all on `Without Me|Con Calma|Houdini|I Gotta Feeling` (`20260321T010836Z`), so that helper is also unlikely to be the active `Without Me` culprit
  - toy-path probe: `snap_first_word_to_whisper_onset()` also leaves the `Guess who's back` / `Back again` toy pair unchanged, so the stronger remaining suspects are now the baseline/restore passes rather than the obvious onset shifters
  - baseline-restore narrowing: the current weak-evidence restore helpers also look unlikely for line 2, because they mainly restore large late shifts or unsupported early duplicate lines, while the live `Without Me` shape is asymmetric (`Back again` moves earlier than pre-Whisper by `-0.292s`, while `Shady's back` moves later by `+0.598s`)
  - stronger live-path suspect: in the mapped-line postpasses, `_pull_late_lines_to_matching_segments()` now looks like the best next target, because it directly shifts a line earlier toward a matching segment start before the later baseline/restore passes run
- latest mixed-density result:
  - `Con Calma` improved again after enabling a guarded mixed-density coda rebalance for the repeated-response-plus-tail shape
  - representative broad canary run: `benchmarks/results/20260320T232439Z`
  - focused confirmation run: `benchmarks/results/20260320T232236Z`
- latest correction-pass result:
  - `Con Calma` improved again after reanchoring late line starts to earlier in-order Whisper prefix support
  - representative focused canary run: `benchmarks/results/20260321T004151Z`
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
