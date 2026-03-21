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
  - trace note: `Y2K_TRACE_WHISPER_STAGES_JSON` alone only writes the file shell; stage snapshots stay empty unless `Y2K_TRACE_WHISPER_LINE_RANGE` is also set
  - code-path note: the live `Without Me` run reports `whisper_force_dtw: false`, so the first active path to inspect is the direct alignment flow in `whisper_integration_align.py` and its baseline/audio correction passes, not the force-DTW finalize passes
  - direct-path narrowing: in the non-force-DTW flow, the first stage family worth instrumenting is `_apply_baseline_restore_corrections()` / `_apply_audio_alignment_corrections()`, especially `snap_first_word_to_whisper_onset()` and `reanchor_late_supported_lines_to_earlier_whisper()`, because those are the obvious early-start movers on short hook lines
  - failed direct-path probe: blocking `reanchor_late_supported_lines_to_earlier_whisper()` for 2-word lines produced no change at all on `Without Me|Con Calma|Houdini|I Gotta Feeling` (`20260321T010836Z`), so that helper is also unlikely to be the active `Without Me` culprit
  - toy-path probe: `snap_first_word_to_whisper_onset()` also leaves the `Guess who's back` / `Back again` toy pair unchanged, so the stronger remaining suspects are now the baseline/restore passes rather than the obvious onset shifters
  - baseline-restore narrowing: the current weak-evidence restore helpers also look unlikely for line 2, because they mainly restore large late shifts or unsupported early duplicate lines, while the live `Without Me` shape is asymmetric (`Back again` moves earlier than pre-Whisper by `-0.292s`, while `Shady's back` moves later by `+0.598s`)
  - stronger live-path suspect: in the mapped-line postpasses, `_pull_late_lines_to_matching_segments()` now looks like the best next target, because it directly shifts a line earlier toward a matching segment start before the later baseline/restore passes run
  - failed probe: a narrow guard against extreme early pulls on very short handoff lines inside `_pull_late_lines_to_matching_segments()` produced no benchmark movement on `Without Me|Houdini|Con Calma|I Gotta Feeling` (`20260321T011749Z`), so the active `Without Me` miss is likely a subtler condition inside that stage or a neighboring postpass rather than the obvious toy shape alone
  - sub-pass note: `_pull_late_lines_to_matching_segments()` immediately chains into repetitive/cadence helpers (`_realign_repetitive_runs_to_matching_segments()`, `_smooth_adjacent_duplicate_line_cadence()`, `_rebalance_short_question_pairs()`, `_retime_repetitive_question_runs_to_segment_windows()`, `_pull_adjacent_similar_lines_across_long_gaps()`, `_retime_split_short_refrains_to_matching_segments()`), so those are the next best reads for a repeated-hook clip if the top-level pull guard keeps missing live behavior
  - sub-pass ranking note: for `Without Me` line 2 specifically, `_pull_adjacent_similar_lines_across_long_gaps()` and `_rebalance_short_question_pairs()` look effectively irrelevant, while `_retime_split_short_refrains_to_matching_segments()` is a real candidate because it explicitly retimes short repeated refrains to later matching segment windows
  - elimination note: `_retime_split_short_refrains_to_matching_segments()` is not a live fit for `Without Me` line 2 after all, because its shared helper only fires when the normalized line text appears at least twice in the clip, and `Back again` is unique in the current gold clip
  - stronger structural eliminations: `_realign_repetitive_runs_to_matching_segments()` and `_retime_repetitive_question_runs_to_segment_windows()` also look non-viable for the current `Without Me` clip, because both require repeated runs of length at least 3, and the question-run path additionally requires `?` lines
  - practical short list now: for `Without Me` line 2, the remaining plausible movers inside or immediately after the segment-pull stage are the direct top-level pull logic itself, `_smooth_adjacent_duplicate_line_cadence()`, and possibly a later non-repetition postpass rather than the broader repeated-run helpers
  - cadence probe: `_smooth_adjacent_duplicate_line_cadence()` leaves the `Guess who's back` / `Back again` / `Shady's back` toy shape unchanged, and its overlap thresholds look too strict for that handoff, so it is now a weaker suspect than the direct top-level pull logic
  - direct-pull trigger note: on the `Guess who's back` / `Back again` toy shape, line 2 is treated as non-repetitive (`prev_overlap=1`, `next_overlap=0`) and then clears the direct first-pass pull gates with an exact `back again` segment match (`sim=1.0`, token overlap `1.0`, `late_by=1.082` > `min_late=1.0`), so the next real read should focus on that first-pass late-pull thresholding rather than the repetition helpers
  - direct-pull clamp note: in that same toy case, the helper does not snap all the way to the segment start (`3.61`); it clamps to `max(seg.start, prev_end + min_gap)`, yielding about `3.63`, so if the live `Without Me` miss shares this path the next read should inspect both the `min_late` admission gate and the neighbor-bound `window_start` clamp
  - live-report reality check: actual `Without Me` line 2 still ends at `4.400` with `pre_whisper_start=4.692` and `whisper_line_start_delta=-0.79`; the nearest segment context in the timing report is the merged `"Guess who's back, back again"` segment ending at `5.41`, not a direct snap to `3.63`, so the live miss may be a weaker or partially clamped version of the toy path rather than the full first-pass pull shape
  - live-window note: the same line-2 report only shows `whisper_window_word_count=3` across `3.4-7.01`, which fits the merged-segment story and suggests the active miss may be driven by sparse local Whisper support plus segment-window constraints, not just an overly aggressive pull threshold in isolation
  - boundary note: for the merged `Guess who's back, back again` segment shape, the usable line-2 pull window would still be about `3.63-5.41`, so the previous-line clamp is the active start limiter and the next-line boundary is not the thing keeping the live result at `4.400`
  - window-usage note: the live line-2 result (`4.400-5.230`) only uses about `0.83s` of a roughly `1.78s` available merged-segment window, leaving `0.77s` unused before the line and only `0.18s` after it, so the current behavior is tail-biased inside the window rather than simply blocked by hard boundaries
  - placement note: line 2 also sits about `0.295s` late relative to the midpoint of that usable window, which reinforces that the active behavior is anchoring it closer to the segment end than the segment start
  - duration note: line 2 is shorter than pre-Whisper (`0.83s` vs `1.089s`) but still longer than gold (`0.70s`), so the remaining miss looks more like late placement inside the window than a pure duration-collapse problem
  - line-split note: line 2 and line 4 should stay separated in future probes; line 2 shows a real negative Whisper start delta (`-0.79`) against merged-segment context, while line 4 has no Whisper start delta at all and stays near pre-Whisper timing, so they are not the same repeated-hook failure mode
  - line-4 reality check: `Tell a friend` drifts only `-0.011s` from pre-Whisper timing (`8.131 -> 8.120`) despite a 5-word local Whisper window, which is more evidence that line 4 is a weak-seed / weak-onset miss rather than the same mapped-window placement bug as line 2
  - next-step order for `Without Me`: inspect line 2 first in the direct late-pull placement path inside `whisper_mapping_post_segment_pull.py`; only after that is understood should line 4 reopen the weaker seed/onset path
  - do-not-start-here note: for the next `Without Me` pass, do not begin by changing repeated-run helpers, broad duration restores, or line-4 seed heuristics; the first code read should stay on line 2 placement inside the direct segment-pull path
  - exact first read: start with the first-pass late-pull gates in `whisper_mapping_post_segment_pull.py:147-185`, especially the `min_sim_required` / `best_token_overlap` / `late_by` / `window_start` block, because that is the narrowest code path still consistent with the current line-2 evidence
  - exact first artifact: keep `benchmarks/results/20260321T004829Z/06_eminem-without-me-repeated-hook_timing_report.json` open while reading that block, because its line-2 merged-segment context and line-4 no-delta context are the current anchor facts
  - exact first baseline to compare against: in `20260321T004829Z`, `Without Me` is `0.2792 / 0.2243`
  - exact line targets: line 2 `Back again` is currently `4.400-5.230` vs gold `4.950-5.650` (pre-Whisper `4.692-5.781`); line 4 `Tell a friend` is currently `8.120-9.990` vs gold `9.050-10.150` (pre-Whisper `8.131-9.793`)
  - exact first success criterion: a line-2 change should move `Back again` later than `4.400` without pushing `Tell a friend` earlier than its current `8.120`, and the focused canary should beat the current `Without Me` baseline `0.2792 / 0.2243`
  - exact first failure criterion: if a change leaves `Back again` at `4.400` or earlier, or regresses `Houdini`, `Con Calma`, or `I Gotta Feeling` on the same focused canary, revert it immediately rather than broadening the experiment
  - exact first rerun command: `PYTHONPATH=src ./.venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --match "Without Me|Houdini|Con Calma|I Gotta Feeling" --offline`
  - exact first trace env: pair `Y2K_TRACE_WHISPER_STAGES_JSON=/tmp/without_me_trace.json` with `Y2K_TRACE_WHISPER_LINE_RANGE=2-2` if stage snapshots are needed for line 2; the JSON path alone will create an empty shell file
  - exact first traced rerun command: `Y2K_TRACE_WHISPER_STAGES_JSON=/tmp/without_me_trace.json Y2K_TRACE_WHISPER_LINE_RANGE=2-2 PYTHONPATH=src ./.venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --match "Without Me|Houdini|Con Calma|I Gotta Feeling" --offline`
  - exact first local checks after edits: `PYTHONPATH=src ./.venv/bin/pytest tests/unit/whisper/test_whisper_mapping_post.py -q` and `flake8 src/y2karaoke/core/components/whisper/whisper_mapping_post_segment_pull.py --count --max-complexity=15 --max-line-length=127 --statistics`
  - exact second read for line 4: if line 2 is understood and line 4 still needs work, start in `whisper_integration_align_corrections.py:65-113` (`_apply_baseline_constraint_and_snap`) and then `whisper_integration_align_experimental.py:190-273` (`reanchor_late_supported_lines_to_earlier_whisper`), because those are the concrete weak-seed / onset-style start movers already known to touch this family
  - exact second trace env: if line 4 needs stage snapshots later, use the same JSON env with `Y2K_TRACE_WHISPER_LINE_RANGE=4-4` so the trace stays focused on the weak-seed path rather than the line-2 placement path
  - exact second traced rerun command: `Y2K_TRACE_WHISPER_STAGES_JSON=/tmp/without_me_trace.json Y2K_TRACE_WHISPER_LINE_RANGE=4-4 PYTHONPATH=src ./.venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --match "Without Me|Houdini|Con Calma|I Gotta Feeling" --offline`
  - exact second local checks after edits: `PYTHONPATH=src ./.venv/bin/pytest tests/unit/whisper/test_whisper_integration_align_experimental_unit.py -q` and `flake8 src/y2karaoke/core/components/whisper/whisper_integration_align_experimental.py --count --max-complexity=15 --max-line-length=127 --statistics`
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
