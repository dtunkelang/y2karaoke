# Next Session TODO

Last updated: 2026-03-25

Use this file as a session handoff, not as a second backlog.

For the reusable iteration pattern behind the current handoff style, see:
- `docs/development.md`
- `docs/curated_clips.md`

This handoff collected unusually frequent tiny updates because iteration budget was tight.
In normal mode, batch note-only updates unless they would be expensive to rediscover.

## Current Position

- The major lyrics / Whisper / sync architecture cleanup is largely complete.
- The main reference doc is `docs/tech_debt_backlog.md`.
- The main structure doc is `ARCHITECTURE.md`.
- The active work is quality iteration on curated short clips, not further broad architecture cleanup.
- Latest kept quality win:
  - `Rap God|Royals` improved from `0.213s` in `benchmarks/results/20260326T044631Z` to `0.195s` in `benchmarks/results/20260326T045058Z`
  - driver: timing reports were being written after `_apply_splash_offset()`, which added a presentation-only global delay to benchmarked starts; reports now use the pre-splash line set while rendered karaoke keeps the splash offset
  - concrete `Rap God` effect: line 1 now reports `0.40-2.95` instead of `0.50-3.05`, and line 2 now reports `2.99-4.09` instead of `3.09-4.18`
- Latest kept quality win:
  - `Houdini` improved from `0.2209s / 0.2394s` to `0.1683s / 0.2164s` in `benchmarks/results/20260326T042945Z`
  - focused 4-song canary improved from `0.2160s` in `20260325T210658Z` to `0.1970s` in `20260326T043025Z`
  - broad cached canary improved from `0.2050s` in `20260325T231558Z` to `0.2010s` in `20260326T043136Z`
  - driver: after `shift_restored_low_support_runs_to_onset()` moves the repeated `Houdini` tail as a block, a new narrow suffix-tail onset reanchor can move only the late compact repeated tail lines further right while preserving their ends
- Latest negative learning:
  - preserving the mapped repeated tail through `_constrain_line_starts_to_baseline()` was a false lead and regressed `Houdini` badly to `0.442s` in `benchmarks/results/20260326T042027Z`
  - implication: the baseline snap itself is not the right intervention target; the usable signal is downstream in the low-support onset repair layer
- Current blocker after the kept `Houdini` win:
  - the old `Rap God` boundary mystery was partly diagnostic, not alignment: the clean forced-alignment boundary was being shifted later by splash offset before report serialization
  - next session should rerank the broad cached canary from the new report baseline before taking another narrow alignment branch

## Next Week Plan

1. Start from the newest broad cached canary, not the older `Con Calma` ordering:
   `PYTHONPATH=src ./.venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --match "Houdini|Con Calma|Sweet Caroline|Take On Me|Taste|Without Me|I Gotta Feeling|Time After Time|Total Eclipse|Stayin' Alive|Rap God|Royals" --offline`
2. `Taste` is healthier again after the accepted-forced-alignment prefix repair. The cleaner next code-side targets are now `Rap God`, `Houdini`, and possibly `Time After Time`.
3. `Without Me` is no longer the best next driver after the latest forced-alignment repair. Keep it as a protected control unless a new failure family appears.
4. If continuing the best-understood narrow failure instead, start with `Rap God` before reopening `Taste` or `Without Me`.
5. Ask for manual curation verification when the remaining miss looks plausibly like gold drift rather than a clean pipeline failure. No new clips are needed right now.
6. The newest companion clips are now:
   - `Please Please Please` (`chorus-setup-tail`)
   - `Espresso` (`opening-hook`)
   Rerun with:
   `PYTHONPATH=src ./.venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --match "Please Please Please|Espresso" --offline`

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
- `Without Me` improved from `0.2792s / 0.2243s` to `0.2387s / 0.2764s` in `benchmarks/results/20260325T200610Z`
  - driver: after accepted WhisperX forced alignment, a new compact 3-word suffix-match repair now rebuilds lines like `Tell a friend` later from exact raw-Whisper suffix timing instead of letting a weak unmatched prefix consume most of the line
  - kept focused win: the 4-song canary (`Without Me|Houdini|Con Calma|I Gotta Feeling`) improved from `0.2395s` in `20260321T011749Z` to `0.2369s` weighted start mean in `20260325T200610Z`
  - positive learning: the dominant `Without Me` miss was not the mapped-line pull helper anymore; it was the forced-alignment repair layer
  - negative learning: the first version of the suffix repair over-tightened the line tail and worsened end timing; preserving the later forced tail recovered most of that regression
  - kept broad win after tightening the gate away from balanced sustained 3-word lines: the 12-song cached canary improved from `0.2205s` in `20260325T195522Z` to `0.2193s` in `20260325T201408Z`
- `Houdini` improved from `0.2463s / 0.2562s` to `0.2209s / 0.2394s` in `benchmarks/results/20260325T205644Z`
  - driver: `shift_restored_low_support_runs_to_onset()` was stopping at the first half of the song and skipping late compact repeated tails entirely; allowing late repeated-tail runs through that gate moved the final `Guess who's back` run later together
  - kept focused win: the 4-song canary improved from `0.2369s` in `20260325T200610Z` to `0.2320s` in `20260325T205102Z`
  - kept broad win: the 12-song cached canary improved from `0.2193s` in `20260325T201408Z` to `0.2170s` in `20260325T205644Z`
  - positive learning: the real `Houdini` blocker was an early scan cutoff in the onset-shift repair stage, not the repeated-line postpass itself
  - negative learning: two intermediate branches were false leads
- `Taste` improved from `0.2240s / 0.2626s` to `0.2080s / 0.2730s` in `benchmarks/results/20260325T230140Z`
  - driver: after accepted WhisperX forced alignment, a new medium-line exact-prefix repair now reanchors starts earlier when the first 3 tokens have an earlier exact Whisper match that immediately follows the previous line boundary
  - kept Sabrina companion win: `Taste|Please Please Please|Espresso` improved from `0.108s` in `20260325T225720Z` to `0.101s` weighted start mean in `20260325T230140Z`
  - kept mixed guardrail win: `Taste|Please Please Please|Espresso|Royals|Without Me` improved to `0.135s` in `20260325T231425Z`, with `Royals` back at `0.1882 / 0.1555`
  - kept broad win: the 12-song cached canary improved from `0.2100s` in `20260325T210830Z` to `0.2050s` in `20260325T231558Z`
  - positive learning: the Sabrina companion clips were the right guardrails for this family; they isolated the accepted-forced-alignment late-start miss without dragging `Sweet Caroline` into the probe
  - negative learning: two earlier gates were too broad
  - an uncapped exact-prefix snap overshot the intended start and regressed `Royals` to `0.4357 / 0.3362` in `20260325T230407Z`
  - boundary carry-over by itself was still too broad and moved `Royals` line 3 to the earlier Whisper onset in `20260325T231141Z`
  - the kept gate also requires a tight prior-line boundary gap, which removed the `Royals` leak while preserving the `Taste` win
    baseline-preservation around `_constrain_line_starts_to_baseline()` worsened the 4-song canary to `0.252s` in `20260325T203921Z`
    family-aware split-short-refrain restore had zero live effect on `Houdini`, even though the isolated helper test passed
- `Con Calma` improved from `0.2430s / 0.2662s` to `0.2148s / 0.2399s` in `benchmarks/results/20260325T210830Z`
  - driver: mapped lines with a single light leading token can now reanchor to a later locally supported content word, which moved `La noche...` off the carryover pickup without disturbing the rest of the broad cached pack
  - kept focused win: the 4-song canary improved from `0.2320s` in `20260325T205102Z` to `0.2160s` in `20260325T210658Z`
  - kept broad win: the 12-song cached canary improved from `0.2170s` in `20260325T205644Z` to `0.2100s` in `20260325T210830Z`
  - positive learning: `Con Calma`'s tail issue was a mapped-line weak-opening/content-anchor miss, not a later correction-stage regression
  - negative learning: this helper only moved line 10 live; lines 11-12 still need a different repair family
- `Without Me` improved again from `0.2387s / 0.2764s` to `0.1972s / 0.1969s` in `benchmarks/results/20260325T211941Z`
  - driver: after post-normalization forced-alignment repair, compact 2-word lines now restore from the pre-forced source timing when forced alignment starts them materially earlier and compresses their duration
  - kept focused win: the 4-song canary improved from `0.2160s` in `20260325T210658Z` to `0.2130s` in `20260325T211734Z`
  - kept broad win: the 12-song cached canary improved from `0.2100s` in `20260325T210830Z` to `0.2090s` in `20260325T211941Z`
  - positive learning: the remaining `Back again` miss was still inside the forced-alignment post-normalization layer, not the mapped-line pull path
  - negative learning: the raw Whisper word windows for `Back again` still start earlier than gold, so this family was only safe to fix by restoring from the stronger pre-forced baseline rather than chasing later Whisper suffix support
- `Rap God` exploratory branch did not land live after the `Without Me` fix:
  - failed focused canary: `Rap God|Royals` stayed flat at `0.213s` in `benchmarks/results/20260325T213017Z`
  - negative learning: a new dense-line exact-prefix handoff repair looked correct in a local unit repro but had zero live effect, even after moving it later in forced finalize
  - implication: the current `Rap God` miss is not exposed by the simple line-1/line-2 compressed-handoff toy and needs a different read before another risky code branch
- `Time After Time` manual gold check partially reduced ambiguity:
  - manual gold edit moved line 3 (`If you fall, I will catch you, I'll be waiting`) `0.15s` earlier
  - focused sparse/falsetto pack rerun: `benchmarks/results/20260325T214458Z`
  - result: `Time After Time` moved from `0.2261 / 0.2892` to `0.2289 / 0.2833`
  - positive learning: line-3 timing was not the main driver; end error improved slightly
  - negative learning: start error did not improve, so line 4 (`Time after time`) remains the primary verification question before reopening code work there
- `Taste` exploratory branch also failed to keep:
  - direct trace confirmed the live clip uses accepted WhisperX forced alignment, not the mapped-line postpass path
  - raw Whisper gives exact earlier 3-token prefixes for lines 2-4 (`You'll just have`, `If you want`, `Just know you'll`)
  - failed focused canary: `Taste|Sweet Caroline|Con Calma` regressed to `0.236s` in `benchmarks/results/20260325T215438Z`
  - negative learning: an exact-prefix forced reanchor that looked defensible on `Taste` overfired on `Sweet Caroline`, so this repair family is too broad in its current form
  - implication: `Taste` line 3 still looks like a real code-side miss, but the next probe must be narrower than generic forced exact-prefix reanchoring
- New curation/workflow fixes landed around fresh Sabrina companion clips:
  - `yt-dlp` is now installed globally via Homebrew, so URL-based clip vetting no longer depends on the repo venv
  - `tools/curated_clip_helper.py` now bootstraps missing curated gold files from the default rebaseline root into the canonical curated clip root
  - `tools/run_benchmark_suite.py` now automatically switches to the curated clip gold root when `benchmarks/curated_clip_songs.yaml` is used, so new clip reruns score against the curated files instead of the default full-song gold root
  - new curated companions added:
    - `32_sabrina-carpenter-please-please-please-chorus-setup-tail.gold.json`
    - `33_sabrina-carpenter-espresso-opening-hook.gold.json`
  - locked-lyrics rule for these clips:
    - changing audio boundaries must not change lyric text unless explicitly requested
  - corrected apples-to-apples rerun on the curated gold root: `benchmarks/results/20260325T225138Z`
  - result: `gold_start_abs_mean_weighted=0.000s` for the two-clip pack after aligning the saved gold timings to the actual locked lyric windows

## Current Cached Canary Baseline

- Latest broad cached canary subset after the compact 2-word forced-alignment restore:
  - `benchmarks/results/20260325T211941Z/benchmark_report.json`
  - curated canary weighted start mean: `0.2090s`
  - key ordering:
    - `Rap God`: `0.2329 / 0.2128`
    - `Time After Time`: `0.2261 / 0.2892`
    - `Taste`: `0.2240 / 0.2626`
    - `Houdini`: `0.2209 / 0.2394`
    - `Sweet Caroline`: `0.2154 / 0.2513`
    - `Con Calma`: `0.2148 / 0.2399`
    - `I Gotta Feeling`: `0.2034 / 0.1510`
    - `Without Me`: `0.1972 / 0.1969`
    - `Total Eclipse of the Heart`: `0.1961 / 0.2979`
    - `Royals`: `0.1882 / 0.1555`
    - `Take On Me`: `0.1513 / 0.3345`
    - `Stayin' Alive`: `0.1374 / 0.6253`
  - only material movement from `20260325T210830Z` was the `Without Me` improvement
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
- Latest broad cached canary rerank on the current code:
  - `benchmarks/results/20260325T195522Z/benchmark_report.json`
  - curated canary weighted start mean: `0.2205s`
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
  - only material movement from `20260321T004829Z` was baseline stability; `Without Me` remained the top broad-return start target before the new forced-alignment repair landed
- Latest focused 4-song canary after the compact forced-alignment suffix repair:
  - `benchmarks/results/20260325T200610Z/benchmark_report.json`
  - curated canary weighted start mean: `0.2369s`
  - key results:
    - `Without Me`: `0.2387 / 0.2764`
    - `Houdini`: `0.2463 / 0.2562`
    - `Con Calma`: `0.2430 / 0.2662`
    - `I Gotta Feeling`: `0.2034 / 0.1510`
  - kept focused win: `Without Me` improved while the 3 controls stayed flat
- Latest broad cached canary rerank after tightening the suffix-repair gate:
  - `benchmarks/results/20260325T201408Z/benchmark_report.json`
  - curated canary weighted start mean: `0.2193s`
  - updated ordering:
    - `Houdini`: `0.2463 / 0.2562`
    - `Con Calma`: `0.2430 / 0.2662`
    - `Without Me`: `0.2387 / 0.2764`
    - `Rap God`: `0.2329 / 0.2128`
    - `Time After Time`: `0.2261 / 0.2892`
    - `Taste`: `0.2240 / 0.2626`
    - `Sweet Caroline`: `0.2154 / 0.2513`
    - `I Gotta Feeling`: `0.2034 / 0.1510`
    - `Total Eclipse of the Heart`: `0.1961 / 0.2979`
    - `Royals`: `0.1882 / 0.1555`
    - `Take On Me`: `0.1513 / 0.3345`
    - `Stayin' Alive`: `0.1374 / 0.6253`
  - only material movement from `20260325T195522Z` was the `Without Me` improvement
  - broad target after that fix was `Houdini`, with `Without Me` close behind
- Latest focused 4-song canary after the late-tail onset-run fix:
  - `benchmarks/results/20260325T205102Z/benchmark_report.json`
  - curated canary weighted start mean: `0.2320s`
  - key results:
    - `Houdini`: `0.2209 / 0.2394`
    - `Con Calma`: `0.2430 / 0.2662`
    - `Without Me`: `0.2387 / 0.2764`
    - `I Gotta Feeling`: `0.2034 / 0.1510`
  - only material movement from `20260325T200610Z` was the `Houdini` improvement
- Latest broad cached canary rerank after the late-tail onset-run fix:
  - `benchmarks/results/20260325T205644Z/benchmark_report.json`
  - curated canary weighted start mean: `0.2170s`
  - updated ordering:
    - `Con Calma`: `0.2430 / 0.2662`
    - `Without Me`: `0.2387 / 0.2764`
    - `Rap God`: `0.2329 / 0.2128`
    - `Time After Time`: `0.2261 / 0.2892`
    - `Taste`: `0.2240 / 0.2626`
    - `Houdini`: `0.2209 / 0.2394`
    - `Sweet Caroline`: `0.2154 / 0.2513`
    - `I Gotta Feeling`: `0.2034 / 0.1510`
    - `Total Eclipse of the Heart`: `0.1961 / 0.2979`
    - `Royals`: `0.1882 / 0.1555`
    - `Take On Me`: `0.1513 / 0.3345`
    - `Stayin' Alive`: `0.1374 / 0.6253`
  - only material movement from `20260325T201408Z` was the `Houdini` improvement
- Latest focused 4-song canary after the light-leading content-word reanchor:
  - `benchmarks/results/20260325T210658Z/benchmark_report.json`
  - curated canary weighted start mean: `0.2160s`
  - key results:
    - `Without Me`: `0.2387 / 0.2764`
    - `Houdini`: `0.2209 / 0.2394`
    - `Con Calma`: `0.2148 / 0.2399`
    - `I Gotta Feeling`: `0.2034 / 0.1510`
  - only material movement from `20260325T205102Z` was the `Con Calma` improvement
- Latest broad cached canary rerank after the light-leading content-word reanchor:
  - `benchmarks/results/20260325T210830Z/benchmark_report.json`
  - curated canary weighted start mean: `0.2100s`
  - updated ordering:
    - `Without Me`: `0.2387 / 0.2764`
    - `Rap God`: `0.2329 / 0.2128`
    - `Time After Time`: `0.2261 / 0.2892`
    - `Taste`: `0.2240 / 0.2626`
    - `Houdini`: `0.2209 / 0.2394`
    - `Sweet Caroline`: `0.2154 / 0.2513`
    - `Con Calma`: `0.2148 / 0.2399`
    - `I Gotta Feeling`: `0.2034 / 0.1510`
    - `Total Eclipse of the Heart`: `0.1961 / 0.2979`
    - `Royals`: `0.1882 / 0.1555`
    - `Take On Me`: `0.1513 / 0.3345`
    - `Stayin' Alive`: `0.1374 / 0.6253`
  - only material movement from `20260325T205644Z` was the `Con Calma` improvement

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

- `Houdini` is no longer the broad canary leader after the late-tail onset-run fix.
- `Con Calma` is materially healthier after the mixed-density coda rebalance and is no longer the obvious top target on seed layout alone.
- `Taste` is materially healthier after the recent gold refresh and is no longer the top broad-return target.
- `Sweet Caroline` is materially healthier after the latest short-title chorus rebalance and is no longer the top broad-return target.
- `Con Calma` is materially healthier again after the late-start Whisper reanchor and is no longer the top broad-return target after the full rerank.
- `Without Me` was the clearest broad-return start-error target before the latest forced-alignment repair stack.
- `Without Me` is materially healthier again after the compact 2-word source restore and is no longer the broad canary leader.
- `Without Me` line 2 now has a direct mapped-line postpass repro:
  - `_pull_late_lines_to_matching_segments()` can pull a `Guess who's back` / `Back again` toy pair from `4.692s` to about `3.63s`
  - a characterization test for that shape now lives in `tests/unit/whisper/test_whisper_mapping_post.py`
  - live-path ordering note: in `whisper_integration_stages.py`, `_pull_late_lines_to_matching_segments()` runs twice in the mapped-line postpasses (`postpass_pull_late_segments`, then `postpass_pull_late_segments_second`) before the onset snap stages
- Important latest finding:
  - the live `Without Me` path is usually using accepted WhisperX forced alignment, so mapped-line toy wins do not directly explain the remaining production miss
  - the saved forced-alignment debug shape was: line 2 too early (`4.404s` vs gold `4.95s`) and line 4 much too early (`8.122s` vs gold `9.05s`)
  - the new repair fixed the line-4 family but did not move line 2, which is useful evidence that those two misses come from different causes
- Broad canary status after the latest fixes:
  - the tightened forced-alignment suffix helper no longer regresses `Time After Time`
  - the late-tail onset-run fix improved `Houdini` without moving the other 11-song cached controls
  - the compact 2-word source restore improved `Without Me` without moving the other 11-song cached controls
- `Con Calma` is materially healthier after the light-leading content-word reanchor and is no longer the broad canary leader
- `Rap God` is back in the top broad-return slot, with `Time After Time`, `Taste`, and `Houdini` next
- `Rap God` is still numerically high, but the first new live probe there was a false lead and did not move the 2-song canary at all.
- `Time After Time` is now the strongest verification candidate:
  - line 4 (`Time after time`) is about `0.98s` early against gold in `20260325T211941Z`
  - line 3 ends about `0.98s` early too, but both the pre-Whisper seed and the raw Whisper support are already much earlier than the current gold
  - that makes gold drift a live possibility, not just a pipeline bug
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

## 2026-03-26 Time After Time paired-refrain keep

- kept change:
  - moved `_enforce_repeated_short_refrain_followup_gap()` earlier on the accepted forced-alignment path in [whisper_integration_forced_fallback.py](/Users/dtunkelang/y2karaoke/src/y2karaoke/core/components/whisper/whisper_integration_forced_fallback.py), so repeated short refrain lines shift later before generic non-overlap compresses the preceding line tails
  - retained the word-sequence and split-refrain restores as post-finalize repairs
- why it worked:
  - on `Time After Time`, raw forced alignment already had the right coupled shape for lines 3-4
  - the old order let generic non-overlap compress line 3 early because line 4 still started too early
  - shifting the refrain follow-up gap before finalization preserves the later line-3 tail and yields the right later line-4 start without a special-case tail restore
- positive results:
  - focused sparse/falsetto pack improved in `benchmarks/results/20260326T150957Z`
  - `Time After Time`: `0.2289 / 0.2833 -> 0.1706 / 0.1457`
  - broad cached canary improved in `benchmarks/results/20260326T151137Z`
  - broad aggregate `avg_abs_word_start_delta_sec_word_weighted_mean`: `0.2007 -> 0.1924`
  - broad aggregate `gold_end_mean_abs_sec_mean`: `0.2693 -> 0.2565`
  - broad aggregate `gold_start_p95_abs_sec_mean`: `0.5086 -> 0.4481`
  - `Rap God` also improved on the same broad rerun: `0.2329 / 0.2128 -> 0.1994 / 0.1894`
- negative learnings:
  - the earlier “restore line-3 suffix after refrain gap shift” branch was a false lead
  - it moved `Time After Time` line 3 later exactly as expected, but left line 4 behind and regressed the focused pack in `benchmarks/results/20260326T050355Z`
  - the right fix axis was stage ordering on accepted forced alignment, not another local suffix helper
- accepted tradeoff:
  - `Total Eclipse of the Heart` regressed slightly on the broad rerun: `0.1961 / 0.2979 -> 0.2016 / 0.3054`
  - that regression is small relative to the `Time After Time` and `Rap God` gains, and the broad aggregate still improved materially
- next broad-return target:
  - `Con Calma`, then `Taste`
  - `Rap God` is improved enough that it is no longer the cleanest next target

## 2026-03-26 Con Calma corrected-tail keep

- corrected gold first:
  - [27_daddy-yankee-snow-con-calma-first-chorus-bilingual.gold.json](/Users/dtunkelang/y2karaoke/benchmarks/clip_gold_candidate/20260312T_curated_clips/27_daddy-yankee-snow-con-calma-first-chorus-bilingual.gold.json) line 12 now matches the actual clip cutoff as `De guayarte, ma...` ending at `30.05`
  - this exposed the real tail miss instead of the old flattering overlong gold
- what the corrected gold revealed:
  - with the accurate `Con Calma` tail, the current pipeline was much worse than it looked
  - focused rerun before new code: `0.4394 / 0.4301` in `benchmarks/results/20260326T165226Z`
  - live tail shape after the gold fix:
    - line 10 ended late at `27.245` vs gold `26.85`
    - line 11 started late at `27.379` vs gold `26.85`
    - line 12 started late at `28.889` vs gold `28.40`
  - trace note: the mapped tail first jumped to absurd late interpolation (`37-41s`) and then `restore_weak_evidence_large_start_shifts()` snapped it back to the baseline-ish late shape; nothing later corrected the line-10/line-11 carryover
- kept code change:
  - added `rebalance_short_followup_boundaries_from_whisper()` in [whisper_integration_align_experimental.py](/Users/dtunkelang/y2karaoke/src/y2karaoke/core/components/whisper/whisper_integration_align_experimental.py)
  - wired it into [whisper_integration_align_corrections.py](/Users/dtunkelang/y2karaoke/src/y2karaoke/core/components/whisper/whisper_integration_align_corrections.py) right after the light-leading reanchor pass
  - the helper trims a long previous line back to its last locally supported Whisper end and pulls a short light-leading followup earlier from stemmed Whisper prefix support
  - the repeated-fragment followup (`dan-dan-dan` vs Whisper `dam dam dam`) needed an extra short-fragment token matcher; the first version only improved line 11 start and was too weak on the tail
- kept results:
  - focused `Con Calma|Taste` rerun improved in `benchmarks/results/20260326T171739Z`
  - `Con Calma`: `0.4394 / 0.4301 -> 0.4019 / 0.3951`
  - `Taste` stayed flat at `0.2080 / 0.2730`
  - broad cached canary with corrected gold finished in `benchmarks/results/20260326T171826Z`
  - broad controls stayed flat:
    - `Houdini` `0.1683 / 0.2164`
    - `Sweet Caroline` `0.2154 / 0.2513`
    - `Take On Me` `0.1513 / 0.3345`
    - `Without Me` `0.1972 / 0.1969`
    - `I Gotta Feeling` `0.1700 / 0.1446`
    - `Time After Time` `0.1706 / 0.1457`
    - `Total Eclipse of the Heart` `0.2016 / 0.3054`
    - `Stayin' Alive` `0.1374 / 0.6253`
    - `Rap God` `0.1994 / 0.1894`
    - `Royals` `0.1882 / 0.1555`
- still unresolved on `Con Calma`:
  - line 12 is still late at about `+0.49s` on start
  - local Whisper still hears the tail more like `degollarte mami` than the saved text, so the helper only safely fixed the line-10/line-11 carryover and left the line-12 lexical mismatch alone
- next best target:
  - `Con Calma` line 12 if a clean lexical variant strategy can be proven
  - otherwise pivot back to the next broad-return clip with real support, likely `Taste` or `Rap God`

## 2026-03-26 Con Calma truncated-tail phonetic keep

- kept follow-on change:
  - added `reanchor_truncated_followup_lines_from_phonetic_variants()` in [whisper_integration_align_experimental.py](/Users/dtunkelang/y2karaoke/src/y2karaoke/core/components/whisper/whisper_integration_align_experimental.py)
  - wired it into [whisper_integration_align_corrections.py](/Users/dtunkelang/y2karaoke/src/y2karaoke/core/components/whisper/whisper_integration_align_corrections.py) immediately after the short-followup boundary rebalance
  - this helper is intentionally narrow:
    - line has at most 3 words
    - exactly one light leading token
    - final token is truncated with `...`
    - content token must have phonetic similarity to an earlier Whisper word
    - truncated tail token must also phonetically match the following Whisper word
- why it worked:
  - on corrected `Con Calma`, line 12 is `De guayarte, ma...`
  - Whisper still hears `degollarte mami`
  - phonetic similarity is strong enough to justify a narrow reanchor:
    - `guayarte` vs `degollarte` about `0.56`
    - `ma` vs `mami` about `0.67`
  - the helper pulls line 12 from the baseline-ish late start `28.889` back to the supported variant start `28.64`
- kept results:
  - focused `Con Calma|Taste` rerun improved again in `benchmarks/results/20260326T174540Z`
  - `Con Calma`: `0.4019 / 0.3951 -> 0.3978 / 0.3940`
  - line 12 start improved from `+0.49s` late to `+0.24s`
  - `Taste` stayed flat at `0.2080 / 0.2730`
  - broad cached canary improved again in `benchmarks/results/20260326T174631Z`
  - broad aggregate `avg_abs_word_start_delta_sec_word_weighted_mean`: `0.2362 -> 0.2352`
  - broad aggregate `gold_end_mean_abs_sec_mean`: `0.2694 -> 0.2693`
  - all 11 controls stayed flat
- negative learning:
  - a generic lexical-overlap helper would not have reached this case safely; the useful signal here was the combination of truncation plus phonetic variant support, not ordinary token overlap
- next best target:
  - `Rap God` or `Taste`
  - `Con Calma` still has residual miss on line 12 end, but the remaining gain looks smaller and riskier now
