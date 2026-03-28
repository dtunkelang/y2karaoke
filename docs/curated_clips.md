# Curated Clip Workflow

Last updated: 2026-03-27

Use curated clips to isolate specific timing or alignment behaviors without paying whole-song iteration costs on every pass.

## When To Add A Clip

Add a clip when a full-song metric is hiding a distinct failure mode or when you need a stable short control.

High-value clip categories:
- repeated-hook comparability
- duet or backing-vocal overlap
- weak-onset or clipped-onset starts
- tail and sparse-support endings
- source-text or source-timing mismatch stress cases
- clean control clips for sanity checks

## Manifest Rules

Curated clips live in `benchmarks/curated_clip_songs.yaml`.

Each clip entry should include:
- `clip_id`
- `clip_tags`
- `audio_start_sec`
- `clip_duration_sec`
- `notes`

Recommended tag vocabulary:
- `control`
- `stress`
- `chorus`
- `verse`
- `verse-hook`
- `repeated-hook`
- `duet`
- `tail`
- `source-text`
- `source-timing`
- `comparability`
- `clean`

Use a small set of stable tags rather than inventing one-off labels.

## Recommended Loop

1. Vet the source URL if it is new.
2. Add or update the clip entry in `benchmarks/curated_clip_songs.yaml`.
3. Validate the manifest:
   `./venv/bin/python tools/validate_benchmark_manifest.py benchmarks/curated_clip_songs.yaml`
4. Open the saved gold file, not a timing-report seed:
   `make curated-open MATCH="Song Or Clip Id"`
   Direct helper form:
   `PYTHONPATH=src ./.venv/bin/python tools/curated_clip_helper.py --match "Song Or Clip Id" --open-editor`
   The helper now starts the local gold editor server automatically when needed before opening the browser.
5. If the song is cold, use a quick first probe:
   `PYTHONPATH=src ./.venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --clip-tag stress --fast-clip-probe --max-songs 1`
6. For apples-to-apples measurement, rerun on the normal path and let the runner reuse full-song results where possible:
   `PYTHONPATH=src ./.venv/bin/python tools/run_benchmark_suite.py --manifest benchmarks/curated_clip_songs.yaml --clip-tag duet --offline`

## Selection

Use regex matching for specific songs or clip ids:
- `--match "Blinding Lights|hook-repeat"`

Use tag selection for clip packs:
- `--clip-tag control`
- `--clip-tag duet --clip-tag tail`

Tag filters are additive at the CLI level: a song is selected if it matches any requested tag.

## Cost Model

- Prefer normal offline reruns once a song has cached audio and stems.
- The runner can now score many clip entries directly from a compatible cached full-song result, which avoids rerunning generation for those clips.
- `--fast-clip-probe` is still useful for a cold first pass, but it is a triage path, not the final benchmark path.
- Broad offline tag runs are still noisy if the selected pack includes uncached clips; use known cached match-based subsets when you need a clean quality signal.

## Curation Discipline

- Always open the editor from the saved gold JSON and canonical clip audio path.
- Prefer the stable shortcut:
  `make curated-open MATCH="Song Or Clip Id"`
- Use `tools/curated_clip_helper.py` instead of hand-building editor URLs or filenames.
- After any manual curation change:
  - verify the saved gold JSON on disk
  - verify the actual clip audio duration/path on disk
  - reopen the saved gold file once to confirm the editor is loading the right artifact
  - commit and push immediately before any further code or benchmark work
- If source and lyrics disagree structurally, fix the source before curating around it.
- Avoid changing the clip window and lyric span in the same pass unless you are deliberately reseeding the clip.
- If a hard clip remains the only outlier after other clips improve, add one or two companion clips in the same failure family before tuning further.

## Current Learnings

- Short curated clips are strong quality drivers when the pipeline stays clip-scoped:
  - bounded clip audio
  - clip-scoped lyric text
  - clip-scoped scoring against gold
- Repeated-hook clips should be optimized as families, not single songs. The current useful family is:
  - `Houdini`
  - `Without Me`
  - `I Gotta Feeling`
- For longer repeated-hook clips where the dominant repeated block starts after one or two unique setup lines, the prefix gap before the repeated block needs to be wider than the simpler compact-hook layout gives it. That improved `Houdini` without disturbing `Without Me` or `I Gotta Feeling`.
- Dense non-repeated short rap verses like `Rap God` need a different seed from both the repeated-hook path and the generic dense spread:
  - keep the opening anchor early
  - weight the first two dense lines heavily
  - preserve enough tail span for the final long line instead of letting the generic spread collapse it
- Plain-text clip lyrics need an audio-window-aware timing seed. Starting every plain-text clip at `0.0s` hides useful structure and biases repeated hooks toward early collapse.
- Short-title chorus clips like `Sweet Caroline` need a different seed from the generic compact spread:
  - give the title line less span
  - widen the setup gap into line 2
  - leave more room for the tail line
- Mixed-density chorus clips like `Con Calma` still need a bit more span on the repeated long lines than the generic chorus weighting gives them. A small increase there, plus a slightly looser long-line to short-response gap, improved the clip without hurting the focused lyrics tests.
- `Con Calma` then exposed a second failure mode after the seed improved:
  - some late lines had real earlier Whisper support, but rollback/correction behavior still left them pinned to later baseline starts
  - a narrow earlier-Whisper reanchor for lines with in-order prefix support improved `Con Calma` again without regressing the `Houdini|Without Me|I Gotta Feeling` canary
  - once a clip reaches that stage, inspect post-map and correction-pass traces before tuning seed helpers again
- `Con Calma` later exposed a third failure mode after the obvious start fixes landed:
  - gold start/end metrics became decent while DTW / agreement coverage stayed poor
  - the practical signal was `dtw_line_coverage = 0.667` with `gold_start_mean_abs_sec = 0.2242`
  - implication: stop assuming the next gain is another boundary retime; the remaining issue is likely in mapping / coverage policy
- A deeper agreement diagnostic sharpened that read:
  - the remaining `Con Calma` agreement failures split into `anchor_outside_window` and `low_text_similarity`
  - but several `low_text_similarity` cases are not true text mismatches
  - they are exact local lyric phrases being compared against over-merged anchor text that spans multiple lines
  - implication: before relaxing similarity thresholds on bilingual / chorus-merging clips, inspect anchor granularity and anchor-selection policy
- A follow-up clipping simulation made that more concrete:
  - on the kept `Con Calma` baseline, clipping merged anchor text down to the best contiguous phrase would recover most of the `low_text_similarity` lines
  - implication: anchor-text clipping is now a more credible next strategy than another boundary retime or broad threshold relaxation
  - the stronger quantified read is:
    - baseline agreement coverage `0/5`
    - clipped-anchor simulated coverage `4/5`
  - that is strong enough to justify an explicit clipped-anchor comparability experiment before more song-specific retiming work
- A pack-level simulation showed the same idea is not automatically isolated:
  - on a mixed kept pack, clipped-anchor simulated agreement improves `Con Calma` strongly, but also recovers `Take On Me` and one `Without Me` line
  - implication: if this becomes a real comparability policy, it will need guards rather than a blanket enable
- The first simple guard that looked clean was:
  - `line_words >= 6`
  - `anchor_words - line_words >= 15`
  - on the mixed kept pack, that still recovers `Con Calma` while dropping the `Take On Me` / `Without Me` spillover
- On `Con Calma` itself, that guarded policy cleanly separates the remaining issues:
  - clipped-anchor recovery handles most of the `low_text_similarity` cases
  - the rest are still `anchor_outside_window`
  - implication: if you adopt clipped-anchor comparability, treat it as one layer, not the whole fix
- A follow-up window-phrase diagnostic tightened the second layer too:
  - most of the remaining `anchor_outside_window` lines in `Con Calma` already have strong local phrases inside the Whisper window
  - implication: the remaining blocker is largely anchor-start selection, not lack of local lexical evidence, except for the final `De guayarte, ma...` line
- The combined two-layer simulation is the current best strategic read:
  - `Con Calma` moves from `0/5` comparable lines to `10/11` under:
    - guarded clipped-anchor text for merged-anchor mismatches
    - local window phrase anchoring for recoverable outside-window lines
  - implication: the next benchmark-only policy experiment is justified; further local timing heuristics are not the best use of time here
- On a mixed kept pack, the same two-layer simulation is still mostly driven by `Con Calma`, but not perfectly isolated:
  - `Con Calma` improves heavily
  - `Take On Me` still picks up one recovery
  - implication: benchmark-only integration is reasonable next, but keep treating guard design as part of the experiment
- A stricter guard improved isolation:
  - `line_words >= 6`
  - `anchor_words - line_words >= 15`
  - `anchor_words >= 20`
  - on the mixed pack, that keeps a strong `Con Calma` gain while dropping the `Take On Me` spillover
  - implication: the first benchmark-only policy prototype should probably start from this stricter version
- A small guard-tradeoff sweep confirmed that choice:
  - the current best benchmark-only candidate is still `line_words >= 6`, `anchor_surplus >= 15`, `anchor_words >= 20`
  - implication: use that exact setting for the first real comparability-policy prototype, not a looser version
- The first additive benchmark-side prototype now exists in `tools/analyze_two_layer_benchmark_prototype.py`:
  - it is now directly runnable from repo root without extra `PYTHONPATH` setup
  - on mixed kept pack `benchmarks/results/20260327T180340Z`
    - baseline `agreement_coverage_ratio_total = 0.3514`
    - prototype `agreement_coverage_ratio_total = 0.6562`
    - baseline `agreement_bad_ratio_total = 0.1081`
    - prototype `agreement_bad_ratio_total = 0.1081`
  - the whole simulated gain is currently isolated to `Con Calma`:
    - `0/5 -> 8/9`
  - hotspot ordering changes in the expected direction:
    - baseline worst tier: `Take On Me`, `Con Calma`
    - prototype worst hotspot: `Take On Me`
    - prototype `Con Calma` moves to `8/9`, ahead of only `Total Eclipse`
  - prototype assumption is explicit:
    - recovered lines count as good matches
    - bad/warn/severe counts stay fixed
  - implication: the next step can stay benchmark-side, but it is now grounded in runner-compatible aggregate math rather than raw line recovery counts
- Direct integration into `tools/run_benchmark_suite.py` was intentionally not kept:
  - the file still carries large pre-existing `flake8` complexity debt
  - modifying it causes pre-commit to re-evaluate those old `C901` failures
  - rather than weaken that gate, the kept integration path is a wrapper
- Benchmark-side wrapper now exists:
  - `tools/run_benchmark_suite_with_two_layer_prototype.py`
  - it runs the normal benchmark suite, then writes sidecars into the benchmark run dir:
    - `two_layer_benchmark_prototype.json`
    - `two_layer_benchmark_prototype.md`
  - it also supports resumed runs by honoring `--resume-run-dir`
  - validated on cached mixed pack `benchmarks/results/20260327T180340Z`:
    - coverage `0.3514 -> 0.6562`
    - bad ratio `0.1081 -> 0.1081`
    - worst hotspot `a-ha - Take On Me`
  - broader cached canary `benchmarks/results/20260327T165543Z` strengthens the same read:
    - coverage `0.4211 -> 0.7292`
    - bad ratio `0.0702 -> 0.0702`
    - recovered songs:
      - `Con Calma` `0/5 -> 8/9`
      - `I Gotta Feeling` `0/3 -> 3/4`
    - worst hotspot still `a-ha - Take On Me`
  - implication:
    - the benchmark-only comparability idea is broader than just `Con Calma`
    - but it still does not solve the contamination / zero-support family, which keeps `Take On Me` as the clearest remaining hotspot
- A new contamination-family pack analyzer now exists:
  - `tools/analyze_contaminated_gap_pack.py`
  - focused cached subset on `benchmarks/results/20260327T165543Z` with `Take On Me|Stayin' Alive`:
    - `gaps_total = 4`
    - `prev_line_truncated_total = 3`
    - `next_line_delayed_total = 0`
    - `Take On Me`: `2/3` gaps are `prev_line_truncated`
    - `Stayin' Alive`: `1/1` gap is `prev_line_truncated`
  - implication:
    - the contamination family is mostly previous-line end loss, not next-line delay
    - `Take On Me` is the stronger contamination case and remains the best target in that family
- A diagnostics-only sparse-forced-fallback switch now exists:
  - `Y2K_WHISPER_ENABLE_SPARSE_FORCED_FALLBACK=0`
  - this is not intended as a production default change
  - it exists so sparse clips can be inspected on the raw mapped DTW path when forced fallback would otherwise hide the stage behavior
- Using that switch on `Take On Me` produced the first usable mapped-stage trace:
  - run dir: `benchmarks/results/20260327T233011Z`
  - trace: `/tmp/take_on_me_mapped_trace.json`
  - outcome is much worse than the kept forced path:
    - gold start/end means move from about `0.126 / 0.278` to `0.870 / 1.028`
  - so the sparse forced fallback is still justified as the shipped default
  - but the mapped trace changes the diagnosis in an important way:
    - line 2 `Take me on` is badly damaged by `postpass_extend_trailing`
    - it jumps from `4.22-8.22` to `9.86-12.34` before later passes partially recover it
  - implication:
    - `Take On Me` is not only about previous-line truncation
    - the mapped contamination family also includes repeated-phrase overextension on the next line
    - the next code-side experiment should target contamination-aware guards on trailing-phrase extension rather than another broad fallback toggle
- A second diagnostics-only split now exists inside continuous-vocals refinement:
  - `Y2K_WHISPER_CONTINUOUS_SHIFT_LONG_GAPS=0`
  - `Y2K_WHISPER_CONTINUOUS_EXTEND_ACTIVE_GAPS=0`
  - these are for isolating subpasses, not for a production default change
- Isolating only long-gap shifting on `Take On Me` did not solve the mapped failure:
  - run dir: `benchmarks/results/20260327T233810Z`
  - result remains very poor: about `0.921 / 1.113`
  - trace implication:
    - line 2 no longer gets pulled early by long-gap shifting
    - but the combination of repeated trailing extension, later restore logic, and active-gap extension still leaves the mapped layout badly wrong
  - implication:
    - the mapped `Take On Me` family is not attributable to a single continuous-vocals subpass
    - it is still a multi-stage interaction problem, so the next probe should isolate active-gap extension rather than shipping a guard based on this run
- That next isolation is now done:
  - `Y2K_WHISPER_CONTINUOUS_EXTEND_ACTIVE_GAPS=0`
  - run dir: `benchmarks/results/20260327T234002Z`
  - result is unchanged from the original forced-off mapped baseline
  - implication: active-gap extension is not the main mover on `Take On Me`
- Baseline constraint was also isolated directly:
  - `Y2K_WHISPER_DISABLE_BASELINE_CONSTRAINT=1`
  - run dir: `benchmarks/results/20260327T234136Z`
  - result gets worse, not better: about `1.056 / 1.302`
  - trace implication:
    - line 2 remains at the too-early continuous-vocals position `5.387-7.867`
    - baseline constraint is currently a net repair
  - implication:
    - the main mapped-path error is earlier than baseline restoration
    - `postpass_extend_trailing` remains the first stage that makes the layout clearly wrong
- Weak-evidence restore diagnostics now distinguish `suffix_only_support` from true missing lexical support:
  - on the reverted short-repeat diagnostic trace `/tmp/take_on_me_short_repeat_skip.json`
  - line 2 `Take me on` is classified as `suffix_only_support`, not `lexical_support_missing`
  - local support window still contains `me`, `on`, `I'll`
  - implication:
    - the restore logic is currently over-penalizing short repeated hooks whose first token disappears but whose suffix remains locally supported
    - the next `Take On Me` experiment should focus on that weak-evidence-restore shape rather than broad lexical-support threshold changes
- A broad diagnostics-only bypass for that suffix-only weak-support shape was tested and reverted:
  - run dir: `benchmarks/results/20260327T235218Z`
  - outcome got much worse on forced-off mapped `Take On Me`: about `1.803 / 1.974`
  - the failure was narrow and informative:
    - line 2 stayed unchanged
    - line 3 regressed from about `11.91-15.49` to `8.01-11.17`
  - implication:
    - the right future policy, if any, must be narrower than “do not restore suffix-only support”
    - line-2-style repeated hooks and line-3-style sparse tails cannot share the same weak-evidence bypass
- A new analyzer now splits that `suffix_only_support` bucket into narrower families:
  - tool: `tools/analyze_weak_evidence_restore_families.py`
  - on the reverted `Take On Me` short-repeat trace:
    - line 2 `Take me on` -> `repeated_short_hook_suffix_support`
    - line 3 `I'll be gone` -> `sparse_tail_suffix_support`
    - line 4 `In a day or two` -> `sparse_tail_suffix_support`
  - implication:
    - the next viable mapped-path experiment is no longer a generic weak-evidence rule
    - it is a repeated-short-hook-specific exception, likely requiring strong overlap with a neighboring repeated hook
- That narrower repeated-short-hook bypass was also tested and reverted:
  - run dir: `benchmarks/results/20260327T235954Z`
  - outcome was an exact no-op versus the forced-off mapped baseline: about `0.870 / 1.028`
  - all final line timings matched `20260327T233011Z`
  - implication:
    - even the repeated-short-hook weak-evidence restore is not the active live lever
    - the remaining `Take On Me` mapped-path error is still being set earlier or elsewhere than this restore stage
- A new trailing-extension candidate analyzer now reproduces the real live word source:
  - tool: `tools/analyze_trailing_extension_candidates.py`
  - true word source for this case is the cached vocals transcription:
    - `~/.cache/karaoke/NaQ083rNUwc/trimmed_from_50.00s_for_22.00s_(Vocals)_htdemucs_ft_whisper_large_auto.json`
  - on `Take On Me` line 2 `Take me on`, the later candidate wins because soft token matching allows:
    - `take -> take`
    - `me -> me`
    - `on -> gone`
  - so `postpass_extend_trailing` is not just choosing the later phrase; it is being misled by a short-token substring false positive
  - a direct production patch to forbid short substring matches in that helper was tested and reverted:
    - run dir: `benchmarks/results/20260328T000700Z`
    - result got slightly worse: about `0.961 / 1.195`
    - line 2 end over-extended to about `11.9`
  - implication:
    - the root cause inside trailing extension is now known
    - but the correct fix still needs an additional end-selection guard, not just stricter token matching
- A follow-up tail-extension scoring simulator now exists:
  - tool: `tools/simulate_trailing_extension_candidate_scoring.py`
  - it rewards exact matches and penalizes:
    - short soft-only matches
    - large distance from the current line end
    - crossing into the next lyric phrase
  - on the real cached `Take On Me` transcription, this correctly prefers the early `Take me on` phrase over the later `on -> gone` false positive
  - but the first production-style ranking attempt was reverted:
    - run dir: `benchmarks/results/20260328T001400Z`
    - result got slightly worse again: about `0.961 / 1.195`
  - implication:
    - keep this as a diagnostic lens for tail-extension candidate quality
    - but do not treat simple candidate reordering as sufficient; the next safe probe needs a stronger end-selection policy
- A second tail-extension simulator now compares the current chosen end against a guarded policy:
  - tool: `tools/simulate_trailing_extension_end_guards.py`
  - on the real cached `Take On Me` trace:
    - line 2 flips from the bad late phrase to the early local `Take me on` phrase
    - line 3 stays unchanged
    - line 4 has no candidate
  - implication:
    - this looks more isolated than the earlier broad ranking attempt
    - the next production probe should target final end selection inside `postpass_extend_trailing`, not broad token-match strictness
- A diagnostics-only runtime trace now exists for the actual tail-extension helper:
  - env: `Y2K_TRACE_TAIL_EXTENSION_JSON`
  - this captures the real `all_words` candidate set used by `postpass_extend_trailing`
  - on forced-off `Take On Me`, it confirmed the live helper really does see both:
    - the early local phrase ending near `9.7`
    - the late false-positive phrase ending at `15.5`
- A narrow production tie-break based on near-anchor candidates was tested and reverted:
  - run dir: `benchmarks/results/20260328T002856Z`
  - result got worse again: about `0.961 / 1.195`
  - traced reason:
    - `postpass_extend_trailing` stayed early as intended
    - but `postpass_pull_continuous_vocals` then overgrew line 2 and squeezed lines 3-4
  - implication:
    - `Take On Me` is now clearly a multi-stage interaction, not a single tail-extension bug
    - the next safe probe must consider the coupling between tail extension and continuous-vocals expansion
- A follow-up handoff analyzer now makes that coupling more concrete:
  - tool: `tools/analyze_continuous_vocals_handoff.py`
  - on the reverted `Take On Me` probe trace:
    - line 2 grows right by `+2.50s`
    - line 3 grows right by `+1.95s`
    - starts do not move
  - implication:
    - the harmful interaction is right-end growth across active gaps
    - not long-gap start shifting
    - the next code-side probe, if taken, should target `_extend_line_ends_across_active_gaps()` specifically
- A diagnostics-only trace now exists for active-gap right extension:
  - env: `Y2K_TRACE_ACTIVE_GAP_EXTENSION_JSON`
  - on the current bad forced-off `Take On Me` baseline, that trace is empty
  - implication:
    - active-gap extension is not part of the current shipped bad layout
    - it only becomes relevant after the earlier tail-extension choice is improved
    - so the eventual fix is likely a paired cross-stage policy, not an isolated active-gap change
- A narrow compact-handoff guard in `_extend_line_ends_across_active_gaps()` was tested and reverted:
  - forced-off `Take On Me` stayed exactly flat in `benchmarks/results/20260328T005614Z` at about `0.870 / 1.028`
  - the normal mixed control pack in that same run dir also stayed unchanged:
    - `Take On Me` `0.1257 / 0.2781`
    - `Stayin' Alive` `0.1374 / 0.1827`
    - `Total Eclipse of the Heart` `0.1697 / 0.2465`
    - `Royals` `0.2093 / 0.1278`
  - implication:
    - a static active-gap shape guard is not enough
    - if a future fix uses this stage, it will need cross-stage state from the preceding tail-extension choice
- Diagnostics-only traces now exist for:
  - baseline constraint decisions via `Y2K_TRACE_BASELINE_CONSTRAINT_JSON`
  - continuous-vocals subpass snapshots via `Y2K_TRACE_CONTINUOUS_VOCALS_JSON`
- On the clean forced-off `Take On Me` baseline in `benchmarks/results/20260328T012205Z`:
  - `postpass_extend_trailing` first makes line 2 late: `9.86-12.34`
  - the first `postpass_pull_continuous_vocals` call then collapses lines 2-4 during `after_shift_long_activity_gaps` to:
    - line 2 `5.387-7.867`
    - line 3 `8.011-11.171`
    - line 4 `11.308-14.808`
  - `after_extend_active_gaps` does nothing on that baseline
  - the second continuous-vocals call is also a no-op
  - baseline constraint only nudges line 2 later to `6.451-8.931`
- That means the clean-baseline mover is the first `_shift_lines_across_long_activity_gaps()` call, not active-gap extension.
- A narrow compact-handoff guard in `_shift_lines_across_long_activity_gaps()` was then tested and reverted:
  - forced-off `Take On Me` got worse in `benchmarks/results/20260328T012630Z` from about `0.870 / 1.028` to `0.921 / 1.113`
  - the normal mixed control pack in `benchmarks/results/20260328T012657Z` stayed unchanged:
    - `Take On Me` `0.1257 / 0.2781`
    - `Stayin' Alive` `0.1374 / 0.1827`
    - `Total Eclipse of the Heart` `0.1697 / 0.2465`
    - `Royals` `0.2093 / 0.1278`
  - implication:
    - the culprit stage is now known more precisely
    - but a static tight-handoff guard is still the wrong abstraction
    - the next credible fix needs a repeated-hook / sequence-level policy for that first long-gap-shift pass
- A paired sequence-level policy was then tried and kept:
  - after a large repeated-hook shift in `_shift_lines_across_long_activity_gaps()`, stop the nonmatching cascade
  - in the same continuous-vocals call, skip active-gap end growth on that shifted line and its immediate follower
- Results:
  - forced-off `Take On Me` improved in `benchmarks/results/20260328T013645Z` from about `0.870 / 1.028` to `0.780 / 1.001`
  - normal shipped-path control pack in `benchmarks/results/20260328T013718Z` stayed unchanged:
    - `Take On Me` `0.1257 / 0.2781`
    - `Stayin' Alive` `0.1374 / 0.1827`
    - `Total Eclipse of the Heart` `0.1697 / 0.2465`
    - `Royals` `0.2093 / 0.1278`
  - broader cached canary `benchmarks/results/20260328T013956Z` completed `OK`
- Takeaway:
  - the first keepable `Take On Me` mapped-path gain came from a paired sequence policy across continuous-vocals subpasses
  - this is stronger evidence for carrying local stage state than for adding more static shape gates
- Two-line falsetto/refrain clips exposed a different failure mode from longer repeated-hook clips:
  - WhisperX forced alignment previously could not help 2-line clips at all
  - weak onset detection could incorrectly fall back to a generic spread seed
  - subset-refrain clips need their own plain-text seed layout when line 2 is a shorter tail of line 1
- `Take On Me` revealed a different sparse/falsetto issue:
  - the accepted forced alignment can have correct line boundaries but poor within-line word distribution
  - the worst case is a short-function-word lead-in followed by a held final word
  - compare raw forced word timings against gold before changing line-level seeding again
  - a narrow within-line redistribution fix can improve this family without disturbing `Stayin' Alive`, `Time After Time`, or `Total Eclipse`
- `Clocks` exposed a different forced-alignment family from both `Take On Me` and the held-tail clips:
  - raw forced alignment already gets line 3 (`Confusion that never stops`) right
  - the bad output came from restoring short lines toward an overlong source baseline
  - the keepable fix was two short-line guards in forced fallback:
    - skip exact baseline restore for extreme sustained collapse on short lines
    - skip sparse-support duration restore on those same short-line cases
  - kept result:
    - `benchmarks/results/20260328T_clocks_trace3`
    - `Coldplay - Clocks` improved from `3.6572 / 2.7231` to `0.5000 / 0.6065`
    - gold coverage stayed `1.0`
  - guardrails stayed clean:
    - `benchmarks/results/20260328T_clocks_guard_pack2`
    - `Take On Me`, `Total Eclipse`, and both `Royals` clips stayed effectively unchanged
    - `benchmarks/results/20260328T_clocks_stayin_sanity`
    - `Stayin' Alive` stayed flat at `0.1374 / 0.1827`
  - implication:
    - do not treat all repeated-hook companions as one family
    - `Clocks` is now a forced-duration-rollback control, not a mapped contamination control
- `Take On Me` is now constrained more tightly than before:
  - a diagnostic-only onset analyzer now exists:
    - `tools/analyze_long_gap_shift_candidates.py`
  - on the kept mapped trace, the first long-gap shift picks the earliest onset, but the gold-nearest candidate is much later:
    - line 2 `Take me on`: chosen `5.387`, gold-nearest `6.803`
    - line 3 `I'll be gone`: chosen `8.011`, gold-nearest `12.144`
  - the failed later-onset branch in `benchmarks/results/20260328T015630Z` explains why onset-only changes are still wrong:
    - line 2 moved to `6.45-11.43`
    - the start was less wrong, but the end overgrew badly
  - implication:
    - the next `Take On Me` probe has to score onset and target end together
    - do not spend another branch on onset-only selection
- When a live clip still looks wrong after a plausible fix, compare:
  - the helper-generated seed on the real cached clip audio
  - the accepted forced-alignment output
  - the final timing report
  This is faster than guessing which postpass is to blame.
- If a focused canary improves cleanly, commit and push before widening the benchmark set. That keeps the next step recoverable when iteration budget is tight.
- Do not drop difficult clips just because they are difficult. Keep them if they reflect real production failures, but add companion clips when a single clip is too underdetermined to tune against safely.
- For repeated-hook mapped failures like `Take On Me`, inspect the long-gap candidate set as windows, not just starts:
  - multiple later onsets can be safe against the next line and still differ materially in end quality
  - if several candidate windows are safe, the right next rule is sequence-level line-pair selection, not a per-line onset tweak
- Even sequence-level onset pairing may still be insufficient when the carried line duration is too short:
  - the best valid `Take On Me` line-2/line-3 pair under current durations prefers a later line-2 onset than the gold-nearest one
  - that means the next probe likely has to combine paired onset choice with a repeated-hook duration policy
- `Take On Me` also does not fit the existing repetition-run realigner family:
  - `Take on me` / `Take me on` has perfect normalized token overlap
  - but the current helper only triggers on 3+ line runs with an exact duplicate
  - so this family needs either a new alternating-hook repetition path or a later baseline-aware sequence correction
- `Take On Me` also exposes a merged-segment anchoring problem:
  - the real vocals transcription collapses lines 1-3 into one segment starting at `0.0`
  - segment-start anchoring is therefore unusable for line 2
  - but subphrase windows inside that merged segment are still good enough to anchor:
    - `Take me on` local phrase window `4.22-9.7`
    - `I'll be gone` local phrase window `12.34-15.5`
  - so a phrase-window anchor inside merged segments is now a plausible next architecture step
- A direct phrase-window pull is still too blunt:
  - if you simply replace merged-segment starts with local phrase-window starts, `Take On Me` line 1 also moves earlier
  - that then forces line 2 early as well (`4.99-8.79`)
  - so the useful version of this idea has to be selective, likely targeting the alternating-hook line rather than the whole merged segment family
- A baseline-clamped selective version looks materially better:
  - if only the middle alternating hook line is retimed
  - and its start is clamped near baseline while its end is allowed to follow the local phrase window
  - `Take me on` simulates to `6.45-10.25`
  - that is the first merged-segment path that looks plausibly ship-worthy enough to test in production
- That selective merged-segment repair is now kept in production:
  - implementation lives in `whisper_integration_align_corrections.py`
  - it runs after weak-evidence restore and before the adjacent late-shift cleanup
  - it only targets the middle alternating 3-word hook line, using:
    - exact normalized phrase matching inside `all_words`
    - baseline-clamped starts
    - phrase-window-informed end recovery
  - forced-off mapped `Take On Me` improved in `benchmarks/results/20260328T034813Z`:
    - `0.780 / 1.001 -> 0.777 / 0.963`
  - shipped-path rerun in `benchmarks/results/20260328T035014Z` stayed effectively stable:
    - `Take On Me` `0.1257 / 0.2781 -> 0.1359 / 0.2834`
    - `Clocks` unchanged at `0.5000 / 0.6065`
  - read:
    - the merged-segment subphrase path is now a real architecture lever for the alternating-hook mapped family
    - this is stronger than any remaining onset-only long-gap idea
- `Take On Me` line 4 is now better understood too:
  - forced-off trace `benchmarks/results/20260328T035902Z` shows its tail loss happens at the initial baseline constraint
  - later correction passes do not move it at all
  - a narrow full-line baseline restore was tested and reverted:
    - it made forced-off `Take On Me` worse (`0.866 / 0.991` in `benchmarks/results/20260328T040136Z`)
    - line-level shape looked nicer, but word-level timing got much worse
  - a new diagnostic simulator, `tools/simulate_final_tail_last_word_extension.py`, shows why:
    - the real cached Whisper segment ends at `18.6`
    - there is no local Whisper tail evidence for extending `In a day or two` toward the gold end at `22.05`
  - implication:
    - line 4 is not another merged-subphrase recovery candidate
    - do not spend the next branch on a local Whisper-tail extension there
- `Take On Me` line 3 is now better localized:
  - `tools/analyze_restored_later_onset_candidates.py` shows the real cached segment still has an exact phrase window for `I'll be gone`
  - current forced-off mapped line: `11.91-15.49`
  - exact phrase window: `12.34-15.5`
  - baseline start: `12.35`
  - the existing later-onset reanchors never reach it because they require near-baseline anchoring first
  - line 3 is `0.44s` off baseline, so it is blocked before exact phrase support is considered
  - implication:
    - the next sensible `Take On Me` branch is a narrow restored-line exception for exact later phrase windows
    - not a broad relaxation of the baseline-anchor tolerance
- That restored-line exact-phrase branch is now kept too:
  - implementation lives in `whisper_integration_align_corrections.py`
  - it only touches compact 3-4 word lines when:
    - an exact later phrase window exists in `all_words`
    - the current line end is already close to the phrase-window end
    - the line is late at the start but still near its baseline anchor
  - forced-off mapped `Take On Me` improved again in `benchmarks/results/20260328T041703Z`:
    - `0.7767 / 0.9634 -> 0.7239 / 0.9458`
    - line 3 `I'll be gone` moved from `11.91-15.49` to `12.34-15.39`
  - shipped-path controls in `benchmarks/results/20260328T041950Z` stayed unchanged for the validated songs:
    - `Take On Me` `0.1359 / 0.2834`
    - `Clocks` `0.5000 / 0.6065`
  - read:
    - `Take On Me` now has two keepable narrow repairs in the same architecture family:
      - merged-subphrase recovery for line 2
      - restored exact-phrase late-start recovery for line 3
    - broad later-onset relaxations are still the wrong tool; exact phrase support is the useful signal
- A third narrow `Take On Me` repair is now kept for the final line:
  - implementation also lives in `whisper_integration_align_corrections.py`
  - it extends only the last word of the final line to baseline end
  - it only fires when:
    - the final line start is still near baseline
    - the baseline tail gain is modest
    - there is no later exact Whisper phrase window for that final line
  - forced-off mapped `Take On Me` improved again in `benchmarks/results/20260328T042557Z`:
    - `0.7239 / 0.9458 -> 0.7239 / 0.9069`
    - line 4 `In a day or two` moved from `17.37-20.87` to `17.37-21.42`
  - shipped-path controls in `benchmarks/results/20260328T042657Z` stayed unchanged:
    - `Take On Me` `0.1359 / 0.2834`
    - `Clocks` `0.5000 / 0.6065`
  - read:
    - the earlier full-line baseline restore was too blunt
    - last-word-only baseline tail repair is the safe version for this final-line shape
- The leading hook now has a kept repair too:
  - stage trace showed the live line-1 regression clearly:
    - `after_initial_baseline_constraint` put `Take on me` at `0.99-4.57`
    - `after_reanchor_late_supported_lines_to_earlier_whisper` pulled it back to `0.64-4.57`
  - the kept fix lives in `whisper_integration_align_corrections.py`
  - it restores the leading alternating 3-word hook start after that generic earlier-Whisper reanchor
  - only when:
    - the following baseline line forms the alternating pair
    - the line end is already near baseline
    - exact phrase support exists for the current line text
  - forced-off mapped `Take On Me` improved again in `benchmarks/results/20260328T203409Z`:
    - `0.7239 / 0.9069 -> 0.6214 / 0.8550`
    - line 1 `Take on me` moved from `0.64-4.57` to `0.99-4.45`
  - shipped-path controls in `benchmarks/results/20260328T203450Z` stayed unchanged:
    - `Take On Me` `0.1359 / 0.2834`
    - `Clocks` `0.5000 / 0.6065`
  - read:
    - `Take On Me` now has a full four-line stack of keepable narrow repairs
    - the next useful move is probably not more local `Take On Me` surgery, but extracting what generalizes
- Family-scope check:
  - `tools/analyze_alternating_hook_family.py` now scans the curated gold set for adjacent alternating 3-word hook pairs
  - current result is intentionally narrow:
    - only `33_a-ha-take-on-me-first-chorus.gold.json` line 1/2 matches
  - read:
    - the alternating-hook repairs are now well-validated on `Take On Me`
    - but they are not yet a demonstrated broad family in the current curated pack
    - any wider rollout should wait for another real clip or a bigger corpus scan

## Process Learnings

- See also `docs/development.md` for the broader local workflow and documentation-maintenance rules around this loop.
- Use the narrow iteration loop when:
  - one clip is clearly the top outlier
  - a clip has split into separate line-level failure modes
  - a plausible fix already improved the seed, and the remaining miss looks downstream
  - a broad canary is clean enough that you can afford to localize the next read
- Do not use the narrow loop as the default when:
  - the broad canary has not been reranked recently
  - the current top clip may still be curation drift rather than pipeline behavior
  - multiple clips are moving at once and you do not yet know the shared failure family
- The recent time-pressure loop was useful, and not only because of the deadline.
- The parts worth keeping even when time pressure is lower are:
  - narrow the next step to one concrete code path plus one concrete artifact before editing
  - state explicit success and failure criteria before broadening a fix
  - save negative results, not just wins, when they materially eliminate a suspect path
  - keep the current target clip split into separate failure modes when the lines are clearly failing for different reasons
  - pin the exact rerun command, trace env, unit check, and lint check next to the current hypothesis
- The part not worth keeping at full intensity is constant commit/push churn after every tiny note.
- Preferred normal mode:
  - commit/push after a clean behavioral win
  - commit/push after a meaningful diagnostic artifact that would be expensive to rediscover
  - batch small handoff-note updates together unless there is a real risk of losing context
- In practice, the good default loop is:
  1. identify one clip, one line, one likely code path
  2. name the exact artifact and rerun command
  3. run one focused probe
  4. either keep the win or record the elimination
  5. only then widen to the broader canary
