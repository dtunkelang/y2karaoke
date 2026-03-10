# Next Session TODO (Post-0.35 Baseline Refresh)

## 0. Current Baseline (frozen reference)
- Latest strong run: `benchmarks/results/20260305T231015Z`
- Strategy: `hybrid_whisper`
- Current default agreement thresholds:
  - `Y2KARAOKE_BENCH_AGREEMENT_MIN_TEXT_SIM=0.58`
  - `Y2KARAOKE_BENCH_AGREEMENT_MIN_TOKEN_OVERLAP=0.50`
- Key aggregate metrics:
  - `agreement_coverage_ratio_total=0.3524`
  - `agreement_start_p95_abs_sec_mean=1.0675`
  - `agreement_bad_ratio_mean=0.049`
  - `timing_quality_score_line_weighted_mean=0.7907`
  - `dtw_line_coverage_line_weighted_mean=0.9342`
  - `dtw_word_coverage_line_weighted_mean=0.8522`

## 1. Primary Quality Goals
- [ ] Reduce benchmark warning pressure while preserving >=`0.35` agreement coverage total.
- [ ] Improve `agreement_start_p95_abs_sec_mean` from ~`1.07` toward <=`0.95` without reducing coverage.
- [ ] Keep `agreement_bad_ratio_mean` <=`0.050` while improving timing quality.
- [ ] Raise `agreement_coverage_ratio_mean` from current ~`0.329` toward >=`0.350`.

## 2. Human-Guided Correction Execution
- [x] Generate current correction handoff artifacts:
  - `benchmarks/results/human_guidance_tasks_20260305T231015Z.md`
  - `benchmarks/results/human_guidance_ready_20260305T231015Z.tsv`
- [x] Generate proxy delta report artifact to validate comparison pipeline wiring:
  - `benchmarks/results/20260305T231015Z/human_correction_delta_proxy.md`
  - Baseline `20260305T225648Z` vs candidate `20260305T231015Z` (non-human proxy).
- [ ] Run a focused manual-correction pass for current top priority songs:
  - `J Balvin - Mi Gente`
  - `Bruno Mars - Uptown Funk`
  - `ROSALIA - DESPECHA`
- [x] Complete first curated canary pair:
  - `benchmarks/gold_set_candidate/20260305T231015Z/08_indila-derniere-danse.gold.json`
  - `benchmarks/gold_set_candidate/20260305T231015Z/06_the-weeknd-blinding-lights.gold.json`
- [x] Produce a focused auto-vs-corrected canary comparison using `tools/compare_benchmark_correction.py`.
  - Artifacts:
    - `benchmarks/results/20260306T_two_song_curated_fresh/human_correction_delta.json`
    - `benchmarks/results/20260306T_two_song_curated_fresh/human_correction_delta.md`
  - Current 2-song canary outcome:
    - `avg_abs_word_start_delta_sec_word_weighted_mean: 2.57s -> 1.019s`
    - `agreement_start_p95_abs_sec_mean: 0.880s -> 0.8473s`
    - `agreement_bad_ratio_mean: 0.1177 -> 0.0732`
    - `agreement_coverage_ratio_mean: 0.2206 -> 0.2236`
    - tradeoff to investigate: `dtw_word_coverage_line_weighted_mean: 0.9422 -> 0.7887`
- [ ] Quantify human edit ROI:
  - coverage delta
  - p95 delta
  - bad-ratio delta
  - edits-per-minute / snap-usage from editor telemetry

## 3. Pipeline Improvement Goals
- [x] Investigate large-anchor-span mismatch behavior (especially songs with long noisy anchor segments).
  - Findings artifact: `benchmarks/results/20260305T231015Z/anchor_span_investigation.md`
  - Highest-risk cases by long-anchor + low-sim count: `The Weeknd - Blinding Lights`, `Queen - Bohemian Rhapsody`, `Indila - Derniere danse`, `J Balvin - Mi Gente`.
- [x] Add/validate another conservative agreement rescue that improves comparability only under tight timing constraints.
  - Implemented weak-lexical tight-timing rescue; validated on `20260305T231015Z` (`agreement_coverage_ratio_total=0.3524`, guard pass).
- [x] Add a regression test fixture for long noisy `nearest_segment_start_text` anchors (Blinding Lights-like behavior).
- [ ] Explore syllable-level alignment signal as a fallback for lexical-comparability gaps.
  - Hypothesis: syllable overlap can recover valid matches when word-level text similarity is low but timing is close.
  - Scope: fallback-only path behind a feature flag (do not replace primary word-level logic).
  - Guardrails for accepting syllable-based matches:
    - tight start delta (e.g., <=`0.20s`)
    - strong local Whisper evidence
    - preserve/avoid regression in `agreement_bad_ratio_mean` and p95.
  - Evaluation plan:
    - add focused unit tests for syllable fallback acceptance/rejection branches
    - run 10-song benchmark A/B (`flag off` vs `flag on`)
    - require tradeoff guard pass (`min_coverage_gain=0.005`, `max_bad_ratio_increase=0.002`) before promotion.
- [ ] Explore repetition-aware timing transfer within a song.
  - Hypothesis: repeated chorus/refrain lines can borrow or regularize timing from other occurrences when local Whisper evidence is sparse.
  - Initial targets:
    - repeated chorus pairs in `The Weeknd - Blinding Lights`
    - repeated refrain blocks in `Indila - Derniere danse`
  - Candidate approaches:
    - align repeated lyric blocks and compare inter-line spacing consistency
    - use high-confidence repeated occurrences as timing priors for low-support repeats
    - detect verse/chorus block structure and preserve relative cadence within repeated sections
  - Guardrails:
    - only apply when text match is strong and local structure is clearly repeated
    - do not let repetition transfer override strong local Whisper/onset evidence
    - measure both main curated canary metrics and interjection/repetition-specific deltas
- [ ] Investigate gold-vs-audio interpretation gaps on repeated chorus lines.
  - Earlier source-vs-gold comparison suggested `Blinding Lights` chorus starts were about `0.7s-1.2s` earlier than curated gold.
  - Artifact:
    - `benchmarks/results/20260308T_canary_eval_interjection_markdown/blinding_lights_chorus_start_comparison.md`
    - `benchmarks/results/20260308T_canary_eval_interjection_markdown/gold_onset_bias.md`
  - Updated finding:
    - measured curated gold for `Blinding Lights` is not globally late against nearby onsets
    - aggregate onset-minus-gold-start is about `0.168s`
    - remaining chorus lines are also close to onset (`~0.01s-0.41s`)
  - Next question:
    - why is the pipeline still early on those lines relative to both gold and onset evidence?
  - Favor pipeline-stage diagnosis over more gold-target skepticism here.
- [ ] Instrument and patch the audio-only LRC refinement path before Whisper.
  - Current strongest evidence: the remaining late-chorus distortion for `Blinding Lights` is introduced before Whisper alignment.
  - Trace tool:
    - `tools/analyze_lrc_refinement_stages.py`
  - Current diagnostic artifacts:
    - `benchmarks/results/20260309T_canary_eval_no_before_i_said/blinding_lights_lrc_refinement_trace.json`
    - `benchmarks/results/20260309T_canary_eval_no_before_i_said/blinding_lights_lrc_refinement_trace.md`
  - Confirmed stage-level finding:
    - `fix_spurious_gaps()` collapses the late `Blinding Lights` chorus block from `35` lines to `24` by merging lines like `I said, ooh, I'm drowning in the night`, `Oh, when I'm like this, you're the one I trust`, and `(Hey, hey, hey)`.
  - Next branch should patch the first refinement-stage divergence that survives the real offline benchmark path, not add more Whisper-side heuristics.
  - Update from live benchmark-path trace:
    - `tools/analyze_pre_whisper_live_path.py` shows the actual pre-Whisper `get_lyrics_with_quality` path stays structurally intact for the same late chorus block.
    - Artifacts:
      - `benchmarks/results/20260309T_canary_eval_no_before_i_said/blinding_lights_live_pre_whisper_trace.json`
      - `benchmarks/results/20260309T_canary_eval_no_before_i_said/blinding_lights_live_pre_whisper_trace.md`
    - This means the helper-path `fix_spurious_gaps()` collapse is real, but it is not currently the active blocker in the offline curated-canary benchmark.
    - The remaining benchmark miss is therefore downstream of the live pre-Whisper path, likely in Whisper alignment / post-alignment handling.
    - Direct comparison of live pre-Whisper vs current best final output shows Whisper already fixes most of this block:
      - lines `23-24` move from `115.38/121.32` to `117.73/123.67`, essentially matching gold `117.70/123.75`
      - line `25` remains the main residual miss, moving from `126.51` pre-Whisper to `130.01` final while gold is `128.65`
    - Next downstream branch should target why `I said, ooh, I'm drowning in the night` is still shifted late after Whisper/post-alignment handling.
    - Additional tracing note:
      - `tools/analyze_whisper_postpasses.py` can replay `align_lrc_text_to_whisper_timings`, but direct replay is not fidelity-equal to the offline benchmark path because it may enter a different Whisper/WhisperX fallback branch.
      - Future line-25 tracing should hook the actual benchmark subprocess path or force the same cached branch before trusting stage-level deltas.
    - Fidelity-correct benchmark-path trace:
      - using the real separated vocals stem (`.cache/fHI8X4OXluQ/The Weeknd - Blinding Lights (Official Audio)_(Vocals)_htdemucs_ft.wav`), the live pre-Whisper path keeps line `25` at `128.174s`, close to gold `128.65s`.
      - the exact replayed offline benchmark output then moves that same line to `130.010s`.
      - artifact:
        - `benchmarks/results/20260309T_blinding_exact_replay_delta/blinding_lights_pre_vs_final_delta_vocals.md`
      - active downstream delta on line `25` is therefore `+1.836s`, and the next branch should focus specifically on which Whisper/post-alignment stage applies that late shift.
    - Mapper-stage update:
      - new env-controlled trace hooks now expose per-line accepted matches, per-word candidate scores, and per-word preselected assignments:
        - `Y2K_TRACE_MAPPER_DETAILS_JSON`
        - `Y2K_TRACE_MAPPER_CANDIDATES_JSON`
        - `Y2K_TRACE_MAPPER_LINE_RANGE`
      - latest focused trace on `Blinding Lights` lines `9-10` shows the main issue is upstream of candidate scoring:
        - line `9` lexical words (`Sin`, `City's`, `cold`, `and`, `empty`) are pre-assigned only to `[VOCAL]` pseudo-words around `61.66-66.66s`
        - line `10` words are pre-assigned to far-ahead unrelated words like `maybe`, `clearly`, `didn't`, `don't`, `long,`, `me` around `97-112s`
      - candidate selection is therefore not the primary bug on those lines; it only receives one assigned word per lyric word in the `assigned_words` phase
      - next mapper branch should inspect the phoneme/DTW assignment entry path and add a confidence gate before those assignments are converted back to word indices
- [ ] Use multi-source timed-lyrics disagreement as a routing signal.
  - Hypothesis: provider disagreement is useful evidence that line timestamps are untrustworthy and we should rely more on audio/Whisper scoring.
  - Initial evidence:
    - `Blinding Lights`: `lyriq`, `NetEase`, and `Lrclib` agree closely, but `Lrclib` still scores slightly better against audio than `lyriq`.
    - `Derniere danse`: sources disagree structurally (`41` vs `44` lines), suggesting provider choice matters.
    - `Mi Gente`: sources disagree heavily on duration, line count, tail timing, and text structure.
  - Artifacts:
    - `benchmarks/results/20260308T_source_disagreement/blinding_lights.json`
    - `benchmarks/results/20260308T_source_disagreement/derniere_danse.json`
    - `benchmarks/results/20260308T_source_disagreement/mi_gente.json`
  - Candidate routing:
    - low disagreement: run cheap audio scoring across candidates and pick the best source
    - high disagreement: distrust provider timing and lean harder on Whisper/audio alignment
    - extremely high disagreement: flag as likely reference-divergence/watchlist case

## 4. Gold Set and Benchmark Data Quality
- [ ] Increase gold comparable-word coverage from ~`0.747` toward >=`0.800`.
- [ ] Prioritize gold-set expansion for songs with highest human-guidance priority and poor comparability.
- [x] Establish curated canary slice for hard-song analysis.
  - Current canaries: `Indila - Derniere danse`, `The Weeknd - Blinding Lights`
  - Use curated-gold metrics as a separate promotion lens for difficult songs.
- [ ] Keep gold updates traceable (song-level changelog note per rebaseline).

## 5. Runtime and Efficiency
- [ ] Keep full 10-song offline benchmark wall-time <=`110s` while applying quality changes.
- [ ] Track runtime regressions when agreement diagnostics change (candidate vs baseline runtime diff report).
- [ ] Continue avoiding recomputation for diagnostics-only iterations.

## 6. Tooling and UX Improvements
- [x] Extend human-guidance report with direct “first 2 suggested correction targets” per song (line index + reason).
- [x] Add optional JSON schema validation for `human_guidance_tasks.json` artifacts.
- [x] Add a compact “ready-to-edit” export list for editor workflow handoff.

## 7. CI/CD Checklist (reset for next push cycle)
- [x] `black --check src tests`
- [x] `flake8 src/y2karaoke --count --max-complexity=15 --max-line-length=127 --statistics`
- [x] `mypy src`
- [x] targeted unit tests for touched modules
- [x] `python tools/quality_guardrails.py`
- [x] if benchmark logic changed: run `tools/main_benchmark_guardrails.py`

## 8. Success Criteria for This New Queue
- [ ] At least one new run that keeps `agreement_coverage_ratio_total >= 0.35` and improves p95 meaningfully.
- [ ] At least one validated human-corrected run demonstrating measurable quality lift vs auto baseline.
- [ ] No net regression in bad-ratio or major runtime regressions.
