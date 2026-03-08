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
