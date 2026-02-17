# Bootstrap Improvement Progress

## Goals
- [x] Candidate auto-discovery and ranking for karaoke source videos
- [x] Visual suitability quality gates integrated into bootstrap
- [x] Visual refinement efficiency improvements
- [x] Confidence metadata in generated gold JSON
- [x] Test coverage for bootstrap/suitability paths

## Milestones

### Milestone 1: Candidate Discovery + Quality Gates
Status: Completed
- [x] Add candidate search when `--candidate-url` is omitted
- [x] Add optional `--show-candidates` output
- [x] Integrate suitability scoring into candidate selection
- [x] Add threshold flags and override path
- [x] Align docs (`README.md`, `docs/karaoke_visual_bootstrap.md`) with implemented CLI

### Milestone 2: Efficiency + Confidence
Status: Completed
- [x] Refactor refinement loops to reduce repeated frame processing costs
- [x] Add per-word confidence and per-line aggregate confidence in output

### Milestone 3: Tests + Validation
Status: Completed
- [x] Add/expand unit tests for bootstrap logic and suitability decisions
- [x] Ensure `black`, `flake8`, `mypy`, and unit tests pass

### Milestone 4: Reproducibility + Runtime Caching
Status: Completed
- [x] Add explicit `--work-dir` control for bootstrap artifacts
- [x] Cache coarse OCR frame extraction for fast reruns
- [x] Add optional `--report-json` output for candidate rankings and run settings

### Milestone 5: Bootstrap Guardrails
Status: Completed
- [x] Add `tools/bootstrap_quality_guardrails.py` for bootstrap-output validation
- [x] Add unit tests for guardrail checks
- [x] Integrate bootstrap guardrails into `make check` / `ci-fast`

### Milestone 6: Threshold Calibration
Status: Completed
- [x] Add `tools/bootstrap_calibrate_thresholds.py` to derive threshold suggestions from bootstrap reports
- [x] Add unit tests for calibration utility
- [x] Add `make bootstrap-calibrate` helper and README usage note

### Milestone 7: Benchmark Strategy Matrix
Status: Completed
- [x] Add `--strategy` to benchmark runner (`hybrid_dtw`, `hybrid_whisper`, `whisper_only`, `lrc_only`)
- [x] Wire strategy into generated command flags and run-signature tracking
- [x] Add unit tests for strategy command generation

### Milestone 8: Multi-Strategy Benchmark Runner
Status: Completed
- [x] Add `tools/run_benchmark_strategy_matrix.py` for one-command multi-strategy runs
- [x] Emit consolidated JSON + Markdown matrix reports
- [x] Emit recommendation hints for best strategy by key metrics
- [x] Add unit tests and `make benchmark-matrix`

### Milestone 9: Default Recommendation Utility
Status: Completed
- [x] Add `tools/recommend_benchmark_defaults.py` to combine matrix + calibration outputs
- [x] Add unit tests for recommendation scoring and output
- [x] Add `make benchmark-recommend`

### Milestone 10: Matrix Summary Robustness
Status: Completed
- [x] Add fallback extraction for agreement/low-confidence keys in matrix summary generation
- [x] Add fallback extraction for gold word-start delta metric in matrix summaries
- [x] Ignore zero/non-positive calibration thresholds in recommendation output
- [x] Validate recommendation on real matrix artifacts (`liveforce4-20260217T071510Z-matrix`)

### Milestone 11: OCR Sampling Efficiency
Status: Completed
- [x] Replace frame-by-frame decode loop with `grab`/`retrieve` sampling in bootstrap OCR collection
- [x] Keep sampled timing behavior stable while reducing decode work at low visual FPS
- [x] Add unit test coverage for decode-sparing sampling path

### Milestone 12: OCR Cache Invalidation Control
Status: Completed
- [x] Add `--raw-ocr-cache-version` to control OCR-frame cache keying
- [x] Include cache version in cache signature to avoid stale reuse after algorithm updates
- [x] Add unit test ensuring cache version changes trigger recalculation
- [x] Update bootstrap docs with selective cache invalidation guidance

### Milestone 13: Suitability Sampling Efficiency
Status: Completed
- [x] Replace full-frame decode loop with sampled `grab`/`retrieve` in suitability frame collection
- [x] Preserve sampled frame outputs while reducing decode overhead during candidate ranking
- [x] Add unit test coverage for suitability sampler decode behavior

### Milestone 14: Refinement Window Reuse
Status: Completed
- [x] Add line refinement job/window builders in visual refinement module
- [x] Merge overlapping line windows to avoid repeated decode passes
- [x] Reuse decoded window frames across multiple lines while preserving per-line filtering
- [x] Add unit tests for job/window grouping behavior

### Milestone 15: Suitability Cache Key Hardening
Status: Completed
- [x] Include full file identity (path + mtime + size) in suitability cache keys
- [x] Prevent cache collisions for same-named videos in different locations
- [x] Add unit test coverage for cache-key identity behavior

## Notes
- Keep changes in significant, validated commits.
- Prefer deterministic ranking and quality decisions for reproducibility.
