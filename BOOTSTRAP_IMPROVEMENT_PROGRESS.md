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

### Milestone 16: Candidate Audio Reuse
Status: Completed
- [x] Add local audio extraction from already-downloaded candidate video
- [x] Fall back to direct YouTube audio download if extraction fails
- [x] Add unit test coverage for one-time extraction + cache reuse behavior

### Milestone 17: Refinement Window Slice Efficiency
Status: Completed
- [x] Replace per-line full group-frame scans with bisect-based time slicing
- [x] Preserve inclusive window semantics on start/end bounds
- [x] Add unit test coverage for frame-window slicing behavior

### Milestone 18: Bootstrap Post-Processing Separation
Status: Completed
- [x] Extract line/word post-processing from CLI `main` into `_build_refined_lines_output`
- [x] Preserve existing fallback/interpolation behavior and line confidence aggregation
- [x] Add unit tests for title/artist filtering and missing-word-start fallback behavior

### Milestone 19: Refinement Per-Line Extraction
Status: Completed
- [x] Extract per-line visual refinement logic into `_refine_line_with_frames`
- [x] Keep group-window orchestration focused on window management only
- [x] Add unit test coverage for per-line word timing population behavior

### Milestone 20: Refined-Line Builder Cleanup
Status: Completed
- [x] Precompute normalized title/artist once in refined-line output builder
- [x] Remove repeated per-line normalization of static metadata values

### Milestone 21: Bootstrap Main Orchestration Split
Status: Completed
- [x] Extract argument parsing into `_parse_args`
- [x] Extract media resolution, suitability validation, visual bootstrap, and report writing into dedicated helpers
- [x] Keep CLI behavior unchanged while reducing `main` complexity

### Milestone 22: Word-Fill Interpolation Efficiency
Status: Completed
- [x] Add precomputed nearest-known-word index helper for missing word-start interpolation
- [x] Remove per-word list scans for previous/next known timing anchors
- [x] Add unit test coverage for nearest-known index mapping behavior

### Milestone 23: Shared Post-Processing Module
Status: Completed
- [x] Move refined-line post-processing logic into `core.visual.bootstrap_postprocess`
- [x] Keep tool-level compatibility wrappers for existing callers/tests
- [x] Add direct unit tests for shared post-processing module behavior

### Milestone 24: Shared Runtime Helpers
Status: Completed
- [x] Move suitability gating and report payload/write logic into `core.visual.bootstrap_runtime`
- [x] Rewire tool wrappers to delegate to shared runtime module
- [x] Add direct unit tests for runtime suitability/report helper behavior

### Milestone 25: Shared Candidate Discovery/Ranking
Status: Completed
- [x] Move candidate search/ranking logic into `core.visual.bootstrap_candidates`
- [x] Rewire tool wrappers to delegate to shared candidate helpers
- [x] Add direct unit tests for candidate search/ranking helper behavior

### Milestone 26: Shared OCR Sampling/Caching
Status: Completed
- [x] Move raw OCR frame sampling and cache-key/cache-load logic into `core.visual.bootstrap_ocr`
- [x] Rewire tool OCR wrappers to delegate to shared OCR helpers
- [x] Add direct unit tests for OCR cache key/version and cache reuse behavior

### Milestone 27: Shared Media Resolution
Status: Completed
- [x] Move audio extraction and media path resolution logic into `core.visual.bootstrap_media`
- [x] Rewire tool media wrappers to delegate to shared media helpers
- [x] Add direct unit tests for extraction cache behavior and fallback download behavior

### Milestone 28: Shared Candidate Selection
Status: Completed
- [x] Move candidate selection orchestration into `core.visual.bootstrap_selection`
- [x] Rewire tool candidate-selection wrapper to delegate to shared selector helper
- [x] Add direct unit tests for explicit-URL and low-quality rejection selection behavior

### Milestone 29: Documentation Sync
Status: Completed
- [x] Update `ARCHITECTURE.md` with shared bootstrap module boundaries
- [x] Update `docs/karaoke_visual_bootstrap.md` to reflect delegated shared implementation modules

### Milestone 30: Candidate Logging Severity Fix
Status: Completed
- [x] Split candidate ranking logging into info vs warning channels in shared candidate helper
- [x] Restore warning-level logging for skipped candidate errors in CLI wrapper

### Milestone 31: Stable Candidate Cache Paths
Status: Completed
- [x] Use per-video-id candidate work directories instead of rank-index-only paths
- [x] Improve cache reuse across reruns when search result ordering changes
- [x] Add unit coverage that candidate directories are keyed by YouTube video id

### Milestone 32: Intro Non-Lyric Artifact Suppression
Status: Completed
- [x] Add pre-lyric reconstruction filter for intro/title-card artifacts (branding and metadata-like text)
- [x] Add regression coverage for `KAI AOK`/`SingKING`-style intro leakage
- [x] Validate that early non-lyric entries are removed while preserving short real lyrics when no intro gap exists
- [x] Add recurrent bottom-edge fragment-family suppression for clipped branding shards (`KIN/KIR/KAPA`-style)
- [x] Add low-FPS overlap-block surrogate sequencing fallback for unresolved short visibility blocks (prevents early line starts when full cycle detection is absent)

## Notes
- Keep changes in significant, validated commits.
- Prefer deterministic ranking and quality decisions for reproducibility.
