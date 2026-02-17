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

## Notes
- Keep changes in significant, validated commits.
- Prefer deterministic ranking and quality decisions for reproducibility.
