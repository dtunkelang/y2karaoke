# Bootstrap Improvement Progress

## Goals
- [ ] Candidate auto-discovery and ranking for karaoke source videos
- [ ] Visual suitability quality gates integrated into bootstrap
- [ ] Visual refinement efficiency improvements
- [ ] Confidence metadata in generated gold JSON
- [ ] Test coverage for bootstrap/suitability paths

## Milestones

### Milestone 1: Candidate Discovery + Quality Gates
Status: Completed
- [x] Add candidate search when `--candidate-url` is omitted
- [x] Add optional `--show-candidates` output
- [x] Integrate suitability scoring into candidate selection
- [x] Add threshold flags and override path
- [x] Align docs (`README.md`, `docs/karaoke_visual_bootstrap.md`) with implemented CLI

### Milestone 2: Efficiency + Confidence
Status: Pending
- [ ] Refactor refinement loops to reduce repeated frame processing costs
- [ ] Add per-word confidence and per-line aggregate confidence in output

### Milestone 3: Tests + Validation
Status: In progress
- [x] Add/expand unit tests for bootstrap logic and suitability decisions
- [x] Ensure `black`, `flake8`, `mypy`, and unit tests pass

## Notes
- Keep changes in significant, validated commits.
- Prefer deterministic ranking and quality decisions for reproducibility.
