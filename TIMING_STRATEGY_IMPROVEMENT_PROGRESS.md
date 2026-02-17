# Timing Strategy Improvement Progress

Last updated: 2026-02-17

## Priority Roadmap

### P0 - Benchmark Validity and Measurement Integrity
- [x] Replace index-based gold-word comparison with monotonic text-aware matching.
- [x] Split line-start agreement into:
  - [x] Independent metric (usable for cross-strategy comparison)
  - [x] Whisper-anchor diagnostic metric (self-referential, debug-only)
- [x] Prevent recommendation logic from over-weighting unavailable/diagnostic metrics.

### P0 - LRC Timing Robustness Guardrails
- [ ] Add safer fallback behavior when LRC timing is clearly duration-mismatched.
- [ ] Ensure timing reports and docs clearly separate trusted vs degraded timing paths.

### P1 - Lyrics Without Timings
- [ ] Default to audio-aware alignment when lyrics exist but no reliable timings exist.
- [ ] Add benchmark scenarios to isolate this path and compare cost/quality.

### P1 - No-Lyrics (Whisper-Only) Quality/Cost Tradeoff
- [ ] Add adaptive Whisper strategy (fast pass + selective high-quality re-pass).
- [ ] Evaluate quality gain per added runtime.

### P1 - Karaoke Video Scraping Efficiency
- [ ] Reduce candidate ranking cost (prefilter + lightweight visual probe).
- [ ] Improve ROI/suitability sampling to avoid expensive seek-heavy decoding.

### P2 - Architecture and Maintainability
- [ ] Continue splitting high-complexity Whisper orchestration hotspots.
- [ ] Tighten module boundaries and contracts for easier regression isolation.

### P2 - Documentation Sync
- [ ] Remove stale “WhisperX” references and align docs to current implementation.
- [ ] Document independent vs diagnostic metrics in benchmark outputs.

## Work Log

- [x] Captured prioritized improvement plan in-repo.
- [x] Completed: P0 benchmark validity fixes.
- [ ] In progress: P0 LRC timing robustness guardrails.
