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
- [x] Add safer fallback behavior when LRC timing is clearly duration-mismatched.
- [x] Ensure timing reports and docs clearly separate trusted vs degraded timing paths.

### P1 - Lyrics Without Timings
- [x] Default to audio-aware alignment when lyrics exist but no reliable timings exist.
- [x] Add benchmark scenarios to isolate this path and compare cost/quality.

### P1 - No-Lyrics (Whisper-Only) Quality/Cost Tradeoff
- [x] Add adaptive Whisper strategy (fast pass + selective high-quality re-pass).
- [x] Evaluate quality gain per added runtime.

### P1 - Karaoke Video Scraping Efficiency
- [x] Reduce candidate ranking cost (prefilter + lightweight visual probe).
- [x] Improve ROI/suitability sampling to avoid expensive seek-heavy decoding.
- [x] Add line-level visual fallback when word-level highlight transitions are absent.

### P2 - Architecture and Maintainability
- [ ] Continue splitting high-complexity Whisper orchestration hotspots.
- [ ] Tighten module boundaries and contracts for easier regression isolation.

### P2 - Documentation Sync
- [x] Remove stale “WhisperX” references and align docs to current implementation.
- [x] Document independent vs diagnostic metrics in benchmark outputs.

## Work Log

- [x] Captured prioritized improvement plan in-repo.
- [x] Completed: P0 benchmark validity fixes.
- [x] Completed: P0 LRC timing robustness guardrails.
- [x] Completed: P1 lyrics-without-timings benchmark scenario isolation and runtime-aware strategy matrix reporting.
- [x] Completed: P1 karaoke visual fallback for line-level highlight videos (line transition detection + weighted per-word allocation).
- [x] Completed: P1 karaoke visual runtime guard that skips native-FPS word refinement on near-zero word-level suitability candidates.
- [x] Completed: P1 karaoke OCR overlay filtering (top-band ignore + static watermark token suppression during reconstruction).
- [x] Completed: P1 karaoke OCR raw-cache filtering for persistent right/bottom-edge watermark artifacts (pre-cache suppression + short edge-fragment cleanup).
- [x] Removed fixed top-band OCR suppression in raw frame sampling; recovered valid top-line lyrics on videos where active lyrics render near the top of the ROI.
- [x] Validated on Billie Eilish - "bad guy" (`GsFlbMS7UIc`) with fresh cache versions:
  - `raw_frames_463cf7852a3083257907dae10e9b4399.json`: residual branded tokens reduced to intro cards only (`SingKING/KARAOKE/Karaoke`, 7 total).
  - Removed recurring edge fragments (`KIN/KII/KAPA/KARAO`) from sampled raw frames.
- [x] Completed: P2 architectural cleanup step by isolating LRC timing-trust policy in lyrics quality orchestration.
- [x] Completed: P2 boundary cleanup by centralizing Whisper auto-enable gating across simple and quality lyrics paths.
