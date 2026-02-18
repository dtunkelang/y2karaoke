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
- [x] Added contraction-aware OCR token normalization and apostrophe fragment merging in visual reconstruction output (e.g., `you ' re` -> `you're`, `don ' t` -> `don't`, `sleepin ' ` -> `sleepin'`).
- [x] Added short-lived duplicate re-entry suppression in visual reconstruction to prevent stale OCR line echoes from re-inserting already-consumed lyrics while preserving legitimate repeated lines.
- [x] Validation snapshot (`Billie Eilish - bad guy`, `GsFlbMS7UIc`, v20):
  - First 39 OCR tokens align exactly with target sequence.
  - Comparable-word coverage improved to `0.9724` (247/254).
  - p95 absolute start-time delta reduced to `11.25s`.
- [x] Added low-FPS line-level visual timing refinement fallback when word-level suitability is too low for high-FPS word refinement.
- [x] Validation snapshot (`Billie Eilish - bad guy`, `GsFlbMS7UIc`, v21):
  - Comparable-word coverage improved to `0.9764` (248/254).
  - Mean absolute start-time delta improved from `7.942s` (v20) to `7.805s`.
  - p95 absolute start-time delta improved from `11.25s` (v20) to `10.9s`.
- [x] Added strict line lifecycle gating (`inactive -> active -> consumed`) for low-FPS line-level refinement so repeated text is only reusable after a full highlight cycle.
- [x] Validation snapshot (`Billie Eilish - bad guy`, `GsFlbMS7UIc`, v25 lifecycle gate):
  - Mean absolute word-start delta improved from `7.819s` (v21) to `7.284s`.
  - p95 absolute word-start delta improved from `10.9s` (v21) to `10.7s`.
  - Residual after global-shift correction improved from `p95=8.645s` (v21) to `p95=8.101s` (v25).
- [x] Added lane-aware duplicate suppression in visual reconstruction so concurrent repeated text in different on-screen lyric lanes is preserved.
- [x] Added reconstruction regression test for concurrent repeated `Duh` lines in separate lanes.
- [x] Added lane-continuity merge for same-text near-lane segments to prevent false line splits when OCR y-jitter crosses lane-bin boundaries.
- [x] Validation snapshot (`Billie Eilish - bad guy`, `GsFlbMS7UIc`, v30 lane continuity):
  - Removed spurious extra same-lane `duh` in repeated chorus block (`I'm the bad guy / duh / I'm the bad guy / duh`).
  - Reduced reconstructed initial line count in this run from `73` to `68` by collapsing boundary-jitter re-splits.
- [x] Validated on Billie Eilish - "bad guy" (`GsFlbMS7UIc`) with fresh cache versions:
  - `raw_frames_463cf7852a3083257907dae10e9b4399.json`: residual branded tokens reduced to intro cards only (`SingKING/KARAOKE/Karaoke`, 7 total).
  - Removed recurring edge fragments (`KIN/KII/KAPA/KARAO`) from sampled raw frames.
- [x] Completed: P2 architectural cleanup step by isolating LRC timing-trust policy in lyrics quality orchestration.
- [x] Completed: P2 boundary cleanup by centralizing Whisper auto-enable gating across simple and quality lyrics paths.
- [x] Reduced line-level mask strictness for low-FPS line refinement (`_line_fill_mask`) to retain unselected-text samples in long visibility windows.
- [x] Clustered persistent-block onset overrides by visibility interval before enforcing top-to-bottom ordering (avoids cross-section onset pushing in long merged groups).
- [x] Added guardrail to avoid stretching a line to the next onset across large gaps (>6s), preserving instrumental breaks.
- [x] Added visual-line visibility-span metadata (`visibility_start` / `visibility_end`) to reconstructed target lines and enforced refinement windows that are never shorter than on-screen visibility.
- [x] Increased low-FPS line-refinement merged window duration to keep persistent multi-line blocks in a single analysis window.
- [x] Fixed low-FPS `min_start_time` gating to be start-based (not end-based) and to honor each line's visibility floor, with regression tests for overlapping-line behavior.
