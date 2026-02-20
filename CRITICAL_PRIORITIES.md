# Critical Priorities & Roadmap

This document outlines the critical areas of the `y2karaoke` codebase and the immediate priorities for development and maintenance.

## 1. Rendering Performance (High Priority)
**Context:** Video generation is CPU-intensive. The current `frame_renderer.py` recalculates text layouts (bounding boxes, widths) for every single frame (e.g., 30 times/sec), even though lyrics layout is static for seconds at a time.

**Status (2026-02-17):**
*   **Complete:** Implemented `layout_cache` in `frame_renderer.py` and `video_writer.py`. Verified with unit tests.

**Critical Areas:**
*   `src/y2karaoke/core/components/render/frame_renderer.py`: Main rendering loop.
*   `src/y2karaoke/core/components/render/video_writer.py`: MoviePy integration.

**Action Plan:**
*   **Monitor:** Check generation speed on longer songs.

## 2. Gold Set Quality Automation
**Context:** Updating gold files manually (copying timing reports) is error-prone and tedious. Some songs have large errors due to version mismatches (e.g., radio edit vs album).

**Critical Areas:**
*   `tools/run_benchmark_suite.py`: The benchmark runner.

**Status (2026-02-19):**
*   **Complete:** Added `--rebaseline` to `tools/run_benchmark_suite.py` with test coverage and README documentation.
*   **Complete:** Rebaselined "Anti-Hero" and "Blinding Lights" gold files to resolve systematic offsets detected in validation.

**Action Plan:**
*   **Monitor:** Validate that rebaselined benchmarks remain stable in future runs.

## 3. Technical Debt & Cleanup
**Context:** `lyrics_renderer.py` appears to be a placeholder or legacy module with unused parameters. `frame_renderer.py` mixes timing logic (`_get_lines_to_display`) with pixel drawing.

**Critical Areas:**
*   `src/y2karaoke/core/components/render/lyrics_renderer.py`: Unused logic?
*   `src/y2karaoke/core/components/render/frame_renderer.py`: High complexity.

**Status (2026-02-19):**
*   **Complete:** Timing/visibility logic extracted from `frame_renderer.py` into `render/lyric_timeline.py` to separate timeline decisions from pixel drawing.
*   **Complete:** Removed unused parameter from `lyrics_renderer.get_singer_colors` and updated callsites/tests.
*   **Complete:** Further split `frame_renderer.py` by extracting active-line resolution, per-line highlight-width computation, and visible-line rendering pass into focused helpers.
*   **Complete:** Extracted frame mode/state computation (`outro` / `splash` / `progress` / `lyrics`) from `render_frame` to reduce orchestration complexity.
*   **Complete:** Introduced explicit per-frame render plan model in `frame_renderer.py` to formalize coordinator state.
*   **Complete:** Moved lyric text/highlight drawing primitives to `render/render_text.py` to narrow `frame_renderer.py` responsibilities.
*   **Complete:** Moved cue-indicator drawing primitive to `render/cue_indicator.py` with compatibility wrapper in `frame_renderer.py`.
*   **Complete:** Moved line layout/cache primitive to `render/layout.py` with compatibility wrapper in `frame_renderer.py`.
*   **Complete:** Moved render-plan/state computation to `render/frame_plan.py` with compatibility wrappers in `frame_renderer.py`.
*   **Complete:** Moved visible-line rendering pass orchestration to `render/line_pass.py` with compatibility wrappers in `frame_renderer.py`.
*   **Complete:** Moved unresolved overlap surrogate-timing clustering/assignment into `visual/surrogate_timing.py` with compatibility wrappers in `visual/refinement.py`.
*   **Complete:** Moved repeated-line retiming postpasses into `visual/refinement_repetition_postpasses.py` with compatibility wrappers in `visual/refinement_postpasses.py`.
*   **Complete:** Moved static-overlay suppression logic out of `visual/reconstruction.py` into `visual/reconstruction_overlay.py` to isolate OCR-noise filtering from line sequencing logic.
*   **Complete:** Moved shared-visibility gap/followup retiming heuristics into `visual/refinement_shared_visibility_postpasses.py` with compatibility wrappers in `visual/refinement_postpasses.py`.
*   **Complete:** Moved transition/interstitial retiming heuristics into `visual/refinement_transition_postpasses.py` with compatibility wrappers in `visual/refinement_postpasses.py`.
*   **Complete:** Started Whisper post-processing decomposition by moving interjection-line retiming into `components/whisper/whisper_mapping_post_interjections.py` with compatibility wrapper in `whisper_mapping_post.py`.
*   **Complete:** Continued Whisper post-processing decomposition by moving overlap resolution into `components/whisper/whisper_mapping_post_overlaps.py` with compatibility wrapper in `whisper_mapping_post.py`.
*   **Complete:** Started Whisper integration decomposition by moving low-confidence word filtering into `components/whisper/whisper_integration_filters.py` with compatibility wrapper in `whisper_integration_pipeline.py`.
*   **Complete:** Continued Whisper integration decomposition by moving mapped-line stage orchestration/invariant helpers into `components/whisper/whisper_integration_stages.py` with compatibility wrappers in `whisper_integration_pipeline.py`.
*   **Complete:** Continued Whisper post-processing decomposition by moving segment-based late/early line pulling into `components/whisper/whisper_mapping_post_segment_pull.py` with compatibility wrapper in `whisper_mapping_post.py`.
*   **Complete:** Continued Whisper integration decomposition by moving baseline-alignment/rollback helpers into `components/whisper/whisper_integration_baseline.py` with compatibility wrappers in `whisper_integration_pipeline.py`.
*   **Complete:** Continued Whisper integration decomposition by moving low-quality-segment and finalization orchestration into `components/whisper/whisper_integration_finalize.py` with compatibility wrappers in `whisper_integration_pipeline.py`.
*   **Complete:** Continued Whisper repetition-postprocessing decomposition by moving short-question minimum-duration enforcement into `components/whisper/whisper_mapping_post_question_duration.py` with compatibility wrapper in `whisper_mapping_post_repetition.py`.
*   **Complete:** Continued Whisper alignment decomposition by moving short-line silence/onset refinement helpers into `components/whisper/whisper_alignment_short_lines.py` with compatibility wrappers in `whisper_alignment_refinement.py`.
*   **Complete:** Continued Whisper alignment decomposition by moving vocal-activity gap-fill and timing dedupe helpers into `components/whisper/whisper_alignment_activity.py` with compatibility wrappers in `whisper_alignment_refinement.py`.
*   **Complete:** Continued Whisper repetition-postprocessing decomposition by moving trailing-line extension logic into `components/whisper/whisper_mapping_post_tail_extension.py` with compatibility wrapper in `whisper_mapping_post_repetition.py`.
*   **Complete:** Hardened GitHub Actions apt bootstrap in `.github/workflows/test.yml` to remove any stale `packages.microsoft.com` source entries before `apt-get update` (fixes intermittent Ubuntu Noble 403 failures).
*   **Complete:** Continued Whisper mapping decomposition by moving line-context and drift-clamp helpers into `components/whisper/whisper_mapping_pipeline_line_context.py` with compatibility wrappers in `whisper_mapping_pipeline.py`.
*   **Complete:** Continued Whisper integration decomposition by moving transcription/cache orchestration into `components/whisper/whisper_integration_transcribe.py` with compatibility wrapper in `whisper_integration_pipeline.py`.
*   **Complete:** Continued Whisper post-processing decomposition by moving repeated-line shift + monotonic-start helpers into `components/whisper/whisper_mapping_post_repeat_shift.py` with compatibility wrappers in `whisper_mapping_post.py`.
*   **Complete:** Continued Whisper integration decomposition by moving DTW LRC-to-Whisper alignment orchestration into `components/whisper/whisper_integration_align.py` with compatibility wrapper in `whisper_integration_pipeline.py`.
*   **Complete:** Continued Whisper mapping decomposition by moving assigned-word and gap-fill matching passes into `components/whisper/whisper_mapping_pipeline_matching.py` with compatibility wrappers in `whisper_mapping_pipeline.py`.
*   **Complete:** Continued Whisper mapping decomposition by moving candidate scoring/ordering/registration into `components/whisper/whisper_mapping_pipeline_candidates.py` with compatibility wrappers in `whisper_mapping_pipeline.py`.
*   **Complete:** Continued Whisper integration decomposition by moving hybrid/DTW correction orchestration into `components/whisper/whisper_integration_correct.py` with compatibility wrapper in `whisper_integration_pipeline.py`.
*   **Complete:** Continued Whisper post-processing decomposition by moving first-word onset snap logic into `components/whisper/whisper_mapping_post_onset.py` with compatibility wrapper in `whisper_mapping_post.py`.
*   **Complete:** Continued Whisper mapping decomposition by moving mapped-line assembly logic into `components/whisper/whisper_mapping_pipeline_assembly.py` with compatibility wrapper in `whisper_mapping_pipeline.py`.
*   **Complete:** Continued Whisper mapping decomposition by moving top-level line orchestration into `components/whisper/whisper_mapping_pipeline_orchestration.py` with compatibility wrapper in `whisper_mapping_pipeline.py`.
*   **Complete:** Continued visual reconstruction decomposition by moving mirrored-lane cycle extrapolation helpers into `visual/reconstruction_mirrored_cycles.py` with compatibility wrappers in `visual/reconstruction.py`.
*   **Complete:** Continued visual reconstruction decomposition by moving intro-credits and bottom-fragment suppression helpers into `visual/reconstruction_intro_filters.py` with compatibility wrappers in `visual/reconstruction.py`.
*   **Complete:** Continued visual refinement decomposition by moving frame-window capture/sampling/slicing helpers into `visual/refinement_frame_windows.py` with compatibility wrappers in `visual/refinement.py`.
*   **Complete:** Continued visual refinement decomposition by moving persistent-overlap block selection/clustering/onset assignment helpers into `visual/refinement_persistent_blocks.py` with compatibility wrappers in `visual/refinement.py`.
*   **Complete:** Continued visual refinement decomposition by moving unresolved-overlap onset-hint helpers into `visual/refinement_overlap_hints.py` with compatibility wrappers in `visual/refinement.py`.
*   **Complete:** Continued visual refinement decomposition by moving highlight detection algorithms into `visual/refinement_detection.py` with compatibility wrappers in `visual/refinement.py`.
*   **Complete:** Continued visual reconstruction decomposition by moving context-transition epoch inference into `visual/reconstruction_context_transitions.py` with compatibility wrapper in `visual/reconstruction.py`.
*   **Complete:** Continued visual reconstruction decomposition by moving overlapped same-text repetition expansion into `visual/reconstruction_overlap_repetitions.py` with compatibility wrapper in `visual/reconstruction.py`.
*   **Complete:** Continued visual refinement decomposition by moving line-job window build/merge helpers into `visual/refinement_jobs.py` with compatibility wrappers in `visual/refinement.py`.
*   **Complete:** Continued visual reconstruction decomposition by moving short-refrain detection/collapse helpers into `visual/reconstruction_refrain.py` with compatibility wrappers in `visual/reconstruction.py`.
*   **Complete:** Continued visual refinement decomposition by moving line-level word timing distribution into `visual/refinement_line_assignment.py` with compatibility wrapper in `visual/refinement.py`.
*   **Complete:** Continued visual reconstruction decomposition by moving lane-proximity and overlapping-lane merge helpers into `visual/reconstruction_lane_merge.py` with compatibility wrappers in `visual/reconstruction.py`.
*   **Complete:** Continued visual reconstruction decomposition by moving short-duplicate suppression and same-lane reentry merge helpers into `visual/reconstruction_reentry.py` with compatibility wrappers in `visual/reconstruction.py`.
*   **Complete:** Continued visual refinement decomposition by moving foreground mask and line color-value collection helpers into `visual/refinement_masks.py` with compatibility wrappers in `visual/refinement.py`.
*   **Complete:** Continued visual refinement decomposition by moving visibility-window onset estimators into `visual/refinement_onset_estimation.py` with compatibility wrappers in `visual/refinement.py`.
*   **Complete:** Continued visual refinement decomposition by moving line-level highlight-cycle/onset combination into `visual/refinement_line_highlight.py` with compatibility wrapper in `visual/refinement.py`.
*   **Complete:** Continued visual refinement decomposition by moving per-line frame refinement orchestration into `visual/refinement_line_refine.py` with compatibility wrapper in `visual/refinement.py`.
*   **Complete:** Continued visual reconstruction decomposition by moving top-level OCR-to-line reconstruction orchestration into `visual/reconstruction_pipeline.py` with compatibility wrapper in `visual/reconstruction.py`.
*   **Complete:** Continued visual refinement decomposition by moving high-FPS orchestration loop into `visual/refinement_high_fps_pipeline.py` with compatibility wrapper in `visual/refinement.py`.
*   **Complete:** Continued visual refinement decomposition by moving dense shared-visibility run retiming heuristics into `visual/refinement_dense_run_postpasses.py` with compatibility wrappers in `visual/refinement_postpasses.py`.
*   **Complete:** Continued visual reconstruction decomposition by moving persistent-line frame accumulation logic into `visual/reconstruction_frame_accumulation.py` with compatibility wrapper in `visual/reconstruction_pipeline.py`.
*   **Complete:** Continued visual reconstruction decomposition by moving persistent-line deduplication logic into `visual/reconstruction_deduplication.py` with compatibility wrapper in `visual/reconstruction_pipeline.py`.
*   **Complete:** Continued visual reconstruction decomposition by moving persistent-line to target-line conversion logic into `visual/reconstruction_target_conversion.py` with compatibility wrapper in `visual/reconstruction_pipeline.py`.
*   **Complete:** Consolidated singer-color policy into `render/singer_style.py` and updated `frame_renderer.py` and `lyrics_renderer.py`.
*   **Complete:** Extracted frame generation logic from `video_writer.py` into `render/frame_generation.py` to decouple MoviePy integration from rendering state management.
*   **Complete:** Consolidate progress reporting logic into `render/progress.py` and remove inline progress printing from `video_writer.py`.
*   **Complete:** Extracted scene detection logic from `render/backgrounds.py` into a standalone script `utils/scene_detector_script.py` to avoid embedded string code and import conflicts.

**Action Plan:**
*   **Monitor:** Check generation speed on longer songs.

## 4. Visual Refinement Stability
**Status (2026-02-19):**
*   **Complete:** Refactoring and test coverage improvements done. Logic extracted to `src/y2karaoke/core/visual/`.

## 5. Benchmark Execution & Validation
**Status (2026-02-19):**
*   **Complete:** Full suite run passed (`validation_run_u`) with 100% song success. Identified and rebaselined offset issues for "Anti-Hero" and "Blinding Lights".

---

**Next Immediate Step:**
Release candidate preparation or further feature development (e.g. CLI usability improvements, new render modes).
