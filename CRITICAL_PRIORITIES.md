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

**Action Plan:**
*   **Audit:** Use this tool to fix gold files for songs with systematic offsets.

## 3. Technical Debt & Cleanup
**Context:** `lyrics_renderer.py` appears to be a placeholder or legacy module with unused parameters. `frame_renderer.py` mixes timing logic (`_get_lines_to_display`) with pixel drawing.

**Critical Areas:**
*   `src/y2karaoke/core/components/render/lyrics_renderer.py`: Unused logic?
*   `src/y2karaoke/core/components/render/frame_renderer.py`: High complexity.

**Status (2026-02-19):**
*   **In progress:** Timing/visibility logic extracted from `frame_renderer.py` into `render/lyric_timeline.py` to separate timeline decisions from pixel drawing.
*   **In progress:** Removed unused parameter from `lyrics_renderer.get_singer_colors` and updated callsites/tests.
*   **In progress:** Further split `frame_renderer.py` by extracting active-line resolution, per-line highlight-width computation, and visible-line rendering pass into focused helpers.
*   **In progress:** Extracted frame mode/state computation (`outro` / `splash` / `progress` / `lyrics`) from `render_frame` to reduce orchestration complexity.
*   **In progress:** Introduced explicit per-frame render plan model in `frame_renderer.py` to formalize coordinator state.
*   **In progress:** Moved lyric text/highlight drawing primitives to `render/render_text.py` to narrow `frame_renderer.py` responsibilities.
*   **In progress:** Moved cue-indicator drawing primitive to `render/cue_indicator.py` with compatibility wrapper in `frame_renderer.py`.
*   **In progress:** Moved line layout/cache primitive to `render/layout.py` with compatibility wrapper in `frame_renderer.py`.
*   **In progress:** Moved render-plan/state computation to `render/frame_plan.py` with compatibility wrappers in `frame_renderer.py`.
*   **In progress:** Moved visible-line rendering pass orchestration to `render/line_pass.py` with compatibility wrappers in `frame_renderer.py`.
*   **In progress:** Moved unresolved overlap surrogate-timing clustering/assignment into `visual/surrogate_timing.py` with compatibility wrappers in `visual/refinement.py`.

**Action Plan:**
*   **Consolidate:** Keep singer-color policy in a dedicated module and avoid duplicating it in frame drawing code.
*   **Decouple:** Continue extracting render-time orchestration hotspots into focused helpers.

## 4. Visual Refinement Stability (Completed)
**Status (2026-02-17):**
*   **Complete:** Refactoring and test coverage improvements done. Logic extracted to `src/y2karaoke/core/visual/`.

## 5. Benchmark Execution & Validation (Completed)
**Status (2026-02-17):**
*   **Complete:** Full suite run passed. "Anti-Hero" and "Bohemian Rhapsody" fixes verified.

---

**Next Immediate Step:**
Continue render technical-debt cleanup by splitting remaining orchestration in `frame_renderer.py` into a focused coordinator model.
