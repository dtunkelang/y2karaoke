# Critical Priorities & Roadmap

This document outlines the critical areas of the `y2karaoke` codebase and the immediate priorities for development and maintenance.

## 1. Rendering Performance (High Priority)
**Context:** Video generation is CPU-intensive. The current `frame_renderer.py` recalculates text layouts (bounding boxes, widths) for every single frame (e.g., 30 times/sec), even though lyrics layout is static for seconds at a time.

**Critical Areas:**
*   `src/y2karaoke/core/components/render/frame_renderer.py`: Main rendering loop.
*   `src/y2karaoke/core/components/render/video_writer.py`: MoviePy integration.

**Risks:**
*   **Slow Generation:** `generate` command takes significantly longer than necessary.
*   **Resource Usage:** High CPU usage limits parallel processing capacity.

**Action Plan:**
*   **Cache Layouts:** Implement a caching mechanism in `render_frame` (or a helper class) to store word positions and widths per `Line`. Invalidate only when visible lines change.
*   **Refactor:** Move layout logic out of the hot loop (`make_frame`).

## 2. Gold Set Quality Automation
**Context:** Updating gold files manually (copying timing reports) is error-prone and tedious. Some songs have large errors due to version mismatches (e.g., radio edit vs album).

**Critical Areas:**
*   `tools/run_benchmark_suite.py`: The benchmark runner.

**Action Plan:**
*   **Add --rebaseline:** Implement a CLI flag to automatically update the gold file for a specific song (or all successful songs) using the current run's output.
*   **Audit:** Use this tool to fix the gold files for songs like "Indila - Derniere danse" that have systematic offsets.

## 3. Technical Debt & Cleanup
**Context:** `lyrics_renderer.py` appears to be a placeholder or legacy module with unused parameters. `frame_renderer.py` mixes timing logic (`_get_lines_to_display`) with pixel drawing.

**Critical Areas:**
*   `src/y2karaoke/core/components/render/lyrics_renderer.py`: Unused logic?
*   `src/y2karaoke/core/components/render/frame_renderer.py`: High complexity.

**Action Plan:**
*   **Consolidate:** Merge useful logic from `lyrics_renderer.py` into `frame_renderer.py` or delete if redundant.
*   **Decouple:** Extract timing/visibility logic into a `LyricTimeline` model to separate "what to show" from "how to draw it".

## 4. Visual Refinement Stability (Completed)
**Status (2026-02-17):**
*   **Complete:** Refactoring and test coverage improvements done. Logic extracted to `src/y2karaoke/core/visual/`.

## 5. Benchmark Execution & Validation (Completed)
**Status (2026-02-17):**
*   **Complete:** Full suite run passed. "Anti-Hero" and "Bohemian Rhapsody" fixes verified.

---

**Next Immediate Step:**
Implement layout caching in `frame_renderer.py` to optimize rendering performance.
