# Critical Priorities & Roadmap

This document outlines the critical areas of the `y2karaoke` codebase and the immediate priorities for development and maintenance.

## 1. Visual Refinement Stability (High Priority)
**Context:** The project recently introduced "visual refinement" (`src/y2karaoke/core/refine_visual.py`) using computer vision to align lyrics with millisecond precision. This uses "Glyph-Interval Proximity" logic (gradient-of-distance, symmetric sweep).

**Critical Areas:**
*   `src/y2karaoke/core/refine_visual.py`: specifically `refine_word_timings_at_high_fps`.
*   `src/y2karaoke/vision/`: `ocr.py`, `roi.py`, `color.py`.

**Risks:**
*   **OCR Fragility:** Dependence on `paddleocr` or Apple Vision means results can vary by platform or video quality.
*   **Complexity:** The refinement logic is complex (cyclomatic complexity > 15) and hard to debug.

**Action Plan:**
*   **Test Coverage:** Ensure `refine_visual.py` has high test coverage, especially with mock video data.
*   **Benchmark:** Run benchmarks regularly (see section 2) to catch regressions in visual alignment.
*   **Refactor:** Break down `refine_word_timings_at_high_fps` into smaller, testable components.

## 2. Benchmark Execution & Validation
**Context:** A set of 12 songs is defined in `benchmarks/benchmark_songs.yaml` to measure alignment accuracy.

**Status (2026-02-16):**
*   **Fixed & Verified:** "Anti-Hero" 3.38s offset issue resolved. Auto-offset logic in `lyrics_helpers.py` AND `lyrics_whisper.py` now disabled for offsets between 0.3s and 5.0s. Confirmed start time ~5.39s (vs expected ~5.45s).
*   **Pending:** Investigation of "Bohemian Rhapsody" gold file mismatch (showing 106s error).

**Critical Areas:**
*   `tools/run_benchmark_suite.py`: The runner script.
*   `benchmarks/`: The configuration and gold standard files.

**Action Plan:**
*   **Investigate Bohemian Rhapsody:** Determine why gold file timings differ so drastically from generated results (likely radio edit vs full album version).
*   **Expand Verification:** Run full suite to ensure no other regressions.

## 3. Alignment Heuristics
**Context:** The core alignment logic (`src/y2karaoke/core/components/alignment/`) uses complex heuristics to score and correct timings.

**Status (2026-02-16):**
*   **Refactored:** `timing_evaluator.py` converted to strict public facade. Internal logic moved to `timing_evaluator_core.py`, `timing_evaluator_correction.py`, etc.
*   **Tests Fixed:** 250+ unit tests updated to import from implementation modules, resolving circular dependencies and private access issues.

**Critical Areas:**
*   `timing_evaluator_*.py`: Multiple files handling corrections, pauses, and scoring.

**Action Plan:**
*   **Document:** Add docstrings explaining the specific heuristic being applied (e.g., "why do we punish missing pauses?").
*   **Simplify:** Continue identifying complex logic blocks for further decomposition.

## 4. OCR Dependency Management
**Context:** The project optionally uses Apple Vision (macOS) or PaddleOCR.

**Risks:**
*   **Installation:** PaddleOCR can be difficult to install on some systems.
*   **Fallback:** Ensure the fallback mechanism in `ocr.py` is robust and logs clearly when it switches engines.

**Action Plan:**
*   **Verify Fallback:** Test the system in an environment without `vision` framework (e.g., Linux container) to ensure PaddleOCR path works.
*   **Error Handling:** Improve error messages when no OCR engine is available.

---

**Next Immediate Step:**
Investigate the "Bohemian Rhapsody" gold file mismatch. It seems the gold file timings (e.g., "Mama" at 18s) do not match the official video audio (where it starts at ~55s). This likely indicates a version mismatch (radio edit vs full) or a corrupted gold file.
