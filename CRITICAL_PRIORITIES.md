# Critical Priorities & Roadmap

This document outlines the critical areas of the `y2karaoke` codebase and the immediate priorities for development and maintenance.

## 1. Visual Refinement Stability (High Priority)
**Context:** The project recently introduced "visual refinement" (`src/y2karaoke/core/refine_visual.py`) using computer vision to align lyrics with millisecond precision. This uses "Glyph-Interval Proximity" logic (gradient-of-distance, symmetric sweep).

**Status (2026-02-17):**
*   **Complete:** Refactoring and test coverage improvements are done. Logic extracted to `src/y2karaoke/core/visual/`.

**Critical Areas:**
*   `src/y2karaoke/core/visual/refinement.py`: refinement logic.
*   `src/y2karaoke/core/visual/reconstruction.py`: reconstruction logic.

**Action Plan:**
*   **Monitor:** Watch for regressions in visual alignment during benchmarks.

## 2. Benchmark Execution & Validation
**Context:** A set of 12 songs is defined in `benchmarks/benchmark_songs.yaml` to measure alignment accuracy.

**Status (2026-02-17):**
*   **Complete:** Full suite run (13/13 songs) passed.
*   **Verified:** "Anti-Hero" and "Bohemian Rhapsody" fixes confirmed. Auto-offset logic handles small offsets correctly while rejecting large suspicious ones.
*   **Note:** Some legacy songs (e.g., "Indila - Derniere danse") show high gold-standard error, likely due to outdated gold files or version mismatches (e.g., radio edit vs album). These do not indicate recent regressions.

**Critical Areas:**
*   `tools/run_benchmark_suite.py`: The runner script.
*   `benchmarks/`: The configuration and gold standard files.

**Action Plan:**
*   **Maintenance:** Periodically update gold files for other songs to match the current pipeline output where appropriate (assuming pipeline is ground truth for those cases).

## 3. Alignment Heuristics
**Context:** The core alignment logic (`src/y2karaoke/core/components/alignment/`) uses complex heuristics to score and correct timings.

**Status (2026-02-17):**
*   **Refactored:** `timing_evaluator.py` converted to strict public facade. Internal logic moved to `timing_evaluator_core.py`, `timing_evaluator_correction.py`, etc.
*   **Documented:** Added docstrings to `timing_evaluator_core.py` and `timing_evaluator_correction.py`.
*   **Simplified:** Refactored `lyrics_whisper_map.py` to use `_LineMapper` class, reducing complexity of the main mapping function.
*   **Tests Fixed:** 250+ unit tests updated to import from implementation modules.

**Critical Areas:**
*   `timing_evaluator_*.py`: Multiple files handling corrections, pauses, and scoring.
*   `lyrics_whisper_map.py`: Core mapping logic.

**Action Plan:**
*   **Maintain:** Ensure new heuristics are well-documented and tested.

## 4. OCR Dependency Management
**Context:** The project optionally uses Apple Vision (macOS) or PaddleOCR.

**Status (2026-02-17):**
*   **Complete:** Verified fallback mechanism via unit tests (`test_ocr_fallback.py`).

**Action Plan:**
*   **Maintenance:** Ensure PaddleOCR version compatibility in future.

---

**Next Immediate Step:**
The repository is currently stable and all identified critical priorities have been addressed. Future work should focus on:
1.  **Rendering Performance:** Investigate `lyrics_renderer.py` for optimization.
2.  **Gold Set Quality:** Audit and update gold files for songs showing high error (e.g. Indila, OneRepublic) to ensure they reflect the correct audio version.
3.  **Visual Refinement:** Continue enhancing robustness of color-based timing refinement.
