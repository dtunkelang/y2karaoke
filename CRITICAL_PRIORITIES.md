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
*   Full suite run shows large systematic offset (~21s weighted mean).
*   "Anti-Hero" (priority song) shows **3.38s offset** against gold standard.
*   "Visual Refinement" logic *is* running, but an auto-offset of ~3.78s might be applying incorrectly.

**Critical Areas:**
*   `tools/run_benchmark_suite.py`: The runner script.
*   `benchmarks/`: The configuration and gold standard files.

**Action Plan:**
*   **Investigate Offset:** Focus on `y2karaoke.core.components.lyrics.lyrics_whisper` and why it applies a `+3.78s` offset that results in a 3.4s error.
*   **Analyze Results:** Compare `anti_hero` results specifically.

## 3. Alignment Heuristics
**Context:** The core alignment logic (`src/y2karaoke/core/components/alignment/`) uses complex heuristics to score and correct timings.

**Critical Areas:**
*   `timing_evaluator_*.py`: Multiple files handling corrections, pauses, and scoring.

**Action Plan:**
*   **Simplify:** Review the split of logic across multiple `timing_evaluator` files. Merge or clarify responsibilities where possible.
*   **Document:** Add docstrings explaining the specific heuristic being applied (e.g., "why do we punish missing pauses?").

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
Run the benchmark suite to baseline the current performance, especially for "Anti-Hero" and "Bohemian Rhapsody".

```bash
python tools/run_benchmark_suite.py
```
