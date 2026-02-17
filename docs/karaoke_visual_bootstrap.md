# Karaoke Visual Bootstrap Tool

The `tools/bootstrap_gold_from_karaoke.py` tool is used to generate or refine word-level "gold" timings by visually analyzing existing karaoke videos from YouTube using a **Glyph-Core Departure** strategy.

## Overview

Instead of relying on audio alignment alone, this tool uses computer vision and OCR to "watch" a karaoke video and detect exactly when each word begins its visual transition. By focusing on the inner glyph fill and ignoring background noise, it provides a robust ground truth for benchmarking the alignment pipeline.

## Vision-Based Extraction Algorithm

The tool uses a multi-stage process to extract timings with **0.05s snapping precision**.

### 1. Line Identification (OCR)
The process starts by sampling the video at **2.0 FPS** using Apple Vision OCR (on macOS) or PaddleOCR:
- **ROI Focus:** Extraction is focused on a generous region where lyrics typically appear.
- **Line Reconstruction:** Individual words are grouped into logical lines based on their vertical Y-coordinate and temporal appearance.
- **Normalization:** Lyrics are normalized (lower-cased, punctuation stripped, hyphens resolved) to facilitate deduplication and merging of multi-line blocks.

### 2. Native-FPS Refinement
Once lines are identified, the tool scans the relevant line windows at the source video's native FPS:
- **Glyph Masking:** For every word, a dynamic mask is created to isolate the "text core" pixels that deviate significantly from the local background.
- **Stability Anchoring:** The tool identifies the most stable "Bright" period (Unselected text) and the first stable "Dark" period (Highlighted text) within the word's visible interval to establish $C_{initial}$ and $C_{final}$ prototypes.
- **Departure-Onset Detection:** The word's **Start Time** is triggered at the exact frame where the glyph color consistently departs from the noise floor of its initial stable state.
- **Midpoint Termination:** The word's **End Time** is triggered when its color becomes mathematically closer to the final highlighted state than the initial state.

### 3. Logical Sequence Enforcement
Reconstructed timings are subject to strict invariants:
- **Vertical Priority:** Within simultaneous multi-line blocks, top-to-bottom vertical order is strictly preserved.
- **Temporal Monotonicity:** A line or word $N+1$ can never start before Line/Word $N$ has finished.
- **Duration Constraints:** Words are capped at **0.8s** for natural flow, with a minimum duration of **0.10s**.
- **Gap Filling:** Any words missed by visual triggers are filled using linear interpolation within the detected line duration.

### Output Confidence Signals
- Each word includes a `confidence` score in `[0, 1]` based on transition strength and trigger quality.
- Each line includes an aggregate `confidence` score (mean of its words).
- The top-level JSON includes `visual_suitability` metrics and `candidate_url` for traceability.

## Usage

```bash
python tools/bootstrap_gold_from_karaoke.py 
  --artist "Artist Name" 
  --title "Song Title" 
  --show-candidates
  --output path/to/refined.gold.json
```

Optional arguments:
- `--candidate-url`: Specify a YouTube URL directly (skips candidate search).
- `--visual-fps`: Frame rate for initial OCR sampling (default: 2.0).
- `--work-dir`: Cache/work directory for downloads and OCR artifacts (default: `.cache/karaoke_bootstrap`).
- `--raw-ocr-cache-version`: Version stamp for OCR frame cache keys (default: `2`); bump to invalidate stale OCR-frame caches after logic changes.
- `--report-json`: Write a structured run report with candidate rankings and selected metrics.
- `--max-candidates`: Max YouTube candidates to evaluate when auto-searching (default: 5).
- `--suitability-fps`: Sampling rate used for suitability scoring (default: 1.0).
- `--min-detectability`: Minimum detectability score required by quality gate (default: 0.45).
- `--min-word-level-score`: Minimum word-level score required by quality gate (default: 0.15).
- `--allow-low-suitability`: Bypass suitability gates (use with caution).
- `--show-candidates`: Print ranked candidates with suitability metrics.
- `--strict-sequential`: Enforce that lines appearing together are processed one-by-one.

Runtime note:
- When a candidate is selected from evaluated rankings, the tool attempts to extract audio from the already-downloaded candidate video before falling back to a direct audio download.
