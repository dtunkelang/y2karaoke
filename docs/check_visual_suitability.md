# Visual Suitability Checker

The `tools/check_visual_suitability.py` tool evaluates whether a karaoke video is suitable for automated "gold" timing bootstrapping. It analyzes the video for clear word-level highlighting and high OCR reliability.

## Overview

Automated bootstrapping depends on being able to visually "see" when individual words are highlighted. If a video only highlights one line at a time, or if the text contrast is too low for OCR, the resulting timings will be poor. This tool provides a pre-flight check to ensure high-quality results.

## Key Metrics

The tool produces a **Detectability Score** (0.0 to 1.0) based on:

1.  **Word-Level Score (30% weight):**
    - Measures the frequency of frames where some words in a line are highlighted while others are not.
    - A high score indicates true word-level karaoke.
    - A low score suggests line-level or non-standard highlighting.
2.  **OCR Avg Confidence (70% weight):**
    - The raw confidence score from the PaddleOCR engine.
    - Detects issues like low contrast, blurry text, or heavy background noise.

State classification is computed from **foreground text pixels** inside each OCR
word box (not the full box). This avoids dark/bright video backgrounds
dominating the selected/unselected decision.

## Usage

You can run the tool against a local file or a YouTube URL:

```bash
# Check a YouTube video
python tools/check_visual_suitability.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Check a local file
python tools/check_visual_suitability.py path/to/video.mp4

# Run with debug output to see detected lyrics and states
python tools/check_visual_suitability.py "URL" --debug-lyrics
```

## Features

### 1. Automated Caching
To avoid redundant compute-intensive OCR, the tool caches results in `.cache/visual_suitability/`:
- **ROI:** The detected lyric region.
- **Colors:** The inferred unselected and selected (highlight) colors.
- **Raw Frames:** The full OCR results and word states for the entire video.

### 2. Debug Mode (`--debug-lyrics`)
Prints a per-frame breakdown of what the "robot" sees:
- `[Word(S:0.99)]`: **S**elected with 99% confidence.
- `[Word(U:1.00)]`: **U**nselected with 100% confidence.
- `[Word(M:0.95)]`: **M**ixed (partially highlighted).

## Quality Interpretation

- **EXCELLENT (>0.8):** Perfect for automated word-level bootstrapping.
- **GOOD (>0.5):** Reliable, though may have minor OCR noise.
- **FAIR (>0.3):** Likely only useful for line-level alignment.
- **POOR (≤0.3):** Unsuitable; manual timing or a different source is recommended.
