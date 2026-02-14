# Karaoke Visual Bootstrap Tool

The `tools/bootstrap_gold_from_karaoke.py` tool is used to generate or refine high-precision word-level "gold" timings by visually analyzing existing karaoke videos from YouTube using a **Visual Evidence + Fixed Anchor** hybrid strategy.

## Overview

Instead of relying on audio alignment alone, this tool uses computer vision and OCR to "watch" a karaoke video and detect exactly when each word is highlighted. By anchoring these visual cues to high-quality LRC line starts, it provides a robust ground truth for benchmarking the alignment pipeline.

## Vision-Based Extraction Algorithm

The tool uses a multi-stage process to extract timings with **0.05s precision**.

### 1. Anchor Definition (LRC)
The process starts with an LRC file, which serves as the **Ground Truth for line starts**. We trust the LRC's vertical alignment (when a line begins) but use the video to solve the horizontal alignment (how long each word lasts).

### 2. Intelligent Visual Extraction (PaddleOCR)
The tool processes video frames at a configurable rate (typically 2.0–5.0 FPS):
- **Windowing:** To optimize performance and reduce noise, the tool only scans "active windows" (e.g., 2s before to 3s after each LRC line timestamp).
- **ROI Focus:** Extraction is focused on the lower-middle region of the frame where lyrics typically appear.
- **PaddleOCR:** Uses PaddleOCR for its high resilience to drop shadows, textured backgrounds, and "glow" effects common in karaoke videos.

### 3. State Classification (The "Highlight" Rule)
For every word detected by OCR, the tool analyzes the pixels within its bounding box:
- **Color Clustering:** K-Means identifies the video's specific "Text Palette" (e.g., White for unselected, Green for selected, Black for borders).
- **Classification:** A word is marked as **Selected** if its internal pixel distribution matches the highlight color. The tool tracks the "Leader"—the furthest word in the sequence currently in a selected state.

### 4. Global Monotonic Alignment
The entire song is treated as two parallel sequences: the **Target Words** (LRC) and the **Visual Events** (OCR highlights):
- **Sequential Matching:** Target words are matched to visual events one-by-one. Once a visual event is matched, it is consumed.
- **Lookahead:** The tool searches forward in the video sequence for the next best text match, limited to a ~10s window past the expected LRC time to prevent "runaway" jumps.

### 5. Temporal Fitting & Consistency
Reconstructed timings are subject to strict constraints:
- **LRC Anchoring:** Word durations and relative gaps are derived from the video, but the entire line block is shifted so the first word aligns **exactly** with the LRC timestamp.
- **Sequential Consistency:** Ensures that Line $N+1$ never starts before Line $N$ has ended.
- **Duration Floors:** If a word was missed visually, it falls back to a proportional heuristic (e.g., 0.4s per word) to ensure a minimum duration.
- **Snapping:** All timestamps are snapped to 0.05s increments.

## Usage

```bash
python tools/bootstrap_gold_from_karaoke.py 
  --artist "Artist Name" 
  --title "Song Title" 
  --lrc-in path/to/lyrics.lrc
  --output path/to/refined.gold.json
```

Optional arguments:
- `--candidate-url`: Specify a YouTube URL directly instead of searching.
- `--visual-fps`: Frame rate for visual analysis (e.g., 2.0 or 5.0).
- `--gold-in`: Use an existing gold JSON as the structure and timing base.
- `--work-dir`: Directory for temporary video/audio downloads and caching.
