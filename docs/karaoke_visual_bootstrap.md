# Karaoke Visual Bootstrap Tool

The `tools/bootstrap_gold_from_karaoke.py` tool is used to generate or refine high-precision word-level "gold" timings by visually analyzing existing karaoke videos from YouTube.

## Overview

Instead of relying on audio alignment alone, this tool uses computer vision and OCR to "watch" a karaoke video and detect exactly when each word is highlighted. This provides a high-quality ground truth for benchmarking the alignment pipeline.

## Vision-Based Extraction Algorithm

The tool uses a two-pass process to extract timings with **0.05s precision** (20 FPS).

### 1. Color Inference Pass
The tool first discovers the specific color scheme used in the video:
- Samples frames throughout the video looking for centered text.
- Extracts foreground pixels and uses **K-Means Clustering** to find the two dominant colors.
- Identifies the **Unhighlighted** (neutral/bright) and **Highlighted** (vibrant) states.

### 2. OCR Transition Tracking Pass
The tool processes the video at 20 frames per second:
- **Continuous OCR:** Runs Tesseract OCR on every frame to find word text and exact bounding box coordinates (`x, y, w, h`).
- **State Machine:** Tracks each word through three states based on pixel ratios inside its box:
    - `unselected`: Mostly unhighlighted color.
    - `mixed`: Transitioning (captures the moment the highlight sweep hits the word).
    - `selected`: Mostly highlighted color.
- **Timing Capture:**
    - **Start Time:** The moment a word transitions from `unselected` to `mixed` or `selected`.
    - **End Time:** The moment a word becomes fully `selected`.

### 3. Positional Deduplication
To handle OCR noise and "blinking" detections:
- Words are tracked by their screen coordinates.
- If a word at the same position is detected multiple times, the tool merges the intervals using the **earliest start** and **latest end** times.

### 4. Audio-Visual Alignment
Once visual timings are extracted, they are mapped back to the audio timeline:
- Anchors the visual timings to the confirmed line-start timestamps from the source LRC or Gold file.
- Enforces a **0.05s minimum word duration**.
- Ensures monotonic order and non-overlapping word intervals.

## Usage

```bash
python tools/bootstrap_gold_from_karaoke.py 
  --artist "Artist Name" 
  --title "Song Title" 
  --gold-in path/to/existing.gold.json 
  --output path/to/refined.gold.json
```

Optional arguments:
- `--candidate-url`: Specify a YouTube URL directly instead of searching.
- `--visual-fps`: Defaults to 20.0 for 0.05s precision.
- `--work-dir`: Directory for temporary video/audio downloads.
