# Gold Timing Editor

The Gold Timing Editor is a local web application for manual refinement of word-level karaoke timings.

## Features

- **Interactive Timeline**: Drag words to shift their timing or use handles to resize intervals.
- **Line Mode**: Shift entire lines while preserving relative word durations and internal gaps.
- **High-Performance Playback**: Uses `requestAnimationFrame` for perfectly synchronized visual feedback at 60 FPS.
- **Sticky Controls**: The play button and edit modes stay pinned to the top of the viewport during scrolling.
- **Autoscroll**: The timeline automatically follows the current line during playback.
- **LRC Integration**: Synchronizes with LRC files to provide a robust starting point.

## Standard Workflow

1. **Bootstrap**: Generate a candidate gold set using `tools/bootstrap_gold_from_karaoke.py`.
2. **Launch Editor**: Start the editor server:
   ```bash
   python tools/gold_timing_editor.py
   ```
3. **Refine**: Open the provided URL in Chrome.
4. **Save**: Click "Save Gold JSON" to commit your changes back to the benchmark set.

## Hotkeys

- **Arrow Left/Right**: Move selected word/line by 0.1s.
- **Shift + Arrow**: Adjust the end timestamp only.
- **Alt + Arrow**: Adjust the start timestamp only.
- **Ctrl/Cmd + Z**: Undo the last edit.
- **Space**: Play/Pause.

## URL Parameters

You can pre-load the editor by passing query parameters:
- `timing`: Absolute path to the timing JSON.
- `audio`: Absolute path to the audio file (wav/mp3/m4a).
- `save`: Absolute path where changes should be saved.
