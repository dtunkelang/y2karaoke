# Gold Timing Editor

The Gold Timing Editor is a local web application for manual refinement of word-level karaoke timings.

## Features

- **Interactive Timeline**: Drag words to shift their timing or use handles to resize intervals.
- **Line Mode**: Shift entire lines while preserving relative word durations and internal gaps.
- **Onset Anchor Snap**: Audio is analyzed for likely onset peaks; snap words/lines to nearby anchors for fast micro-corrections.
- **Anchor Jump Navigation**: Jump directly to the previous/next detected anchor to speed repetitive retiming.
- **Session Telemetry**: Live session stats report edit throughput and correction-loop behavior (edits/min, undo rate, snap/jump usage).
- **High-Performance Playback**: Uses `requestAnimationFrame` for perfectly synchronized visual feedback at 60 FPS.
- **Sticky Controls**: The play button and edit modes stay pinned to the top of the viewport during scrolling.
- **Autoscroll**: Playback and seek actions auto-scroll so the current lyric line stays visible near the top of the timeline window.
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

- **Arrow Left/Right**: Move selected word/line by 0.05s.
- **Ctrl/Cmd + Arrow**: Coarse nudge by 0.2s.
- **Shift + Arrow**: Adjust the end timestamp only.
- **Alt + Arrow**: Adjust the start timestamp only.
- **S**: Snap selected word start (or line start in line mode) to nearest audio anchor.
- **E**: Snap selected word end to nearest audio anchor.
- **A**: Snap selected word or line as a whole to nearest audio anchor.
- **[ / ]**: Jump selected word/line to previous/next audio anchor.
- **Shift + [ / ]**: Jump selected word end to previous/next anchor.
- **Alt + [ / ]**: Jump selected word start to previous/next anchor.
- **Ctrl/Cmd + Z**: Undo the last edit.
- **Space**: Play/Pause.

## URL Parameters

You can pre-load the editor by passing query parameters:
- `timing`: Absolute path to the timing JSON.
- `audio`: Absolute path to the audio file (wav/mp3/m4a).
- `save`: Absolute path where changes should be saved.
