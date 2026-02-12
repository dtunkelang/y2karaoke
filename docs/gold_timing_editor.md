# Gold Timing Editor

This tool lets you create and maintain a manually corrected word-level timing gold set.

## Run

```bash
python tools/gold_timing_editor.py --host 127.0.0.1 --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

## Workflow

1. Generate a seed timing report with existing pipeline timing (`--timing-report ...json`).
2. In the editor, load that timing JSON by path.
3. Load local audio file by path.
4. Adjust word intervals:
- Drag word block to move entire interval.
- Drag left/right handle to adjust start/end.
- Keyboard arrows for 0.1s steps:
- `Left/Right`: move whole interval
- `Alt+Left/Right`: adjust start
- `Shift+Left/Right`: adjust end
- `Ctrl+Z` or `Cmd+Z`: undo
5. Save to one gold file per song (`*.gold.json`).

## Canonical Gold Format

`*.gold.json` is the canonical format.

```json
{
  "schema_version": "1.0",
  "title": "bad guy",
  "artist": "Billie Eilish",
  "audio_path": "/absolute/path/to/audio.wav",
  "source_timing_path": "/absolute/path/to/seed_timing_report.json",
  "lines": [
    {
      "line_index": 1,
      "text": "White shirt now red, my bloody nose",
      "start": 6.1,
      "end": 9.6,
      "words": [
        {"word_index": 1, "text": "White", "start": 6.1, "end": 6.6},
        {"word_index": 2, "text": "shirt", "start": 6.6, "end": 7.1}
      ]
    }
  ]
}
```

## Rules Enforced

- Word-level `start`/`end` are required.
- Values are snapped to `0.1` seconds.
- Overlaps are forbidden across all words.
- Gaps are allowed.
- Punctuation stays attached to words (no token splitting).
- Line timing is derived from first/last word.

## Input Support

The editor can load either:

- Existing pipeline timing report JSON (`*_timing_report.json`) with `lines[].words[].{text,start,end}`.
- Existing canonical `*.gold.json` files.
