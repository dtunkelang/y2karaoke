# Karaoke Pipeline

Primary entrypoint:
- `src/y2karaoke/core/karaoke.py` (`KaraokeGenerator`)

Execution flow:
1. `generate()` delegates to `core/karaoke_generate.py`.
2. Media preparation (`_prepare_media`) downloads/loads audio/video and separates stems.
3. Lyrics pipeline (`_get_lyrics`) resolves lines + quality report.
4. Audio pipeline (`_process_audio_track`) applies effects and optional break shortening.
5. Timing/post-processing applies scaling, break edits, splash offset, ordering fixes.
6. Rendering writes final output (or skips rendering with `--no-render`).

Helper modules:
- `src/y2karaoke/core/karaoke_generate.py`: orchestration body.
- `src/y2karaoke/core/karaoke_audio_helpers.py`: outro insertion + break-shortening implementation.
- `src/y2karaoke/core/karaoke_utils.py`: timing/quality utility helpers.
