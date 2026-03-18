# Karaoke Pipeline

Last updated: 2026-03-18

Primary entrypoint:
- `src/y2karaoke/core/karaoke.py` (`KaraokeGenerator`)

## Flow

1. `KaraokeGenerator.generate()` delegates to `src/y2karaoke/core/karaoke_generate.py`.
2. Media preparation downloads or loads source media, extracts audio, and separates stems when needed.
3. The lyrics pipeline resolves line text, timing, and a quality report.
4. Audio processing applies optional effects, key/tempo changes, and break shortening.
5. Timing post-processing applies offsets, scaling, ordering fixes, and splash/outro handling.
6. Rendering writes the final video unless `--no-render` is set.

## Module Ownership

- `src/y2karaoke/core/karaoke_generate.py`
  - top-level orchestration for media, lyrics, timing post-processing, and rendering
- `src/y2karaoke/core/karaoke_audio_helpers.py`
  - audio-specific helpers such as outro insertion and break shortening
- `src/y2karaoke/core/karaoke_utils.py`
  - shared timing, line-ordering, and quality utility helpers
- `src/y2karaoke/core/components/lyrics/`
  - lyrics acquisition, sync, Whisper-assisted timing, and quality policy
- `src/y2karaoke/core/rendering/`
  - frame generation, styling, and final output assembly

## Related Docs

- `docs/pipelines/lyrics.md`
- `docs/pipelines/whisper.md`
- `ARCHITECTURE.md`
