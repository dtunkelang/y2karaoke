# Y2Karaoke Architecture

This document describes the current (February 2026) structure of the y2karaoke codebase.

## Entry Points

- CLI: `src/y2karaoke/cli.py`
- CLI helpers: `src/y2karaoke/cli_commands.py`
- Main orchestrator: `src/y2karaoke/core/karaoke.py` (`KaraokeGenerator`)

## High-Level Generation Flow

`y2karaoke generate ...` executes this pipeline:

1. Identify track metadata (artist/title/url/duration) via `pipeline/identify`.
2. Download audio (and optional video backgrounds) via `pipeline/audio`.
3. Separate vocals/instrumental stems.
4. Resolve lyrics + timing via `pipeline/lyrics` (optionally with Whisper refinement).
5. Apply audio transforms (key/tempo) and optional break shortening.
6. Post-process timings (offsets, scaling, ordering, splash/outro behavior).
7. Render the karaoke video via `pipeline/render` (or skip with `--no-render`).

Primary orchestration implementation lives in:
- `src/y2karaoke/core/karaoke_generate.py`
- `src/y2karaoke/core/karaoke_audio_helpers.py`
- `src/y2karaoke/core/karaoke_utils.py`

## Pipeline Facades

The `src/y2karaoke/pipeline/` package provides stable subsystem interfaces:

- `pipeline/identify`: track discovery, title parsing, metadata quality checks.
- `pipeline/lyrics`: provider fetch, LRC parsing, quality scoring, Whisper-enabled alignment paths.
- `pipeline/audio`: media download, source separation, audio effects.
- `pipeline/alignment`: timing evaluators and alignment reporting helpers.
- `pipeline/render`: frame rendering, background processing, and video writing entrypoints.

Internals for these subsystems live mostly under `src/y2karaoke/core/components/`.

## Whisper and Timing Alignment

Whisper-related components are grouped under:
- `src/y2karaoke/core/components/whisper/`

Key responsibilities:
- transcription caching and invocation,
- LRC-to-Whisper mapping,
- DTW/phonetic alignment,
- timing corrections and fallback strategies.

The compatibility layer is `whisper_integration.py`; orchestration is split into focused modules such as `whisper_integration_pipeline.py` and mapping/alignment helpers.

## Rendering and Visual Processing

Rendering stack is in:
- `src/y2karaoke/core/components/render/`

Notable modules:
- `frame_renderer.py`
- `lyrics_renderer.py`
- `backgrounds.py` / `backgrounds_static.py`
- `video_writer.py`

Visual bootstrap/refinement utilities live in:
- `src/y2karaoke/core/visual/`
- `src/y2karaoke/core/refine_visual.py`

## Configuration and Environment

- Global settings: `src/y2karaoke/config.py`
- Cache default: `~/.cache/karaoke`
- Python version: 3.12 (`pyproject.toml`)

## Testing and Quality Gates

- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- End-to-end tests: `tests/e2e/`

Primary local quality gate:
- `make check`

This runs dependency checks, formatting/lint/type checks, fast unit tests, performance smoke checks, guardrail checks, and benchmark manifest validation.
