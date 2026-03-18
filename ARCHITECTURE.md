# Y2Karaoke Architecture

Last updated: 2026-03-18

This document describes the current structure of the codebase after the recent lyrics, Whisper, and sync cleanup work.

## Top-Level Entry Points

- CLI: `src/y2karaoke/cli.py`
- CLI command wiring: `src/y2karaoke/cli_commands.py`
- Main orchestration object: `src/y2karaoke/core/karaoke.py` (`KaraokeGenerator`)
- Stable pipeline facades: `src/y2karaoke/pipeline/`

## End-to-End Flow

`y2karaoke generate ...` runs this high-level sequence:

1. Resolve track metadata and source URLs through `pipeline/identify`.
2. Download media and prepare audio through `pipeline/audio`.
3. Separate vocals/instrumental stems.
4. Resolve lyrics and timing through `pipeline/lyrics`.
5. Apply key/tempo changes and timing-side postprocessing.
6. Render the final video through `pipeline/render`.

Main generation orchestration lives in:
- `src/y2karaoke/core/karaoke_generate.py`
- `src/y2karaoke/core/karaoke_audio_helpers.py`
- `src/y2karaoke/core/karaoke_utils.py`

## Stable Facade Layer

`src/y2karaoke/pipeline/` is the boundary that higher-level orchestration should target:

- `pipeline/identify`
- `pipeline/audio`
- `pipeline/lyrics`
- `pipeline/alignment`
- `pipeline/render`

Most implementation detail sits under `src/y2karaoke/core/components/`.

## Lyrics and Sync Structure

Lyrics-related logic is centered in:
- `src/y2karaoke/core/components/lyrics/api.py`
- `src/y2karaoke/core/components/lyrics/lyrics_whisper.py`
- `src/y2karaoke/core/components/lyrics/lyrics_whisper_pipeline.py`
- `src/y2karaoke/core/components/lyrics/lyrics_whisper_quality.py`
- `src/y2karaoke/core/components/lyrics/sync.py`
- `src/y2karaoke/core/components/lyrics/sync_pipeline.py`

Recent ownership splits:
- `lyrics_source_routing.py`
  - provider disagreement routing and source-selection policy
- `lyrics_offset_quality.py`
  - offset detection and quality-scoring policy
- `lyrics_quality_sources.py`
  - source acquisition, trust policy, Genius fallback coordination
- `lyrics_quality_alignment.py`
  - Whisper-assisted alignment and fallback application
- `lyrics_quality_tail_guardrail.py`
  - tail-completeness guardrail and duration clipping
- `runtime_config.py`
  - typed runtime config for lyrics-provider/tolerance behavior
- `sync_cache.py`
  - sync cache and disk-cache behavior

The main module files remain compatibility surfaces for older imports and tests, but the major policy and orchestration concerns are now split into narrower ownership modules.

## Whisper Structure

Whisper-related logic lives under:
- `src/y2karaoke/core/components/whisper/`

Primary compatibility and entry modules:
- `whisper_integration.py`
- `whisper_mapping.py`
- `whisper_mapping_post.py`

Major ownership areas:
- transcription and retry
- DTW / phonetic mapping
- line correction passes
- segment assignment
- fallback / forced alignment

Recent structural changes:
- `whisper_runtime_config.py`
  - typed runtime config for profile-driven heuristics
- `whisper_mapping_runtime_config.py`
  - typed config for segment-assignment and trace controls
- `whisper_integration_align.py`
  - still the main aligner facade, but substantially reduced by moving dense helper groups outward
- `whisper_integration_align_heuristics.py`
  - line-shape / heuristic extension helpers
- `whisper_integration_align_corrections.py`
  - correction-pass orchestration helpers
- `whisper_segment_assignments.py`
  - segment-level text-overlap assignment logic
- `whisper_block_dtw_assignments.py`
  - block-scoped syllable-DTW assignment logic
- `whisper_assignment_trace.py`
  - shared trace serialization helpers
- `whisper_mapping_post_core.py`
  - implementation side of mapping postprocess behavior, leaving `whisper_mapping_post.py` as a thinner compatibility facade

## Rendering and Visual Bootstrap

Rendering stack:
- `src/y2karaoke/core/components/render/`

Visual bootstrap and refinement:
- `src/y2karaoke/core/visual/`
- `src/y2karaoke/core/refine_visual.py`

Bootstrap helpers are split by concern:
- `bootstrap_candidates.py`
- `bootstrap_selection.py`
- `bootstrap_media.py`
- `bootstrap_ocr.py`
- `bootstrap_postprocess.py`
- `bootstrap_runtime.py`

## Configuration and Runtime State

- Global config: `src/y2karaoke/config.py`
- Default cache root: `~/.cache/karaoke`
- Python version target: 3.12

The recent cleanup moved several formerly ambient env/state controls to typed config or context-local state:
- lyrics runtime config
- Whisper runtime config
- Whisper mapping runtime config
- `ContextVar`-backed default hook/state flows for lyrics and sync

## Tests and Quality Gates

- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- End-to-end tests: `tests/e2e/`

Primary local gate:
- `make check`

This gate is expected to cover formatting, lint, typing, fast tests, perf smoke, and custom guardrails such as file-size / complexity limits.
