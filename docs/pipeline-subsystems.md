# Pipeline Subsystems

This project uses a **subsystem facade** layout under `src/y2karaoke/pipeline/`.

## Goal

Keep orchestration boundaries explicit while allowing `core/` modules to remain implementation-focused.

## Core Component Layer

Component-organized core facades now live under `src/y2karaoke/core/components/`:

- `core/components/identify`
- `core/components/lyrics`
- `core/components/audio`
- `core/components/alignment`
- `core/components/render`

These facades provide stable component-level imports while implementation
modules live under each component package. Remaining top-level `core/*.py`
modules are cross-cutting/shared utilities or orchestration entrypoints.

## Subsystems

- `pipeline/identify`
  Exposes track identification entrypoints (`TrackIdentifier`, `TrackInfo`).
- `pipeline/lyrics`
  Exposes lyrics acquisition and quality-aware timing entrypoints.
- `pipeline/audio`
  Exposes media download, vocal separation, and audio transform entrypoints.
- `pipeline/alignment`
  Exposes timing evaluation and Whisper alignment entrypoints.
- `pipeline/render`
  Exposes rendering entrypoints (background segmentation and final video render).

## Dependency Rules

- `pipeline/*` is a facade layer over `core/*`.
  - `pipeline/*` may import `y2karaoke.core.*`.
  - `pipeline/*` should not contain heavy business logic.
- `core/*` should not import `pipeline/*`.
  - Current explicit orchestration exception: `core/karaoke.py` (main pipeline coordinator).
- `cli.py` should use `pipeline/*` facades for orchestration-level flows.

## Why This Helps

- Clear ownership per subsystem.
- Fewer brittle tests that patch deep internals.
- Safer refactors: implementation modules can change while facade contracts stay stable.

## Enforcement

`tests/test_pipeline_architecture.py` enforces these rules in CI.
