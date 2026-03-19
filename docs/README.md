# Documentation Index

This directory is the canonical entry point for project documentation.

## Core Docs

- `../README.md`
  - product overview, installation, CLI usage, benchmarking basics
- `../ARCHITECTURE.md`
  - current system structure and major module boundaries
- `development.md`
  - local workflow, quality gates, CI lanes, artifact policy
- `tech_debt_backlog.md`
  - current debt assessment, completed cleanup, remaining priorities
- `../NEXT_SESSION_TODO.md`
  - lightweight handoff for the next coding session

## Pipeline Docs

- `pipelines/karaoke.md`
  - end-to-end generation flow
- `pipelines/lyrics.md`
  - lyrics acquisition, quality, and sync ownership
- `pipelines/whisper.md`
  - Whisper transcription, mapping, and correction ownership
- `pipelines/alignment_policy.md`
  - centralized decision-policy layer for alignment heuristics

## Tooling Docs

- `curated_clips.md`
  - workflow for adding and running short curated benchmark clips
- `gold_timing_editor.md`
  - manual gold-timing refinement workflow
- `karaoke_visual_bootstrap.md`
  - visual bootstrap workflow for generating benchmark timing data
- `check_visual_suitability.md`
  - preflight suitability scoring for karaoke-video bootstrap sources

## Documentation Rules

- Keep status and workflow guidance here, not in ad hoc root notes.
- Prefer stable ownership docs over change logs.
- Update `ARCHITECTURE.md`, `development.md`, and `tech_debt_backlog.md` when structural changes land.
