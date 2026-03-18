# Tech Debt Backlog

Last updated: 2026-03-18

This file tracks the current debt assessment, not a historical log of every cleanup commit.

## Current Assessment

The highest-value cleanup in the lyrics / Whisper / sync slice is mostly done.

Completed structural improvements:
- explicit runtime-config boundaries for lyrics and Whisper heuristics
- `ContextVar`-backed hook/state defaults instead of mutable process-global test state
- narrower public facade surfaces for lyrics and Whisper entry modules
- `lyrics_whisper_quality.py` split into source / alignment / guardrail modules
- `whisper_blocks.py` split into segment-assignment and block-DTW modules
- `sync.py` cache subsystem moved into `sync_cache.py`
- large Whisper aligner orchestration decomposed into focused helper modules

## What Is Still Worth Watching

1. Prevent boundary regression
- Keep orchestration modules from accreting policy and helper logic again.
- Treat `tools/quality_guardrails.py` as the primary enforcement point.

2. Finish config centralization only where it still improves ownership
- Some isolated env-driven helper and trace toggles remain.
- Further cleanup should be justified by readability or safety, not completeness for its own sake.

3. Utility-heavy modules
- `src/y2karaoke/core/components/lyrics/helpers.py`
- `src/y2karaoke/core/components/lyrics/lrc.py`
- These are still large, but should only be split if a real ownership seam emerges.

4. Benchmark- and visual-sensitive areas
- Visual bootstrap and benchmark tooling still have debt, but those paths need evidence-backed refactors.
- Prefer benchmark-guided work over opportunistic restructuring there.

## Completed High-Return Work

- Complexity regression checks in CI and guardrails
- Whisper runtime-config and mapping-runtime-config boundaries
- Lyrics runtime-config boundary
- Lyrics quality orchestration split
- Lyrics source-routing / offset-quality extraction
- Whisper facade cleanup and lazy compatibility exports
- Whisper block/segment assignment ownership split
- Sync cache-layer extraction
- Focused unit-test cleanup to avoid accidental real audio/provider work

## Working Rule

Do not continue open-ended architecture cleanup by default.

Resume debt work when one of these is true:
- a module is clearly slowing down feature work
- a benchmark or CI regression points to a design bottleneck
- hidden state or ambiguous ownership creates a concrete testing or maintenance problem
