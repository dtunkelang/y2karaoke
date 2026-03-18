# Lyrics Pipeline

Primary facade:
- `src/y2karaoke/pipeline/lyrics/__init__.py`

Primary implementation boundary:
- `src/y2karaoke/core/components/lyrics/api.py`

## Responsibilities

The lyrics pipeline is responsible for:
- acquiring synced lyrics and fallback text
- parsing LRC and plain-text sources
- scoring source quality and timing trust
- applying optional Whisper-assisted timing refinement
- returning lines, metadata, and quality diagnostics

## Main Modules

- `lyrics_whisper.py`
  - compatibility surface and hook resolution
- `lyrics_whisper_pipeline.py`
  - simple / non-quality orchestration
- `lyrics_whisper_quality.py`
  - quality-aware orchestration
- `sync.py`
  - synced-provider orchestration and state
- `sync_pipeline.py`
  - provider-routing / fetch orchestration

Focused ownership modules introduced during cleanup:
- `lyrics_source_routing.py`
  - source disagreement routing and provider-selection policy
- `lyrics_offset_quality.py`
  - offset detection and quality scoring
- `lyrics_quality_sources.py`
  - source acquisition and timing-trust policy
- `lyrics_quality_alignment.py`
  - Whisper-assisted alignment and fallback application
- `lyrics_quality_tail_guardrail.py`
  - tail-completeness guardrail and duration clipping
- `runtime_config.py`
  - typed runtime config for provider preference and duration tolerance
- `sync_cache.py`
  - sync cache and disk-cache behavior

## Quality-Aware Flow

1. Resolve LRC and fallback text sources.
2. Apply timing-trust policy to provider timestamps.
3. Build line objects from LRC or plain text.
4. Optionally auto-enable or explicitly apply Whisper alignment.
5. Apply clipping, singer metadata, romanization, and quality reporting.

## Notes

- Main facade modules still preserve compatibility wrappers and test seams.
- Benchmark reporting distinguishes general agreement metrics from Whisper-anchor diagnostics.
- Further cleanup in this slice should be driven by concrete ownership problems, not just file length.
