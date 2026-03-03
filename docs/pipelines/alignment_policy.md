# Alignment Policy Layer

This project now uses an explicit policy layer for high-level alignment decisions.

## Purpose

The goal is to keep "what decision should we make?" separate from
"how do we compute evidence?". This reduces heuristic sprawl and makes behavior
easier to review.

## Boundaries

1. Evidence extraction
- Owned by lyrics/whisper/alignment components.
- Produces measurable signals (`duration_mismatch`, `matched_ratio`, `line_coverage`, etc.).

2. Policy arbitration
- Owned by `src/y2karaoke/core/components/alignment/alignment_policy.py`.
- Consumes evidence and returns explicit decisions.
- Must be deterministic and lightweight.

3. Execution / postprocess
- Pipeline modules apply policy decisions to mutate timings.

## Current centralized decisions

1. LRC timing trust under duration mismatch
- `decide_lrc_timing_trust(...)`
- Decides keep/drop behavior for provider LRC line timings.

2. Aggressive Whisper retry eligibility
- `should_retry_aggressive_whisper_dtw_map(...)`
- Decides if a second pass with aggressive Whisper settings is warranted.

## Engineering rules

1. New global heuristics should be added to the policy module first.
2. Pipeline code should call policy functions instead of duplicating threshold logic.
3. Policy functions must have unit tests independent of full pipeline tests.
