# Next Session TODO

Last updated: 2026-03-18

## Current State

Recent high-value architecture cleanup completed:

- `lyrics_whisper_quality.py`
  - Split the quality flow into explicit submodules:
    - `lyrics_quality_sources.py`
    - `lyrics_quality_alignment.py`
    - `lyrics_quality_tail_guardrail.py`
  - Kept compatibility by re-exporting the old private helper names from `lyrics_whisper_quality.py`.
  - Preserved existing monkeypatch seams used by tests by routing internal helper calls back through `lyrics_whisper.py` wrappers where needed.
- `lyrics_whisper.py`
  - Hooks now support explicit `hooks=` injection plus `ContextVar`-backed compatibility.
  - Source-routing policy is already extracted to `lyrics_source_routing.py`.
  - Offset / quality policy is already extracted to `lyrics_offset_quality.py`.
- `sync.py`
  - Mutable process-wide default state was replaced with `ContextVar`-backed state and compatibility proxies.
- `runtime_config.py`
  - Lyrics provider / duration-tolerance env reads now live behind `LyricsRuntimeConfig`.
- `whisper_runtime_config.py`
  - Profile and major Whisper runtime toggles now live behind `WhisperRuntimeConfig`.
- `whisper_mapping_runtime_config.py`
  - Segment-assignment and trace config now live behind typed runtime config.
- `whisper_integration_align.py`
  - Main aligner orchestration was decomposed heavily and the entrypoint is much smaller.
- `lyrics/api.py`
  - Public API surface is now explicit, with private compatibility exports resolved lazily.
- `whisper_integration.py`, `whisper_mapping.py`, `whisper_mapping_post.py`
  - Eager alias-installation / eager import-facade behavior was replaced with lazy compatibility facades.

## Most Recent Changes

1. Lyrics quality boundary split:

- `lyrics_whisper_quality.py` is now mostly orchestration.
- Source acquisition / trust policy / offset detection moved to `lyrics_quality_sources.py`.
- Whisper alignment / fallback / LRC mapping moved to `lyrics_quality_alignment.py`.
- Tail-completeness guardrail and duration clipping moved to `lyrics_quality_tail_guardrail.py`.
- Compatibility-sensitive helpers like `_apply_whisper_with_quality` and `_fetch_genius_with_quality_tracking` still behave correctly when patched through either `lyrics_whisper.py` or `lyrics_whisper_quality.py`.

2. `whisper_blocks.py` ownership split:

- Segment-level text-overlap assignment logic moved to `whisper_segment_assignments.py`.
- Block-scoped syllable-DTW assignment logic moved to `whisper_block_dtw_assignments.py`.
- Shared trace serialization moved to `whisper_assignment_trace.py`.
- `whisper_blocks.py` is now a compatibility facade that re-exports the old private helpers instead of owning both subsystems directly.

3. `sync.py` cache-layer split:

- Disk-cache and all-sources-cache behavior moved to `sync_cache.py`.
- `sync.py` now keeps state/runtime/provider orchestration while re-exporting the old cache helpers as compatibility aliases.
- Focused sync regression coverage stayed green after the split.

Focused verification that passed after these splits:

- `./.venv/bin/python -m flake8 src/y2karaoke/core/components/lyrics/lyrics_quality_tail_guardrail.py src/y2karaoke/core/components/lyrics/lyrics_quality_sources.py src/y2karaoke/core/components/lyrics/lyrics_quality_alignment.py src/y2karaoke/core/components/lyrics/lyrics_whisper_quality.py`
- `./.venv/bin/pytest tests/unit/lyrics/test_lyrics_quality_unit.py tests/unit/lyrics/test_lyrics_additional.py tests/unit/lyrics/test_lyrics_pipeline.py tests/unit/lyrics/test_lyrics_api.py tests/unit/pipeline/test_compat_monkeypatch_seams.py -q`
  - `63 passed`
- `./.venv/bin/python -m flake8 src/y2karaoke/core/components/whisper/whisper_assignment_trace.py src/y2karaoke/core/components/whisper/whisper_segment_assignments.py src/y2karaoke/core/components/whisper/whisper_block_dtw_assignments.py src/y2karaoke/core/components/whisper/whisper_blocks.py`
- `./.venv/bin/pytest tests/unit/whisper/test_whisper_mapping_post.py tests/unit/whisper/test_whisper_segment_assignment_experimental.py tests/unit/whisper/test_whisper_integration_pipeline_unit.py tests/unit/whisper/test_whisper_integration_align_stage_unit.py tests/unit/whisper/test_whisper_integration_align_stage_followup_unit.py -q`
  - `66 passed`
- `./.venv/bin/python -m flake8 src/y2karaoke/core/components/lyrics/sync_cache.py src/y2karaoke/core/components/lyrics/sync.py`
- `./.venv/bin/pytest tests/unit/lyrics/test_sync.py tests/unit/lyrics/test_sync_providers.py tests/unit/lyrics/test_sync_quality_unit.py tests/unit/lyrics/test_sync_unit.py tests/unit/lyrics/test_sync_additional.py tests/unit/lyrics/test_sync_more.py tests/unit/lyrics/test_sync_lyriq.py tests/unit/lyrics/test_lyrics_pipeline.py -q`
  - `90 passed`

## Current Priorities

1. **`sync.py` remaining ownership cleanup**
- Why:
  - State/config cleanup is done, and the cache layer is now extracted, but the file still mixes state container behavior, provider orchestration, and compatibility proxy surface.
  - It remains large enough that future changes will still have high navigation cost.
- Goal:
  - Only continue if there is still a clean split between provider routing and compatibility/state plumbing.

2. **`helpers.py` / `lrc.py` review only if the above stop paying off**
- Why:
  - They are still large, but some of that size may be legitimate utility density rather than boundary confusion.
- Goal:
  - Only refactor if there is real concern mixing, not just line count.

## Recommended Next Pass

1. Read:
- `src/y2karaoke/core/components/lyrics/sync.py`
- `src/y2karaoke/core/components/lyrics/helpers.py`
- `src/y2karaoke/core/components/lyrics/lrc.py`
- the related sync-focused tests under `tests/unit/lyrics/`

2. Decide whether the next pass is still worthwhile in `sync.py` or whether the bigger remaining return has moved to `helpers.py` / `lrc.py`.

3. Keep compatibility and current runtime-config flow intact.

4. Verify with focused Whisper suites before widening coverage.

## Guardrails

- Keep heuristic behavior unchanged.
- Do not reintroduce process-global mutable state.
- Do not add new ad hoc `os.getenv()` reads in core logic.
- Preserve compatibility seams that tests still rely on unless the tests are updated in the same pass.
- Leave unrelated dirty files alone:
  - `tools/run_benchmark_suite.py`
  - `1`
  - `benchmarks/gold_set_candidate/...`
