# Logic Audit: Track Selection, Lyrics Sources, Alignment

This document summarizes the core decision logic and the main test coverage for the three critical pillars:

- YouTube track selection (prefer audio-only, avoid non-studio when possible)
- Best lyrics/timing source selection across providers
- Alignment quality (onsets, pauses, timing issues)

Pipeline facade note:
- `src/y2karaoke/pipeline/identify/`, `src/y2karaoke/pipeline/lyrics/`,
  `src/y2karaoke/pipeline/audio/`, `src/y2karaoke/pipeline/alignment/`, and
  `src/y2karaoke/pipeline/render/` provide subsystem entrypoints over the
  underlying `core/` implementations.

## 1) YouTube Track Selection (Prefer Audio-Only)

**Primary code paths**
- `src/y2karaoke/core/track_identifier.py`
  - facade to `src/y2karaoke/core/track_identifier_impl.py`
  - helper logic in `src/y2karaoke/core/track_identifier_helpers.py`
  - pure YouTube heuristics in `src/y2karaoke/core/track_identifier_youtube_rules.py`
  - `_search_youtube_single`
  - `_search_youtube_verified`
  - `_is_likely_non_studio`
  - `_is_preferred_audio_title`

**Decision rules (summary)**
- If the query does **not** explicitly request non-studio variants, filter out
  non-studio candidates (live, remix, cover, karaoke, etc.).
- Prefer audio-only titles (e.g., "Official Audio", "Audio Only") when durations
  are comparable.
- If the first search yields a poor duration match, fall back to searching with
  `"{query} audio"`.

**Key tests**
- `tests/test_track_identifier_youtube_search.py`
  - `test_search_youtube_single_prefers_official_audio`
  - `test_search_youtube_verified_prefers_audio_title`
  - `test_search_youtube_single_keeps_non_studio_when_query_requests`

## 2) Best Lyrics/Timing Source Selection

**Primary code paths**
- `src/y2karaoke/core/lyrics.py::_fetch_lrc_text_and_timings`
- `src/y2karaoke/core/timing_evaluator.py::select_best_source`
- `src/y2karaoke/core/timing_evaluator.py::compare_sources`
- `src/y2karaoke/core/sync_search.py`
  - provider retry/backoff and fallback search orchestration

**Decision rules (summary)**
- If `evaluate_sources=True` and `vocals_path` is available, compare all sources
  using audio timing evaluation.
- Select by highest `overall_score`, with deterministic tie-breakers:
  1) Closest duration to `target_duration`
  2) Prefer known duration over unknown
  3) Higher `(line_alignment_score + pause_alignment_score)`

**Key tests**
- `tests/test_timing_evaluator_select_best.py`
  - Tie-breakers for duration closeness and known duration
- `tests/test_lyrics_pipeline.py`
  - Evaluation fallback behavior when `select_best_source` yields nothing

## 3) Alignment Quality

**Primary code paths**
- `src/y2karaoke/core/timing_evaluator.py`
  - `evaluate_timing`
  - `_check_pause_alignment`
  - `_calculate_pause_score`
  - `_find_closest_onset`
- `src/y2karaoke/core/whisper_phonetic_tokens.py`
  - phoneme and syllable tokenization for DTW
- `src/y2karaoke/core/whisper_phonetic_paths.py`
  - DTW path/cost construction helpers
- `src/y2karaoke/core/whisper_alignment_pull_rules.py`
  - segment pull/merge adjustments used by Whisper alignment
- `src/y2karaoke/core/whisper_alignment_pull_helpers.py`
  - shared nearest-segment and word-reflow helpers

**Decision rules (summary)**
- Line alignment: onsets within 0.5s are counted as matched.
- Pause alignment: long silences (>= 2s) should be covered by lyric gaps.
- Issues flagged:
  - `spurious_gap`: lyrics show a gap but vocals are active
  - `missing_pause`: lyrics gap has no silence
  - `unexpected_pause`: silence not reflected in lyrics timing

**Key tests**
- `tests/test_timing_evaluator.py`
- `tests/test_timing_evaluator_helpers.py`
- `tests/test_timing_evaluator_pauses_more.py`
- `tests/test_whisper_alignment_pull_helpers.py`

## Notes / Follow-ups

- Integration tests that require live network/audio are marked and intentionally
  excluded from this logic audit.
- If additional robustness is needed, consider adding controlled integration
  tests with cached audio and LRC fixtures.
