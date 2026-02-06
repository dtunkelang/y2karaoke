# Whisper-First Alignment Strategy (Draft)

## Goal
Improve timing for tracks where LRC timings are clearly broken by relying more on Whisper word timings, while keeping confidence gating and preserving hard constraints:
- Line order must be preserved.
- No line should appear more than once (no duplicates).

## Modes
- **Hybrid alignment** (`--whisper`): Uses Whisper to fix LRC timing when it looks broken.
- **Forced DTW** (`--whisper --whisper-force-dtw`): Always run DTW alignment, even if LRC looks acceptable.
- **Whisper-only** (`--whisper-only`): Skip LRC/Genius and build lines directly from Whisper transcription.
- **Whisper map LRC** (`--whisper-map-lrc`): Keep LRC text but assign timing from Whisper segments.

## Proposed Approach (Pragmatic Hybrid)
1. **Transcribe vocals with Whisper** to get segments + word timings.
2. **Assess LRC quality** using current `_assess_lrc_quality`.
3. **If quality >= 0.7**: keep LRC, only targeted fixes (current hybrid).
4. **If 0.4 <= quality < 0.7**: hybrid corrections (current).
5. **If quality < 0.4**: switch to a Whisper-first path with confidence gating:
   - Run DTW alignment at the word level.
   - Evaluate alignment confidence (share of words matched above similarity threshold, and monotonicity).
   - If confidence is high enough, **derive line timings from aligned word clusters**.
   - If confidence is low, fall back to conservative segment-based corrections.

## Whisper-First Word Alignment (DTW)
- Use existing `align_dtw_whisper` to map LRC words → Whisper words.
- Add a **confidence score**:
  - `matched_ratio = matched_words / total_lrc_words`
  - `avg_similarity` from phonetic similarity of matched pairs
  - `coverage` = number of lines with >=1 matched word
- Example gate: `matched_ratio >= 0.6` and `avg_similarity >= 0.5` and `coverage >= 0.6`.

## Building Line Timings From Word Matches
- For each line, collect aligned whisper word timestamps.
- If a line has >= 2 matched words:
  - Set line start = min(matched words start)
  - Set line end = max(matched words end)
- If a line has 1 matched word:
  - Use word start/end + min duration clamp
- If a line has 0 matched words:
  - Leave as-is or mark for segment-based adjustment

## Handling Merge/Split Scenarios
- If **adjacent lines map into a single continuous Whisper window** (overlap or very small gap),
  - keep both lines but allow their timings to be continuous and within the same segment.
- If one line’s words map far apart (large internal gap),
  - allow that line to “split” by distributing timings across two adjacent lines (or mark for review).

## Confidence-Gated Pipeline (Summary)
- If LRC quality poor:
  1. Run DTW word alignment
  2. Compute confidence
  3. If confidence high → build timings from aligned words
  4. Else → segment-based corrections only (no aggressive changes)
  5. Always enforce hard invariants: order, no duplicates

## Diagnostics to Add
- Report: `dtw_matched_ratio`, `dtw_word_coverage`, `dtw_line_coverage` in timing report
- Report coverage in `dtw_word_coverage` so the matched whisper-to-provider word ratio surfaces in diagnostics
- Debug list of lines with 0 matched words
- Count of lines adjusted via DTW vs segments

## Next Steps
- Implement confidence metrics in `align_dtw_whisper` and return them.
- Add a `build_lines_from_dtw_alignments` helper.
- Gate use of DTW-derived timings on confidence.
- Add tests for:
  - low similarity but high coverage (allowed)
  - low coverage (fallback)
  - order preservation and no duplicates
