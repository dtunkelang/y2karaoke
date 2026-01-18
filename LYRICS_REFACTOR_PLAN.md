# Lyrics Module Refactoring Plan

## Current State

`lyrics.py` is 3659 lines with 48 functions/classes. Problems:
- **Duplicate functions**: `normalize_text`, `parse_lrc_timestamp`, `parse_lrc_with_timing`, `_is_metadata_line`, `extract_artists_from_title`, `create_lines_from_lrc` all appear twice
- **Mixed concerns**: fetching, parsing, alignment, timing correction, validation all in one file
- **Ad-hoc heuristics**: Many magic numbers and special cases for timing fixes
- **No clear data flow**: Hard to understand how different sources combine

## Core Insight: Forced Alignment, Not Transcription

The current approach conflates two different problems:
1. **Transcription** (ASR): audio → text (error-prone)
2. **Forced Alignment**: given known text + audio → timing (reliable)

**Key simplification**: Use Genius as canonical text, use audio only for timing.

```
Genius text (canonical)  ──┐
                          ├──► Forced Alignment ──► Timed lyrics
Audio (timing source)    ──┘
```

This eliminates:
- ASR errors from Whisper transcription
- Complex text reconciliation between sources
- Fuzzy matching heuristics

## Why Audio Analysis is Essential

1. **Offset detection** - YouTube audio often has silence/intro before vocals start
2. **Word-level timing** - LRC only provides line-level timing, karaoke needs word-level

## Proposed Architecture

### Clean Module Separation

```
lyrics/
  __init__.py          # Public API: get_lyrics(), Line, Word, SongMetadata
  models.py            # Data classes: Word, Line, SongMetadata, SingerID
  genius.py            # Fetch lyrics + singer annotations from Genius
  forced_align.py      # Align known text to audio (word-level timing)
  offset.py            # Detect song start in audio
  romanization.py      # Script conversion (CJK, Arabic, etc.)
  serialization.py     # JSON load/save
```

### Simplified Pipeline

```python
def get_lyrics(audio_path: str, title: str, artist: str) -> tuple[list[Line], SongMetadata]:
    # 1. Get canonical text + singer info from Genius
    lines_text, singers, metadata = fetch_genius(title, artist)

    # 2. Detect where vocals actually start in audio
    offset = detect_song_start(audio_path)

    # 3. Forced alignment: align known text to audio
    #    (no transcription - we know what words to expect)
    word_timings = forced_align(lines_text, audio_path, offset)

    # 4. Build Line/Word objects with timing and singer info
    lines = build_lines(lines_text, word_timings, singers)

    # 5. Apply romanization if needed
    if needs_romanization(lines):
        lines = romanize(lines)

    return lines, metadata
```

### Forced Alignment Details

Forced alignment takes known text and finds when each word occurs in audio:

```python
def forced_align(text: list[str], audio_path: str, offset: float) -> list[list[WordTiming]]:
    """
    Given known lyrics text, find word-level timing in audio.

    Uses whisperx in alignment mode (not transcription mode):
    - Input: text we know is correct + audio
    - Output: (start_time, end_time) for each word

    This is much more reliable than transcription because:
    - No ASR errors (we provide the text)
    - Aligner searches for expected phonemes
    - Works well even with accents/noise
    """
    # WhisperX align mode or Montreal Forced Aligner
    ...
```

### Loss Function for Validation

Even with forced alignment, we should validate results:

```python
def alignment_quality(lines: list[Line], audio_path: str) -> float:
    """
    Score alignment quality. Low score indicates problems.

    Components:
    - timing_smoothness: Penalize unrealistic word durations (<50ms or >3s)
    - gap_penalty: Penalize large gaps between consecutive words
    - vocal_activity_match: Words should coincide with detected vocal energy
    - coverage: All text should be aligned
    """
    ...
```

### Fallback Strategy

If forced alignment fails (rare), fall back gracefully:

```python
def get_lyrics(...):
    # Try forced alignment first
    try:
        word_timings = forced_align(text, audio_path, offset)
        if alignment_quality(word_timings) > THRESHOLD:
            return build_lines(text, word_timings, singers)
    except AlignmentError:
        pass

    # Fallback 1: Use LRC line timing + even word distribution
    lrc = fetch_lrc(title, artist)
    if lrc:
        return build_lines_from_lrc(text, lrc, singers)

    # Fallback 2: Estimate timing from audio duration
    duration = get_audio_duration(audio_path)
    return build_lines_estimated(text, duration, singers)
```

## Migration Path

### Phase 1: Extract and Deduplicate (do first)
1. Remove duplicate functions
2. Extract `models.py` (Word, Line, SongMetadata, SingerID)
3. Extract `romanization.py`
4. Extract `serialization.py` (JSON load/save)
5. Keep `lyrics.py` working but smaller

### Phase 2: Clean Up Genius Fetching
1. Extract `genius.py` with clean interface
2. Remove text merging complexity - Genius is canonical
3. Simplify singer annotation extraction

### Phase 3: Implement Clean Forced Alignment
1. Extract `forced_align.py`
2. Use whisperx alignment mode (or similar)
3. Implement `detect_song_start()` for offset
4. Implement quality scoring

### Phase 4: Simplify Main Pipeline
1. Rewrite `get_lyrics()` with clean flow
2. Remove transcription code paths
3. Remove complex reconciliation logic
4. Add fallback strategies

### Phase 5: Cleanup
1. Delete dead code (should be substantial)
2. Add tests for each module
3. Document the algorithm

## Expected Outcome

- **~500 lines** instead of 3659
- **Clear data flow**: Genius → Forced Align → Output
- **No ASR errors**: Text is always from Genius
- **Principled timing**: Forced alignment + validation
- **Easy to debug**: Each step is isolated and testable
