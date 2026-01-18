# Timing Improvement Plan

## Goal
Improve word-level timing accuracy while respecting lyrics as a hard constraint.

## Core Principles
1. **Lyrics are sacred** - The extracted lyrics text and order are never modified
2. **Line timing from LRC** - Use syncedlyrics database as the starting point for line boundaries
3. **Audio offset detection** - Auto-detect when vocals start, with manual override option
4. **Word timing from audio** - Use vocal audio analysis to optimize word positions within lines

---

## Phase 1: Fix Audio Offset Detection ✓ (partially done)

**Current state:** `detect_song_start()` in `forced_align.py` exists but may not be accurate enough.

**Tasks:**
- [ ] 1.1 Improve `detect_song_start()` to better detect first vocal activity
- [ ] 1.2 Add `--audio-offset` CLI flag for manual override
- [ ] 1.3 Compute offset between detected vocal start and first LRC timestamp

**Files:** `forced_align.py`, `cli.py`, `karaoke.py`

---

## Phase 2: Robust Line Timing from LRC

**Current state:** `sync.py` now fetches LRC, but it's only used as "hints" for forced alignment.

**Tasks:**
- [ ] 2.1 Make LRC line timing the primary source (not just hints)
- [ ] 2.2 Apply computed audio offset to all LRC timestamps
- [ ] 2.3 Validate LRC timing against audio energy (detect gross misalignments)
- [ ] 2.4 Fall back to even distribution if LRC unavailable or unreliable

**Files:** `lyrics.py`, `sync.py`

---

## Phase 3: Word Timing via Audio Analysis

**Current state:** Uses WhisperX forced alignment, which can be slow and sometimes inaccurate.

**Tasks:**
- [ ] 3.1 Create `word_timing.py` module for audio-based word timing
- [ ] 3.2 Implement onset detection to find word boundaries within lines
- [ ] 3.3 Distribute words across detected onsets while respecting line boundaries
- [ ] 3.4 Use energy envelope to estimate word durations
- [ ] 3.5 Handle edge cases (rapid words, held notes, silence gaps)

**Files:** `word_timing.py` (new), `lyrics.py`

---

## Phase 4: Integration & Quality Scoring

**Tasks:**
- [ ] 4.1 Create unified timing pipeline: LRC → offset → word timing
- [ ] 4.2 Implement quality scoring to detect timing issues
- [ ] 4.3 Add logging/debug mode to visualize timing decisions
- [ ] 4.4 Update `get_lyrics_simple()` to use new pipeline

**Files:** `lyrics.py`, `forced_align.py`

---

## Phase 5: Optional - WhisperX as Fallback

**Tasks:**
- [ ] 5.1 Keep WhisperX forced alignment as fallback when audio analysis fails
- [ ] 5.2 Use WhisperX word timing only within LRC line boundaries
- [ ] 5.3 Constrain WhisperX output to match exact lyrics text

**Files:** `forced_align.py`

---

## Architecture Summary

```
Input: title, artist, vocals_path
                │
                ▼
┌─────────────────────────────────────┐
│  1. Get Lyrics (Genius)             │  ◄── Hard constraint: exact text
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  2. Get Line Timing (syncedlyrics)  │  ◄── LRC timestamps
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  3. Detect Audio Offset             │  ◄── When do vocals actually start?
│     (auto + manual override)        │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  4. Adjust Line Timing              │  ◄── Apply offset to LRC
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  5. Word Timing (audio analysis)    │  ◄── Onset detection within lines
│     - Onset detection               │
│     - Energy envelope               │
│     - Constrained to line bounds    │
└─────────────────────────────────────┘
                │
                ▼
Output: List[Line] with word-level timing
```

---

## Implementation Order

1. **Phase 1** - Audio offset (foundation for everything else)
2. **Phase 2** - Line timing from LRC (establish line boundaries)
3. **Phase 3** - Word timing (the main improvement)
4. **Phase 4** - Integration (tie it all together)
5. **Phase 5** - Optional WhisperX fallback

---

## Success Criteria

- [ ] Lyrics text matches Genius exactly (no word changes)
- [ ] Line timing matches LRC within ±0.5s after offset adjustment
- [ ] Word highlighting feels natural and in sync with audio
- [ ] Manual offset override works when auto-detection fails
- [ ] Processing time is reasonable (no slower than current WhisperX approach)
