# Y2Karaoke - Technical Architecture for Claude

This document provides a comprehensive overview of the y2karaoke codebase for Claude to understand at the start of a session.

## Project Overview

Y2Karaoke generates karaoke videos with word-by-word highlighting from YouTube URLs or song search queries. The system downloads audio, separates vocals, fetches synced lyrics, aligns timing using audio analysis, and renders video with KaraFun-style highlighting.

**Entry Points:**
- `karaoke.py` - Legacy wrapper, delegates to `y2karaoke.cli:cli()`
- `src/y2karaoke/cli.py` - Main CLI with `generate`, `cache`, and `evaluate-timing` commands

## Development Environment

**IMPORTANT:** Always use the virtual environment for running commands.

```bash
# Activate venv before running any Python commands
source venv/bin/activate

# Run tests
source venv/bin/activate && pytest tests/ -v

# Run specific test file
source venv/bin/activate && pytest tests/test_models.py -v

# Run linting
source venv/bin/activate && black --check src/y2karaoke tests
source venv/bin/activate && flake8 src/y2karaoke
```

The project uses Python 3.12 with dependencies installed via `pip install -e ".[dev]"`.

## 1. Overall Pipeline

The main orchestrator is `KaraokeGenerator` in `src/y2karaoke/core/karaoke.py`.

**Generation Flow** (`generate()` method, lines 27-364):

```
Input (URL or search query)
    ↓
Track Identification → TrackInfo (artist, title, duration, youtube_url)
    ↓
Download audio/video from YouTube
    ↓
Separate vocals from instrumental (demucs)
    ↓
Fetch synced lyrics (LRC) with duration validation
    ↓
Align lyrics timing using audio analysis
    ↓
Apply audio effects (key shift, tempo) + optional break shortening
    ↓
Scale lyrics timing for tempo changes
    ↓
Render video with word-by-word highlighting
    ↓
Output: karaoke_video.mp4
```

**Key CLI options** (defined in `cli.py:83-195`):
- `--offset`: Manual timing adjustment
- `--key`: Pitch shift (-12 to +12 semitones)
- `--tempo`: Speed multiplier (0.1x to 3.0x)
- `--audio-start`: Skip intro seconds
- `--backgrounds`: Use YouTube video frames
- `--shorten-breaks`: Compress long instrumental sections
- `--whisper`: Use Whisper for severely broken LRC timing
- `--whisper-temperature`: Adjust Whisper creativity (0.0-1.0)
- `--lenient-vocal-activity-threshold`: Vocal energy required for DTW leniency
- `--evaluate-lyrics`: Score all LRC sources and select best

## 2. Track Identification (Title & Artist)

**Files:** `src/y2karaoke/core/track_identifier.py` (facade), `src/y2karaoke/core/track_identifier_impl.py` (entry workflow), `src/y2karaoke/core/track_identifier_helpers.py` (shared split/LRC/url helpers)

Two identification paths:

### Path A: Search Query → Track Info
`identify_from_search(query)` (lines 259-434):
1. Parse query for artist/title hints (separators: `-`, `–`, `—`, `: `, `by`)
2. Query MusicBrainz for candidate recordings
3. Score candidates by LRC availability and duration match
4. Search YouTube for matching videos
5. Return `TrackInfo` with validated LRC

**Key helpers:**
- `_parse_query()` (658-684): Extracts artist/title from separators
- `_try_artist_title_splits()` (686-717): Tries word splits for unseparated queries
- `_query_musicbrainz()` (813-889): Scores recordings by studio likelihood
- `_score_recording_studio_likelihood()` (891-944): Filters live/remix/demo versions

### Path B: YouTube URL → Track Info
`identify_from_url(url, artist_hint, title_hint)` (lines 440-652):
1. Extract YouTube metadata (title, uploader, duration)
2. Use explicit artist/title if provided (overrides parsing)
3. Parse video title for hints
4. Query MusicBrainz and validate with LRC duration
5. If LRC duration differs >15s, search for alternative YouTube video

**Key data structure:**
```python
@dataclass
class TrackInfo:
    artist: str
    title: str
    duration: int           # Canonical duration (seconds)
    youtube_url: str
    youtube_duration: int   # Actual YouTube video duration
    source: str             # "musicbrainz", "syncedlyrics", "youtube"
    lrc_duration: Optional[int]
    lrc_validated: bool     # LRC duration matches canonical
```

**Important validation:**
- `_is_likely_non_studio()` (1338-1437): Detects live/remix/cover versions
- `_check_lrc_and_duration()` (1186-1231): Validates LRC quality
- `_find_best_lrc_by_duration()` (1233-1336): Scores by duration + title similarity

## 3. Lyrics and Timing from Providers

### Synced Lyrics (LRC Format)
**Files:** `src/y2karaoke/core/sync.py` (provider/cache orchestration), `src/y2karaoke/core/sync_quality.py` (timestamp/duration/quality helpers)

**Main functions:**
- `fetch_lyrics_multi_source(title, artist, synced_only=True)`: Tries providers in order
- `fetch_lyrics_for_duration(title, artist, target_duration, tolerance=20)`: Duration-aware fetch
- `get_lrc_duration(lrc_text)`: Extracts implied duration from timestamps
- `validate_lrc_quality(lrc_text, expected_duration)`: Quality validation

**Provider priority** (lines 29-36):
1. lyriq (LRCLib API) - primary
2. Musixmatch - best quality, rate-limited
3. NetEase - good for Asian music
4. Megalobiz
5. Lrclib
6. Genius - plain text fallback only

### LRC Parsing
**File:** `src/y2karaoke/core/lrc.py` (~200 lines)

- `parse_lrc_timestamp(ts_str)`: Parses `[MM:SS.MS]` format
- `parse_lrc_with_timing(lrc_text)`: Returns `[(timestamp, text), ...]`
- `create_lines_from_lrc(lrc_text, romanize=True)`: Creates `Line` objects
- `split_long_lines(lines, max_width_ratio)`: Breaks long lines for display

### Genius (Fallback + Singer Detection)
**File:** `src/y2karaoke/core/genius.py` (~200 lines)

- `fetch_genius_lyrics_with_singers(title, artist)`: Returns lines with singer labels
- `parse_genius_html(html, artist)`: Extracts lyrics with `[Verse]`, `[Chorus]` sections
- Detects duet information and maps to singer1/singer2/both

## 4. Audio Retrieval from YouTube

**File:** `src/y2karaoke/core/downloader.py` (~150 lines)

### YouTubeDownloader Class
- `download_audio(url, output_dir)`: Downloads best audio → WAV (192kbps)
- `download_video(url, output_dir)`: Downloads video for backgrounds
- `get_video_title(url)`: Metadata without full download
- `get_video_uploader(url)`: Channel/uploader name

**Implementation:**
- Uses `yt_dlp` with FFmpeg postprocessor
- Caches in `~/.cache/karaoke/{video_id}/`
- Returns `{"audio_path": ..., "title": ..., "artist": ..., "duration": ...}`

**Metadata utilities** (`src/y2karaoke/core/youtube_metadata.py`):
- `extract_video_id(url)`: Extracts 11-character video ID
- `validate_youtube_url(url)`: Format validation

## 5. Alignment Process

The alignment system combines multiple techniques to achieve accurate word-level timing.

### Main Lyrics Pipeline
**File:** `src/y2karaoke/core/lyrics.py` (~300 lines)

`get_lyrics_simple()` (136-299) - 7-step process:

1. **Fetch LRC**: Duration-aware multi-source fetch
2. **Fetch Genius fallback**: For singer/duet metadata
3. **Detect vocal offset**: `detect_song_start(vocals_path)`
   - Finds first vocal onset, compares to LRC first line
   - Auto-applies offset if difference > 0.3s and < 30s
4. **Create Line objects**: Words distributed by character count
5. **Refine word timing**: `refine_word_timing()` matches to audio onsets
6. **Adjust duration mismatch**: Handles radio edit vs album version
7. **Optional Whisper alignment**: For severely broken LRC

### Audio Analysis
**File:** `src/y2karaoke/core/alignment.py` (~200 lines)

- `detect_song_start(vocals_path)`: First vocal onset using librosa
- `detect_audio_silence_regions()`: Finds instrumental breaks via RMS energy
- `detect_lrc_gaps()`: Identifies large gaps between LRC lines
- `adjust_timing_for_duration_mismatch()`: Scales timing proportionally

**Onset detection settings:**
- Uses librosa `onset_detect()` with backtracking
- Hop length: 512 samples
- Energy threshold: noise_floor + 15% of dynamic range

### Word-Level Refinement
**File:** `src/y2karaoke/core/refine.py` (~150 lines)

`refine_word_timing(lines, vocals_path)`:
1. Detect all onsets in vocals
2. For each line, find onsets within line boundaries
3. Refine word start times to nearest onsets
4. Distribute durations based on onsets and vocal end
5. **Critical:** Respects line boundaries to avoid cross-line matching

### Vocal Separation
**File:** `src/y2karaoke/core/separator.py` (~100 lines)

`AudioSeparator.separate_vocals(audio_path, output_dir)`:
- Uses demucs AI model (htdemucs_ft)
- Separates: vocals, bass, drums, other
- Returns `{"vocals_path": ..., "instrumental_path": ...}`
- Caches separated stems

## 6. Audio Analysis for Timing Quality

**File:** `src/y2karaoke/core/timing_evaluator.py` (~400 lines)

### Audio Feature Extraction
`extract_audio_features(vocals_path)` returns `AudioFeatures`:
- Detected onsets (librosa)
- RMS energy envelope
- Silence regions
- Vocal start/end times

### Timing Scoring
`score_lrc_timing(lrc_text, audio_features)` returns `TimingReport`:
```python
@dataclass
class TimingReport:
    overall_score: float      # 0-100, higher is better
    line_alignment_score: float  # Line starts vs onsets
    pause_alignment_score: float # Pauses vs silence regions
    avg_line_offset: float
    std_line_offset: float
    matched_onsets: int
    total_lines: int
```

**Scoring algorithm:**
1. **Line alignment**: Distance between LRC times and detected onsets (±100ms tolerance)
2. **Pause alignment**: LRC gaps should match audio silence regions
3. **Overall**: Weighted combination (70% line, 30% pause)

### Source Selection
`select_best_source(title, artist, vocals_path)`:
- Fetches from all LRC providers
- Scores each against audio features
- Returns highest-scoring source
- Enabled with `--evaluate-lyrics` flag

### Whisper Fallback
`correct_timing_with_whisper(lines, vocals_path, temperature=0.0, **kwargs)`:
- Transcribes vocals using Whisper model
- Aligns transcription to LRC text via DTW (Dynamic Time Warping)
- Supports phonetic leniency based on vocal activity energy
- Returns adjusted line timings
- Enabled with `--whisper` flag

## Supporting Components

### Data Models
**File:** `src/y2karaoke/core/models.py`

```python
class Word:
    text: str
    start_time: float
    end_time: float
    singer: str = ""

class Line:
    words: List[Word]
    singer: SingerID = SingerID.UNKNOWN
    # Properties: start_time, end_time, text (computed)

class SongMetadata:
    singers: List[str]
    is_duet: bool
    title: Optional[str]
    artist: Optional[str]

class SingerID(Enum):
    SINGER1, SINGER2, BOTH, UNKNOWN
```

### Audio Effects
**File:** `src/y2karaoke/core/audio_effects.py`

`AudioProcessor.process_audio(input, output, semitones, tempo_multiplier)`:
- Pitch shift: librosa `pitch_shift()` (-12 to +12 semitones)
- Tempo change: librosa `time_stretch()` (0.1x to 3.0x)
- No pitch change on tempo (time-stretching preserves pitch)

### Break Shortening
**File:** `src/y2karaoke/core/break_shortener.py` (~200 lines)

- `detect_instrumental_breaks(vocals_path, min_duration=5.0)`: RMS energy analysis
- `shorten_instrumental_breaks(audio, vocals, max_duration)`: Beat-aligned cutting
- `adjust_lyrics_timing(lines, break_edits)`: Adjusts lyrics after shortening

### Video Rendering
**Files:**
- `src/y2karaoke/core/video_writer.py`: Main rendering with MoviePy
- `src/y2karaoke/core/frame_renderer.py` (~400 lines): Frame generation
- `src/y2karaoke/core/lyrics_renderer.py`: Text rendering

**Rendering features:**
- Splash screen (0-4s): Title and artist
- 4 lines visible with smart scrolling
- Word-by-word highlighting: gold for sung, white for upcoming
- Duet colors: blue (singer1), pink (singer2), purple (both)
- Cue indicator: Pulsing dots 3s before line (for gaps ≥4s)
- Break indicator: Progress bar during breaks ≥8s

### Romanization
**File:** `src/y2karaoke/core/romanization.py` (~200 lines)

Automatic romanization for non-Latin scripts:
- Korean: `korean-romanizer` (Revised Romanization)
- Chinese: `pypinyin` (Pinyin without tones)
- Japanese: `pykakasi` (romaji)
- Arabic/Hebrew: Simple transliteration

### Configuration
**File:** `src/y2karaoke/config.py`

Key constants:
```python
# Video
RESOLUTION = (1920, 1080)
FPS = 30
FONT_SIZE = 72

# Timing
SPLASH_DURATION = 4.0
INSTRUMENTAL_BREAK_THRESHOLD = 8.0
LYRICS_LEAD_TIME = 1.0
HIGHLIGHT_LEAD_TIME = 0.15
MAX_LINE_WIDTH_RATIO = 0.75

# Audio
SAMPLE_RATE = 44100
KEY_SHIFT_RANGE = (-12, 12)
TEMPO_RANGE = (0.1, 3.0)
```

### Caching
**File:** `src/y2karaoke/utils/cache.py`

`CacheManager` class:
- Location: `~/.cache/karaoke/{video_id}/`
- Stores: audio, separated stems, processed audio, lyrics metadata
- Methods: `load_metadata()`, `save_metadata()`, `get_cache_stats()`, `cleanup_old_files()`

**Cached operations:**

| Operation | Cache Type | Cache Key | Notes |
|-----------|------------|-----------|-------|
| YouTube download | File | video_id | WAV audio and video files |
| Vocal separation | File | audio filename pattern | `*Vocals*.wav`, `*instrumental*.wav` |
| Audio effects | File | `audio_{suffix}_key{N}_tempo{N}.wav` | Includes key shift and tempo |
| Audio trimming | File | `trimmed_from_{N}s.wav` | Start time in filename |
| Break shortening | File | `shortened_breaks_{N}s{suffix}.wav` | Duration and track suffix |
| LRC lyrics | In-memory | `(artist, title)` + duration validation | Checks cached duration matches target |
| Whisper transcription | File | `{stem}_whisper_{model}_{lang}{_aggr}{_tempN}.json` | Stored alongside vocals |
| Audio features | File | `{stem}_audio_features.npz` | Onsets, silence regions, energy |
| IPA transliteration | In-memory | `{language}:{text}` | For phonetic matching |

**Cache consistency:**
- LRC cache validates duration when `target_duration` is specified
- Audio effects cache includes all parameters in filename
- Whisper and audio features are cached alongside the source audio file

## File Structure Summary

```
y2karaoke/
├── karaoke.py                    # Legacy entry point
├── src/y2karaoke/
│   ├── cli.py                    # Main CLI (generate, cache, evaluate-timing)
│   ├── config.py                 # Constants and settings
│   ├── exceptions.py             # Custom exceptions
│   ├── core/
│   │   ├── karaoke.py            # KaraokeGenerator orchestrator
│   │   ├── karaoke_timing_report.py # JSON timing report helpers
│   │   ├── track_identifier.py   # Track identifier facade
│   │   ├── track_identifier_impl.py # Track identification flow entry points
│   │   ├── track_identifier_helpers.py # Shared scoring/LRC helper logic
│   │   ├── lyrics.py             # Main lyrics pipeline
│   │   ├── sync.py               # LRC provider integration
│   │   ├── sync_quality.py       # LRC quality scoring helpers
│   │   ├── lrc.py                # LRC format parsing
│   │   ├── genius.py             # Genius lyrics + singer detection
│   │   ├── alignment.py          # Audio analysis for timing
│   │   ├── refine.py             # Word-level timing refinement
│   │   ├── timing_evaluator.py   # LRC quality scoring + Whisper
│   │   ├── downloader.py         # YouTube audio/video download
│   │   ├── separator.py          # Vocal separation (demucs)
│   │   ├── audio_effects.py      # Key shift and tempo
│   │   ├── break_shortener.py    # Instrumental break compression
│   │   ├── video_writer.py       # Video rendering
│   │   ├── frame_renderer.py     # Frame generation
│   │   ├── lyrics_renderer.py    # Text rendering
│   │   ├── backgrounds.py        # Dynamic video backgrounds
│   │   ├── romanization.py       # Non-Latin script conversion
│   │   └── models.py             # Word, Line, SongMetadata
│   └── utils/
│       ├── cache.py              # Caching system
│       ├── logging.py            # Logging setup
│       └── validation.py         # Input validation
└── tests/                        # Test suite
```

## Quality Reporting

The pipeline includes comprehensive quality evaluation and reporting at each step.

### Quality Data Structures
**File:** `src/y2karaoke/core/models.py`

- `StepQuality`: Base class for step quality reports (quality_score 0-100, status, issues)
- `TrackIdentificationQuality`: Match confidence, source agreement, fallback status
- `LyricsQuality`: Source, coverage, timestamp density, duration match
- `TimingAlignmentQuality`: Method used, lines aligned, average offset
- `PipelineQualityReport`: Aggregated report with overall score and recommendations

### Quality Reporting Functions

**LRC Quality** (`sync_quality.py`, re-exported from `sync.py`):
- `get_lyrics_quality_report(lrc_text, source, target_duration)`: Returns quality metrics

**Lyrics with Quality** (`lyrics.py`):
- `get_lyrics_with_quality(...)`: Returns (lines, metadata, quality_report)
- Reports: lyrics source, alignment method, whisper usage, issues

**Pipeline Output** (`karaoke.py`, timing report writer in `karaoke_timing_report.py`):
The `generate()` method returns quality information:
```python
{
    "output_path": str,
    "quality_score": float,      # 0-100
    "quality_level": str,        # "high", "medium", "low"
    "quality_issues": List[str],
    "lyrics_source": str,
    "alignment_method": str,     # "lrc_only", "onset_refined", "whisper_hybrid"
}
```

**CLI Display** (`cli.py`):
After generation, displays quality summary with color-coded indicator (🟢/🟡/🔴).

## Common Debugging Scenarios

### Lyrics timing is off
1. Check `track_identifier.py` - is the correct version being identified?
2. Check LRC duration vs audio duration - mismatch triggers adjustment
3. Try `--offset` for manual correction
4. Try `--evaluate-lyrics` to select best LRC source
5. Try `--whisper` for severely broken timing

### Wrong song identified
1. Use explicit `--title` and `--artist` flags
2. Check `_is_likely_non_studio()` for live/cover detection
3. Verify MusicBrainz query results

### Vocals not separating well
1. Check demucs model version (htdemucs_ft)
2. Audio quality affects separation
3. Heavily processed audio may not separate cleanly

### Missing lyrics
1. Check provider order in `sync.py`
2. Try different artist/title spelling
3. Check if song has LRC anywhere (some songs have no synced lyrics)
