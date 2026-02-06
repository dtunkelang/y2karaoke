# y2karaoke

Generate karaoke videos from YouTube URLs or song titles with word-by-word highlighting.

## Features

- **Smart search**: Use YouTube URLs or just search by song title and artist
- Downloads audio from YouTube
- Separates vocals from instrumental using AI (demucs)
- **Hybrid timing system**: Combines synced lyrics (line timing) with WhisperX (word timing) for maximum accuracy
- **Multi-language support**: Automatically romanizes Japanese, Korean, Chinese, Arabic, and Hebrew lyrics with proper spacing
- Renders KaraFun-style karaoke videos with word-by-word highlighting
- **Smart line splitting**: Automatically breaks long lines for better readability
- Optional dynamic video backgrounds generated from the original YouTube video
- Duet-friendly visuals with singer-specific coloring and instrumental-break progress bar
- **Key shifting** (-12 to +12 semitones) to match your vocal range
- **Tempo control** (speed up or slow down) with synchronized lyrics
- Caches intermediate files so expensive steps (separation, transcription) are reused across runs

## Installation

### Prerequisites

- **Python 3.12** (required - onnxruntime is not yet available for Python 3.13+)
- ffmpeg

### Setup

```bash
# Clone the repository
git clone https://github.com/dtunkelang/y2karaoke.git
cd y2karaoke

# Create virtual environment with Python 3.12
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt

# Optional: Install romanization libraries for non-Latin scripts
pip install korean-romanizer pypinyin pykakasi pyarabic
```

### Troubleshooting

If you see `ModuleNotFoundError: No module named 'onnxruntime'`, your Python version is likely too new. Verify you're using Python 3.12:

```bash
python --version  # Should show Python 3.12.x
```

If not, recreate the virtual environment with `python3.12 -m venv venv`.

### Phonetic matching and the FLite fallback

To keep Epitran’s phonetic DTW as accurate as possible we ship a lightweight `lex_lookup`
shim that is installed automatically via the cache directory, so you generally no longer see
`lex_lookup (from flite) is not installed` warnings. Under the hood the shim uses the
`pronouncing` package (headers for which are pulled in via `requirements.txt`) to
emulate a CMU-style dictionary whenever the real `lex_lookup` binary is absent.

If you need the absolute best phonetic coverage (and prefer upstream FLite), feel free to
install the real `flite` package for your OS:

```bash
# macOS (Homebrew)
brew install flite

# Debian/Ubuntu
sudo apt install flite

# Fedora / RHEL
sudo dnf install flite
```

If you build from source, make sure the `flite` binary and its `lex` data directory are on
your `PATH`. Epitran will defer to the real binary when it is available, so rerunning the
pipeline after installing FLite gives you the noise-free warning and the most accurate
lexical pronunciations.

## Usage

### New CLI Interface (Recommended)

```bash
# Basic usage with YouTube URL
y2karaoke generate "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4

# Or search by song title and artist
y2karaoke generate "taste sabrina carpenter" -o output.mp4

# Using Python module
python -m y2karaoke.cli generate "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4

# Adjust timing offset (negative = highlight earlier, positive = later)
y2karaoke generate "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4 --offset -0.5

# Shift key down 3 semitones (for a lower vocal range)
y2karaoke generate "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4 --key -3

# Slow down to 80% speed (great for learning)
y2karaoke generate "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4 --tempo 0.8

# Skip the first 8 seconds of audio (e.g., to remove a long intro)
y2karaoke generate "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4 --audio-start 8

# Use dynamic video backgrounds from the original YouTube video
y2karaoke generate "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4 --backgrounds

# Use lyrics from a different title/artist (e.g., cover of an original)
y2karaoke generate "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4 \
  --title "Song Title" --artist "Artist Name"

# Combine options: shift key up, slow down, adjust timing
y2karaoke generate "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4 --key +2 --tempo 0.9 --offset -0.5

# Force reprocess (ignore cache)
y2karaoke generate "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4 --force

```

### Cache Management

```bash
# Show cache statistics
y2karaoke cache stats

# Clean up old files (older than 30 days)
y2karaoke cache cleanup --days 30

# Clear cache for specific video
y2karaoke cache clear VIDEO_ID
```

### Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output video path (default: `{title}_karaoke.mp4`) |
| `--offset` | Timing offset in seconds (negative = highlight earlier, positive = later, default: 0.0) |
| `--key` | Shift key by N semitones (-12 to +12, default: 0) |
| `--tempo` | Tempo multiplier (0.5 = half speed, 2.0 = double, default: 1.0) |
| `--audio-start` | Start audio processing from this many seconds into the track (skip intro; default: 0.0) |
| `--title` | Override song title used when searching for lyrics |
| `--artist` | Override artist used when searching for lyrics (useful for covers) |
| `--lyrics-file"` | Use lyrics from a local text or .lrc file as the text source |
| `--backgrounds` | Use video backgrounds extracted from the original YouTube video |
| `--work-dir` | Working directory for intermediate files (default: `~/.cache/karaoke/{video_id}`) |
| `--keep-files` | Keep intermediate files (audio, stems, etc.) |
| `--force` | Force re-download and re-process even if cached files exist |
| `--offline` | Run entirely from cached metadata/audio. When provided alongside `--force`, downloads are still skipped but processing (Whisper, lyrics alignment, rendering) reruns over the cached stems |

## Testing

```bash
source venv/bin/activate
pytest tests -v
```

Network/integration tests are skipped by default. To run them:

```bash
RUN_INTEGRATION_TESTS=1 pytest tests -v
# or
pytest tests -v --run-network
```

## Advanced: module-level CLIs

For debugging or experimenting with individual steps, you can run the core modules directly:

- **Download only (audio metadata + WAV)**
  ```bash
  python -m y2karaoke.core.downloader "https://youtube.com/watch?v=VIDEO_ID"
  ```
  Downloads the audio track to `./output` and prints basic metadata (title, artist, duration).

- **Separate vocals and instrumental for a local WAV file**
  ```bash
  python -m y2karaoke.core.separator path/to/audio.wav
  ```
  Writes separated stems into `./output` and prints the paths for vocals and instrumental.

- **Fetch and inspect lyrics/timing**
  ```bash
  python -m y2karaoke.core.lyrics "Song Title" "Artist Name" [optional_vocals.wav]
  ```
  Prints detected lines (and duet information when available). If you provide a vocals WAV, it can fall back to WhisperX transcription.

- **Apply audio effects (key/tempo) to an existing WAV**
  ```bash
  python -m y2karaoke.core.audio_effects input.wav output.wav <semitones> [tempo_multiplier]
  ```
  Example: `python -m y2karaoke.core.audio_effects input.wav output_key-3_tempo0.8.wav -3 0.8`

- **Test rendering (single preview frame)**
  ```bash
  python -m y2karaoke.core.renderer
  ```
  Renders a sample frame to `test_frame.png` for quickly checking fonts/layout.

- **Inspect video backgrounds / scene detection**
  ```bash
  python -m y2karaoke.core.backgrounds path/to/video.mp4
  ```
  Runs scene detection and saves a sample processed background to `test_background.png`.

## How It Works

1. **Download**: Downloads audio from YouTube using yt-dlp
2. **Separate**: Removes vocals using demucs AI model, keeps instrumental
3. **Lyrics**: Fetches synced lyrics when available and combines them with WhisperX transcription for hybrid alignment - synced lyrics provide accurate line timing, WhisperX provides precise word-level timing
4. **Audio Effects**: Applies key shift and/or tempo changes (if requested)
5. **Render**: Creates video with word-by-word highlighting, lyrics timing adjusted to match tempo

## International lyrics and romanization

For songs with non-Latin scripts, y2karaoke automatically romanizes lyrics before rendering:

- **Korean (Hangul)**: Romanized using `korean-romanizer` (Revised Romanization style) when available.
- **Chinese (Han characters)**: Romanized to **pinyin without tone marks** using `pypinyin`, while leaving non-Chinese text untouched.
- **Japanese (hiragana/katakana/kanji)**: Romanized to **romaji** using `pykakasi`, preserving word boundaries and spacing.
- **Arabic**: Romanized using simple transliteration mapping to Latin script.
- **Hebrew**: Romanized using simple transliteration mapping to Latin script.

Romanization is **best-effort** and only applied when the corresponding libraries are installed. If synced lyrics are in non-Latin scripts but Genius has romanized versions, the system will automatically romanize the synced lyrics to match for optimal hybrid alignment. Works seamlessly with mixed-language songs (e.g., English/Korean, English/Japanese).

## Video Style

- 1920x1080 resolution at 30fps
- Dark gradient background (blue to purple) by default, or dynamic video backgrounds when `--backgrounds` is enabled
- Four lines visible at a time with smart scrolling
- Long lines automatically split for better readability (75% screen width threshold)
- Current and already-sung words highlighted in gold (KaraFun-style — they stay gold once sung)
- White for upcoming words on the current and next line
- Optional singer-specific colors in duet songs
- Progress bar during long instrumental intros and breaks
- Intro splash screen and outro logo screen

## Caching

Intermediate files are cached in `~/.cache/karaoke/{video_id}/` by default, including downloaded audio, separated stems, processed instrumentals for particular key/tempo settings, and lyrics/timing metadata. This allows fast re-runs when adjusting timing offsets, key, or tempo without repeating the expensive separation and transcription steps.

## Technical Documentation

For detailed technical architecture, see [CLAUDE.md](CLAUDE.md). This document covers:
- Overall pipeline and orchestration
- Track identification (artist/title) from URLs and search queries
- Lyrics and timing fetching from multiple providers
- Audio retrieval and vocal separation
- Alignment algorithms and audio analysis
- Timing quality evaluation

For a concise summary of the selection logic and test coverage across the three core pillars, see [docs/logic-audit.md](docs/logic-audit.md).

## License

MIT
