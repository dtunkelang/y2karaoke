# y2karaoke

Generate karaoke videos from YouTube URLs or song titles with word-by-word highlighting.

## Features

- **Smart search**: Use YouTube URLs or just search by song title and artist
- Downloads audio from YouTube
- Separates vocals from instrumental using AI (demucs)
- **Hybrid timing system**: Combines synced lyrics (line timing) with Whisper-based alignment (word timing) for maximum accuracy
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

# Install with development tooling (pytest/black/flake8/mypy)
pip install -e ".[dev]"
pip check

# Or bootstrap everything in one step
./tools/bootstrap_dev.sh

# Optional: Install romanization libraries for non-Latin scripts
pip install korean-romanizer pypinyin pykakasi pyarabic

# Optional (macOS): Use fast Apple Vision OCR for visual bootstrap workflows
pip install -e ".[vision_macos]"
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
`cmudict` package (installed from project dependencies) to emulate a CMU-style
dictionary whenever the real `lex_lookup` binary is absent.

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
| `--lyrics-file` | Use lyrics from a local text or .lrc file as the text source |
| `--backgrounds` | Use video backgrounds extracted from the original YouTube video |
| `--work-dir` | Working directory for intermediate files (default: `~/.cache/karaoke/{video_id}`) |
| `--keep-files` | Keep intermediate files (audio, stems, etc.) |
| `--force` | Force re-download and re-process even if cached files exist |
| `--offline` | Run entirely from cached metadata/audio. When provided alongside `--force`, downloads are still skipped but processing (Whisper, lyrics alignment, rendering) reruns over the cached stems |

Artist/title strict mode:
- When both `--artist` and `--title` are provided for a YouTube URL, identification now stays strict on that pair (no candidate expansion to unrelated artists/titles).
- Minor provider-side normalization still applies (punctuation/case/format differences).

Whisper timing refinement notes:
- `Y2K_WHISPER_SILENCE_REFINEMENT=1` (default) enables silence-aware short-line retiming in the Whisper pipeline.
- Set `Y2K_WHISPER_SILENCE_REFINEMENT=0` to disable that pass for debugging/regression checks.
- `Y2KARAOKE_SINGER_COLOR_MODE=single` disables singer-specific line colors and uses one consistent text/highlight palette.

## Testing

```bash
source venv/bin/activate
pip install -e ".[dev]"
pytest tests -v
```

Benchmark seed set for timing quality work:
- `benchmarks/benchmark_songs.yaml` contains a curated core list (IDs, provider preference, tolerance hints).
- Validate it with: `make benchmark-validate`
- Run the benchmark suite and aggregate report:
  - `make benchmark-run`
  - Outputs are written to `benchmarks/results/<timestamp>/benchmark_report.{json,md}`
  - `benchmarks/results/latest.json` points to the latest JSON report path.
- Resume support (interruption-friendly):
  - The runner writes per-song checkpoints (`*_result.json`) and `benchmark_progress.json` after each song.
  - The runner writes per-song command logs (`*_generate.log`) in the run directory.
  - Resume the most recent run: `./venv/bin/python tools/run_benchmark_suite.py --resume-latest`
  - Resume a specific run directory: `./venv/bin/python tools/run_benchmark_suite.py --resume-run-dir benchmarks/results/<run_id>`
  - Safety: cached per-song results are reused only when core run options match (`offline/force/DTW mode/cache-dir/manifest`).
  - Override only if intentional: `--reuse-mismatched-results`
  - By default, resumed runs skip already-completed songs; use `--rerun-failed` or `--rerun-completed` to override.
- Background helper script (recommended for long runs):
  - `make benchmark-run-bg` (or `tools/run_benchmark_suite_bg.sh`)
  - This starts a nohup run with `--resume-latest` and prints a log file path to follow.
  - Heartbeat interval is configurable: `--heartbeat-sec 30` (default), and each heartbeat includes an inferred current stage when available.
  - Heartbeats also include CPU-aware hints; if logs are stale but CPU is high, they infer `separation` / `whisper` / `alignment` when possible.
  - Per-song output now includes phase transitions (`phase_start ...`), phase totals (`phase_summary ...`), and cache decisions (`cache_decisions ...`).
  - Quick status snapshot: `make benchmark-status` (or `./venv/bin/python tools/benchmark_status.py`)
  - Stop running benchmark suites: `make benchmark-kill` (preview only: `tools/kill_benchmark_suites.sh --dry-run`)
- Useful benchmark flags:
  - Strategy matrix:
    - `--strategy hybrid_dtw` (default): audio + lyrics + LRC timing with DTW map
    - `--strategy hybrid_whisper`: audio + lyrics without DTW LRC mapping
    - `--strategy whisper_only`: audio-only lyrics/timing
    - `--strategy lrc_only`: synced lyrics timing only
  - Scenario isolation:
    - `--scenario default` (default): normal provider behavior
    - `--scenario lyrics_no_timing`: ignore provider LRC timestamps and derive timing from lyrics text + audio
  - Offline cached-only run: `./venv/bin/python tools/run_benchmark_suite.py --offline`
  - Run one song for debugging: `./venv/bin/python tools/run_benchmark_suite.py --match "Papaoutai" --max-songs 1`
  - Disable DTW mapping for A/B checks: `./venv/bin/python tools/run_benchmark_suite.py --no-whisper-map-lrc-dtw`
  - Rebaseline gold from successful reports (all selected songs): `./venv/bin/python tools/run_benchmark_suite.py --rebaseline`
  - Rebaseline one song safely: `./venv/bin/python tools/run_benchmark_suite.py --match "bad guy" --max-songs 1 --rebaseline`
  - Run strategy matrix and emit combined report: `make benchmark-matrix`
  - Matrix JSON now includes `recommendations` (best strategy by p95/mean start error, low-confidence ratio, DTW coverage, runtime, and quality/runtime balance)
  - Recommend default strategy/thresholds from prior reports: `make benchmark-recommend`

Benchmark metric interpretation:
- `dtw_line_coverage`: fraction of lyric lines with usable DTW anchor/match. Lower values often mean noisy or duration-mismatched LRC.
- `dtw_word_coverage`: fraction of words matched through DTW/Whisper alignment. This is typically lower than line coverage.
- `dtw_phonetic_similarity_coverage`: matched words with sufficiently strong phonetic similarity; useful for cross-language or misspelling-heavy cases.
- `agreement_start_mean_abs_sec` and `agreement_start_p95_abs_sec`: independent line-start agreement metrics (when DTW anchors are available); p95 is the better regression guard.
- `whisper_anchor_start_mean_abs_sec` / `whisper_anchor_start_p95_abs_sec`: diagnostic-only line-start deltas against nearest Whisper segments (use for debugging, not strategy ranking).
- `low_confidence_lines`: lines where Whisper confidence is weak; inspect these first during debugging.
- `null` metrics: expected when a song path used onset/LRC timing without DTW-based reference comparisons.

## Gold timing editor

For benchmark ground-truth curation, use the local gold timing editor:

```bash
python tools/gold_timing_editor.py --host 127.0.0.1 --port 8765
```

Then open `http://127.0.0.1:8765`.

- Canonical format: `*.gold.json` (human-readable JSON with word-level `start`/`end` timings).
- Input: existing timing report JSON or existing `*.gold.json`.
- Editing: drag intervals/handles plus keyboard nudging at 0.1s increments.
- Validation: forbids word overlaps and allows gaps.

See `docs/gold_timing_editor.md` for schema and workflow details.

## Karaoke bootstrap tool

To auto-seed/refine word timings from a karaoke YouTube version (visual-only):

```bash
./venv/bin/python tools/bootstrap_gold_from_karaoke.py \
  --artist "Billie Eilish" \
  --title "bad guy" \
  --show-candidates \
  --output benchmarks/gold_set/02_billie-eilish-bad-guy.karaoke-seed.gold.json \
  --report-json benchmarks/gold_set/02_billie-eilish-bad-guy.bootstrap-report.json \
  --min-detectability 0.45 \
  --min-word-level-score 0.15
```

- If `--candidate-url` is omitted, the tool searches YouTube for karaoke candidates, scores each for visual suitability, and picks the best one.
- `--show-candidates` prints ranked candidates with detectability/word-level metrics.
- By default, the tool enforces suitability gates (`--min-detectability`, `--min-word-level-score`). Use `--allow-low-suitability` to override.
- OCR frame sampling is cached under `--work-dir` (default `.cache/karaoke_bootstrap`) to speed reruns.
- Use `--raw-ocr-cache-version` to invalidate only OCR-frame caches when bootstrap extraction logic changes (downloaded video/audio/LRC caches can remain reused).
- If the selected candidate video was already downloaded during ranking, audio is extracted locally from that file first, with direct audio download as fallback.
- `--report-json` writes a structured report with candidate rankings, selected candidate metrics, and runtime settings.
- The tool detects line windows from karaoke frames via OCR and infers word timing from lyric highlight progress in those windows (no audio transcription).
- When word-level highlight transitions are not observable, refinement falls back to line-level transition timing and distributes per-word timings within the detected line window.
- For low word-level suitability candidates, the tool skips expensive native-FPS per-word refinement and relies on the line-level fallback path.

See [docs/karaoke_visual_bootstrap.md](docs/karaoke_visual_bootstrap.md) for algorithm and technical details.

Preferred local workflow:

```bash
make check
```

Bootstrap quality guardrails for visual-seeded gold files:

```bash
make bootstrap-quality-guardrails
```

Deterministic visual-extraction metric guardrails (runs `run_visual_eval.py` against the
visual benchmark manifest (`benchmarks/visual_benchmark_songs.yaml`) and seeded visual gold set,
snapshots/uses local LRC references in `benchmarks/reference_lrc/`,
and enforces committed F1 thresholds from `benchmarks/visual_eval_guardrails.json`):

```bash
make visual-eval-guardrails
```

`Counting Stars` is intentionally kept in `benchmarks/visual_dev_songs.yaml` for
block-first extractor development, but excluded from the guardrail manifest because
this karaoke source's late-section lyric sequence does not match the LRC reference.

To refresh only the metrics summary (without enforcing guardrails):

```bash
make visual-eval
```

Token-order quality comparison of visual extraction vs synced LRC
(line-boundary agnostic; parenthetical LRC words treated as optional by default):

```bash
./venv/bin/python tools/evaluate_visual_lyrics_quality.py \
  --gold-json /tmp/shape_nextpass.gold.json \
  --title "Shape of You" \
  --artist "Ed Sheeran" \
  --output-json /tmp/shape_nextpass.lyrics-quality.json
```

The tool prints two scores:
- `strict`: direct token-order comparison to LRC (parenthetical words optional by default).
- `repeat_capped`: same comparison with excessive repeated LRC lines capped to extracted repeat counts (helps separate true extraction misses from karaoke-vs-LRC repeat-count differences).

Threshold calibration from prior bootstrap reports:

```bash
make bootstrap-calibrate
```

Test organization:
- `tests/unit/` for subsystem unit tests
- `tests/integration/` for network/integration coverage
- `tests/e2e/` for end-to-end checks

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
  python -m y2karaoke.core.components.audio.downloader "https://youtube.com/watch?v=VIDEO_ID"
  ```
  Downloads the audio track to `./output` and prints basic metadata (title, artist, duration).

- **Separate vocals and instrumental for a local WAV file**
  ```bash
  python -m y2karaoke.core.components.audio.separator path/to/audio.wav
  ```
  Writes separated stems into `./output` and prints the paths for vocals and instrumental.

- **Fetch and inspect lyrics/timing**
  ```bash
  python -m y2karaoke.core.components.lyrics.api "Song Title" "Artist Name" [optional_vocals.wav]
  ```
  Prints detected lines (and duet information when available). If you provide a vocals WAV, it can fall back to Whisper transcription.

- **Apply audio effects (key/tempo) to an existing WAV**
  ```bash
  python -m y2karaoke.core.components.audio.audio_effects input.wav output.wav <semitones> [tempo_multiplier]
  ```
  Example: `python -m y2karaoke.core.components.audio.audio_effects input.wav output_key-3_tempo0.8.wav -3 0.8`

- **Inspect video backgrounds / scene detection**
  ```bash
  python -m y2karaoke.core.components.render.backgrounds path/to/video.mp4
  ```
  Runs scene detection and saves a sample processed background to `test_background.png`.

## How It Works

1. **Download**: Downloads audio from YouTube using yt-dlp
2. **Separate**: Removes vocals using demucs AI model, keeps instrumental
3. **Lyrics**: Fetches synced lyrics when available and combines them with Whisper transcription for hybrid alignment - synced lyrics provide accurate line timing, Whisper provides precise word-level timing
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

For detailed technical architecture, see [ARCHITECTURE.md](ARCHITECTURE.md). This document covers:
- Overall pipeline and orchestration
- Track identification (artist/title) from URLs and search queries
- Lyrics and timing fetching from multiple providers
- Audio retrieval and vocal separation
- Alignment algorithms and audio analysis
- Timing quality evaluation

Subsystem facades now live under `src/y2karaoke/pipeline/`:
- `pipeline/identify` for track metadata resolution and candidate selection
- `pipeline/lyrics` for lyrics acquisition + quality-aware timing flows
- `pipeline/audio` for media download, separation, and audio transforms
- `pipeline/alignment` for timing evaluation + Whisper alignment orchestration
- `pipeline/render` for video/background rendering entrypoints

For subsystem documentation and development workflow, see:
- [docs/pipelines/karaoke.md](docs/pipelines/karaoke.md)
- [docs/pipelines/lyrics.md](docs/pipelines/lyrics.md)
- [docs/pipelines/whisper.md](docs/pipelines/whisper.md)
- [docs/development.md](docs/development.md)

## License

MIT
