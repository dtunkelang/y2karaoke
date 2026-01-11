# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project overview

This repository (`y2karaoke`) generates karaoke-style videos from YouTube URLs with word-by-word lyric highlighting. The main entrypoint is `karaoke.py`, which orchestrates a multi-step pipeline:

1. Download audio from YouTube.
2. Separate vocals from instrumental using an AI model (demucs via `audio-separator`).
3. Transcribe vocals with whisperx for accurate word-level timing via forced alignment.
4. Optionally apply audio effects (key shifting and tempo changes) to the instrumental.
5. Render a 1080p video with KaraFun-style word highlighting.
6. Optionally upload the rendered video to YouTube as unlisted.

There is currently no automated test suite configured (no `tests/` directory or test runner configuration).

## Environment and setup

This project targets **Python 3.12** and requires `ffmpeg` to be installed on the system.

Create and activate a virtual environment and install dependencies:

```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Key Python dependencies (see `requirements.txt`) include:

- `yt-dlp` for YouTube downloading
- `audio-separator` (demucs) for stem separation
- `syncedlyrics` for fetching lyrics text
- `whisperx` for word-level timing via forced alignment (uses wav2vec2)
- `moviepy`, `Pillow`, `numpy` for video rendering
- `librosa`, `soundfile`, `pydub` for audio processing
- Google API client libraries for YouTube uploads

## Common commands

### Run the main tool

Basic usage (main CLI, from the repo root with the venv activated):

```bash
python karaoke.py "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4
```

Important CLI options (see `karaoke.py` and README):

- `-o, --output`: Output video path (default: `{title}_karaoke.mp4`).
- `--offset`: Timing offset in seconds (negative = highlight earlier, default: `0.0`).
- `--key`: Key shift in semitones (`-12` to `+12`, default `0`).
- `--tempo`: Tempo multiplier (e.g., `0.8` = slower, `1.2` = faster; default `1.0`).
- `--work-dir`: Working directory for intermediate files (default: `~/.cache/karaoke/{video_id}`).
- `--keep-files`: Keep intermediate files instead of cleaning up (when cleanup is enabled).
- `--force`: Ignore cached files and re-download / re-process.
- `--upload`: Upload to YouTube as unlisted after rendering.
- `--no-upload`: Suppress the upload prompt (useful for batch or scripted runs).

Example combinations (all from repo root):

```bash
# Adjust timing offset
python karaoke.py "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4 --offset -0.5

# Shift key and adjust tempo
python karaoke.py "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4 --key -3 --tempo 0.8

# Force reprocessing and upload automatically
python karaoke.py "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4 --force --upload
```

### Module-level CLIs (for debugging individual steps)

Each major module can be run directly for focused debugging:

```bash
# Download only
python downloader.py "https://youtube.com/watch?v=VIDEO_ID"

# Separate vocals/instrumental for a local WAV file
python separator.py path/to/audio.wav

# Fetch and inspect lyrics/timing
python lyrics.py "Song Title" "Artist Name" [optional_vocals.wav]

# Apply audio effects
python audio_effects.py input.wav output.wav <semitones> [tempo_multiplier]

# Upload an existing MP4 to YouTube
python uploader.py output.mp4 [title] [artist]
```

`renderer.py` includes a small self-test that writes `test_frame.png` if run as a script; this is useful for verifying font/loading and layout locally.

## Architecture and data flow

### High-level pipeline

The central orchestration lives in `karaoke.py`:

1. **Argument parsing and validation**
   - Uses `argparse` to parse CLI options (URL, output path, offset, key, tempo, caching and upload flags).
   - Validates `--key` is between `-12` and `+12` and that `--tempo` is positive.

2. **Video ID and caching setup**
   - Extracts a YouTube video ID from the URL via regex; if no ID is found, uses an MD5 hash prefix of the URL.
   - Constructs a working directory as either:
     - `args.work_dir`, if provided, or
     - `~/.cache/karaoke/{video_id}` by default.
   - Stores basic metadata (`title`, `artist`) in `metadata.json` within the work dir for reuse across runs.

3. **Download step (`downloader.download_audio`)**
   - If a non-stem `.wav` already exists in `work_dir` and `metadata.json` is present (and `--force` is not set), it reuses the cached audio and metadata.
   - Otherwise, `download_audio` uses `yt-dlp` to:
     - Extract video info (title, artist/uploader, duration).
     - Download the best audio-only stream and convert to WAV via ffmpeg.
   - Filenames are sanitized via `sanitize_filename` to avoid unsupported characters.

4. **Separation step (`separator.separate_vocals`)**
   - If previously-separated stems exist (`*_(Vocals)_*.wav` and `*_instrumental.wav`) and `--force` is not set, they are reused.
   - Otherwise, `audio_separator.Separator` is initialized with `output_format="wav"` and the `htdemucs_ft.yaml` model is loaded.
   - The demucs model produces multiple stems; the code identifies:
     - A vocals stem (by name containing `vocals`).
     - An instrumental stem (by name containing `instrumental` or `no_vocals`).
   - If no explicit instrumental stem is found, non-vocal stems (e.g., bass, drums, other) are mixed into an instrumental using `pydub` (see `mix_stems`).

5. **Lyrics step (`lyrics.get_lyrics`)**
   - Optionally fetches lyrics text via `syncedlyrics.search` for reference (currently not used for filtering).
   - Uses `whisperx` to transcribe the vocals track with accurate word-level timing:
     - Loads the `medium` Whisper model for transcription.
     - Applies `whisperx.align()` with wav2vec2 for forced alignment to get precise word boundaries.
     - Handles words missing timestamps by interpolating from segment timing.
   - Post-processes the lyrics:
     - `fix_word_timing`: Fixes unrealistic word durations (e.g., first words that are too long) by deriving timing from neighboring words.
     - `split_long_lines`: Recursively splits lines longer than 45 characters to fit on screen.

6. **Audio effects step (`audio_effects.process_audio`)**
   - If key and tempo are unchanged (`key == 0` and `tempo == 1.0`), the instrumental is copied unchanged.
   - Otherwise, `process_audio` loads the instrumental with `librosa`, optionally:
     - Applies pitch shift via `librosa.effects.pitch_shift`.
     - Applies tempo adjustment via `librosa.effects.time_stretch`.
   - The processed track is written to a new WAV file in the work dir, named according to the key and tempo settings.
   - When tempo changes, `scale_lyrics_timing` adjusts all lyric timestamps by dividing by the tempo multiplier so that highlighting stays aligned with the new audio speed.

7. **Rendering step (`renderer.render_karaoke_video`)**
   - Loads the instrumental with `moviepy.AudioFileClip` to determine the final duration.
   - Precomputes a 1920x1080 gradient background and selects a font (tries several common system fonts, falling back to the default).
   - For each frame (30 fps), `render_frame`:
     - Determines the current and next lyric lines relative to the adjusted time (`t - offset`).
     - Renders up to two lines of text, centered vertically.
     - Measures word widths to center lines horizontally.
     - Uses KaraFun-style word highlighting:
       - Words stay gold once they've been sung (no gray for past words).
       - Future words on the current line: white.
       - Next line: all white.
   - Assembles a `VideoClip` from the frame function, attaches the audio, and writes an H.264/AAC MP4 file.

8. **YouTube upload step (`uploader.upload_video`)**
   - If `--upload` is passed, or if the user answers `y/yes` to the interactive prompt (and `--no-upload` is not set), the tool uploads the rendered MP4.
   - Upload uses the YouTube Data API v3 and returns an unlisted video URL.

### Core modules and responsibilities

- `karaoke.py`: CLI entrypoint and top-level orchestration of the entire pipeline, including caching logic, error handling, and user prompts for upload.
- `downloader.py`: Wraps `yt-dlp` to extract metadata and download audio-only content as WAV.
- `separator.py`: Handles vocal/instrumental separation via `audio-separator`/demucs and, when necessary, mixes non-vocal stems into a synthesized instrumental.
- `lyrics.py`: Encapsulates lyrics fetching and timing using whisperx for forced alignment. Includes post-processing for word timing fixes and line splitting. Defines the `Word` and `Line` dataclasses used across the codebase.
- `audio_effects.py`: Applies key shifting and tempo changes to audio, including a combined `process_audio` helper for the typical pipeline.
- `renderer.py`: Renders the final karaoke video frames and composes the movie file with audio.
- `uploader.py`: Manages YouTube OAuth credentials and uploads MP4 files with generated metadata (title/description).

These modules are designed to be individually runnable for debugging and reusable from the main pipeline.

## Caching and working directories

- Default cache root: `~/.cache/karaoke` (defined as `CACHE_DIR` in `karaoke.py`).
- For each input URL, a `video_id` is extracted and used to create a subdirectory: `~/.cache/karaoke/{video_id}`.
- Cached artifacts include:
  - Original downloaded audio WAV.
  - Separated stems (vocals and instrumental) from demucs.
  - Processed instrumentals for particular key/tempo settings.
  - `metadata.json` containing `title` and `artist`.

Key behaviors:

- On subsequent runs for the same URL (and without `--force`), the pipeline reuses existing audio and stems, which is critical because separation and Whisper transcription are expensive.
- `--work-dir` can override the default cache location to use a custom directory (useful for experimentation or debugging without polluting the global cache).
- `--force` forces re-download and re-processing even if cached files exist.

## YouTube upload configuration

YouTube authentication and configuration are handled in `uploader.py`:

- OAuth client secrets must be saved as:
  - `~/.cache/karaoke/client_secrets.json`
- After the first successful OAuth flow, user credentials are cached at:
  - `~/.cache/karaoke/youtube_credentials.pickle`
- The upload scope is `https://www.googleapis.com/auth/youtube.upload`.
- Videos are uploaded with:
  - Privacy status: `unlisted`.
  - Category: Music (`categoryId = 10`).
  - `selfDeclaredMadeForKids = False`.

On first use, `upload_video` will open a local browser window for user consent and then cache credentials for future uploads.

## Operational notes for agents

- Several steps are **computationally heavy** and/or **network-dependent**:
  - Demucs-based separation (`audio-separator`) can be slow and memory-intensive, especially for long tracks.
  - whisperx transcription and alignment may be time-consuming; the alignment step downloads wav2vec2 models on first use.
  - Synced lyrics fetching and YouTube uploads depend on external services; they may fail due to network or API issues.
- When modifying the pipeline, consider the caching structure in `karaoke.py` so that repeated operations (download, separation, transcription) are not needlessly redone.
- There is no current automated test harness; if tests are added, prefer to mock external services (YouTube, remote lyrics APIs, demucs, Whisper) and to keep heavy operations out of the fast test path.
