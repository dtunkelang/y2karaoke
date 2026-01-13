# y2karaoke

Generate karaoke videos from YouTube URLs with word-by-word highlighting.

## Features

- Downloads audio from YouTube
- Separates vocals from instrumental using AI (demucs)
- Fetches synced lyrics when available and uses WhisperX for precise word-level timing
- Renders KaraFun-style karaoke videos with word-by-word highlighting
- Optional dynamic video backgrounds generated from the original YouTube video
- Duet-friendly visuals with singer-specific coloring and instrumental-break progress bar
- **Key shifting** (-12 to +12 semitones) to match your vocal range
- **Tempo control** (speed up or slow down) with synchronized lyrics
- **YouTube upload** with unlisted/shareable link
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

# Install dependencies
pip install -r requirements.txt
```

### Troubleshooting

If you see `ModuleNotFoundError: No module named 'onnxruntime'`, your Python version is likely too new. Verify you're using Python 3.12:

```bash
python --version  # Should show Python 3.12.x
```

If not, recreate the virtual environment with `python3.12 -m venv venv`.

## Usage

```bash
# Basic usage
python karaoke.py "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4

# Adjust timing offset (negative = highlight earlier, positive = later)
python karaoke.py "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4 --offset -0.5

# Shift key down 3 semitones (for a lower vocal range)
python karaoke.py "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4 --key -3

# Slow down to 80% speed (great for learning)
python karaoke.py "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4 --tempo 0.8

# Use dynamic video backgrounds from the original YouTube video
python karaoke.py "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4 --backgrounds

# Combine options: shift key up, slow down, adjust timing
python karaoke.py "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4 --key +2 --tempo 0.9 --offset -0.5

# Force reprocess (ignore cache)
python karaoke.py "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4 --force

# Upload to YouTube after rendering (unlisted link)
python karaoke.py "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4 --upload

# Skip the upload prompt (for scripts/batch mode)
python karaoke.py "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4 --no-upload
```

### Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output video path (default: `{title}_karaoke.mp4`) |
| `--offset` | Timing offset in seconds (negative = highlight earlier, positive = later, default: 0.0) |
| `--key` | Shift key by N semitones (-12 to +12, default: 0) |
| `--tempo` | Tempo multiplier (0.5 = half speed, 2.0 = double, default: 1.0) |
| `--upload` | Upload to YouTube as unlisted video after rendering |
| `--no-upload` | Skip the upload prompt (for batch/script mode) |
| `--backgrounds` | Use video backgrounds extracted from the original YouTube video |
| `--work-dir` | Working directory for intermediate files (default: `~/.cache/karaoke/{video_id}`) |
| `--keep-files` | Keep intermediate files (audio, stems, etc.) |
| `--force` | Force re-download and re-process even if cached files exist |

## Advanced: module-level CLIs

For debugging or experimenting with individual steps, you can run the core modules directly:

- **Download only (audio metadata + WAV)**
  ```bash
  python downloader.py "https://youtube.com/watch?v=VIDEO_ID"
  ```
  Downloads the audio track to `./output` and prints basic metadata (title, artist, duration).

- **Separate vocals and instrumental for a local WAV file**
  ```bash
  python separator.py path/to/audio.wav
  ```
  Writes separated stems into `./output` and prints the paths for vocals and instrumental.

- **Fetch and inspect lyrics/timing**
  ```bash
  python lyrics.py "Song Title" "Artist Name" [optional_vocals.wav]
  ```
  Prints detected lines (and duet information when available). If you provide a vocals WAV, it can fall back to WhisperX transcription.

- **Apply audio effects (key/tempo) to an existing WAV**
  ```bash
  python audio_effects.py input.wav output.wav <semitones> [tempo_multiplier]
  ```
  Example: `python audio_effects.py input.wav output_key-3_tempo0.8.wav -3 0.8`

- **Test rendering (single preview frame)**
  ```bash
  python renderer.py
  ```
  Renders a sample frame to `test_frame.png` for quickly checking fonts/layout.

- **Upload an existing MP4 to YouTube**
  ```bash
  python uploader.py output.mp4 [title] [artist]
  ```
  Uses your saved OAuth credentials to upload as an unlisted video.

- **Inspect video backgrounds / scene detection**
  ```bash
  python backgrounds.py path/to/video.mp4
  ```
  Runs scene detection and saves a sample processed background to `test_background.png`.

## How It Works

1. **Download**: Downloads audio from YouTube using yt-dlp
2. **Separate**: Removes vocals using demucs AI model, keeps instrumental
3. **Lyrics**: Fetches lyrics text from online sources when available and uses WhisperX-based transcription and forced alignment on the vocal track for robust word-level timing.
4. **Audio Effects**: Applies key shift and/or tempo changes (if requested)
5. **Render**: Creates video with word-by-word highlighting, lyrics timing adjusted to match tempo

## Video Style

- 1920x1080 resolution at 30fps
- Dark gradient background (blue to purple) by default, or dynamic video backgrounds when `--backgrounds` is enabled
- Two lines visible at a time
- Current and already-sung words highlighted in gold (KaraFun-style — they stay gold once sung)
- White for upcoming words on the current and next line
- Optional singer-specific colors in duet songs
- Progress bar during long instrumental intros and breaks
- Intro splash screen and outro logo screen

## Caching

Intermediate files are cached in `~/.cache/karaoke/{video_id}/` by default, including downloaded audio, separated stems, processed instrumentals for particular key/tempo settings, and lyrics/timing metadata. This allows fast re-runs when adjusting timing offsets, key, or tempo without repeating the expensive separation and transcription steps.

## YouTube Upload Setup

To enable YouTube uploads, you need to set up Google OAuth credentials:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select existing)
3. Enable the **YouTube Data API v3**
4. Go to **Credentials** → **Create Credentials** → **OAuth client ID**
5. Select **Desktop app** as application type
6. Download the JSON file
7. Save it as `~/.cache/karaoke/client_secrets.json`

On first upload, a browser window will open for authentication. Credentials are cached for future use.

## License

MIT
