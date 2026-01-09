# y2karaoke

Generate karaoke videos from YouTube URLs with word-by-word highlighting.

## Features

- Downloads audio from YouTube
- Separates vocals from instrumental using AI (demucs)
- Fetches time-synced lyrics automatically
- Falls back to Whisper for word-level timing if needed
- Renders KaraFun-style karaoke videos with word highlighting
- Caches intermediate files for fast re-rendering

## Installation

### Prerequisites

- Python 3.10+
- ffmpeg

### Setup

```bash
# Clone the repository
git clone https://github.com/dtunkelang/y2karaoke.git
cd y2karaoke

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Basic usage
python karaoke.py "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4

# Adjust timing offset (negative = highlight earlier)
python karaoke.py "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4 --offset -0.5

# Force reprocess (ignore cache)
python karaoke.py "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4 --force
```

### Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output video path (default: `{title}_karaoke.mp4`) |
| `--offset` | Timing offset in seconds (default: -0.3) |
| `--work-dir` | Working directory for intermediate files |
| `--keep-files` | Keep intermediate files |
| `--force` | Force re-download and re-process |

## How It Works

1. **Download**: Downloads audio from YouTube using yt-dlp
2. **Separate**: Removes vocals using demucs AI model, keeps instrumental
3. **Lyrics**: Fetches synced lyrics from online sources (LRCLIB, etc.)
4. **Fallback**: If no synced lyrics found, uses Whisper for word timing
5. **Render**: Creates video with word-by-word highlighting

## Video Style

- 1920x1080 resolution at 30fps
- Dark gradient background (blue to purple)
- Two lines visible at a time
- Gold highlight for current word
- Gray for already-sung words
- White for upcoming words

## Caching

Intermediate files are cached in `~/.cache/karaoke/{video_id}/` by default. This allows fast re-rendering when adjusting timing offsets.

## License

MIT
