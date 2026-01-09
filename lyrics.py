"""Lyrics fetching and timing using syncedlyrics + whisper fallback."""

import re
from dataclasses import dataclass
from typing import Optional

import syncedlyrics


@dataclass
class Word:
    """A word with timing information."""
    text: str
    start_time: float  # seconds
    end_time: float    # seconds


@dataclass
class Line:
    """A line of lyrics with words."""
    words: list[Word]
    start_time: float
    end_time: float

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def parse_lrc_timestamp(ts: str) -> float:
    """Parse LRC timestamp [mm:ss.xx] to seconds."""
    match = re.match(r'\[(\d+):(\d+)\.(\d+)\]', ts)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        centiseconds = int(match.group(3))
        return minutes * 60 + seconds + centiseconds / 100
    return 0.0


def parse_lrc(lrc_text: str) -> list[Line]:
    """Parse LRC format lyrics into Lines with word timing."""
    lines = []
    raw_lines = []

    for line in lrc_text.strip().split('\n'):
        # Match timestamp and text
        match = re.match(r'(\[\d+:\d+\.\d+\])\s*(.*)', line)
        if match:
            timestamp = parse_lrc_timestamp(match.group(1))
            text = match.group(2).strip()
            if text:  # Skip empty lines
                raw_lines.append((timestamp, text))

    # Create Lines with estimated word timing
    for i, (start_time, text) in enumerate(raw_lines):
        # Get end time from next line or add 3 seconds
        if i + 1 < len(raw_lines):
            end_time = raw_lines[i + 1][0]
        else:
            end_time = start_time + 3.0

        # Split into words and estimate timing
        word_texts = text.split()
        if not word_texts:
            continue

        line_duration = end_time - start_time
        word_duration = line_duration / len(word_texts)

        words = []
        for j, word_text in enumerate(word_texts):
            word_start = start_time + j * word_duration
            word_end = word_start + word_duration
            words.append(Word(text=word_text, start_time=word_start, end_time=word_end))

        lines.append(Line(words=words, start_time=start_time, end_time=end_time))

    return lines


def fetch_synced_lyrics(title: str, artist: str) -> Optional[str]:
    """Fetch synced lyrics using syncedlyrics library."""
    search_term = f"{artist} {title}"
    print(f"Searching for synced lyrics: {search_term}")

    try:
        lrc = syncedlyrics.search(search_term, synced_only=True)
        if lrc:
            print("Found synced lyrics!")
            return lrc
    except Exception as e:
        print(f"Error fetching lyrics: {e}")

    return None


def get_lyrics_with_whisper(vocals_path: str, plain_lyrics: Optional[str] = None) -> list[Line]:
    """
    Use Whisper to get word-level timestamps from vocals.

    If plain_lyrics is provided, uses them as a guide.
    Otherwise, transcribes the audio directly.
    """
    print("Using Whisper for word-level timing...")

    import whisper_timestamped as whisper

    # Load model (medium is good balance of speed/accuracy)
    print("Loading Whisper model...")
    model = whisper.load_model("medium")

    # Transcribe with word timestamps
    print("Transcribing audio...")
    result = whisper.transcribe(model, vocals_path, language="en")

    lines = []
    for segment in result.get("segments", []):
        words = []
        for word_data in segment.get("words", []):
            words.append(Word(
                text=word_data["text"].strip(),
                start_time=word_data["start"],
                end_time=word_data["end"],
            ))

        if words:
            lines.append(Line(
                words=words,
                start_time=segment["start"],
                end_time=segment["end"],
            ))

    return lines


def get_lyrics(title: str, artist: str, vocals_path: Optional[str] = None) -> list[Line]:
    """
    Get lyrics with timing.

    1. Try to fetch synced LRC lyrics
    2. Fall back to Whisper if needed
    """
    # Try synced lyrics first
    lrc = fetch_synced_lyrics(title, artist)

    if lrc:
        lines = parse_lrc(lrc)
        if lines:
            return lines

    # Fall back to Whisper
    if vocals_path:
        return get_lyrics_with_whisper(vocals_path)

    raise RuntimeError("Could not get lyrics: no synced lyrics found and no vocals path provided")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python lyrics.py <title> <artist> [vocals_path]")
        sys.exit(1)

    title = sys.argv[1]
    artist = sys.argv[2]
    vocals_path = sys.argv[3] if len(sys.argv) > 3 else None

    lines = get_lyrics(title, artist, vocals_path)

    print(f"\nFound {len(lines)} lines:")
    for line in lines[:10]:  # Show first 10 lines
        print(f"[{line.start_time:.2f}s] {line.text}")
