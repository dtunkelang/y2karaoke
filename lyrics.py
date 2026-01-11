"""Lyrics fetching with forced alignment for accurate word-level timing."""

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


def extract_lyrics_text(lrc_text: str) -> list[str]:
    """Extract plain text lines from LRC format (no timing)."""
    lines = []
    for line in lrc_text.strip().split('\n'):
        match = re.match(r'\[\d+:\d+\.\d+\]\s*(.*)', line)
        if match:
            text = match.group(1).strip()
            if text:
                lines.append(text)
    return lines


def fetch_synced_lyrics(title: str, artist: str) -> Optional[str]:
    """Fetch synced lyrics using syncedlyrics library."""
    search_term = f"{artist} {title}"
    print(f"Searching for lyrics: {search_term}")

    try:
        lrc = syncedlyrics.search(search_term, synced_only=True)
        if lrc:
            print("Found lyrics online!")
            return lrc
    except Exception as e:
        print(f"Error fetching lyrics: {e}")

    return None


def transcribe_and_align(vocals_path: str, lyrics_text: Optional[list[str]] = None) -> list[Line]:
    """
    Transcribe audio with word-level alignment.

    Uses whisperx for accurate word-level timestamps via forced alignment.

    Args:
        vocals_path: Path to vocals audio file
        lyrics_text: Optional list of lyrics lines (not used - kept for API compatibility)

    Returns:
        List of Line objects with word timing
    """
    import whisperx
    import torch

    # Fix for PyTorch 2.6+ weights_only=True compatibility with pyannote
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        # Force weights_only=False to handle pyannote/omegaconf models
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"Loading whisperx model (device: {device})...")
    model = whisperx.load_model("medium", device, compute_type=compute_type, language="en")

    print("Transcribing audio...")
    audio = whisperx.load_audio(vocals_path)
    result = model.transcribe(audio, batch_size=16)

    # Load alignment model for word-level timestamps
    print("Aligning words to audio...")
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device)

    # Convert to our Line/Word format
    # Handle words missing timestamps by interpolating from segment timing
    lines = []
    for segment in result.get("segments", []):
        segment_start = segment.get("start", 0.0)
        segment_end = segment.get("end", segment_start + 1.0)
        segment_words = segment.get("words", [])

        if not segment_words:
            continue

        # First pass: collect words and identify which have timestamps
        word_infos = []
        for word_data in segment_words:
            word_text = word_data.get("word", "").strip()
            if not word_text:
                continue
            has_timing = "start" in word_data and "end" in word_data
            word_infos.append({
                "text": word_text,
                "start": word_data.get("start"),
                "end": word_data.get("end"),
                "has_timing": has_timing,
            })

        if not word_infos:
            continue

        # Second pass: interpolate missing timestamps
        # Use segment timing and distribute evenly for words without timestamps
        words = []

        # Find runs of words without timing and interpolate
        i = 0
        while i < len(word_infos):
            wi = word_infos[i]

            if wi["has_timing"]:
                words.append(Word(
                    text=wi["text"],
                    start_time=wi["start"],
                    end_time=wi["end"],
                ))
                i += 1
            else:
                # Find the run of words without timing
                run_start_idx = i
                while i < len(word_infos) and not word_infos[i]["has_timing"]:
                    i += 1
                run_end_idx = i

                # Determine time bounds for this run
                if run_start_idx == 0:
                    # Run starts at beginning - use segment start
                    time_start = segment_start
                else:
                    # Use end time of previous word
                    time_start = words[-1].end_time + 0.05

                if run_end_idx >= len(word_infos):
                    # Run goes to end - use segment end
                    time_end = segment_end
                else:
                    # Use start time of next timed word
                    time_end = word_infos[run_end_idx]["start"] - 0.05

                # Distribute time evenly among words in run
                run_count = run_end_idx - run_start_idx
                if time_end > time_start and run_count > 0:
                    duration_per_word = (time_end - time_start) / run_count
                    for j in range(run_count):
                        word_start = time_start + j * duration_per_word
                        word_end = word_start + duration_per_word - 0.02
                        words.append(Word(
                            text=word_infos[run_start_idx + j]["text"],
                            start_time=word_start,
                            end_time=max(word_end, word_start + 0.1),
                        ))

        if words:
            lines.append(Line(
                words=words,
                start_time=words[0].start_time,
                end_time=words[-1].end_time,
            ))

    # Fix bad word timing (first words, words after long gaps)
    lines = fix_word_timing(lines)

    # Split lines that are too wide for the screen
    lines = split_long_lines(lines)

    return lines


def fix_word_timing(lines: list[Line], max_word_duration: float = 2.0, max_gap: float = 2.0) -> list[Line]:
    """
    Fix unrealistic word timing from whisperx alignment.

    Strategy: Keep original timestamps for well-aligned words, but fix
    words with bad timing by deriving from neighboring words.
    """
    # First pass: collect durations of "good" words to calculate average
    good_durations = []
    for line in lines:
        for word in line.words:
            duration = word.end_time - word.start_time
            # Consider a word "good" if duration is reasonable
            if 0.05 < duration < max_word_duration:
                good_durations.append(duration)

    # Calculate average duration from good words (fallback to 0.4s)
    if good_durations:
        avg_duration = sum(good_durations) / len(good_durations)
    else:
        avg_duration = 0.4

    # Second pass: fix problematic words
    fixed_lines = []

    for line in lines:
        if not line.words or len(line.words) < 2:
            fixed_lines.append(line)
            continue

        fixed_words = []

        for i, word in enumerate(line.words):
            duration = word.end_time - word.start_time
            is_first = (i == 0)
            is_last = (i == len(line.words) - 1)

            # Check if this word needs fixing
            needs_fix = False
            fix_from_next = False
            fix_from_prev = False

            # First word with bad duration or big gap to next word
            if is_first and not is_last:
                next_word = line.words[i + 1]
                gap_to_next = next_word.start_time - word.end_time
                if duration > max_word_duration or gap_to_next > max_gap:
                    needs_fix = True
                    fix_from_next = True

            # Last word with big gap from previous word - derive from PREVIOUS
            if is_last and not is_first:
                prev_word = fixed_words[-1] if fixed_words else line.words[i - 1]
                gap_from_prev = word.start_time - prev_word.end_time
                if gap_from_prev > max_gap:
                    needs_fix = True
                    fix_from_prev = True

            # Middle word after a long gap - derive from next word
            if not is_first and not is_last:
                prev_word = fixed_words[-1] if fixed_words else line.words[i - 1]
                gap_from_prev = word.start_time - prev_word.end_time
                if gap_from_prev > max_gap:
                    needs_fix = True
                    fix_from_next = True

            if needs_fix:
                if fix_from_next:
                    # Derive timing from the NEXT word
                    next_word = line.words[i + 1]
                    new_end = next_word.start_time - 0.05
                    new_start = new_end - avg_duration
                    fixed_words.append(Word(
                        text=word.text,
                        start_time=max(new_start, 0),
                        end_time=new_end,
                    ))
                elif fix_from_prev:
                    # Derive timing from the PREVIOUS word
                    prev_word = fixed_words[-1]
                    new_start = prev_word.end_time + 0.05
                    new_end = new_start + avg_duration
                    fixed_words.append(Word(
                        text=word.text,
                        start_time=new_start,
                        end_time=new_end,
                    ))
                else:
                    fixed_words.append(word)
            else:
                # Keep original timing
                fixed_words.append(word)

        if fixed_words:
            fixed_lines.append(Line(
                words=fixed_words,
                start_time=fixed_words[0].start_time,
                end_time=fixed_words[-1].end_time,
            ))

    return fixed_lines


def split_long_lines(lines: list[Line], max_chars: int = 45) -> list[Line]:
    """
    Split lines that are too wide to fit on screen.

    Args:
        lines: List of Line objects
        max_chars: Maximum characters per line (approximate)

    Returns:
        List of Line objects with long lines split
    """
    split_lines = []

    for line in lines:
        line_text = " ".join(w.text for w in line.words)

        if len(line_text) <= max_chars:
            # Line fits, keep as-is
            split_lines.append(line)
            continue

        # Need to split - find a good break point near the middle
        words = line.words
        total_words = len(words)

        if total_words < 2:
            split_lines.append(line)
            continue

        # Find split point - try to split roughly in half by character count
        best_split = total_words // 2
        half_chars = len(line_text) // 2

        char_count = 0
        for i, word in enumerate(words):
            char_count += len(word.text) + 1  # +1 for space
            if char_count >= half_chars:
                best_split = i + 1
                break

        # Ensure we don't create tiny splits
        best_split = max(1, min(best_split, total_words - 1))

        # Create two lines from the split
        first_words = words[:best_split]
        second_words = words[best_split:]

        first_line = None
        second_line = None

        if first_words:
            first_line = Line(
                words=first_words,
                start_time=first_words[0].start_time,
                end_time=first_words[-1].end_time,
            )

        if second_words:
            second_line = Line(
                words=second_words,
                start_time=second_words[0].start_time,
                end_time=second_words[-1].end_time,
            )

        # Recursively split if still too long
        if first_line:
            split_lines.extend(split_long_lines([first_line], max_chars))
        if second_line:
            split_lines.extend(split_long_lines([second_line], max_chars))

    return split_lines


def filter_lines_by_lyrics(transcribed_lines: list[Line], lyrics_text: list[str]) -> list[Line]:
    """
    Filter transcribed lines to only include those that match the provided lyrics.

    This helps remove hallucinated or extra content that Whisper might add.
    """
    import difflib

    # Create a set of normalized lyrics words for matching
    lyrics_words = set()
    for line in lyrics_text:
        for word in line.lower().split():
            # Remove punctuation
            clean = re.sub(r'[^\w]', '', word)
            if clean:
                lyrics_words.add(clean)

    # Filter lines - keep only if most words appear in lyrics
    filtered_lines = []
    for line in transcribed_lines:
        line_words = [re.sub(r'[^\w]', '', w.text.lower()) for w in line.words]
        matches = sum(1 for w in line_words if w in lyrics_words)

        # Keep line if at least 60% of words match lyrics
        if len(line_words) > 0 and matches / len(line_words) >= 0.6:
            filtered_lines.append(line)

    return filtered_lines


def get_lyrics(title: str, artist: str, vocals_path: Optional[str] = None) -> list[Line]:
    """
    Get lyrics with accurate word-level timing.

    Strategy:
    1. Fetch lyrics text from online sources
    2. Use whisperx to transcribe audio with word-level alignment
    3. Filter transcription to match fetched lyrics (removes extra content)

    If no online lyrics found, falls back to pure transcription.
    """
    # Try to fetch lyrics text
    lrc = fetch_synced_lyrics(title, artist)
    lyrics_text = None

    if lrc:
        lyrics_text = extract_lyrics_text(lrc)
        print(f"Got {len(lyrics_text)} lines of lyrics text")
    else:
        print("No lyrics found online, will transcribe from audio")

    # Transcribe and align
    if vocals_path:
        return transcribe_and_align(vocals_path, lyrics_text)

    raise RuntimeError("Could not get lyrics: no vocals path provided")


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
