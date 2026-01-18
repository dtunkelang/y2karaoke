"""Forced alignment: align known text to audio for word-level timing.

This module implements the core principle: we use Genius as the canonical text
and audio only for timing. No transcription/ASR - we know the words, we just
need to find when they occur in the audio.
"""

import warnings
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..utils.logging import get_logger
from .models import Word, Line

logger = get_logger(__name__)


@dataclass
class WordTiming:
    """Timing information for a single word."""
    text: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class AlignmentResult:
    """Result of forced alignment."""
    lines: List[Line]
    quality_score: float
    offset_applied: float


def detect_song_start(audio_path: str, min_duration: float = 0.3) -> float:
    """
    Detect where vocals actually start in the audio.

    YouTube audio often has silence or intro before vocals begin.
    This function finds the first sustained vocal activity.

    Uses adaptive thresholding based on the audio's noise floor,
    and requires sustained activity to avoid false positives.

    Args:
        audio_path: Path to vocals audio file
        min_duration: Minimum duration of vocal activity to count as "start" (seconds)

    Returns:
        Time in seconds when vocals start
    """
    try:
        import librosa
        import numpy as np

        # Load audio at 16kHz for faster processing
        y, sr = librosa.load(audio_path, sr=16000)

        # Compute RMS energy in frames
        frame_length = int(0.05 * sr)  # 50ms frames
        hop_length = frame_length // 2  # 25ms hop

        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        # Adaptive threshold: use percentile of RMS values
        # This adapts to different recording levels
        noise_floor = np.percentile(rms, 10)  # Bottom 10% is noise
        peak_level = np.percentile(rms, 95)   # Top 5% is vocal peaks

        # Threshold is between noise and peaks
        threshold = noise_floor + 0.15 * (peak_level - noise_floor)

        # Find frames above threshold
        above_threshold = rms > threshold

        # Require sustained activity (min_duration consecutive frames)
        min_frames = int(min_duration * sr / hop_length)

        # Find first sustained region
        consecutive = 0
        for i, is_active in enumerate(above_threshold):
            if is_active:
                consecutive += 1
                if consecutive >= min_frames:
                    # Found sustained activity - return start of this region
                    start_frame = i - consecutive + 1
                    start_time = start_frame * hop_length / sr
                    logger.info(f"Detected vocal start at {start_time:.2f}s (threshold: {threshold:.4f})")
                    return start_time
            else:
                consecutive = 0

        logger.warning("No sustained vocal activity detected, assuming start at 0")
        return 0.0

    except Exception as e:
        logger.warning(f"Could not detect song start: {e}")
        return 0.0


def get_audio_duration(audio_path: str) -> float:
    """Get the duration of an audio file in seconds."""
    try:
        import librosa
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e:
        logger.warning(f"Could not get audio duration: {e}")
        return 180.0  # Default 3 minutes


def forced_align(
    text_lines: List[str],
    audio_path: str,
    offset: float = 0.0,
    line_timings: Optional[List[Tuple[float, str]]] = None,
    language: Optional[str] = None,
) -> List[Line]:
    """
    Align known text to audio for word-level timing.

    This is forced alignment: we know the text is correct (from Genius),
    we just need to find when each word occurs in the audio.

    Args:
        text_lines: List of lyrics lines (canonical text from Genius)
        audio_path: Path to vocals audio file
        offset: Offset to apply to all timings (for intro skip)
        line_timings: Optional LRC-style (timestamp, text) for approximate timing
        language: Language code (auto-detected if None)

    Returns:
        List of Line objects with word-level timing
    """
    import torch

    # Suppress warnings
    warnings.filterwarnings("ignore", message="Lightning automatically upgraded")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    # Set up CPU threads
    import os
    if device == "cpu":
        num_threads = os.cpu_count() or 4
        torch.set_num_threads(num_threads)

    # If we have line timings, use them for segment boundaries
    if line_timings:
        segments = _build_segments_from_lrc(text_lines, line_timings)
    else:
        # Estimate timing by distributing evenly across audio
        duration = get_audio_duration(audio_path)
        segments = _build_segments_estimated(text_lines, duration, offset)

    # Detect language if not provided
    if language is None:
        language = _detect_language(text_lines)

    # Load audio and run alignment
    try:
        import whisperx

        # Fix for PyTorch 2.6+ compatibility
        _original_torch_load = torch.load
        def _patched_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return _original_torch_load(*args, **kwargs)
        torch.load = _patched_torch_load

        logger.info("Loading alignment model...")
        model_a, metadata = whisperx.load_align_model(
            language_code=language,
            device=device
        )

        logger.info("Loading audio...")
        audio = whisperx.load_audio(audio_path)

        logger.info("Running forced alignment...")
        result = whisperx.align(
            segments,
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False
        )

        # Convert result to Line objects
        lines = _convert_alignment_result(result, offset)

        # Apply perceptual timing adjustment
        lines = _apply_perceptual_offset(lines, -0.15)

        return lines

    except Exception as e:
        logger.error(f"Forced alignment failed: {e}")
        # Fall back to estimated timing
        return _build_lines_estimated(text_lines, get_audio_duration(audio_path), offset)


def _build_segments_from_lrc(
    text_lines: List[str],
    line_timings: List[Tuple[float, str]]
) -> List[dict]:
    """
    Build WhisperX segments from Genius text and LRC timing.

    We use Genius for text content but LRC for approximate timing.
    """
    from rapidfuzz import fuzz

    segments = []
    used_timings = set()

    for text in text_lines:
        if not text.strip():
            continue

        # Find best matching LRC line for timing
        best_timing = None
        best_score = 0

        for i, (timestamp, lrc_text) in enumerate(line_timings):
            if i in used_timings:
                continue
            score = fuzz.ratio(text.lower(), lrc_text.lower())
            if score > best_score:
                best_score = score
                best_timing = (i, timestamp)

        if best_timing and best_score > 50:
            used_timings.add(best_timing[0])
            start_time = best_timing[1]
        else:
            # No good match - estimate based on previous segment
            if segments:
                start_time = segments[-1].get("end", 0) + 0.5
            else:
                start_time = 0.0

        # Estimate end time
        end_time = start_time + max(len(text.split()) * 0.4, 1.0)

        segments.append({
            "text": text,
            "start": start_time,
            "end": end_time,
        })

    return segments


def _build_segments_estimated(
    text_lines: List[str],
    duration: float,
    offset: float
) -> List[dict]:
    """Build segments with evenly distributed timing."""
    segments = []
    non_empty_lines = [t for t in text_lines if t.strip()]

    if not non_empty_lines:
        return segments

    # Leave some buffer at start and end
    usable_duration = duration - offset - 5.0
    time_per_line = usable_duration / len(non_empty_lines)

    for i, text in enumerate(non_empty_lines):
        start_time = offset + i * time_per_line
        end_time = start_time + time_per_line * 0.9

        segments.append({
            "text": text,
            "start": start_time,
            "end": end_time,
        })

    return segments


def _detect_language(text_lines: List[str]) -> str:
    """Detect language from lyrics text."""
    sample = " ".join(text_lines[:10]).lower()

    english_words = ["the ", "you ", "and ", "are ", "is ", "it ", "to ", "of "]
    spanish_words = [" el ", " la ", " que ", " con ", " por ", " para "]
    japanese_patterns = [" wa ", " ga ", " wo ", " ni ", "desu", "masu"]

    english_count = sum(1 for w in english_words if w in sample)
    spanish_count = sum(1 for w in spanish_words if w in sample)
    japanese_count = sum(1 for p in japanese_patterns if p in sample)

    if english_count >= 3:
        return "en"
    elif spanish_count >= 2:
        return "es"
    elif japanese_count >= 2:
        return "ja"

    return "en"  # Default


def _convert_alignment_result(result: dict, offset: float) -> List[Line]:
    """Convert WhisperX alignment result to Line objects."""
    lines = []

    for segment in result.get("segments", []):
        segment_words = segment.get("words", [])
        if not segment_words:
            continue

        words = []
        for word_data in segment_words:
            word_text = word_data.get("word", "").strip()
            if not word_text:
                continue

            # Get timing with fallback
            start = word_data.get("start", 0.0) + offset
            end = word_data.get("end", start + 0.3) + offset

            words.append(Word(
                text=word_text,
                start_time=start,
                end_time=max(end, start + 0.1),
            ))

        if words:
            lines.append(Line(words=words))

    return lines


def _build_lines_estimated(
    text_lines: List[str],
    duration: float,
    offset: float
) -> List[Line]:
    """Build lines with evenly distributed word timing (fallback)."""
    lines = []
    non_empty = [t for t in text_lines if t.strip()]

    if not non_empty:
        return lines

    usable_duration = duration - offset - 5.0
    time_per_line = usable_duration / len(non_empty)

    for i, text in enumerate(non_empty):
        line_start = offset + i * time_per_line
        word_texts = text.split()

        if not word_texts:
            continue

        word_duration = (time_per_line * 0.9) / len(word_texts)

        words = []
        for j, word_text in enumerate(word_texts):
            word_start = line_start + j * word_duration
            words.append(Word(
                text=word_text,
                start_time=word_start,
                end_time=word_start + word_duration * 0.9,
            ))

        lines.append(Line(words=words))

    return lines


def _apply_perceptual_offset(lines: List[Line], offset: float) -> List[Line]:
    """
    Apply perceptual timing offset.

    WhisperX detects phoneme onset, but humans perceive words slightly earlier.
    A small negative offset improves perceived sync.
    """
    for line in lines:
        for word in line.words:
            word.start_time += offset
            word.end_time += offset
    return lines


def alignment_quality(lines: List[Line], audio_path: Optional[str] = None) -> float:
    """
    Score the quality of an alignment.

    Components:
    - timing_smoothness: Penalize unrealistic word durations (<50ms or >3s)
    - gap_penalty: Penalize large gaps between consecutive words
    - monotonicity: Words should be in time order
    - coverage: All text should be aligned

    Returns:
        Score from 0.0 (bad) to 1.0 (good)
    """
    if not lines:
        return 0.0

    total_words = sum(len(line.words) for line in lines)
    if total_words == 0:
        return 0.0

    penalties = 0.0

    # Check word durations
    for line in lines:
        for word in line.words:
            duration = word.end_time - word.start_time
            if duration < 0.05:
                penalties += 0.1  # Too short
            elif duration > 3.0:
                penalties += 0.1  # Too long

    # Check gaps between words
    prev_end = 0.0
    for line in lines:
        for word in line.words:
            gap = word.start_time - prev_end
            if gap > 5.0:
                penalties += 0.2  # Large gap
            elif gap < -0.1:
                penalties += 0.3  # Overlap (monotonicity violation)
            prev_end = word.end_time

    # Calculate score
    score = max(0.0, 1.0 - (penalties / total_words))
    return score


def refine_with_onset_detection(
    lines: List[Line],
    audio_path: str,
    max_shift: float = 0.3
) -> List[Line]:
    """
    Refine word timing using vocal onset detection.

    Uses librosa's onset detection to find actual vocal onsets
    and slightly adjust word timing to match.

    Args:
        lines: Lines with initial timing from forced alignment
        audio_path: Path to vocals audio
        max_shift: Maximum timing shift in seconds

    Returns:
        Lines with refined timing
    """
    try:
        import librosa
        import numpy as np

        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)

        # Detect onsets
        onset_times = librosa.onset.onset_detect(
            y=y, sr=sr,
            hop_length=512,
            backtrack=True,
            units='time'
        )

        if len(onset_times) == 0:
            return lines

        # For each word, find the closest onset and adjust
        for line in lines:
            for word in line.words:
                # Find closest onset to word start
                closest_idx = np.argmin(np.abs(onset_times - word.start_time))
                closest_onset = onset_times[closest_idx]

                shift = closest_onset - word.start_time

                # Only apply if shift is small enough
                if abs(shift) <= max_shift:
                    duration = word.end_time - word.start_time
                    word.start_time = closest_onset
                    word.end_time = closest_onset + duration

        return lines

    except Exception as e:
        logger.warning(f"Onset refinement failed: {e}")
        return lines
