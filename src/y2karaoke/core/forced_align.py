"""Forced alignment: align known text to audio for word-level timing.

This module implements the core principle: we use Genius as the canonical text
and audio only for timing. No transcription/ASR - we know the words, we just
need to find when they occur in the audio.
"""

import warnings
from typing import List, Tuple, Optional

from ..utils.logging import get_logger
from .models import Word, Line
from .alignment_segments import build_segments_from_lrc, build_segments_estimated, build_lines_estimated
from .alignment_audio import get_audio_duration

logger = get_logger(__name__)


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
        segments = build_segments_from_lrc(text_lines, line_timings)
    else:
        # Estimate timing by distributing evenly across audio
        duration = get_audio_duration(audio_path)
        segments = build_segments_estimated(text_lines, duration, offset)

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
        return build_lines_estimated(text_lines, get_audio_duration(audio_path), offset)

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


