"""Data models for timing evaluation and Whisper transcription."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class TimingIssue:
    """Represents a timing inconsistency between lyrics and audio."""

    issue_type: str  # "early_line", "late_line", "missing_pause", "unexpected_pause"
    line_index: int
    lyrics_time: float
    audio_time: Optional[float]
    delta: float  # positive = lyrics ahead of audio
    severity: str  # "minor", "moderate", "severe"
    description: str


@dataclass
class AudioFeatures:
    """Extracted audio features for timing evaluation."""

    onset_times: np.ndarray  # Detected onset times
    silence_regions: List[Tuple[float, float]]  # (start, end) of silent regions
    vocal_start: float  # First vocal onset
    vocal_end: float  # Last vocal activity
    duration: float  # Total audio duration
    energy_envelope: np.ndarray  # RMS energy over time
    energy_times: np.ndarray  # Time axis for energy envelope


@dataclass
class TimingReport:
    """Comprehensive timing evaluation report."""

    source_name: str
    overall_score: float  # 0-100, higher is better
    line_alignment_score: float  # How well line starts match onsets
    pause_alignment_score: float  # How well pauses match silence
    issues: List[TimingIssue] = field(default_factory=list)
    summary: str = ""

    # Detailed metrics
    avg_line_offset: float = 0.0  # Average offset between lyrics and audio
    std_line_offset: float = 0.0  # Standard deviation of offsets
    matched_onsets: int = 0  # Lines that matched an onset
    total_lines: int = 0
    lyric_on_vocal_ratio: float = 0.0  # Lyric duration overlapping non-silent audio
    vocal_covered_by_lyrics_ratio: float = 0.0  # Non-silent audio covered by lyrics
    lyric_in_silence_ratio: float = 0.0  # Lyric duration placed over silence


@dataclass
class TranscriptionWord:
    """A word from Whisper transcription with timing."""

    start: float
    end: float
    text: str
    probability: float = 1.0


@dataclass
class TranscriptionSegment:
    """A segment from Whisper transcription."""

    start: float
    end: float
    text: str
    words: Optional[List[TranscriptionWord]] = None

    def __post_init__(self) -> None:
        if self.words is None:
            self.words = []
