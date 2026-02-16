"""Data models for lyrics processing and quality reporting."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class SingerID(str, Enum):
    """Identifier for singer in duets."""

    SINGER1 = "singer1"
    SINGER2 = "singer2"
    BOTH = "both"
    UNKNOWN = ""


@dataclass
class Word:
    """A single word with timing information."""

    text: str
    start_time: float
    end_time: float
    singer: str = ""

    def validate(self) -> None:
        if self.start_time < 0 or self.end_time < 0:
            raise ValueError("Word timing must be non-negative")
        if self.end_time < self.start_time:
            raise ValueError("Word end_time must be >= start_time")


@dataclass
class Line:
    """A line of lyrics containing multiple words."""

    words: List[Word]
    singer: SingerID = SingerID.UNKNOWN

    @property
    def start_time(self) -> float:
        return self.words[0].start_time if self.words else 0.0

    @property
    def end_time(self) -> float:
        return self.words[-1].end_time if self.words else 0.0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    def validate(self) -> None:
        if not self.words:
            raise ValueError("Line must contain at least one word")
        for w in self.words:
            w.validate()
            if w.start_time < self.start_time or w.end_time > self.end_time:
                raise ValueError("Word timing outside line bounds")


def compute_word_slots(words: List[Word], line_end: float) -> List[float]:
    """Compute per-word slot durations based on start-to-start gaps."""
    slots: List[float] = []
    for i, w in enumerate(words):
        if i + 1 < len(words):
            slots.append(max(words[i + 1].start_time - w.start_time, 0.0))
        else:
            slots.append(max(line_end - w.start_time, 0.0))
    return slots


@dataclass
class SongMetadata:
    """Metadata about a song including singer information."""

    singers: List[str]
    is_duet: bool = False
    title: Optional[str] = None
    artist: Optional[str] = None

    def get_singer_id(self, singer_name: str) -> SingerID:
        """Map a singer name to a SingerID."""
        if not singer_name:
            return SingerID.UNKNOWN
        name = singer_name.lower().strip()
        duet_tokens = ["&", " and ", " feat ", " feat.", " with ", " x "]
        if any(token in name for token in duet_tokens):
            return SingerID.BOTH
        for i, known_singer in enumerate(self.singers):
            ks = known_singer.lower()
            if name == ks or name.startswith(ks) or ks.startswith(name):
                return SingerID(f"singer{i + 1}")
        return SingerID.SINGER1 if self.singers else SingerID.UNKNOWN


# =============================================================================
# Quality Report Data Structures
# =============================================================================


@dataclass
class TrackInfo:
    """Canonical track information from identification pipeline."""

    artist: str
    title: str
    duration: int  # canonical duration in seconds
    youtube_url: str
    youtube_duration: int  # actual YouTube video duration
    source: str  # "musicbrainz", "syncedlyrics", "youtube"
    lrc_duration: Optional[int] = None  # duration implied by LRC lyrics (if found)
    lrc_validated: bool = False  # True if LRC duration matches canonical duration
    # Quality reporting fields
    identification_quality: float = 100.0  # 0-100 confidence score
    quality_issues: Optional[List[str]] = None  # List of quality concerns
    sources_tried: Optional[List[str]] = None  # List of sources attempted
    fallback_used: bool = False  # True if had to fall back from primary source

    def __post_init__(self):
        if self.quality_issues is None:
            object.__setattr__(self, "quality_issues", [])
        if self.sources_tried is None:
            object.__setattr__(self, "sources_tried", [])


@dataclass
class StepQuality:
    """Quality report for a single pipeline step."""

    step_name: str
    quality_score: float  # 0-100
    status: str = "success"  # "success", "degraded", "failed"
    cached: bool = False
    issues: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_good(self) -> bool:
        return self.quality_score >= 70

    @property
    def needs_review(self) -> bool:
        return 40 <= self.quality_score < 70

    @property
    def is_poor(self) -> bool:
        return self.quality_score < 40


@dataclass
class TrackIdentificationQuality(StepQuality):
    """Quality report for track identification step."""

    match_confidence: float = 0.0  # 0-100
    source: str = ""  # "musicbrainz", "syncedlyrics", "youtube"
    duration_agreement: str = ""  # "All sources match", "LRC differs by Ns"
    sources_tried: List[str] = field(default_factory=list)
    fallback_used: bool = False

    def __post_init__(self) -> None:
        self.step_name = "track_identification"


@dataclass
class LyricsQuality(StepQuality):
    """Quality report for LRC lyrics fetching step."""

    source: str = ""  # Provider that succeeded
    sources_tried: List[str] = field(default_factory=list)
    coverage: float = 0.0  # 0-1, % of song duration covered
    timestamp_density: float = 0.0  # Lines per 10 seconds
    duration: Optional[int] = None  # LRC implied duration
    duration_match: bool = True  # Does LRC duration match target?

    def __post_init__(self) -> None:
        self.step_name = "lyrics_fetch"


@dataclass
class TimingAlignmentQuality(StepQuality):
    """Quality report for timing alignment step."""

    method_used: str = (
        "lrc_only"  # "lrc_only", "onset_refined", "whisper_hybrid", "whisper_dtw"
    )
    lines_aligned: int = 0
    total_lines: int = 0
    avg_offset: float = 0.0  # Average timing adjustment
    lines_with_issues: int = 0
    whisper_used: bool = False

    def __post_init__(self) -> None:
        self.step_name = "timing_alignment"

    @property
    def alignment_rate(self) -> float:
        return (
            (self.lines_aligned / self.total_lines * 100)
            if self.total_lines > 0
            else 0.0
        )


@dataclass
class SeparationQuality(StepQuality):
    """Quality report for vocal separation step."""

    model_used: str = "htdemucs"
    # Note: Full audio quality metrics require additional analysis
    # These are placeholders for future enhancement

    def __post_init__(self) -> None:
        self.step_name = "vocal_separation"


@dataclass
class BreakShorteningQuality(StepQuality):
    """Quality report for break shortening step."""

    breaks_detected: int = 0
    breaks_shortened: int = 0
    total_time_removed: float = 0.0
    beat_aligned: bool = True

    def __post_init__(self) -> None:
        self.step_name = "break_shortening"


@dataclass
class PipelineQualityReport:
    """Aggregated quality report for the entire pipeline."""

    overall_score: float  # 0-100, weighted average
    confidence_level: str  # "high", "medium", "low"
    steps: Dict[str, StepQuality] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    @classmethod
    def from_steps(cls, steps: List[StepQuality]) -> "PipelineQualityReport":
        """Create a pipeline report from individual step reports."""
        if not steps:
            return cls(overall_score=0, confidence_level="low")

        # Weighted average based on step importance
        weights = {
            "track_identification": 1.5,
            "lyrics_fetch": 2.0,
            "timing_alignment": 2.5,
            "vocal_separation": 1.0,
            "break_shortening": 0.5,
        }

        total_weight = 0.0
        weighted_sum = 0.0
        steps_dict = {}
        warnings = []
        recommendations = []

        for step in steps:
            steps_dict[step.step_name] = step
            weight = weights.get(step.step_name, 1.0)
            weighted_sum += step.quality_score * weight
            total_weight += weight

            # Collect warnings
            if step.status == "failed":
                warnings.append(f"{step.step_name}: Failed")
            elif step.status == "degraded":
                warnings.append(
                    f"{step.step_name}: Degraded quality ({step.quality_score:.0f}%)"
                )
            warnings.extend(step.issues)

        overall = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Determine confidence level
        if overall >= 80 and not any(s.status == "failed" for s in steps):
            confidence = "high"
        elif overall >= 50:
            confidence = "medium"
            recommendations.append("Review timing alignment before use")
        else:
            confidence = "low"
            recommendations.append("Manual timing review recommended")
            recommendations.append("Consider using --whisper for better alignment")

        return cls(
            overall_score=overall,
            confidence_level=confidence,
            steps=steps_dict,
            warnings=warnings,
            recommendations=recommendations,
        )

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Quality: {self.overall_score:.0f}/100 ({self.confidence_level} confidence)",
        ]

        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
            for w in self.warnings[:3]:  # Show top 3
                lines.append(f"  - {w}")

        if self.recommendations:
            lines.append("Recommendations:")
            for r in self.recommendations:
                lines.append(f"  - {r}")

        return "\n".join(lines)


@dataclass
class TargetLine:
    """A target line for visual alignment, including OCR metadata."""

    line_index: int
    start: float
    end: Optional[float]
    text: str
    words: List[str]
    y: float
    word_starts: Optional[List[Optional[float]]] = None
    word_ends: Optional[List[Optional[float]]] = None
    word_rois: Optional[List[tuple[int, int, int, int]]] = None
    char_rois: Optional[List[Optional[tuple[int, int, int, int]]]] = None
