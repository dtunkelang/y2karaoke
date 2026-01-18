"""Data models for lyrics processing."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


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
