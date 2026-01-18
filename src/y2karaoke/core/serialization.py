"""JSON serialization for lyrics data structures."""

import json
from typing import List, Optional, Tuple

from .models import Word, Line, SongMetadata


def lines_to_json(lines: List[Line]) -> List[dict]:
    """Convert a list of Line objects into JSON-serializable dicts."""
    data: List[dict] = []
    for line in lines:
        data.append({
            "start_time": line.start_time,
            "end_time": line.end_time,
            "singer": line.singer,
            "words": [
                {
                    "text": w.text,
                    "start_time": w.start_time,
                    "end_time": w.end_time,
                    "singer": w.singer,
                } for w in line.words
            ]
        })
    return data


def lines_from_json(data: List[dict]) -> List[Line]:
    """Convert JSON data back into Line objects with Word objects."""
    lines: List[Line] = []
    for item in data:
        words = [
            Word(
                text=w["text"],
                start_time=float(w["start_time"]),
                end_time=float(w["end_time"]),
                singer=w.get("singer", "")
            ) for w in item.get("words", [])
        ]
        lines.append(Line(
            words=words,
            singer=item.get("singer", "")
        ))
    return lines


def metadata_to_json(metadata: Optional[SongMetadata]) -> Optional[dict]:
    """Convert SongMetadata to JSON-serializable dict."""
    if not metadata:
        return None
    return {
        "singers": metadata.singers,
        "is_duet": metadata.is_duet,
        "title": metadata.title,
        "artist": metadata.artist
    }


def metadata_from_json(data: Optional[dict]) -> Optional[SongMetadata]:
    """Convert JSON dict back into SongMetadata."""
    if not data:
        return None
    return SongMetadata(
        singers=list(data.get("singers", [])),
        is_duet=bool(data.get("is_duet", False)),
        title=data.get("title"),
        artist=data.get("artist")
    )


def save_lyrics_to_json(
    filepath: str,
    lines: List[Line],
    metadata: Optional[SongMetadata] = None
) -> None:
    """Save lyrics and metadata to a JSON file."""
    data = {
        "lines": lines_to_json(lines),
        "metadata": metadata_to_json(metadata)
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_lyrics_from_json(filepath: str) -> Tuple[List[Line], Optional[SongMetadata]]:
    """Load lyrics and metadata from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = lines_from_json(data.get("lines", []))
    metadata = metadata_from_json(data.get("metadata"))
    return lines, metadata
