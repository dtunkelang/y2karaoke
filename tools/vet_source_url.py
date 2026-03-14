#!/usr/bin/env python3
"""Quick source vetting for benchmark/clip curation candidates."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from y2karaoke.core.components.identify.implementation import TrackIdentifier
from y2karaoke.core.components.lyrics.sync import fetch_lyrics_for_duration
from y2karaoke.core.text_utils import normalize_title


@dataclass
class VetResult:
    artist: str
    title: str
    youtube_url: str
    youtube_title: str
    youtube_uploader: str
    youtube_duration_sec: int
    canonical_artist: str | None
    canonical_title: str | None
    canonical_duration_sec: int | None
    lrc_source: str | None
    lrc_synced: bool | None
    lrc_duration_sec: int | None
    youtube_vs_canonical_sec: int | None
    youtube_vs_lrc_sec: int | None
    canonical_vs_lrc_sec: int | None
    likely_non_studio: bool
    title_has_official_audio_hint: bool
    verdict: str
    notes: list[str]


def _best_canonical_match(
    identifier: TrackIdentifier, artist: str, title: str
) -> tuple[int | None, str | None, str | None]:
    query = f"{artist} {title}".strip()
    recordings = identifier._query_musicbrainz(query, artist, title)
    best = identifier._find_best_with_artist_hint(recordings, query, artist)
    if not best:
        best = identifier._find_best_title_only(recordings, title)
    if not best:
        return None, None, None
    duration, canonical_artist, canonical_title = best
    if not _title_matches_hint(title, canonical_title):
        return None, None, None
    return duration, canonical_artist, canonical_title


def _duration_delta(a: int | None, b: int | None) -> int | None:
    if a is None or b is None:
        return None
    return abs(a - b)


def _official_audio_hint(title: str) -> bool:
    lowered = title.lower()
    return "official audio" in lowered or lowered.endswith("(audio)")


def _has_feature_version_mismatch(requested_title: str, youtube_title: str) -> bool:
    yt = youtube_title.lower()
    requested = requested_title.lower()
    feature_markers = (" feat", " featuring ", "(feat", " ft.", " with ")
    yt_has_feature = any(marker in yt for marker in feature_markers)
    requested_has_feature = any(marker in requested for marker in feature_markers)
    return yt_has_feature and not requested_has_feature


def _title_matches_hint(title_hint: str, candidate_title: str | None) -> bool:
    if not candidate_title:
        return False
    hint = normalize_title(title_hint, remove_stopwords=False)
    candidate = normalize_title(candidate_title, remove_stopwords=False)
    if not hint or not candidate:
        return False
    return hint == candidate or hint in candidate or candidate in hint


def vet_source(url: str, artist: str, title: str) -> VetResult:
    identifier = TrackIdentifier()
    yt_title, yt_uploader, yt_duration = identifier._get_youtube_metadata(url)
    canonical_duration, canonical_artist, canonical_title = _best_canonical_match(
        identifier, artist, title
    )

    lrc_source = None
    lrc_synced = None
    lrc_duration = None
    if canonical_duration and canonical_artist and canonical_title:
        lrc_text, lrc_synced, lrc_source, lrc_duration = fetch_lyrics_for_duration(
            canonical_title, canonical_artist, canonical_duration, tolerance=8
        )
        if not lrc_text:
            lrc_source = None
            lrc_synced = None
            lrc_duration = None

    yt_vs_canonical = _duration_delta(yt_duration, canonical_duration)
    yt_vs_lrc = _duration_delta(yt_duration, lrc_duration)
    canonical_vs_lrc = _duration_delta(canonical_duration, lrc_duration)
    likely_non_studio = identifier._is_likely_non_studio(yt_title)
    title_has_official_audio_hint = _official_audio_hint(yt_title)
    feature_version_mismatch = _has_feature_version_mismatch(title, yt_title)

    notes: list[str] = []
    if title_has_official_audio_hint:
        notes.append("YouTube title includes official-audio hint.")
    else:
        notes.append("YouTube title is not tagged as official audio.")
    if likely_non_studio:
        notes.append("YouTube title looks non-studio or alternate-version-like.")
    if canonical_duration is None:
        notes.append("No canonical MusicBrainz duration found.")
    if lrc_duration is None:
        notes.append("No synced-lyrics duration found.")
    else:
        notes.append(
            "Synced-lyrics duration is based on the timed lyrics track, not the true song end."
        )

    verdict = "pass"
    if yt_vs_canonical is not None and yt_vs_canonical > 20:
        verdict = "fail"
        notes.append(
            f"YouTube duration differs from canonical duration by {yt_vs_canonical}s."
        )
    elif yt_vs_canonical is not None and yt_vs_canonical > 8:
        verdict = "warn"
        notes.append(
            f"YouTube duration differs from canonical duration by {yt_vs_canonical}s."
        )

    if yt_vs_lrc is not None and yt_vs_lrc > 20:
        verdict = "fail"
        notes.append(
            f"YouTube duration differs from synced-lyrics duration by {yt_vs_lrc}s."
        )
    elif yt_vs_lrc is not None and yt_vs_lrc > 8 and verdict == "pass":
        verdict = "warn"
        notes.append(
            f"YouTube duration differs from synced-lyrics duration by {yt_vs_lrc}s."
        )

    if likely_non_studio and verdict == "pass":
        verdict = "warn"
    if feature_version_mismatch:
        verdict = "fail"
        notes.append(
            "YouTube title appears to be a featured/alternate version that the requested title does not mention."
        )
    if canonical_duration is None and lrc_duration is None and verdict == "pass":
        verdict = "warn"
        notes.append("No canonical or synced-lyrics duration confirmation was found.")
    if not title_has_official_audio_hint and verdict == "pass":
        notes.append("Prefer an official-audio upload when one exists.")

    return VetResult(
        artist=artist,
        title=title,
        youtube_url=url,
        youtube_title=yt_title,
        youtube_uploader=yt_uploader,
        youtube_duration_sec=yt_duration,
        canonical_artist=canonical_artist,
        canonical_title=canonical_title,
        canonical_duration_sec=canonical_duration,
        lrc_source=lrc_source,
        lrc_synced=lrc_synced,
        lrc_duration_sec=lrc_duration,
        youtube_vs_canonical_sec=yt_vs_canonical,
        youtube_vs_lrc_sec=yt_vs_lrc,
        canonical_vs_lrc_sec=canonical_vs_lrc,
        likely_non_studio=likely_non_studio,
        title_has_official_audio_hint=title_has_official_audio_hint,
        verdict=verdict,
        notes=notes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="YouTube source URL")
    parser.add_argument("--artist", required=True, help="Expected artist")
    parser.add_argument("--title", required=True, help="Expected title")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    result = vet_source(args.url, args.artist, args.title)
    if args.json:
        print(json.dumps(asdict(result), indent=2))
        return 0

    print(f"verdict: {result.verdict}")
    print(f"artist/title: {result.artist} - {result.title}")
    print(f"youtube: {result.youtube_title} [{result.youtube_duration_sec}s]")
    print(f"uploader: {result.youtube_uploader}")
    print(
        "canonical:"
        f" {result.canonical_artist or '?'} - {result.canonical_title or '?'}"
        f" [{result.canonical_duration_sec if result.canonical_duration_sec is not None else '?'}s]"
    )
    print(
        "synced lyrics:"
        f" {result.lrc_source or '?'}"
        f" [{result.lrc_duration_sec if result.lrc_duration_sec is not None else '?'}s]"
    )
    print(
        "deltas:"
        f" yt-canonical={result.youtube_vs_canonical_sec}"
        f", yt-lrc={result.youtube_vs_lrc_sec}"
        f", canonical-lrc={result.canonical_vs_lrc_sec}"
    )
    for note in result.notes:
        print(f"- {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
