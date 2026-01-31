"""Integration tests for Genius lyrics fetching, robust to live HTML changes."""

import os
import pickle
from functools import wraps
import pytest
from y2karaoke.core.genius import fetch_genius_lyrics_with_singers as original_fetch

# ------------------------------
# Genius caching decorator
# ------------------------------
CACHE_DIR = ".genius_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def cache_genius(func):
    @wraps(func)
    def wrapper(title, artist, *args, **kwargs):
        safe_name = f"{artist}_{title}".replace(" ", "_").replace("/", "_")
        cache_path = os.path.join(CACHE_DIR, f"{safe_name}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        result = func(title, artist, *args, **kwargs)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        return result

    return wrapper


fetch_genius_lyrics_with_singers = cache_genius(original_fetch)

# ------------------------------
# Test Songs
# ------------------------------
SONGS = [
    {"title": "Somewhere I Belong", "artist": "Linkin Park"},
    {"title": "Numb", "artist": "Linkin Park"},
    {"title": "Bohemian Rhapsody", "artist": "Queen"},
]


# ------------------------------
# Stable Tests
# ------------------------------
@pytest.mark.parametrize("song", SONGS)
def test_genius_fetch_returns_lines(song):
    """Check that fetching lyrics returns at least one line."""
    lines, metadata = fetch_genius_lyrics_with_singers(song["title"], song["artist"])
    assert lines, f"No lyrics returned for {song['title']}"


@pytest.mark.parametrize("song", SONGS)
def test_no_artist_only_lines(song):
    """Ensure that no line consists solely of artist names."""
    lines, metadata = fetch_genius_lyrics_with_singers(song["title"], song["artist"])
    known_singers = set(metadata.singers if metadata else [song["artist"]])
    for text, singer in lines:
        parts = [
            p.strip().lower()
            for p in text.replace("&", "/").replace(",", "/").split("/")
        ]
        assert not all(
            part in (s.lower() for s in known_singers) for part in parts
        ), f"Artist-only line detected in {song['title']}: {text}"


@pytest.mark.parametrize("song", SONGS)
def test_fetch_returns_nonempty_text(song):
    """Ensure no lines are completely empty."""
    lines, metadata = fetch_genius_lyrics_with_singers(song["title"], song["artist"])
    for text, _ in lines:
        assert text.strip(), f"Empty text line returned for {song['title']}"


@pytest.mark.parametrize("song", SONGS)
def test_print_sample_lines(song):
    """Optional: print the first few lines for manual inspection."""
    lines, metadata = fetch_genius_lyrics_with_singers(song["title"], song["artist"])
    if lines:
        print(f"\nSample lines for {song['title']} by {song['artist']}:")
        for i, (line, singer) in enumerate(lines[:10]):
            singer_info = f" [{singer}]" if singer else ""
            print(f"{i+1:02d}: {line}{singer_info}")
