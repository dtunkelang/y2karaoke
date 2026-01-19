# test_genius_fetch.py
from y2karaoke.core.genius import fetch_genius_lyrics_with_singers

def test_fetch_genius_song():
    # Example: pick a song you know exists on Genius
    title = "Bohemian Rhapsody"
    artist = "Queen"

    lines, metadata = fetch_genius_lyrics_with_singers(title, artist)

    assert lines is not None, "Failed to fetch lines"
    assert metadata is not None, "Failed to fetch metadata"
    assert metadata.title.lower() == "bohemian rhapsody", f"Unexpected title: {metadata.title}"
    assert metadata.artist.lower() == "queen", f"Unexpected artist: {metadata.artist}"

    # Print first few lines for visual verification
    print(f"Title: {metadata.title}, Artist: {metadata.artist}, Singers: {metadata.singers}")
    print("First 5 lines with singers:")
    for line, singer in lines[:5]:
        print(f"{singer}: {line}")

if __name__ == "__main__":
    test_fetch_genius_song()
