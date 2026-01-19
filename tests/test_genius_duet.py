# test_genius_duet.py
from y2karaoke.core.genius import fetch_genius_lyrics_with_singers

def test_fetch_genius_duet():
    # Duet example: Linkin Park & Jay-Z
    title = "Numb/Encore"
    artist = "Linkin Park & Jay-Z"

    lines, metadata = fetch_genius_lyrics_with_singers(title, artist)

    assert lines is not None, "Failed to fetch lines"
    assert metadata is not None, "Failed to fetch metadata"
    assert metadata.is_duet, "Song should be detected as a duet"
    assert len(metadata.singers) >= 2, f"Expected 2+ singers, got {metadata.singers}"

    # Print metadata for verification
    print(f"Title: {metadata.title}")
    print(f"Artist: {metadata.artist}")
    print(f"Singers detected: {metadata.singers}")
    print("First 10 lines with singer annotations:")
    for line, singer in lines[:10]:
        print(f"{singer}: {line}")

if __name__ == "__main__":
    test_fetch_genius_duet()
