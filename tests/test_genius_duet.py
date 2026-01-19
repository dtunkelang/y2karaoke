# test_genius_duet.py
from y2karaoke.core.genius import parse_genius_html, fetch_html

def test_fetch_genius_duet_fast():
    # Hardcoded Genius URL for the duet
    song_url = "https://genius.com/Linkin-park-in-the-end-lyrics"
    artist = "Linkin Park"

    html = fetch_html(song_url)
    assert html is not None, f"Failed to fetch HTML from {song_url}"

    lines, metadata = parse_genius_html(html, artist)

    assert lines is not None, "Failed to parse lines"
    assert metadata is not None, "Failed to parse metadata"
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
    test_fetch_genius_duet_fast()
