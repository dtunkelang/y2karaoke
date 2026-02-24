# test_genius_output.py
import sys
from y2karaoke.core.components.lyrics.genius import fetch_genius_lyrics_with_singers


def print_genius_lyrics(title, artist, max_lines=None):
    """
    Fetch lyrics with singers from Genius and print them line by line.

    Args:
        title (str): Song title
        artist (str): Artist name
        max_lines (int, optional): Maximum number of lines to print
    """
    lines_with_singers, metadata = fetch_genius_lyrics_with_singers(title, artist)

    if not lines_with_singers:
        print(f"No lyrics found for {title} by {artist}")
        return

    print(f"Lyrics for '{title}' by {artist} ({len(lines_with_singers)} lines):\n")
    for i, (line, singer) in enumerate(lines_with_singers):
        if max_lines and i >= max_lines:
            print("... (truncated)")
            break
        singer_info = f" [{singer}]" if singer else ""
        print(f"{i+1:02d}: {line}{singer_info}")

    if metadata:
        print(f"\nDetected singers: {metadata.singers}")
        print(f"Is duet: {metadata.is_duet}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 tools/genius_inspect.py <artist> <title>")
        sys.exit(1)
    
    artist = sys.argv[1]
    title = sys.argv[2]
    print_genius_lyrics(title, artist)
