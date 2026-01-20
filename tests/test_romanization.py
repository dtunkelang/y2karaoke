import pytest
from y2karaoke.core.lyrics import Line, Word
from y2karaoke.core.romanization import romanize_line


def test_romanization_line():
    # Sample non-ASCII text (Korean)
    text = "안녕하세요"
    words = [Word(text=text, start_time=0.0, end_time=3.0)]

    # Apply romanization
    for word in words:
        word.text = romanize_line(word.text)

    # Assert all characters are ASCII after romanization
    assert all(ord(c) < 128 for word in words for c in word.text), \
        f"Romanization failed: {[word.text for word in words]}"

    # Check that output is not empty
    assert all(word.text for word in words), "Romanized text should not be empty"
