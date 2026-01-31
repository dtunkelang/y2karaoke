"""Test validation utilities."""

import pytest
from y2karaoke.utils.validation import (
    validate_youtube_url,
    validate_key_shift,
    validate_tempo,
    validate_offset,
    sanitize_filename,
)
from y2karaoke.exceptions import ValidationError


class TestValidation:
    """Test validation functions."""

    def test_validate_youtube_url_valid(self):
        """Test valid YouTube URLs."""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtube.com/embed/dQw4w9WgXcQ",
        ]

        for url in valid_urls:
            assert validate_youtube_url(url) == url

    def test_validate_youtube_url_invalid(self):
        """Test invalid YouTube URLs."""
        invalid_urls = [
            "",
            "not a url",
            "https://example.com",
            "https://vimeo.com/123456",
        ]

        for url in invalid_urls:
            with pytest.raises(ValidationError):
                validate_youtube_url(url)

    def test_validate_key_shift_valid(self):
        """Test valid key shift values."""
        for key in range(-12, 13):
            assert validate_key_shift(key) == key

    def test_validate_key_shift_invalid(self):
        """Test invalid key shift values."""
        invalid_keys = [-13, 13, -100, 100]

        for key in invalid_keys:
            with pytest.raises(ValidationError):
                validate_key_shift(key)

    def test_validate_tempo_valid(self):
        """Test valid tempo values."""
        valid_tempos = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]

        for tempo in valid_tempos:
            assert validate_tempo(tempo) == tempo

    def test_validate_tempo_invalid(self):
        """Test invalid tempo values."""
        invalid_tempos = [0, -1, 3.1, 10]

        for tempo in invalid_tempos:
            with pytest.raises(ValidationError):
                validate_tempo(tempo)

    def test_validate_offset_valid(self):
        """Test valid offset values."""
        valid_offsets = [-10, -5, 0, 5, 10]

        for offset in valid_offsets:
            assert validate_offset(offset) == offset

    def test_validate_offset_invalid(self):
        """Test invalid offset values."""
        invalid_offsets = [-11, 11, -100, 100]

        for offset in invalid_offsets:
            with pytest.raises(ValidationError):
                validate_offset(offset)

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        test_cases = [
            ("normal_file.mp4", "normal_file.mp4"),
            ("file<with>bad:chars", "filewithbadchars"),
            ("file/with\\path|chars", "filewithpathchars"),
        ]

        for input_name, expected in test_cases:
            result = sanitize_filename(input_name)
            assert result == expected
            assert len(result) <= 100

        # Test long filename truncation
        long_name = "very_long_filename_" + "x" * 100
        result = sanitize_filename(long_name)
        assert len(result) <= 100
        assert result.startswith("very_long_filename_")
