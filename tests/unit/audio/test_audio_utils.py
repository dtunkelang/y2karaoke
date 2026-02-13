"""Tests for audio_utils.py module."""

import pytest
from pathlib import Path  # noqa: F401
from unittest.mock import Mock, patch, MagicMock  # noqa: F401
from y2karaoke.core.components.audio.audio_utils import (
    trim_audio_if_needed,
    apply_audio_effects,
    separate_vocals,
)


class TestTrimAudioIfNeeded:
    """Test trim_audio_if_needed function."""

    def test_function_exists(self):
        """Test that trim_audio_if_needed function exists."""
        assert trim_audio_if_needed is not None
        assert callable(trim_audio_if_needed)

    def test_function_signature(self):
        """Test trim_audio_if_needed function signature."""
        import inspect

        sig = inspect.signature(trim_audio_if_needed)
        params = list(sig.parameters.keys())
        expected_params = [
            "audio_path",
            "start_time",
            "video_id",
            "cache_manager",
            "force",
        ]
        for param in expected_params:
            assert param in params

    def test_no_trim_needed_returns_original(self):
        """Test that no trimming returns original path when start_time <= 0."""
        mock_cache = Mock()

        result = trim_audio_if_needed("/audio.wav", 0.0, "video123", mock_cache)
        assert result == "/audio.wav"

        result = trim_audio_if_needed("/audio.wav", -1.0, "video123", mock_cache)
        assert result == "/audio.wav"

    @patch("y2karaoke.core.components.audio.audio_utils.AudioSegment.from_wav")
    def test_trim_audio_with_positive_start_time(self, mock_from_wav):
        """Test trimming audio with positive start time."""
        mock_cache = Mock()
        mock_cache.get_file_path.return_value = "/trimmed_audio.wav"
        mock_cache.file_exists.return_value = False

        mock_audio = Mock()
        mock_audio.__len__ = Mock(return_value=10000)  # 10 seconds in ms
        mock_audio.__getitem__ = Mock(return_value=mock_audio)
        mock_audio.export = Mock()
        mock_from_wav.return_value = mock_audio

        result = trim_audio_if_needed("/audio.wav", 5.0, "video123", mock_cache)

        assert result == "/trimmed_audio.wav"
        mock_from_wav.assert_called_once_with("/audio.wav")

    @patch("y2karaoke.core.components.audio.audio_utils.AudioSegment.from_wav")
    def test_trim_audio_start_beyond_length_returns_original(self, mock_from_wav):
        """Test that start time beyond audio length returns original."""
        mock_cache = Mock()
        mock_cache.file_exists.return_value = False

        mock_audio = Mock()
        mock_audio.__len__ = Mock(return_value=1000)  # 1 second in ms
        mock_from_wav.return_value = mock_audio

        result = trim_audio_if_needed("/audio.wav", 5.0, "video123", mock_cache)

        assert result == "/audio.wav"

    def test_uses_cached_result_when_available(self):
        """Test that cached result is used when available and not forced."""
        mock_cache = Mock()
        mock_cache.get_file_path.return_value = "/cached_trimmed.wav"
        mock_cache.file_exists.return_value = True

        result = trim_audio_if_needed(
            "/audio.wav", 3.0, "video123", mock_cache, force=False
        )

        assert result == "/cached_trimmed.wav"

    @patch("y2karaoke.core.components.audio.audio_utils.AudioSegment.from_wav")
    def test_force_retrim_ignores_cache(self, mock_from_wav):
        """Test that force=True ignores cached result."""
        mock_cache = Mock()
        mock_cache.get_file_path.return_value = "/trimmed_audio.wav"
        mock_cache.file_exists.return_value = True

        mock_audio = Mock()
        mock_audio.__len__ = Mock(return_value=10000)  # 10 seconds in ms
        mock_audio.__getitem__ = Mock(return_value=mock_audio)
        mock_audio.export = Mock()
        mock_from_wav.return_value = mock_audio

        result = trim_audio_if_needed(
            "/audio.wav", 2.0, "video123", mock_cache, force=True
        )

        assert result == "/trimmed_audio.wav"
        mock_from_wav.assert_called_once()


class TestApplyAudioEffects:
    """Test apply_audio_effects function."""

    def test_function_exists(self):
        """Test that apply_audio_effects function exists."""
        assert apply_audio_effects is not None
        assert callable(apply_audio_effects)

    def test_function_signature(self):
        """Test apply_audio_effects function signature."""
        import inspect

        sig = inspect.signature(apply_audio_effects)
        params = list(sig.parameters.keys())
        expected_params = [
            "audio_path",
            "key_shift",
            "tempo",
            "video_id",
            "cache_manager",
            "audio_processor",
            "force",
            "cache_suffix",
        ]
        for param in expected_params:
            assert param in params

    def test_no_effects_returns_original(self):
        """Test that no effects returns original path."""
        mock_cache = Mock()
        mock_processor = Mock()

        result = apply_audio_effects(
            "/audio.wav", 0, 1.0, "video123", mock_cache, mock_processor
        )
        assert result == "/audio.wav"

    def test_applies_effects_when_needed(self):
        """Test that effects are applied when parameters are non-default."""
        mock_cache = Mock()
        mock_cache.file_exists.return_value = False
        mock_cache.get_file_path.return_value = "/processed_audio.wav"

        mock_processor = Mock()
        mock_processor.process_audio.return_value = "/processed_audio.wav"

        result = apply_audio_effects(
            "/audio.wav", 2, 1.2, "video123", mock_cache, mock_processor
        )

        assert result == "/processed_audio.wav"
        mock_processor.process_audio.assert_called_once()

    def test_uses_cached_result_when_available(self):
        """Test that cached result is used when available."""
        mock_cache = Mock()
        mock_cache.file_exists.return_value = True
        mock_cache.get_file_path.return_value = "/cached_processed.wav"
        mock_processor = Mock()

        result = apply_audio_effects(
            "/audio.wav", 1, 0.8, "video123", mock_cache, mock_processor
        )

        assert result == "/cached_processed.wav"

    def test_force_reprocess_ignores_cache(self):
        """Test that force=True ignores cached result."""
        mock_cache = Mock()
        mock_cache.file_exists.return_value = True
        mock_cache.get_file_path.return_value = "/processed_audio.wav"

        mock_processor = Mock()
        mock_processor.process_audio.return_value = "/processed_audio.wav"

        result = apply_audio_effects(
            "/audio.wav", -1, 1.0, "video123", mock_cache, mock_processor, force=True
        )

        assert result == "/processed_audio.wav"
        mock_processor.process_audio.assert_called_once()


class TestSeparateVocals:
    """Test separate_vocals function."""

    def test_function_exists(self):
        """Test that separate_vocals function exists."""
        assert separate_vocals is not None
        assert callable(separate_vocals)

    def test_function_signature(self):
        """Test separate_vocals function signature."""
        import inspect

        sig = inspect.signature(separate_vocals)
        params = list(sig.parameters.keys())
        expected_params = [
            "audio_path",
            "video_id",
            "separator",
            "cache_manager",
            "force",
        ]
        for param in expected_params:
            assert param in params

    def test_separates_vocals_when_not_cached(self):
        """Test vocal separation when not cached."""
        mock_cache = Mock()
        mock_cache.find_files.return_value = []  # No cached files
        mock_cache.get_video_cache_dir.return_value = "/cache/video123"

        mock_separator = Mock()
        mock_separator.separate_vocals.return_value = {
            "vocals_path": "/vocals.wav",
            "instrumental_path": "/instrumental.wav",
        }

        result = separate_vocals("/audio.wav", "video123", mock_separator, mock_cache)

        expected = {
            "vocals_path": "/vocals.wav",
            "instrumental_path": "/instrumental.wav",
        }
        assert result == expected
        mock_separator.separate_vocals.assert_called_once()

    def test_uses_cached_result_when_available(self):
        """Test that cached result is used when available."""
        mock_cache = Mock()
        mock_cache.find_files.side_effect = [
            ["/cached_vocals.wav"],  # vocals files
            ["/cached_instrumental.wav"],  # instrumental files
        ]
        mock_separator = Mock()

        result = separate_vocals("/audio.wav", "video123", mock_separator, mock_cache)

        expected = {
            "vocals_path": "/cached_vocals.wav",
            "instrumental_path": "/cached_instrumental.wav",
        }
        assert result == expected

    def test_uses_existing_split_files_when_named(self, tmp_path):
        """Test using pre-separated files when audio filename indicates splits."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        vocals_path = cache_dir / "Song (Vocals).wav"
        instrumental_path = cache_dir / "Song (Instrumental).wav"
        vocals_path.write_text("v")
        instrumental_path.write_text("i")

        mock_cache = Mock()
        mock_cache.get_video_cache_dir.return_value = str(cache_dir)

        result = separate_vocals(
            str(cache_dir / "song_vocals.wav"), "video123", Mock(), mock_cache
        )

        assert result["vocals_path"] == str(vocals_path)
        assert result["instrumental_path"] == str(instrumental_path)

    def test_missing_split_files_raises(self, tmp_path):
        """Test error when expected split files are missing."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "Song (Vocals).wav").write_text("v")

        mock_cache = Mock()
        mock_cache.get_video_cache_dir.return_value = str(cache_dir)

        with pytest.raises(RuntimeError):
            separate_vocals(
                str(cache_dir / "song_vocals.wav"), "video123", Mock(), mock_cache
            )

    def test_force_reseparate_ignores_cache(self):
        """Test that force=True ignores cached result."""
        mock_cache = Mock()
        mock_cache.find_files.return_value = ["/cached.wav"]  # Has cached files
        mock_cache.get_video_cache_dir.return_value = "/cache/video123"

        mock_separator = Mock()
        mock_separator.separate_vocals.return_value = {
            "vocals_path": "/vocals.wav",
            "instrumental_path": "/instrumental.wav",
        }

        _result = separate_vocals(  # noqa: F841
            "/audio.wav", "video123", mock_separator, mock_cache, force=True
        )

        mock_separator.separate_vocals.assert_called_once()

    def test_returns_dict_format(self):
        """Test that function returns dictionary with expected keys."""
        mock_cache = Mock()
        mock_cache.find_files.side_effect = [["/vocals.wav"], ["/instrumental.wav"]]
        mock_separator = Mock()

        result = separate_vocals("/audio.wav", "video123", mock_separator, mock_cache)

        assert isinstance(result, dict)
        assert "vocals_path" in result
        assert "instrumental_path" in result


class TestAudioUtilsIntegration:
    """Test audio_utils module integration."""

    def test_module_imports(self):
        """Test that all required functions can be imported."""
        from y2karaoke.core.components.audio.audio_utils import (
            trim_audio_if_needed,
            apply_audio_effects,
            separate_vocals,
        )

        assert trim_audio_if_needed is not None
        assert apply_audio_effects is not None
        assert separate_vocals is not None

    def test_pydub_integration(self):
        """Test integration with pydub AudioSegment."""
        from y2karaoke.core.components.audio.audio_utils import AudioSegment

        assert AudioSegment is not None

    def test_logging_integration(self):
        """Test that logging is properly integrated."""
        # Test that logger is available
        from y2karaoke.core.components.audio.audio_utils import logger

        assert logger is not None

    def test_all_functions_callable(self):
        """Test that all functions are callable."""
        functions = [trim_audio_if_needed, apply_audio_effects, separate_vocals]
        for func in functions:
            assert callable(func)

    def test_function_parameter_consistency(self):
        """Test that functions have consistent parameter patterns."""
        import inspect

        # All functions should have cache_manager parameter
        for func in [trim_audio_if_needed, apply_audio_effects, separate_vocals]:
            sig = inspect.signature(func)
            assert "cache_manager" in sig.parameters
            assert "force" in sig.parameters


class TestAudioUtilsErrorHandling:
    """Test error handling in audio_utils module."""

    def test_trim_audio_handles_file_errors(self):
        """Test handling of file loading errors in trim_audio_if_needed."""
        mock_cache = Mock()
        mock_cache.get_file_path.return_value = "/trimmed.wav"
        mock_cache.file_exists.return_value = False

        with patch(
            "y2karaoke.core.components.audio.audio_utils.AudioSegment.from_wav"
        ) as mock_from_wav:
            mock_from_wav.side_effect = Exception("File not found")

            with pytest.raises(Exception):
                trim_audio_if_needed("/nonexistent.wav", 5.0, "video123", mock_cache)

    def test_apply_effects_handles_processor_errors(self):
        """Test handling of processor errors in apply_audio_effects."""
        mock_cache = Mock()
        mock_cache.file_exists.return_value = False
        mock_cache.get_file_path.return_value = "/processed.wav"

        mock_processor = Mock()
        mock_processor.process_audio.side_effect = Exception("Processor failed")

        with pytest.raises(Exception):
            apply_audio_effects(
                "/audio.wav", 2, 1.0, "video123", mock_cache, mock_processor
            )

    def test_separate_vocals_handles_separator_errors(self):
        """Test handling of separator errors in separate_vocals."""
        mock_cache = Mock()
        mock_cache.find_files.return_value = []  # No cached files
        mock_cache.get_video_cache_dir.return_value = "/cache"

        mock_separator = Mock()
        mock_separator.separate_vocals.side_effect = Exception("Separator failed")

        with pytest.raises(Exception):
            separate_vocals("/audio.wav", "video123", mock_separator, mock_cache)

    def test_invalid_cache_manager_handling(self):
        """Test handling of invalid cache manager."""
        # Test with None cache manager
        with pytest.raises(AttributeError):
            trim_audio_if_needed("/audio.wav", 5.0, "video123", None)


class TestAudioUtilsCaching:
    """Test caching behavior in audio_utils module."""

    def test_trim_audio_caching_logic(self):
        """Test caching logic in trim_audio_if_needed."""
        mock_cache = Mock()
        mock_cache.get_file_path.return_value = "/trimmed.wav"

        # Test cache hit
        mock_cache.file_exists.return_value = True
        result = trim_audio_if_needed("/audio.wav", 3.0, "video123", mock_cache)
        assert result == "/trimmed.wav"

    def test_apply_effects_caching_logic(self):
        """Test caching logic in apply_audio_effects."""
        mock_cache = Mock()
        mock_cache.get_file_path.return_value = "/processed.wav"
        mock_processor = Mock()

        # Test cache hit
        mock_cache.file_exists.return_value = True
        result = apply_audio_effects(
            "/audio.wav", 1, 1.0, "video123", mock_cache, mock_processor
        )
        assert result == "/processed.wav"

    def test_separate_vocals_caching_logic(self):
        """Test caching logic in separate_vocals."""
        mock_cache = Mock()
        mock_cache.find_files.side_effect = [["/vocals.wav"], ["/instrumental.wav"]]
        mock_separator = Mock()

        # Test cache hit
        result = separate_vocals("/audio.wav", "video123", mock_separator, mock_cache)
        expected = {
            "vocals_path": "/vocals.wav",
            "instrumental_path": "/instrumental.wav",
        }
        assert result == expected
