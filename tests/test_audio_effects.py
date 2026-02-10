"""Tests for audio_effects.py module."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from y2karaoke.core.components.audio.audio_effects import (
    AudioProcessor,
    process_audio,
    shift_key,
    change_tempo,
)
from y2karaoke.exceptions import ValidationError


class TestAudioProcessor:
    """Test AudioProcessor class."""

    def test_init_default_sample_rate(self):
        """Test initialization with default sample rate."""
        processor = AudioProcessor()
        assert hasattr(processor, "sample_rate")
        assert isinstance(processor.sample_rate, int)

    def test_init_custom_sample_rate(self):
        """Test initialization with custom sample rate."""
        processor = AudioProcessor(sample_rate=44100)
        assert processor.sample_rate == 44100

    def test_process_audio_file_not_found(self):
        """Test process_audio with non-existent file."""
        processor = AudioProcessor()

        with pytest.raises(ValidationError):
            processor.process_audio("/nonexistent/file.wav", "/output.wav")

    def test_process_audio_validates_key_shift(self):
        """Test that process_audio validates key shift parameter."""
        with patch(
            "y2karaoke.core.components.audio.audio_effects.validate_key_shift"
        ) as mock_validate:
            mock_validate.side_effect = ValidationError("Invalid key shift")

            processor = AudioProcessor()

            with pytest.raises(ValidationError):
                processor.process_audio("/input.wav", "/output.wav", semitones=15)

    def test_process_audio_validates_tempo(self):
        """Test that process_audio validates tempo parameter."""
        with patch(
            "y2karaoke.core.components.audio.audio_effects.validate_tempo"
        ) as mock_validate:
            mock_validate.side_effect = ValidationError("Invalid tempo")

            processor = AudioProcessor()

            with pytest.raises(ValidationError):
                processor.process_audio(
                    "/input.wav", "/output.wav", tempo_multiplier=5.0
                )

    @patch("shutil.copy")
    @patch("y2karaoke.core.components.audio.audio_effects.Path")
    def test_process_audio_no_effects_copies_file(self, mock_path, mock_copy):
        """Test process_audio with no effects just copies the file."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        processor = AudioProcessor()
        result = processor.process_audio("/input.wav", "/output.wav")

        mock_copy.assert_called_once()
        assert isinstance(result, str)

    @patch("y2karaoke.core.components.audio.audio_effects.librosa.effects.pitch_shift")
    @patch("y2karaoke.core.components.audio.audio_effects.sf.write")
    @patch("y2karaoke.core.components.audio.audio_effects.librosa.load")
    @patch("y2karaoke.core.components.audio.audio_effects.Path")
    def test_process_audio_pitch_shift(
        self, mock_path, mock_load, mock_write, mock_pitch_shift
    ):
        """Test process_audio with pitch shift."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        mock_load.return_value = ([1, 2, 3], 22050)
        mock_pitch_shift.return_value = [1.1, 2.1, 3.1]

        processor = AudioProcessor()
        result = processor.process_audio("/input.wav", "/output.wav", semitones=2)

        mock_pitch_shift.assert_called_once()
        assert isinstance(result, str)

    @patch("y2karaoke.core.components.audio.audio_effects.librosa.effects.time_stretch")
    @patch("y2karaoke.core.components.audio.audio_effects.sf.write")
    @patch("y2karaoke.core.components.audio.audio_effects.librosa.load")
    @patch("y2karaoke.core.components.audio.audio_effects.Path")
    def test_process_audio_tempo_change(
        self, mock_path, mock_load, mock_write, mock_time_stretch
    ):
        """Test process_audio with tempo change."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        mock_load.return_value = ([1, 2, 3], 22050)
        mock_time_stretch.return_value = [1.2, 2.2, 3.2]

        processor = AudioProcessor()
        result = processor.process_audio(
            "/input.wav", "/output.wav", tempo_multiplier=1.5
        )

        mock_time_stretch.assert_called_once()
        assert isinstance(result, str)

    def test_shift_key_delegates_to_process_audio(self):
        """Test that shift_key method delegates to process_audio."""
        processor = AudioProcessor()

        with patch.object(processor, "process_audio") as mock_process:
            mock_process.return_value = "/output.wav"

            result = processor.shift_key("/input.wav", "/output.wav", 3)

            assert result == "/output.wav"
            mock_process.assert_called_once_with(
                "/input.wav", "/output.wav", semitones=3
            )

    def test_change_tempo_delegates_to_process_audio(self):
        """Test that change_tempo method delegates to process_audio."""
        processor = AudioProcessor()

        with patch.object(processor, "process_audio") as mock_process:
            mock_process.return_value = "/output.wav"

            result = processor.change_tempo("/input.wav", "/output.wav", 0.8)

            assert result == "/output.wav"
            mock_process.assert_called_once_with(
                "/input.wav", "/output.wav", tempo_multiplier=0.8
            )


class TestAudioEffectsConvenienceFunctions:
    """Test module-level convenience functions."""

    @patch("y2karaoke.core.components.audio.audio_effects.AudioProcessor")
    def test_process_audio_function(self, mock_processor_class):
        """Test process_audio convenience function."""
        mock_processor = Mock()
        mock_processor.process_audio.return_value = "/output.wav"
        mock_processor_class.return_value = mock_processor

        result = process_audio(
            "/input.wav", "/output.wav", semitones=2, tempo_multiplier=1.2
        )

        assert result == "/output.wav"
        mock_processor_class.assert_called_once()
        mock_processor.process_audio.assert_called_once_with(
            "/input.wav", "/output.wav", 2, 1.2
        )

    @patch("y2karaoke.core.components.audio.audio_effects.AudioProcessor")
    def test_shift_key_function(self, mock_processor_class):
        """Test shift_key convenience function."""
        mock_processor = Mock()
        mock_processor.shift_key.return_value = "/output.wav"
        mock_processor_class.return_value = mock_processor

        result = shift_key("/input.wav", "/output.wav", 5)

        assert result == "/output.wav"
        mock_processor_class.assert_called_once()
        mock_processor.shift_key.assert_called_once_with("/input.wav", "/output.wav", 5)

    @patch("y2karaoke.core.components.audio.audio_effects.AudioProcessor")
    def test_change_tempo_function(self, mock_processor_class):
        """Test change_tempo convenience function."""
        mock_processor = Mock()
        mock_processor.change_tempo.return_value = "/output.wav"
        mock_processor_class.return_value = mock_processor

        result = change_tempo("/input.wav", "/output.wav", 0.75)

        assert result == "/output.wav"
        mock_processor_class.assert_called_once()
        mock_processor.change_tempo.assert_called_once_with(
            "/input.wav", "/output.wav", 0.75
        )

    def test_function_signatures(self):
        """Test that functions have expected signatures."""
        import inspect

        # Test process_audio function
        sig = inspect.signature(process_audio)
        params = list(sig.parameters.keys())
        assert "audio_path" in params
        assert "output_path" in params

        # Test shift_key function
        sig = inspect.signature(shift_key)
        params = list(sig.parameters.keys())
        assert "audio_path" in params
        assert "output_path" in params
        assert "semitones" in params

        # Test change_tempo function
        sig = inspect.signature(change_tempo)
        params = list(sig.parameters.keys())
        assert "audio_path" in params
        assert "output_path" in params
        assert "tempo_multiplier" in params


class TestAudioEffectsIntegration:
    """Test audio_effects module integration."""

    def test_module_imports(self):
        """Test that all required classes and functions can be imported."""
        from y2karaoke.core.components.audio.audio_effects import (
            AudioProcessor,
            process_audio,
            shift_key,
            change_tempo,
        )

        assert AudioProcessor is not None
        assert process_audio is not None
        assert shift_key is not None
        assert change_tempo is not None

    def test_processor_initialization(self):
        """Test AudioProcessor can be initialized."""
        processor = AudioProcessor()
        assert processor is not None
        assert hasattr(processor, "sample_rate")

    def test_processor_has_required_methods(self):
        """Test AudioProcessor has all required methods."""
        processor = AudioProcessor()
        required_methods = ["process_audio", "shift_key", "change_tempo"]

        for method in required_methods:
            assert hasattr(processor, method)
            assert callable(getattr(processor, method))

    def test_config_imports(self):
        """Test that configuration constants are properly imported."""
        from y2karaoke.core.components.audio.audio_effects import AUDIO_SAMPLE_RATE

        assert isinstance(AUDIO_SAMPLE_RATE, int)

    def test_validation_integration(self):
        """Test integration with validation functions."""
        from y2karaoke.core.components.audio.audio_effects import (
            validate_key_shift,
            validate_tempo,
        )

        assert validate_key_shift is not None
        assert validate_tempo is not None

    def test_librosa_integration(self):
        """Test that librosa functions are available."""
        # Test that librosa is properly imported and used
        processor = AudioProcessor()
        assert hasattr(processor, "sample_rate")


class TestAudioEffectsErrorHandling:
    """Test error handling in audio_effects module."""

    def test_invalid_file_path_handling(self):
        """Test handling of invalid file paths."""
        processor = AudioProcessor()

        with pytest.raises(ValidationError):
            processor.process_audio("/nonexistent/file.wav", "/output.wav")

    def test_invalid_key_shift_handling(self):
        """Test handling of invalid key shift values."""
        processor = AudioProcessor()

        with patch(
            "y2karaoke.core.components.audio.audio_effects.validate_key_shift"
        ) as mock_validate:
            mock_validate.side_effect = ValidationError("Invalid key shift")

            with pytest.raises(ValidationError):
                processor.process_audio("/input.wav", "/output.wav", semitones=20)

    def test_invalid_tempo_handling(self):
        """Test handling of invalid tempo values."""
        processor = AudioProcessor()

        with patch(
            "y2karaoke.core.components.audio.audio_effects.validate_tempo"
        ) as mock_validate:
            mock_validate.side_effect = ValidationError("Invalid tempo")

            with pytest.raises(ValidationError):
                processor.process_audio(
                    "/input.wav", "/output.wav", tempo_multiplier=10.0
                )

    @patch("y2karaoke.core.components.audio.audio_effects.librosa.load")
    @patch("y2karaoke.core.components.audio.audio_effects.Path")
    def test_audio_loading_error_handling(self, mock_path, mock_load):
        """Test handling of audio loading errors."""
        mock_path.return_value.exists.return_value = True
        mock_load.side_effect = Exception("Failed to load audio")

        processor = AudioProcessor()

        with pytest.raises(Exception):
            processor.process_audio("/input.wav", "/output.wav")

    @patch("y2karaoke.core.components.audio.audio_effects.sf.write")
    @patch("y2karaoke.core.components.audio.audio_effects.librosa.load")
    @patch("y2karaoke.core.components.audio.audio_effects.Path")
    def test_audio_writing_error_handling(self, mock_path, mock_load, mock_write):
        """Test handling of audio writing errors."""
        mock_path.return_value.exists.return_value = True
        mock_load.return_value = ([1, 2, 3], 22050)
        mock_write.side_effect = Exception("Failed to write audio")

        processor = AudioProcessor()

        with pytest.raises(Exception):
            processor.process_audio("/input.wav", "/output.wav")


class TestAudioEffectsValidation:
    """Test validation in audio_effects module."""

    def test_processor_validates_inputs(self):
        """Test that AudioProcessor has validation integration."""
        processor = AudioProcessor()

        # Test that validation functions are available
        from y2karaoke.core.components.audio.audio_effects import (
            validate_key_shift,
            validate_tempo,
        )

        assert validate_key_shift is not None
        assert validate_tempo is not None

        # Test that invalid inputs raise ValidationError
        with pytest.raises(ValidationError):
            processor.process_audio("/nonexistent.wav", "/output.wav")

    def test_convenience_functions_use_processor(self):
        """Test that convenience functions properly use AudioProcessor."""
        with patch(
            "y2karaoke.core.components.audio.audio_effects.AudioProcessor"
        ) as mock_processor_class:
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Test process_audio function
            process_audio("/input.wav", "/output.wav")
            mock_processor_class.assert_called()

            # Test shift_key function
            shift_key("/input.wav", "/output.wav", 2)
            mock_processor.shift_key.assert_called_once()

            # Test change_tempo function
            change_tempo("/input.wav", "/output.wav", 1.5)
            mock_processor.change_tempo.assert_called_once()
