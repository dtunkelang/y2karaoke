"""Tests for separator.py - vocal separation functionality (minimal)."""

import pytest
from unittest.mock import patch, MagicMock

from y2karaoke.core.separator import AudioSeparator, separate_vocals, mix_stems


class TestAudioSeparator:
    def test_init(self):
        separator = AudioSeparator()
        assert separator is not None

    def test_setup_torch_exists(self):
        separator = AudioSeparator()
        assert hasattr(separator, '_setup_torch')
        assert callable(separator._setup_torch)

    def test_find_existing_file_exists(self):
        separator = AudioSeparator()
        assert hasattr(separator, '_find_existing_file')
        assert callable(separator._find_existing_file)

    def test_separate_vocals_method_exists(self):
        separator = AudioSeparator()
        assert hasattr(separator, 'separate_vocals')
        assert callable(separator.separate_vocals)

    @patch('pathlib.Path.glob')
    def test_find_existing_file_not_found(self, mock_glob):
        mock_glob.return_value = []
        
        separator = AudioSeparator()
        result = separator._find_existing_file("/test", "*vocals*")
        
        assert result is None

    @patch('y2karaoke.core.separator.AudioSeparator._find_existing_file')
    def test_separate_vocals_uses_existing_files(self, mock_find):
        mock_find.side_effect = ["/test/vocals.wav", "/test/instrumental.wav"]
        
        separator = AudioSeparator()
        result = separator.separate_vocals("/test/audio.wav", "/test")
        
        assert result["vocals_path"] == "/test/vocals.wav"
        assert result["instrumental_path"] == "/test/instrumental.wav"
        assert mock_find.call_count == 2


class TestSeparateVocalsFunction:
    def test_separate_vocals_function_exists(self):
        assert callable(separate_vocals)

    def test_separate_vocals_function_signature(self):
        import inspect
        sig = inspect.signature(separate_vocals)
        params = list(sig.parameters.keys())
        assert "audio_path" in params
        assert "output_dir" in params


class TestMixStems:
    def test_mix_stems_function_exists(self):
        assert callable(mix_stems)

    def test_mix_stems_empty_list_raises_error(self):
        with pytest.raises(RuntimeError, match="No stem files to mix"):
            mix_stems([], "/test/mixed.wav")

    @patch('pydub.AudioSegment.from_wav')
    def test_mix_stems_handles_single_file(self, mock_from_wav):
        mock_segment = MagicMock()
        mock_from_wav.return_value = mock_segment
        
        # Mock the export method
        mock_segment.export = MagicMock()
        
        stem_files = ["/test/stem1.wav"]
        output_path = "/test/mixed.wav"
        
        result = mix_stems(stem_files, output_path)
        
        assert result == output_path
        mock_from_wav.assert_called_once_with("/test/stem1.wav")

    @patch('pydub.AudioSegment.from_wav')
    def test_mix_stems_handles_multiple_files(self, mock_from_wav):
        mock_segment1 = MagicMock()
        mock_segment2 = MagicMock()
        mock_mixed = MagicMock()
        
        mock_from_wav.side_effect = [mock_segment1, mock_segment2]
        mock_segment1.__add__ = MagicMock(return_value=mock_mixed)
        mock_mixed.export = MagicMock()
        
        stem_files = ["/test/stem1.wav", "/test/stem2.wav"]
        output_path = "/test/mixed.wav"
        
        result = mix_stems(stem_files, output_path)
        
        assert result == output_path
        assert mock_from_wav.call_count == 2


class TestModuleIntegration:
    def test_module_imports(self):
        # Test that all required classes and functions can be imported
        from y2karaoke.core.separator import AudioSeparator, separate_vocals, mix_stems
        
        assert AudioSeparator is not None
        assert separate_vocals is not None
        assert mix_stems is not None

    def test_separator_initialization(self):
        separator = AudioSeparator()
        assert separator is not None

    def test_separator_has_required_methods(self):
        separator = AudioSeparator()
        
        assert hasattr(separator, 'separate_vocals')
        assert hasattr(separator, '_find_existing_file')
        assert hasattr(separator, '_setup_torch')
        assert callable(separator.separate_vocals)
        assert callable(separator._find_existing_file)
        assert callable(separator._setup_torch)

    @patch('y2karaoke.core.separator.AudioSeparator._find_existing_file')
    def test_separate_vocals_return_format(self, mock_find):
        mock_find.side_effect = ["/vocals.wav", "/instrumental.wav"]
        
        separator = AudioSeparator()
        result = separator.separate_vocals("audio.wav", ".")
        
        # Test return format
        assert isinstance(result, dict)
        assert "vocals_path" in result
        assert "instrumental_path" in result
        assert result["vocals_path"] == "/vocals.wav"
        assert result["instrumental_path"] == "/instrumental.wav"

    def test_function_signatures(self):
        # Test that functions have expected signatures
        import inspect
        
        # Test separate_vocals function signature
        sig = inspect.signature(separate_vocals)
        params = list(sig.parameters.keys())
        assert "audio_path" in params
        assert "output_dir" in params
        
        # Test mix_stems function signature
        sig = inspect.signature(mix_stems)
        params = list(sig.parameters.keys())
        assert "stem_files" in params
        assert "output_path" in params

    def test_audio_separator_class_structure(self):
        # Test class structure without instantiation issues
        assert hasattr(AudioSeparator, '__init__')
        assert hasattr(AudioSeparator, 'separate_vocals')
        assert hasattr(AudioSeparator, '_find_existing_file')
        assert hasattr(AudioSeparator, '_setup_torch')
