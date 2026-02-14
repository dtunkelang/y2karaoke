import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from y2karaoke.core.audio_analysis import (
    compute_harmonic_chroma,
    calculate_harmonic_suitability,
)


def test_compute_harmonic_chroma_normalizes():
    # Mock librosa load and feature extraction
    mock_librosa = MagicMock()
    mock_librosa.load.return_value = (np.zeros(1000), 22050)
    # 12 pitch classes, 5 frames
    mock_chroma = np.ones((12, 5))
    mock_librosa.feature.chroma_cqt.return_value = mock_chroma

    with patch(
        "y2karaoke.core.audio_analysis._load_librosa", return_value=mock_librosa
    ):
        chroma = compute_harmonic_chroma("fake.wav")
        assert chroma is not None
        assert chroma.shape == (12, 5)
        # Each column should be normalized (sum of squares = 1)
        # For a vector of 12 'ones', L2 norm is sqrt(12)
        # Normalized value should be 1 / sqrt(12)
        expected_val = 1.0 / np.sqrt(12)
        np.testing.assert_allclose(chroma[:, 0], expected_val, atol=1e-5)


def test_calculate_harmonic_suitability_perfect_match():
    # Mock compute_harmonic_chroma to return identical features
    mock_chroma = np.random.rand(12, 10)

    # Mock librosa dtw
    mock_librosa = MagicMock()
    # Perfect path: (0,0), (1,1) ... (9,9)
    mock_path = [(i, i) for i in range(10)]
    mock_librosa.sequence.dtw.return_value = (np.zeros((10, 10)), mock_path)

    with (
        patch(
            "y2karaoke.core.audio_analysis.compute_harmonic_chroma",
            return_value=mock_chroma,
        ),
        patch("y2karaoke.core.audio_analysis._load_librosa", return_value=mock_librosa),
    ):

        metrics = calculate_harmonic_suitability("orig.wav", "kara.wav")

        assert metrics["similarity_cost"] == 0.0
        assert metrics["best_key_shift"] == 0
        assert metrics["structure_jump_count"] == 0
        assert metrics["tempo_variance"] == 0.0
