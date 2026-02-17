import pytest
from unittest.mock import patch, MagicMock
import sys

from y2karaoke.vision.ocr import get_ocr_engine, _OCR_ENGINE, OCRError
import y2karaoke.vision.ocr as ocr_module


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton _OCR_ENGINE before and after each test."""
    ocr_module._OCR_ENGINE = None
    yield
    ocr_module._OCR_ENGINE = None


@pytest.fixture
def mock_paddle():
    """Mock paddleocr module."""
    mock_module = MagicMock()
    mock_class = MagicMock()
    mock_module.PaddleOCR = mock_class
    with patch.dict(sys.modules, {"paddleocr": mock_module}):
        yield mock_class


def test_get_ocr_engine_uses_vision_on_macos_arm64():
    with (
        patch("platform.system", return_value="Darwin"),
        patch("platform.machine", return_value="arm64"),
        patch("y2karaoke.vision.ocr.VisionOCR") as MockVision,
    ):

        engine = get_ocr_engine()

        assert engine is MockVision.return_value
        MockVision.assert_called_once()


def test_get_ocr_engine_falls_back_to_paddle_if_vision_fails(mock_paddle):
    with (
        patch("platform.system", return_value="Darwin"),
        patch("platform.machine", return_value="arm64"),
        patch(
            "y2karaoke.vision.ocr.VisionOCR",
            side_effect=Exception("Vision init failed"),
        ),
    ):

        engine = get_ocr_engine()

        assert engine is mock_paddle.return_value
        mock_paddle.assert_called_once()


def test_get_ocr_engine_skips_vision_on_linux(mock_paddle):
    with (
        patch("platform.system", return_value="Linux"),
        patch("y2karaoke.vision.ocr.VisionOCR") as MockVision,
    ):

        engine = get_ocr_engine()

        assert engine is mock_paddle.return_value
        MockVision.assert_not_called()
        mock_paddle.assert_called_once()


def test_get_ocr_engine_skips_vision_on_intel_mac(mock_paddle):
    with (
        patch("platform.system", return_value="Darwin"),
        patch("platform.machine", return_value="x86_64"),
        patch("y2karaoke.vision.ocr.VisionOCR") as MockVision,
    ):

        engine = get_ocr_engine()

        assert engine is mock_paddle.return_value
        MockVision.assert_not_called()
        mock_paddle.assert_called_once()


def test_get_ocr_engine_raises_error_if_paddle_missing_on_linux():
    # Ensure paddleocr is NOT in sys.modules
    with patch("platform.system", return_value="Linux"):
        with patch.dict(sys.modules):
            if "paddleocr" in sys.modules:
                del sys.modules["paddleocr"]
            # Also ensure it can't be imported
            # Side effect of import is hard to mock if we deleted it, usually it tries to find it.
            # If it's installed, it will be found.
            # We can map it to None to force ImportError.
            sys.modules["paddleocr"] = None

            with pytest.raises(OCRError, match="PaddleOCR not found"):
                get_ocr_engine()
