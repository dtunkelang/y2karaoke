from unittest.mock import MagicMock, patch
from y2karaoke.core.health import SystemDoctor


@patch("shutil.which")
@patch("subprocess.run")
def test_check_ffmpeg_installed(mock_run, mock_which):
    mock_which.return_value = "/usr/bin/ffmpeg"
    mock_run.return_value.stdout = "ffmpeg version 4.4.2 Copyright (c) 2000-2021 the FFmpeg developers"
    
    doc = SystemDoctor()
    doc.check_ffmpeg()
    
    result = doc.results[0]
    assert result.name == "ffmpeg"
    assert result.is_installed is True
    assert result.version == "4.4.2"


@patch("shutil.which")
def test_check_ffmpeg_missing(mock_which):
    mock_which.return_value = None
    
    doc = SystemDoctor()
    doc.check_ffmpeg()
    
    result = doc.results[0]
    assert result.name == "ffmpeg"
    assert result.is_installed is False
    assert result.critical is True


def test_check_yt_dlp_installed():
    # We assume yt-dlp is installed in the test env, but let's mock it to be safe
    with patch.dict("sys.modules", {"yt_dlp": MagicMock(version={"__version__": "2023.01.01"})}):
        doc = SystemDoctor()
        doc.check_yt_dlp()
        
        result = doc.results[0]
        assert result.name == "yt-dlp"
        assert result.is_installed is True
        assert result.version == "2023.01.01"


def test_check_ocr_fallback():
    # Mock platform to NOT be macOS
    with patch("platform.system", return_value="Linux"):
        with patch.dict("sys.modules", {"paddleocr": MagicMock()}):
            doc = SystemDoctor()
            doc.check_ocr()
            
            # Should have 2 results: Vision (False) and Paddle (True)
            vision_res = next(r for r in doc.results if r.name == "Apple Vision OCR")
            paddle_res = next(r for r in doc.results if r.name == "PaddleOCR")
            
            assert vision_res.is_installed is False
            assert paddle_res.is_installed is True
            assert paddle_res.critical is True  # Critical because Vision failed
