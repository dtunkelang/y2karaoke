"""System health checks for external dependencies and environment."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    name: str
    is_installed: bool
    version: Optional[str] = None
    message: str = ""
    critical: bool = True


class SystemDoctor:
    """Diagnose system environment and dependencies."""

    def __init__(self):
        self.results: List[HealthStatus] = []

    def check_all(self) -> List[HealthStatus]:
        """Run all health checks."""
        self.results = []
        self.check_ffmpeg()
        self.check_yt_dlp()
        self.check_ocr()
        self.check_cache_permissions()
        return self.results

    def _add_result(
        self,
        name: str,
        is_installed: bool,
        version: Optional[str] = None,
        message: str = "",
        critical: bool = True,
    ):
        self.results.append(
            HealthStatus(
                name=name,
                is_installed=is_installed,
                version=version,
                message=message,
                critical=critical,
            )
        )

    def check_ffmpeg(self):
        """Check for ffmpeg availability and version."""
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            self._add_result(
                "ffmpeg",
                False,
                message="Required for audio processing. Install via brew/apt.",
            )
            return

        try:
            result = subprocess.run(
                [ffmpeg_path, "-version"], capture_output=True, text=True, check=True
            )
            # Parse first line like "ffmpeg version 6.0 ..."
            version_line = result.stdout.splitlines()[0]
            version = version_line.split()[2]
            self._add_result("ffmpeg", True, version=version)
        except Exception as e:
            self._add_result(
                "ffmpeg", True, message=f"Installed but failed to run: {e}"
            )

    def check_yt_dlp(self):
        """Check for yt-dlp availability."""
        # yt-dlp is a Python dependency, so it should be importable
        try:
            import yt_dlp

            version = getattr(yt_dlp, "version", {}).get("__version__", "unknown")
            self._add_result("yt-dlp", True, version=version)
        except ImportError:
            self._add_result(
                "yt-dlp", False, message="Required for downloading. pip install yt-dlp"
            )

    def check_ocr(self):
        """Check available OCR engines."""
        # Check Apple Vision
        import platform

        is_mac_arm = platform.system() == "Darwin" and platform.machine() == "arm64"

        vision_available = False
        if is_mac_arm:
            try:
                import Vision  # noqa
                import Quartz  # noqa

                vision_available = True
            except ImportError:
                pass

        self._add_result(
            "Apple Vision OCR",
            vision_available,
            message="Only available on macOS (Apple Silicon)",
            critical=False,
        )

        # Check PaddleOCR
        paddle_available = False
        try:
            import paddleocr  # noqa

            paddle_available = True
        except ImportError:
            pass

        self._add_result(
            "PaddleOCR",
            paddle_available,
            message="Cross-platform fallback. pip install paddlepaddle paddleocr",
            critical=not vision_available,
        )

    def check_cache_permissions(self):
        """Check if we can write to the cache directory."""
        from ..config import get_cache_dir

        cache_dir = get_cache_dir()

        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=cache_dir, delete=True) as f:
                f.write(b"test")
            self._add_result("Cache Directory", True, message=f"Writable: {cache_dir}")
        except Exception as e:
            self._add_result(
                "Cache Directory", False, message=f"Cannot write to {cache_dir}: {e}"
            )
