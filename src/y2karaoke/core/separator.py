import os

"""Vocal separation using audio-separator (demucs)."""

from pathlib import Path
from typing import Dict, Optional

from ..exceptions import SeparationError
from ..utils.performance import timing_decorator
from ..utils.logging import get_logger

logger = get_logger(__name__)


class AudioSeparator:
    """Audio separator with caching and error handling."""

    def __init__(self):
        self._setup_torch()

    def _setup_torch(self):
        """Configure torch to avoid MPS issues on Apple Silicon."""
        pass  # The original code handles this in the function

    @timing_decorator
    def separate_vocals(
        self, audio_path: str, output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """Separate vocals from instrumental track."""
        if output_dir is None:
            output_dir = str(Path(audio_path).parent)

        # Never try to separate an already-separated file
        audio_filename = Path(audio_path).name.lower()
        if any(
            marker in audio_filename
            for marker in ["vocals", "instrumental", "drums", "bass", "other"]
        ):
            raise SeparationError(
                f"Cannot separate an already-separated file: {audio_path}"
            )

        # Check for existing cached stems first
        effective_base = Path(audio_path).stem
        existing_vocals = self._find_existing_file(
            output_dir, f"{effective_base}*_(Vocals)_*.wav"
        )
        existing_instrumental = self._find_existing_file(
            output_dir, f"{effective_base}*_instrumental.wav"
        )

        if existing_vocals and existing_instrumental:
            logger.debug("Using cached vocal separation")
            return {
                "vocals_path": existing_vocals,
                "instrumental_path": existing_instrumental,
            }

        try:
            return separate_vocals(audio_path, output_dir)
        except Exception as e:
            raise SeparationError(f"Vocal separation failed: {e}")

    def _find_existing_file(self, directory: str, pattern: str) -> Optional[str]:
        """Find existing file matching pattern."""
        import glob

        matches = glob.glob(str(Path(directory) / pattern))
        return matches[0] if matches else None


def mix_stems(stem_files: list[str], output_path: str) -> str:
    """Mix multiple audio stems into a single file using pydub."""
    from pydub import AudioSegment

    if not stem_files:
        raise RuntimeError("No stem files to mix")

    # Load and mix all stems
    mixed = None  # type: Optional[AudioSegment]
    for stem_file in stem_files:
        stem = AudioSegment.from_wav(stem_file)
        if mixed is None:
            mixed = stem
        else:
            mixed = mixed.overlay(stem)

    # Export the mixed audio
    if mixed is not None:
        mixed.export(output_path, format="wav")
    return output_path


def separate_vocals(audio_path: str, output_dir: str = ".") -> dict:
    """
    Separate vocals from instrumental track.

    Returns:
        dict with keys: vocals_path, instrumental_path
    """
    import torch

    # Temporarily force Torch to report MPS as unavailable so that
    # audio-separator uses CPU instead of MPS. This avoids the
    # "Output channels > 65536 not supported at the MPS device" error
    # seen with Demucs on Apple Silicon.
    mps = getattr(torch.backends, "mps", None)
    orig_is_available = None
    if mps is not None and hasattr(mps, "is_available"):
        orig_is_available = mps.is_available
        mps.is_available = lambda: False

    try:
        from audio_separator.separator import Separator

        os.makedirs(output_dir, exist_ok=True)

        # Initialize separator with the Demucs model (htdemucs).
        separator = Separator(
            output_dir=output_dir,
            output_format="wav",
            demucs_params={
                "segment_size": "Default",
                "shifts": 2,  # Higher quality separation (original setting)
                "overlap": 0.25,
                "segments_enabled": True,
            },
        )

        separator.load_model(model_filename="htdemucs_ft.yaml")

        # Separate the audio
        output_files = separator.separate(audio_path)
    finally:
        if orig_is_available is not None and mps is not None:
            mps.is_available = orig_is_available

    # Convert to full paths
    output_files = [
        os.path.join(output_dir, f) if not os.path.isabs(f) else f for f in output_files
    ]

    # Find vocals and instrumental in output
    vocals_path = None
    instrumental_path = None

    for f in output_files:
        if "vocals" in f.lower():
            vocals_path = f
        elif "instrumental" in f.lower() or "no_vocals" in f.lower():
            instrumental_path = f

    # If we didn't find an instrumental track, we need to create one
    # by combining non-vocal stems (bass + drums + other)
    if instrumental_path is None and vocals_path:
        non_vocal_files = [f for f in output_files if "vocals" not in f.lower()]
        if non_vocal_files:
            # Mix bass + drums + other into instrumental
            base_name = os.path.basename(audio_path).rsplit(".", 1)[0]
            instrumental_path = os.path.join(
                output_dir, f"{base_name}_instrumental.wav"
            )
            logger.debug(
                f"Mixing {len(non_vocal_files)} stems into instrumental track..."
            )
            mix_stems(non_vocal_files, instrumental_path)

    if not vocals_path or not instrumental_path:
        raise RuntimeError(f"Failed to separate tracks. Output files: {output_files}")

    logger.debug(f"Vocals: {vocals_path}")
    logger.debug(f"Instrumental: {instrumental_path}")

    return {
        "vocals_path": vocals_path,
        "instrumental_path": instrumental_path,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        logger.info("Usage: python separator.py <audio_file>")
        sys.exit(1)

    result = separate_vocals(sys.argv[1], output_dir="./output")
    print(f"Vocals: {result['vocals_path']}")
    print(f"Instrumental: {result['instrumental_path']}")
