"""Compatibility helpers for importing whisperx with newer torchaudio builds."""

from __future__ import annotations

import os


def patch_torchaudio_for_whisperx() -> None:
    """Patch removed torchaudio symbols expected by whisperx dependencies."""
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    try:
        import torchaudio  # type: ignore
    except Exception:
        return

    if not hasattr(torchaudio, "AudioMetaData"):

        class AudioMetaData:  # pragma: no cover - trivial compatibility shim
            pass

        setattr(torchaudio, "AudioMetaData", AudioMetaData)

    if not hasattr(torchaudio, "list_audio_backends"):
        setattr(torchaudio, "list_audio_backends", lambda: ["soundfile"])

    if not hasattr(torchaudio, "set_audio_backend"):
        setattr(torchaudio, "set_audio_backend", lambda _backend: None)

    if not hasattr(torchaudio, "get_audio_backend"):
        setattr(torchaudio, "get_audio_backend", lambda: "soundfile")
