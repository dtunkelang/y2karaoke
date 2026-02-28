from types import SimpleNamespace

import y2karaoke.core.components.whisper.whisperx_compat as compat


def test_patch_torchaudio_for_whisperx_adds_missing_symbols(monkeypatch):
    fake_torchaudio = SimpleNamespace()
    fake_torchaudio.__dict__["__name__"] = "torchaudio"

    def fake_import(name, *args, **kwargs):
        if name == "torchaudio":
            return fake_torchaudio
        return orig_import(name, *args, **kwargs)

    orig_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)

    compat.patch_torchaudio_for_whisperx()

    assert hasattr(fake_torchaudio, "AudioMetaData")
    assert callable(fake_torchaudio.list_audio_backends)
    assert callable(fake_torchaudio.set_audio_backend)
    assert callable(fake_torchaudio.get_audio_backend)
