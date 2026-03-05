from y2karaoke.core.components.whisper.whisper_profile import get_whisper_profile


def test_get_whisper_profile_defaults_to_default(monkeypatch):
    monkeypatch.delenv("Y2K_WHISPER_PROFILE", raising=False)
    assert get_whisper_profile() == "default"


def test_get_whisper_profile_accepts_known_values(monkeypatch):
    monkeypatch.setenv("Y2K_WHISPER_PROFILE", "safe")
    assert get_whisper_profile() == "safe"

    monkeypatch.setenv("Y2K_WHISPER_PROFILE", "aggressive")
    assert get_whisper_profile() == "aggressive"


def test_get_whisper_profile_falls_back_on_unknown(monkeypatch):
    monkeypatch.setenv("Y2K_WHISPER_PROFILE", "experimental")
    assert get_whisper_profile() == "default"
