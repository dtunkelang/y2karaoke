from y2karaoke.core.components.whisper.whisper_runtime_config import (
    WhisperRuntimeConfig,
    load_whisper_runtime_config,
    normalize_whisper_profile,
)


def test_load_whisper_runtime_config_defaults(monkeypatch):
    monkeypatch.delenv("Y2K_WHISPER_PROFILE", raising=False)
    monkeypatch.delenv(
        "Y2K_WHISPER_ENABLE_TAIL_SHORTFALL_FORCED_FALLBACK", raising=False
    )
    monkeypatch.delenv("Y2K_WHISPER_ENABLE_LOW_SUPPORT_ONSET_REANCHOR", raising=False)
    monkeypatch.delenv("Y2K_WHISPER_ENABLE_REPEAT_CADENCE_REANCHOR", raising=False)
    monkeypatch.delenv("Y2K_WHISPER_ENABLE_RESTORED_RUN_ONSET_SHIFT", raising=False)

    assert load_whisper_runtime_config() == WhisperRuntimeConfig()


def test_load_whisper_runtime_config_reads_env(monkeypatch):
    monkeypatch.setenv("Y2K_WHISPER_PROFILE", "safe")
    monkeypatch.setenv("Y2K_WHISPER_ENABLE_TAIL_SHORTFALL_FORCED_FALLBACK", "1")
    monkeypatch.setenv("Y2K_WHISPER_ENABLE_LOW_SUPPORT_ONSET_REANCHOR", "1")
    monkeypatch.setenv("Y2K_WHISPER_ENABLE_REPEAT_CADENCE_REANCHOR", "1")
    monkeypatch.setenv("Y2K_WHISPER_ENABLE_RESTORED_RUN_ONSET_SHIFT", "1")
    monkeypatch.setenv("Y2K_REPEAT_DURATION_NORMALIZE", "1")
    monkeypatch.setenv("Y2K_WHISPER_DISABLE_REPEAT_SHIFT", "1")
    monkeypatch.setenv("Y2K_WHISPER_DISABLE_MONOTONIC_START_ENFORCE", "1")

    assert load_whisper_runtime_config() == WhisperRuntimeConfig(
        profile="safe",
        tail_shortfall_forced_fallback=True,
        low_support_onset_reanchor=True,
        repeat_cadence_reanchor=True,
        restored_run_onset_shift=True,
        repeat_duration_normalize=True,
        disable_repeat_shift=True,
        disable_monotonic_start_enforce=True,
    )


def test_load_whisper_runtime_config_explicit_overrides_take_precedence(monkeypatch):
    monkeypatch.setenv("Y2K_WHISPER_PROFILE", "safe")
    monkeypatch.setenv("Y2K_WHISPER_ENABLE_TAIL_SHORTFALL_FORCED_FALLBACK", "1")

    config = load_whisper_runtime_config(
        profile="aggressive",
        tail_shortfall_forced_fallback=False,
    )

    assert config.profile == "aggressive"
    assert not config.tail_shortfall_forced_fallback


def test_normalize_whisper_profile_falls_back_to_default():
    assert normalize_whisper_profile("experimental") == "default"
