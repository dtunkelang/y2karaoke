from y2karaoke.core.components.lyrics.runtime_config import (
    LyricsRuntimeConfig,
    load_lyrics_runtime_config,
)


def test_load_lyrics_runtime_config_reads_env(monkeypatch):
    monkeypatch.setenv("Y2K_PREFERRED_LYRICS_PROVIDER", "lyriq")
    monkeypatch.setenv("Y2K_LRC_DURATION_TOLERANCE_SEC", "18")

    config = load_lyrics_runtime_config()

    assert config == LyricsRuntimeConfig(
        preferred_provider="lyriq",
        lrc_duration_tolerance_sec=18,
    )


def test_load_lyrics_runtime_config_explicit_values_override_env(monkeypatch):
    monkeypatch.setenv("Y2K_PREFERRED_LYRICS_PROVIDER", "syncedlyrics")
    monkeypatch.setenv("Y2K_LRC_DURATION_TOLERANCE_SEC", "18")

    config = load_lyrics_runtime_config(
        preferred_provider="lyriq",
        lrc_duration_tolerance_sec=6,
    )

    assert config == LyricsRuntimeConfig(
        preferred_provider="lyriq",
        lrc_duration_tolerance_sec=6,
    )
