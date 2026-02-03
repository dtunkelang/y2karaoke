import pytest

import y2karaoke.config as config
from y2karaoke.exceptions import ConfigError


def test_validate_config_invalid_tempo(monkeypatch):
    monkeypatch.setattr(config, "TEMPO_RANGE", (0.05, 4.0))
    with pytest.raises(ConfigError):
        config.validate_config()


def test_validate_config_invalid_key_shift(monkeypatch):
    monkeypatch.setattr(config, "KEY_SHIFT_RANGE", (-13, 12))
    with pytest.raises(ConfigError):
        config.validate_config()


def test_validate_config_invalid_dimensions(monkeypatch):
    monkeypatch.setattr(config, "VIDEO_WIDTH", 0)
    with pytest.raises(ConfigError):
        config.validate_config()


def test_validate_config_invalid_fps(monkeypatch):
    monkeypatch.setattr(config, "FPS", 0)
    with pytest.raises(ConfigError):
        config.validate_config()


def test_get_cache_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("Y2KARAOKE_CACHE_DIR", str(tmp_path))
    assert config.get_cache_dir() == tmp_path


def test_parse_resolution_preset():
    assert config.parse_resolution("1080p") == (1920, 1080)


def test_parse_resolution_custom():
    assert config.parse_resolution("800x600") == (800, 600)


def test_parse_resolution_invalid():
    with pytest.raises(ValueError):
        config.parse_resolution("bad")
