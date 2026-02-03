from dataclasses import dataclass
import logging

import click
from click.testing import CliRunner
import pytest

import y2karaoke.cli as cli


@dataclass
class DummyTrackInfo:
    youtube_url: str = "https://youtu.be/abc123def45"
    title: str = "Song"
    artist: str = "Artist"
    duration: float = 180.0
    source: str = "youtube"
    lrc_validated: bool = True


class DummyGenerator:
    last_instance = None

    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir
        self.cleanup_called = False
        self.generate_kwargs = None
        DummyGenerator.last_instance = self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return {
            "output_path": "out.mp4",
            "quality_score": 90,
            "quality_level": "good",
            "quality_issues": [],
        }

    def cleanup_temp_files(self):
        self.cleanup_called = True


def test_resolve_url_or_query_requires_title():
    with pytest.raises(click.BadParameter):
        cli.resolve_url_or_query(None, artist=None, title=None)


def test_resolve_url_or_query_builds_query():
    assert (
        cli.resolve_url_or_query(None, artist="Artist", title="Song")
        == "Artist - Song"
    )
    assert cli.resolve_url_or_query(None, artist=None, title="Song") == "Song"


def test_build_video_settings_parses_resolution():
    settings = cli.build_video_settings(
        "1280x720", fps=24, font_size=64, no_progress=False
    )
    assert settings == {
        "width": 1280,
        "height": 720,
        "fps": 24,
        "font_size": 64,
        "show_progress": True,
    }


def test_resolve_shorten_breaks_disables_when_invalid(caplog):
    track_info = DummyTrackInfo(lrc_validated=False)
    caplog.set_level(logging.WARNING)
    assert (
        cli.resolve_shorten_breaks(logging.getLogger("test"), True, track_info) is False
    )
    assert "disabling break shortening" in caplog.text


def test_generate_command_uses_title_search_and_cleans_up(monkeypatch):
    captured = {}

    def fake_identify_track(logger, identifier, url_or_query, artist, title):
        captured["query"] = url_or_query
        return DummyTrackInfo()

    monkeypatch.setattr(cli, "identify_track", fake_identify_track)
    monkeypatch.setattr(cli, "KaraokeGenerator", DummyGenerator)

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        [
            "generate",
            "--title",
            "Song",
            "--artist",
            "Artist",
            "--output",
            "out.mp4",
        ],
    )
    assert result.exit_code == 0
    assert captured["query"] == "Artist - Song"
    generator = DummyGenerator.last_instance
    assert generator is not None
    assert generator.cleanup_called is True
    assert generator.generate_kwargs["filter_promos"] is True


def test_generate_respects_shorten_breaks_toggle(monkeypatch):
    track_info = DummyTrackInfo(lrc_validated=False)

    def fake_identify_track(logger, identifier, url_or_query, artist, title):
        return track_info

    monkeypatch.setattr(cli, "identify_track", fake_identify_track)
    monkeypatch.setattr(cli, "KaraokeGenerator", DummyGenerator)

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["generate", "--title", "Song", "--shorten-breaks", "--output", "out.mp4"],
    )
    assert result.exit_code == 0
    generator = DummyGenerator.last_instance
    assert generator.generate_kwargs["shorten_breaks"] is False
