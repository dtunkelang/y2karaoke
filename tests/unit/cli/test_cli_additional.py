from dataclasses import dataclass
import logging

from click.testing import CliRunner
import pytest

import y2karaoke.cli as cli
from y2karaoke.exceptions import Y2KaraokeError


@dataclass
class DummyTrackInfo:
    youtube_url: str = "https://youtu.be/abc123def45"
    title: str = "Song"
    artist: str = "Artist"
    duration: float = 180.0
    source: str = "search"
    lrc_validated: bool = True


class DummyIdentifier:
    def __init__(self):
        self.last_url = None
        self.last_query = None

    def identify_from_url(self, url, artist_hint=None, title_hint=None):
        self.last_url = url
        return DummyTrackInfo(source="url")

    def identify_from_search(self, query):
        self.last_query = query
        return DummyTrackInfo(source="search")


def test_parse_resolution_invalid():
    with pytest.raises(ValueError):
        cli.parse_resolution("bad")


def test_identify_track_from_url(caplog):
    identifier = DummyIdentifier()
    caplog.set_level(logging.INFO)
    track = cli.identify_track(
        logging.getLogger("test"),
        identifier,
        "https://youtube.com/watch?v=abc",
        artist=None,
        title=None,
    )
    assert identifier.last_url.startswith("https://youtube.com")
    assert track.source == "url"
    assert "Identifying track from URL" in caplog.text


def test_identify_track_from_search(caplog):
    identifier = DummyIdentifier()
    caplog.set_level(logging.INFO)
    track = cli.identify_track(
        logging.getLogger("test"),
        identifier,
        "some query",
        artist=None,
        title=None,
    )
    assert identifier.last_query == "some query"
    assert track.source == "search"
    assert "Identifying track from search" in caplog.text


def test_build_video_settings_no_progress_only():
    settings = cli.build_video_settings(None, None, None, True)
    assert settings == {"show_progress": False}


def test_log_quality_summary_low_quality(caplog):
    caplog.set_level(logging.INFO)
    result = {
        "quality_score": 40,
        "quality_level": "poor",
        "quality_issues": ["Issue 1", "Issue 2", "Issue 3", "Issue 4"],
        "lyrics_source": "lrc",
        "alignment_method": "whisper",
    }
    cli.log_quality_summary(logging.getLogger("test"), result)
    assert "Quality: 40/100 (poor)" in caplog.text
    assert "Lyrics source: lrc" in caplog.text
    assert "Alignment: whisper" in caplog.text
    assert "Consider using --whisper" in caplog.text


def test_cache_stats_command(monkeypatch, tmp_path):
    class DummyCacheManager:
        def __init__(self, cache_dir):
            self.cache_dir = cache_dir

        def get_cache_stats(self):
            return {
                "cache_dir": str(self.cache_dir),
                "total_size_gb": 1.25,
                "file_count": 10,
                "video_count": 2,
            }

    monkeypatch.setattr("y2karaoke.utils.cache.CacheManager", DummyCacheManager)

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["cache", "stats", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "Cache Directory:" in result.output
    assert "Total Size:" in result.output


def test_cache_cleanup_command(monkeypatch, tmp_path):
    class DummyCacheManager:
        def __init__(self, cache_dir):
            self.cache_dir = cache_dir
            self.cleaned_days = None

        def cleanup_old_files(self, days):
            self.cleaned_days = days

    monkeypatch.setattr("y2karaoke.utils.cache.CacheManager", DummyCacheManager)

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["cache", "cleanup", "--cache-dir", str(tmp_path), "--days", "7"],
        input="y\n",
    )
    assert result.exit_code == 0
    assert "Cache cleanup completed" in result.output


def test_cache_clear_command(monkeypatch, tmp_path):
    class DummyCacheManager:
        def __init__(self, cache_dir):
            self.cache_dir = cache_dir
            self.cleared_id = None

        def clear_video_cache(self, video_id):
            self.cleared_id = video_id

    monkeypatch.setattr("y2karaoke.utils.cache.CacheManager", DummyCacheManager)

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["cache", "clear", "abc123", "--cache-dir", str(tmp_path)],
        input="y\n",
    )
    assert result.exit_code == 0
    assert "Cleared cache for video abc123" in result.output


def test_evaluate_timing_uses_title_search(monkeypatch, tmp_path):
    class DummyDownloader:
        def __init__(self, cache_dir):
            self.cache_dir = cache_dir

        def download_audio(self, url):
            return {"audio_path": str(tmp_path / "audio.wav")}

    def fake_separate_vocals(audio_path, output_dir):
        return {"vocals_path": str(tmp_path / "vocals.wav")}

    captured = {}

    def fake_report(title, artist, vocals_path):
        captured["title"] = title
        captured["artist"] = artist
        captured["vocals_path"] = vocals_path

    monkeypatch.setattr("y2karaoke.pipeline.identify.TrackIdentifier", DummyIdentifier)
    monkeypatch.setattr("y2karaoke.pipeline.audio.YouTubeDownloader", DummyDownloader)
    monkeypatch.setattr(
        "y2karaoke.pipeline.audio.separate_vocals", fake_separate_vocals
    )
    monkeypatch.setattr(
        "y2karaoke.pipeline.alignment.print_comparison_report", fake_report
    )

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        [
            "evaluate-timing",
            "--title",
            "Song",
            "--artist",
            "Artist",
            "--work-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    assert captured["title"] == "Song"
    assert captured["artist"] == "Artist"


def test_generate_handles_y2karaoke_error(monkeypatch):
    class DummyGenerator:
        def __init__(self, cache_dir=None):
            pass

        def generate(self, **_kwargs):
            raise Y2KaraokeError("boom")

    def fake_identify_track(_logger, _identifier, _url_or_query, _artist, _title):
        return DummyTrackInfo()

    monkeypatch.setattr(cli, "identify_track", fake_identify_track)
    monkeypatch.setattr(cli, "KaraokeGenerator", DummyGenerator)

    runner = CliRunner()
    result = runner.invoke(
        cli.cli, ["generate", "--title", "Song", "--output", "out.mp4"]
    )
    assert result.exit_code == 1


def test_evaluate_timing_handles_error(monkeypatch):
    class BadIdentifier:
        def identify_from_search(self, _query):
            raise RuntimeError("nope")

    monkeypatch.setattr("y2karaoke.pipeline.identify.TrackIdentifier", BadIdentifier)

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["evaluate-timing", "query"])
    assert result.exit_code == 1
