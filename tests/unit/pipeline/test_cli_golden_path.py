from click.testing import CliRunner

import y2karaoke.cli as cli
import y2karaoke.cli_commands as cli_commands


class _TrackInfo:
    youtube_url = "https://youtube.com/watch?v=abc123def45"
    title = "Song"
    artist = "Artist"
    duration = 185.0
    source = "search"
    lrc_validated = True


class _DummyIdentifier:
    def identify_from_search(self, _query):
        return _TrackInfo()

    def identify_from_url(self, _url, artist_hint=None, title_hint=None):
        return _TrackInfo()

    def get_cached_youtube_metadata(self, _url):
        return ("Song", "Artist", 185.0)


class _DummyGenerator:
    last_instance = None

    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir
        self.generate_kwargs = None
        self.cleanup_called = False
        _DummyGenerator.last_instance = self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return {
            "output_path": "out.mp4",
            "rendered": False,
            "quality_score": 88.0,
            "quality_level": "high",
            "quality_issues": [],
            "lyrics_source": "mock",
            "alignment_method": "hybrid",
        }

    def cleanup_temp_files(self):
        self.cleanup_called = True


def test_cli_generate_golden_path_wiring(monkeypatch):
    monkeypatch.setattr(cli_commands, "TrackIdentifier", _DummyIdentifier)
    monkeypatch.setattr(cli, "KaraokeGenerator", _DummyGenerator)

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        [
            "generate",
            "song artist query",
            "--whisper",
            "--tempo",
            "1.1",
            "--no-render",
            "--output",
            "out.mp4",
        ],
    )

    assert result.exit_code == 0
    generator = _DummyGenerator.last_instance
    assert generator is not None
    assert generator.cleanup_called is True
    assert generator.generate_kwargs is not None
    assert generator.generate_kwargs["use_whisper"] is True
    assert generator.generate_kwargs["tempo_multiplier"] == 1.1
    assert generator.generate_kwargs["skip_render"] is True
