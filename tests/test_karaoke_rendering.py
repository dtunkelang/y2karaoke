from pathlib import Path

from y2karaoke.core.karaoke import KaraokeGenerator


def test_render_video_passes_video_settings(monkeypatch, tmp_path):
    generator = KaraokeGenerator(cache_dir=tmp_path)

    captured = {}

    def fake_render_karaoke_video(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "y2karaoke.core.video_writer.render_karaoke_video", fake_render_karaoke_video
    )

    generator._render_video(
        lines=[],
        audio_path="audio.wav",
        output_path=Path("out.mp4"),
        title="Title",
        artist="Artist",
        timing_offset=0.0,
        background_segments=None,
        song_metadata=None,
        video_settings={"resolution": "720p"},
    )

    assert captured["resolution"] == "720p"
    assert captured["title"] == "Title"


def test_create_background_segments_uses_audio_duration(monkeypatch, tmp_path):
    generator = KaraokeGenerator(cache_dir=tmp_path)

    class FakeClip:
        def __init__(self, path):
            self.duration = 12.34

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeProcessor:
        def create_background_segments(self, video_path, lines, duration):
            return {
                "video_path": video_path,
                "duration": duration,
                "lines": lines,
            }

    monkeypatch.setattr("moviepy.AudioFileClip", FakeClip)
    monkeypatch.setattr("y2karaoke.core.backgrounds.BackgroundProcessor", FakeProcessor)

    lines = []
    result = generator._create_background_segments("video.mp4", lines, "audio.wav")

    assert result["video_path"] == "video.mp4"
    assert result["duration"] == 12.34
    assert result["lines"] == lines
