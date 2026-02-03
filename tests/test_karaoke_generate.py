from pathlib import Path

import pytest

from y2karaoke.core.karaoke import KaraokeGenerator
from y2karaoke.core.lyrics import Line, Word


def _line(text, start, end):
    return Line(words=[Word(text=text, start_time=start, end_time=end, singer=None)])


def test_generate_offsets_lines_and_uses_vocals_debug(monkeypatch, tmp_path):
    generator = KaraokeGenerator(cache_dir=tmp_path)
    monkeypatch.setattr("y2karaoke.core.karaoke.extract_video_id", lambda _: "vid")
    monkeypatch.setattr(generator.cache_manager, "auto_cleanup", lambda: None)

    monkeypatch.setattr(
        generator,
        "_download_audio",
        lambda *a, **k: {
            "audio_path": "audio.wav",
            "title": "Title",
            "artist": "Artist",
        },
    )
    monkeypatch.setattr(
        "y2karaoke.core.karaoke.trim_audio_if_needed", lambda *a, **k: "trimmed.wav"
    )
    monkeypatch.setattr(
        "y2karaoke.core.karaoke.separate_vocals",
        lambda *a, **k: {
            "vocals_path": "vocals.wav",
            "instrumental_path": "inst.wav",
        },
    )

    lyrics_line = _line("hi", 1.0, 1.5)
    monkeypatch.setattr(
        generator,
        "_get_lyrics",
        lambda *a, **k: {
            "lines": [lyrics_line],
            "metadata": {},
            "quality": {
                "overall_score": 85.0,
                "issues": [],
                "source": "src",
                "alignment_method": "method",
            },
        },
    )

    captured = {}

    def fake_apply(audio_path, *args, **kwargs):
        captured["audio_path"] = audio_path
        return "processed.wav"

    monkeypatch.setattr("y2karaoke.core.karaoke.apply_audio_effects", fake_apply)

    def fake_render_video(**kwargs):
        captured["render"] = kwargs

    monkeypatch.setattr(generator, "_render_video", fake_render_video)

    result = generator.generate(
        url="https://youtube.com/watch?v=test",
        output_path=tmp_path / "out.mp4",
        debug_audio="vocals",
    )

    assert captured["audio_path"] == "vocals.wav"
    rendered_lines = captured["render"]["lines"]
    assert rendered_lines[0].start_time == pytest.approx(3.5)
    assert result["quality_level"] == "high"


def test_generate_uses_original_audio_debug(monkeypatch, tmp_path):
    generator = KaraokeGenerator(cache_dir=tmp_path)
    monkeypatch.setattr("y2karaoke.core.karaoke.extract_video_id", lambda _: "vid")
    monkeypatch.setattr(generator.cache_manager, "auto_cleanup", lambda: None)

    monkeypatch.setattr(
        generator,
        "_download_audio",
        lambda *a, **k: {
            "audio_path": "audio.wav",
            "title": "Title",
            "artist": "Artist",
        },
    )
    monkeypatch.setattr(
        "y2karaoke.core.karaoke.trim_audio_if_needed", lambda *a, **k: "trimmed.wav"
    )
    monkeypatch.setattr(
        "y2karaoke.core.karaoke.separate_vocals",
        lambda *a, **k: {
            "vocals_path": "vocals.wav",
            "instrumental_path": "inst.wav",
        },
    )
    monkeypatch.setattr(
        generator,
        "_get_lyrics",
        lambda *a, **k: {
            "lines": [_line("hi", 4.0, 4.5)],
            "metadata": {},
            "quality": {"overall_score": 50.0, "issues": []},
        },
    )

    captured = {}

    def fake_apply(audio_path, *args, **kwargs):
        captured["audio_path"] = audio_path
        return "processed.wav"

    monkeypatch.setattr("y2karaoke.core.karaoke.apply_audio_effects", fake_apply)
    monkeypatch.setattr(generator, "_render_video", lambda **kwargs: None)

    generator.generate(
        url="https://youtube.com/watch?v=test",
        output_path=Path("out.mp4"),
        debug_audio="original",
    )

    assert captured["audio_path"] == "audio.wav"
