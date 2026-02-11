from pathlib import Path

import pytest

from y2karaoke.core.karaoke import KaraokeGenerator
from y2karaoke.core.components.lyrics.api import Line, Word


def _line(text, start, end):
    return Line(words=[Word(text=text, start_time=start, end_time=end, singer=None)])


def test_generate_offsets_lines_and_uses_vocals_debug(monkeypatch, tmp_path):
    generator = KaraokeGenerator(cache_dir=tmp_path)
    monkeypatch.setattr(generator.cache_manager, "auto_cleanup", lambda: None)

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

    captured = {}

    def fake_apply(audio_path, *args, **kwargs):
        captured["audio_path"] = audio_path
        return "processed.wav"

    monkeypatch.setattr("y2karaoke.core.karaoke.apply_audio_effects", fake_apply)

    def fake_render_video(**kwargs):
        captured["render"] = kwargs

    with generator.use_test_hooks(
        download_audio_fn=lambda *a, **k: {
            "audio_path": "audio.wav",
            "title": "Title",
            "artist": "Artist",
        },
        get_lyrics_fn=lambda *a, **k: {
            "lines": [lyrics_line],
            "metadata": {},
            "quality": {
                "overall_score": 85.0,
                "issues": [],
                "source": "src",
                "alignment_method": "method",
            },
        },
        render_video_fn=fake_render_video,
    ):
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
    monkeypatch.setattr(generator.cache_manager, "auto_cleanup", lambda: None)

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

    captured = {}

    def fake_apply(audio_path, *args, **kwargs):
        captured["audio_path"] = audio_path
        return "processed.wav"

    monkeypatch.setattr("y2karaoke.core.karaoke.apply_audio_effects", fake_apply)
    with generator.use_test_hooks(
        download_audio_fn=lambda *a, **k: {
            "audio_path": "audio.wav",
            "title": "Title",
            "artist": "Artist",
        },
        get_lyrics_fn=lambda *a, **k: {
            "lines": [_line("hi", 4.0, 4.5)],
            "metadata": {},
            "quality": {"overall_score": 50.0, "issues": []},
        },
        render_video_fn=lambda **kwargs: None,
    ):
        generator.generate(
            url="https://youtube.com/watch?v=test",
            output_path=Path("out.mp4"),
            debug_audio="original",
        )

    assert captured["audio_path"] == "audio.wav"


def test_generate_instrumental_backgrounds_and_breaks(monkeypatch, tmp_path):
    generator = KaraokeGenerator(cache_dir=tmp_path)
    monkeypatch.setattr(generator.cache_manager, "auto_cleanup", lambda: None)

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

    lyrics_line = _line("hi", 4.0, 4.5)
    monkeypatch.setattr(
        "y2karaoke.core.break_shortener.adjust_lyrics_timing",
        lambda *_args, **_kwargs: [_line("hi", 5.0, 5.5)],
    )

    monkeypatch.setattr("y2karaoke.core.karaoke.sanitize_filename", lambda _: "safe")

    captured = {}

    def fake_apply(audio_path, *args, **kwargs):
        captured["audio_path"] = audio_path
        return "processed.wav"

    monkeypatch.setattr("y2karaoke.core.karaoke.apply_audio_effects", fake_apply)

    def fake_render_video(**kwargs):
        captured["render"] = kwargs

    with generator.use_test_hooks(
        download_audio_fn=lambda *a, **k: {
            "audio_path": "audio.wav",
            "title": "Title",
            "artist": "Artist",
        },
        download_video_fn=lambda *a, **k: {"video_path": "video.mp4"},
        get_lyrics_fn=lambda *a, **k: {
            "lines": [lyrics_line],
            "metadata": {},
            "quality": {"overall_score": 20.0, "issues": ["bad sync", "promo"]},
        },
        scale_lyrics_timing_fn=lambda lines, tempo: lines,
        shorten_breaks_fn=lambda *a, **k: ("processed.wav", ["edit"]),
        create_background_segments_fn=lambda *_a, **_k: "bg",
        render_video_fn=fake_render_video,
    ):
        result = generator.generate(
            url="https://youtube.com/watch?v=test",
            use_backgrounds=True,
            shorten_breaks=True,
        )

    assert captured["audio_path"] == "inst.wav"
    assert captured["render"]["background_segments"] == "bg"
    assert captured["render"]["output_path"].name == "safe_karaoke.mp4"
    assert captured["render"]["lines"][0].start_time == pytest.approx(5.0)
    assert result["quality_level"] == "low"
