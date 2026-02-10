import json

from y2karaoke.core.karaoke import KaraokeGenerator
from y2karaoke.core.components.lyrics.api import Line, Word
from y2karaoke.core.break_shortener import BreakEdit


def _make_line(text, start, end):
    word = Word(text=text, start_time=start, end_time=end, singer=None)
    return Line(words=[word], singer=None)


def test_scale_lyrics_timing_noop(tmp_path):
    generator = KaraokeGenerator(cache_dir=tmp_path)
    lines = [_make_line("hello", 1.0, 2.0)]
    scaled = generator._scale_lyrics_timing(lines, 1.0)
    assert scaled is lines


def test_scale_lyrics_timing_scales_values(tmp_path):
    generator = KaraokeGenerator(cache_dir=tmp_path)
    lines = [_make_line("hello", 2.0, 4.0)]
    scaled = generator._scale_lyrics_timing(lines, 2.0)

    assert scaled is not lines
    assert scaled[0].words[0].start_time == 1.0
    assert scaled[0].words[0].end_time == 2.0


def test_shorten_breaks_cached_audio_no_edits(tmp_path):
    generator = KaraokeGenerator(cache_dir=tmp_path)
    video_id = "vid"

    shortened_path = generator.cache_manager.get_file_path(
        video_id, "shortened_breaks_30s.wav"
    )
    shortened_path.write_bytes(b"audio")

    shortened, edits = generator._shorten_breaks(
        "audio.wav",
        "vocals.wav",
        "inst.wav",
        video_id,
        max_break_duration=30.0,
        force=False,
    )

    assert shortened == str(shortened_path)
    assert edits == []


def test_shorten_breaks_writes_edits_cache(monkeypatch, tmp_path):
    generator = KaraokeGenerator(cache_dir=tmp_path)
    video_id = "vid"

    def fake_shorten(*args, **kwargs):
        return (
            str(
                generator.cache_manager.get_file_path(
                    video_id, "shortened_breaks_30s.wav"
                )
            ),
            [
                BreakEdit(
                    original_start=5.0,
                    original_end=25.0,
                    new_end=15.0,
                    time_removed=10.0,
                    cut_start=6.0,
                )
            ],
        )

    monkeypatch.setattr(
        "y2karaoke.core.break_shortener.shorten_instrumental_breaks", fake_shorten
    )

    shortened, edits = generator._shorten_breaks(
        "audio.wav",
        "vocals.wav",
        "inst.wav",
        video_id,
        max_break_duration=30.0,
        force=True,
    )

    edits_path = generator.cache_manager.get_file_path(
        video_id, "shortened_breaks_30s_edits.json"
    )

    assert shortened.endswith("shortened_breaks_30s.wav")
    assert len(edits) == 1
    assert edits_path.exists()
    data = json.loads(edits_path.read_text())
    assert data[0]["time_removed"] == 10.0
