from y2karaoke.core.break_shortener import (
    BreakEdit,
    InstrumentalBreak,
    adjust_lyrics_timing,
    shorten_instrumental_breaks,
)
from y2karaoke.core.models import Line, Word


class DummyAudio:
    last_export = None

    def __init__(self, length_ms):
        self.length_ms = length_ms

    def __len__(self):
        return self.length_ms

    def __getitem__(self, item):
        if isinstance(item, slice):
            start = item.start or 0
            stop = item.stop if item.stop is not None else self.length_ms
            return DummyAudio(max(0, stop - start))
        raise TypeError("Unsupported index")

    def fade_out(self, _duration):
        return self

    def fade_in(self, _duration):
        return self

    def overlay(self, other):
        return DummyAudio(max(len(self), len(other)))

    def __add__(self, other):
        return DummyAudio(len(self) + len(other))

    def export(self, path, format="wav"):
        DummyAudio.last_export = (path, format)
        return path


def test_shorten_instrumental_breaks_no_long_breaks(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.break_shortener.detect_instrumental_breaks",
        lambda _path, min_break_duration=5.0: [],
    )
    monkeypatch.setattr(
        "y2karaoke.core.break_shortener.AudioSegment.from_file",
        lambda _path: DummyAudio(10000),
    )

    output_path, edits = shorten_instrumental_breaks(
        "audio.wav", "vocals.wav", "out.wav"
    )
    assert output_path == "audio.wav"
    assert edits == []


def test_shorten_instrumental_breaks_skips_intro(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.break_shortener.detect_instrumental_breaks",
        lambda _path, min_break_duration=5.0: [InstrumentalBreak(start=2.0, end=40.0)],
    )
    monkeypatch.setattr(
        "y2karaoke.core.break_shortener.AudioSegment.from_file",
        lambda _path: DummyAudio(10000),
    )

    output_path, edits = shorten_instrumental_breaks(
        "audio.wav",
        "vocals.wav",
        "out.wav",
        max_break_duration=10.0,
        skip_intro=True,
        intro_threshold=10.0,
    )
    assert output_path == "audio.wav"
    assert edits == []


def test_shorten_instrumental_breaks_applies_edit(monkeypatch):
    dummy_audio = DummyAudio(120000)

    monkeypatch.setattr(
        "y2karaoke.core.break_shortener.detect_instrumental_breaks",
        lambda _path, min_break_duration=5.0: [InstrumentalBreak(start=20.0, end=60.0)],
    )
    monkeypatch.setattr(
        "y2karaoke.core.break_shortener.AudioSegment.from_file",
        lambda _path: dummy_audio,
    )
    monkeypatch.setattr(
        "y2karaoke.core.break_shortener.find_beat_near",
        lambda _path, t: t,
    )

    output_path, edits = shorten_instrumental_breaks(
        "audio.wav",
        "vocals.wav",
        "out.wav",
        max_break_duration=30.0,
        keep_start=5.0,
        keep_end=5.0,
        crossfade_ms=1000,
    )
    assert output_path == "out.wav"
    assert len(edits) == 1
    edit = edits[0]
    assert edit.original_start == 20.0
    assert edit.original_end == 60.0
    assert edit.new_duration == 11.0
    assert edit.time_removed == 29.0
    assert DummyAudio.last_export == ("out.wav", "wav")


def test_adjust_lyrics_timing_uses_fallback_cut_start():
    lines = [
        Line(words=[Word(text="a", start_time=14.0, end_time=14.5)]),
        Line(words=[Word(text="b", start_time=15.0, end_time=15.5)]),
    ]
    edits = [
        BreakEdit(
            original_start=10.0,
            original_end=30.0,
            new_end=0.0,
            time_removed=2.0,
            cut_start=0.0,
        )
    ]
    adjusted = adjust_lyrics_timing(lines, edits, keep_start=5.0)
    assert adjusted[0].words[0].start_time == 14.0
    assert adjusted[1].words[0].start_time == 13.0


def test_adjust_lyrics_timing_respects_cut_start_override():
    lines = [
        Line(words=[Word(text="a", start_time=10.0, end_time=10.5)]),
        Line(words=[Word(text="b", start_time=12.0, end_time=12.5)]),
    ]
    edits = [
        BreakEdit(
            original_start=5.0,
            original_end=20.0,
            new_end=0.0,
            time_removed=3.0,
            cut_start=11.0,
        )
    ]
    adjusted = adjust_lyrics_timing(lines, edits, keep_start=5.0)
    assert adjusted[0].words[0].start_time == 10.0
    assert adjusted[1].words[0].start_time == 9.0
