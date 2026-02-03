import pytest

from y2karaoke.core import lyrics
from y2karaoke.core.models import Line, Word, SongMetadata


def _line_with_words(texts):
    return Line(words=[Word(text=t, start_time=0.0, end_time=0.0) for t in texts])


def test_estimate_singing_duration_clamps_min_and_max():
    short = lyrics._estimate_singing_duration("hi", 1)
    assert short == pytest.approx(0.5)

    long_text = "a" * 300
    long = lyrics._estimate_singing_duration(long_text, 50)
    assert long == pytest.approx(8.0)


def test_create_no_lyrics_placeholder():
    lines, metadata = lyrics._create_no_lyrics_placeholder("Title", "Artist")

    assert len(lines) == 1
    assert lines[0].words[0].text == "Lyrics not available"
    assert metadata.title == "Title"
    assert metadata.artist == "Artist"


def test_detect_and_apply_offset_auto_applies(monkeypatch):
    monkeypatch.setattr("y2karaoke.core.alignment.detect_song_start", lambda _: 5.0)

    line_timings = [(1.0, "Line")]
    updated, offset = lyrics._detect_and_apply_offset(
        "vocals.wav", line_timings, lyrics_offset=None
    )

    assert offset == pytest.approx(4.0)
    assert updated[0][0] == pytest.approx(5.0)


def test_detect_and_apply_offset_respects_manual(monkeypatch):
    monkeypatch.setattr("y2karaoke.core.alignment.detect_song_start", lambda _: 5.0)

    line_timings = [(1.0, "Line")]
    updated, offset = lyrics._detect_and_apply_offset(
        "vocals.wav", line_timings, lyrics_offset=-1.0
    )

    assert offset == pytest.approx(-1.0)
    assert updated[0][0] == pytest.approx(0.0)


def test_detect_and_apply_offset_skips_large_delta(monkeypatch):
    monkeypatch.setattr("y2karaoke.core.alignment.detect_song_start", lambda _: 100.0)

    line_timings = [(1.0, "Line")]
    updated, offset = lyrics._detect_and_apply_offset(
        "vocals.wav", line_timings, lyrics_offset=None
    )

    assert offset == 0.0
    assert updated == line_timings


def test_distribute_word_timing_in_line():
    line = _line_with_words(["hello", "world"])

    lyrics._distribute_word_timing_in_line(line, line_start=0.0, next_line_start=4.0)

    assert line.words[0].start_time == pytest.approx(0.0)
    assert line.words[1].start_time > line.words[0].start_time
    assert line.words[1].end_time <= 3.95


def test_apply_timing_to_lines():
    lines = [_line_with_words(["a", "b"]), _line_with_words(["c"])]
    line_timings = [(1.0, "a b"), (3.0, "c")]

    lyrics._apply_timing_to_lines(lines, line_timings)

    assert lines[0].words[0].start_time == pytest.approx(1.0)
    assert lines[1].words[0].start_time == pytest.approx(3.0)


def test_romanize_lines_only_non_ascii(monkeypatch):
    line = _line_with_words(["hello", "café"])
    monkeypatch.setattr("y2karaoke.core.lyrics.romanize_line", lambda text: "cafe")

    lyrics._romanize_lines([line])

    assert line.words[0].text == "hello"
    assert line.words[1].text == "cafe"


def test_apply_singer_info_sets_word_and_line_singers():
    lines = [_line_with_words(["a", "b"]), _line_with_words(["c"])]
    genius_lines = [("a b", "Singer A"), ("c", "Singer B")]
    metadata = SongMetadata(singers=["Singer A", "Singer B"], is_duet=True)

    lyrics._apply_singer_info(lines, genius_lines, metadata)

    assert lines[0].singer is not None
    assert lines[0].words[0].singer == lines[0].singer
    assert lines[1].singer is not None
    assert lines[1].words[0].singer == lines[1].singer


def test_detect_offset_with_issues_tracks_large_delta(monkeypatch):
    monkeypatch.setattr("y2karaoke.core.alignment.detect_song_start", lambda _: 15.0)

    issues = []
    line_timings = [(1.0, "Line")]
    updated, offset = lyrics._detect_offset_with_issues(
        "vocals.wav", line_timings, lyrics_offset=None, issues=issues
    )

    assert offset == pytest.approx(14.0)
    assert issues
    assert updated[0][0] == pytest.approx(15.0)
