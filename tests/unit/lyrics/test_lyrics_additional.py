import pytest

from y2karaoke.core.components.lyrics import api as lyrics
from y2karaoke.core import lyrics_whisper as lw
from y2karaoke.core.components.lyrics import helpers as lh
from y2karaoke.core.components.lyrics import api as lyrics_api
from y2karaoke.core.models import Line, SongMetadata, Word


def _line(text, start=0.0, end=1.0):
    return Line(words=[Word(text=text, start_time=start, end_time=end)])


def test_detect_and_apply_offset_large_delta_applies(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda *_: 20.0,
    )

    line_timings = [(1.0, "Line")]
    updated, offset = lh._detect_and_apply_offset(
        "vocals.wav", line_timings, lyrics_offset=None
    )

    assert offset == pytest.approx(19.0)
    assert updated[0][0] == pytest.approx(20.0)


def test_detect_offset_with_issues_respects_manual(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda *_: 10.0,
    )

    issues = []
    line_timings = [(1.0, "Line")]
    updated, offset = lw._detect_offset_with_issues(
        "vocals.wav", line_timings, lyrics_offset=2.0, issues=issues
    )

    assert offset == pytest.approx(2.0)
    assert updated[0][0] == pytest.approx(3.0)
    assert issues == []


def test_fetch_lrc_text_and_timings_duration_match(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.components.lyrics.sync.fetch_lyrics_for_duration",
        lambda *a, **k: ("[00:01.00]hi", True, "src", 120),
    )
    monkeypatch.setattr(lw, "parse_lrc_with_timing", lambda *a, **k: [(1.0, "hi")])

    lrc_text, timings, source = lw._fetch_lrc_text_and_timings(
        "Title",
        "Artist",
        target_duration=120,
    )

    assert lrc_text == "[00:01.00]hi"
    assert timings == [(1.0, "hi")]
    assert source == "src"


def test_fetch_lrc_text_and_timings_standard_fetch(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.components.lyrics.sync.fetch_lyrics_multi_source",
        lambda *a, **k: ("[00:01.00]hi", True, "src"),
    )
    monkeypatch.setattr(lw, "parse_lrc_with_timing", lambda *a, **k: [(1.0, "hi")])

    lrc_text, timings, source = lw._fetch_lrc_text_and_timings(
        "Title",
        "Artist",
    )

    assert lrc_text == "[00:01.00]hi"
    assert timings == [(1.0, "hi")]
    assert source == "src"


def test_fetch_lrc_text_and_timings_standard_fetch_not_synced(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.components.lyrics.sync.fetch_lyrics_multi_source",
        lambda *a, **k: (None, False, "src"),
    )

    lrc_text, timings, source = lw._fetch_lrc_text_and_timings(
        "Title",
        "Artist",
    )

    assert lrc_text is None
    assert timings is None
    assert source == ""


def test_fetch_lrc_text_and_timings_handles_exception(monkeypatch):
    def raise_error(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "y2karaoke.core.components.lyrics.sync.fetch_lyrics_multi_source", raise_error
    )

    lrc_text, timings, source = lw._fetch_lrc_text_and_timings(
        "Title",
        "Artist",
    )

    assert lrc_text is None
    assert timings is None
    assert source == ""


def test_calculate_quality_score_no_lyrics_quality():
    report = {
        "lyrics_quality": None,
        "alignment_method": "genius_fallback",
        "issues": ["a"],
    }

    score = lw._calculate_quality_score(report)
    assert score == pytest.approx(5.0)


def test_get_lyrics_simple_whisper_failure(monkeypatch):
    lrc_text = "[00:01.00]First\n[00:03.00]Second"
    line_timings = [(1.0, "First"), (3.0, "Second")]

    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda *_: 0.0,
    )
    monkeypatch.setattr(
        lw,
        "_fetch_lrc_text_and_timings",
        lambda *a, **k: (lrc_text, line_timings, "src"),
    )
    monkeypatch.setattr(
        "y2karaoke.core.components.lyrics.genius.fetch_genius_lyrics_with_singers",
        lambda *a, **k: ([], SongMetadata(singers=[], is_duet=False)),
    )
    monkeypatch.setattr(
        lh,
        "create_lines_from_lrc",
        lambda *a, **k: [_line("First"), _line("Second")],
    )
    monkeypatch.setattr(
        lh, "_refine_timing_with_audio", lambda *a, **k: [_line("First")]
    )

    def raise_whisper(*args, **kwargs):
        raise RuntimeError("whisper down")

    monkeypatch.setattr(lh, "_apply_whisper_alignment", raise_whisper)

    lines, _ = lw.get_lyrics_simple(
        "Title",
        "Artist",
        vocals_path="vocals.wav",
        use_whisper=True,
        romanize=False,
    )

    assert lines


def test_get_lyrics_simple_whisper_success(monkeypatch):
    lrc_text = "[00:01.00]First\n[00:03.00]Second"
    line_timings = [(1.0, "First"), (3.0, "Second")]

    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda *_: 0.0,
    )
    monkeypatch.setattr(
        lw,
        "_fetch_lrc_text_and_timings",
        lambda *a, **k: (lrc_text, line_timings, "src"),
    )
    monkeypatch.setattr(
        "y2karaoke.core.components.lyrics.genius.fetch_genius_lyrics_with_singers",
        lambda *a, **k: ([], SongMetadata(singers=[], is_duet=False)),
    )
    monkeypatch.setattr(
        lh,
        "create_lines_from_lrc",
        lambda *a, **k: [_line("First"), _line("Second")],
    )
    monkeypatch.setattr(
        lh, "_refine_timing_with_audio", lambda *a, **k: [_line("First")]
    )
    monkeypatch.setattr(
        lyrics,
        "_apply_whisper_alignment",
        lambda *a, **k: ([_line("First")], ["fix"]),
    )

    lines, _ = lw.get_lyrics_simple(
        "Title",
        "Artist",
        vocals_path="vocals.wav",
        use_whisper=True,
        romanize=False,
    )

    assert lines


def test_get_lyrics_with_quality_lrc_path(monkeypatch):
    lrc_text = "[00:01.00]First\n[00:03.00]Second"
    line_timings = [(1.0, "First"), (3.0, "Second")]
    metadata = SongMetadata(singers=["A", "B"], is_duet=True)

    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda *_: 0.0,
    )
    monkeypatch.setattr(
        lw,
        "_fetch_lrc_text_and_timings",
        lambda *a, **k: (lrc_text, line_timings, "src"),
    )
    monkeypatch.setattr(
        lw,
        "_fetch_genius_with_quality_tracking",
        lambda *a, **k: ([("First", "A"), ("Second", "B")], metadata),
    )
    monkeypatch.setattr(
        lh,
        "create_lines_from_lrc",
        lambda *a, **k: [_line("First"), _line("Second")],
    )
    monkeypatch.setattr(
        lw,
        "_refine_timing_with_quality",
        lambda *a, **k: ([_line("First")], "onset_refined"),
    )
    monkeypatch.setattr(
        lyrics,
        "_apply_whisper_alignment",
        lambda *a, **k: ([_line("First")], ["fix"]),
    )

    lines, meta, report = lw.get_lyrics_with_quality(
        "Title",
        "Artist",
        vocals_path="vocals.wav",
        use_whisper=True,
        romanize=False,
    )

    assert lines
    assert meta == metadata
    assert report["alignment_method"] in {"onset_refined", "whisper_hybrid"}
    assert report["total_lines"] == len(lines)


def test_get_lyrics_with_quality_fallback_to_genius(monkeypatch):
    monkeypatch.setattr(
        lw, "_fetch_lrc_text_and_timings", lambda *a, **k: (None, None, "")
    )
    monkeypatch.setattr(
        lw,
        "_fetch_genius_with_quality_tracking",
        lambda *a, **k: (
            [("Hello", "Singer")],
            SongMetadata(singers=["Singer"], is_duet=False),
        ),
    )
    monkeypatch.setattr(lw, "create_lines_from_lrc", lambda *a, **k: [_line("Hello")])

    lines, meta, report = lw.get_lyrics_with_quality(
        "Title",
        "Artist",
        vocals_path=None,
        romanize=False,
    )

    assert lines
    assert meta is not None
    assert report["total_lines"] == len(lines)


def test_fetch_genius_with_quality_tracking_lrc_present(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.components.lyrics.genius.fetch_genius_lyrics_with_singers",
        lambda *a, **k: (
            [("Hello", "Singer")],
            SongMetadata(singers=["Singer"], is_duet=False),
        ),
    )
    report = {"alignment_method": "none", "issues": [], "overall_score": 100.0}

    lines, metadata = lw._fetch_genius_with_quality_tracking(
        line_timings=[(1.0, "hello")],
        title="Song",
        artist="Artist",
        quality_report=report,
    )

    assert lines
    assert metadata is not None


def test_lyrics_processor_returns_placeholder(tmp_path):
    processor = lyrics.LyricsProcessor(cache_dir=tmp_path)

    lines, metadata = processor.get_lyrics(title=None, artist=None)

    assert metadata is not None
    assert metadata.title == "Unknown"
    assert metadata.artist == "Unknown"
    assert len(lines) == 1


def test_lyrics_processor_delegates_to_get_lyrics_simple(tmp_path, monkeypatch):
    processor = lyrics.LyricsProcessor(cache_dir=tmp_path)

    captured = {}

    def fake_get_lyrics_simple(*args, **kwargs):
        captured["kwargs"] = kwargs
        return [_line("Hello")], SongMetadata(singers=[], is_duet=False)

    monkeypatch.setattr(lyrics_api, "get_lyrics_simple", fake_get_lyrics_simple)

    lines, metadata = processor.get_lyrics(
        title="Title", artist="Artist", romanize=False
    )

    assert lines
    assert metadata is not None
    assert captured["kwargs"]["cache_dir"] == str(tmp_path)


def test_get_lyrics_convenience_calls_simple(monkeypatch):
    monkeypatch.setattr(
        lyrics_api,
        "get_lyrics_simple",
        lambda *a, **k: ([_line("Hi")], SongMetadata(singers=[], is_duet=False)),
    )

    lines, metadata = lyrics.get_lyrics("Title", "Artist")

    assert lines
    assert metadata is not None
