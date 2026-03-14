import pytest
from pathlib import Path

from y2karaoke.core.components.lyrics import api as lyrics
from y2karaoke.core.components.lyrics import helpers as lh
from y2karaoke.core import lyrics_whisper as lw
from y2karaoke.core.models import Line, Word, SongMetadata
from y2karaoke.core.components.alignment.timing_models import (
    TranscriptionSegment,
    TranscriptionWord,
)


def _line_with_words(texts):
    return Line(words=[Word(text=t, start_time=0.0, end_time=0.0) for t in texts])


def test_estimate_singing_duration_clamps_min_and_max():
    short = lh._estimate_singing_duration("hi", 1)
    assert short == pytest.approx(0.5)

    long_text = "a" * 300
    long = lh._estimate_singing_duration(long_text, 50)
    assert long == pytest.approx(8.0)


def test_create_no_lyrics_placeholder():
    lines, metadata = lyrics._create_no_lyrics_placeholder("Title", "Artist")

    assert len(lines) == 1
    assert lines[0].words[0].text == "Lyrics not available"
    assert metadata.title == "Title"
    assert metadata.artist == "Artist"


def test_detect_and_apply_offset_skips_suspicious_delta(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start", lambda _: 5.0
    )

    line_timings = [(1.0, "Line")]
    updated, offset = lh._detect_and_apply_offset(
        "vocals.wav", line_timings, lyrics_offset=None
    )

    assert offset == 0.0
    assert updated[0][0] == pytest.approx(1.0)


def test_detect_and_apply_offset_respects_manual(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start", lambda _: 5.0
    )

    line_timings = [(1.0, "Line")]
    updated, offset = lh._detect_and_apply_offset(
        "vocals.wav", line_timings, lyrics_offset=-1.0
    )

    assert offset == pytest.approx(-1.0)
    assert updated[0][0] == pytest.approx(0.0)


def test_detect_and_apply_offset_skips_large_delta(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda _: 100.0,
    )

    line_timings = [(1.0, "Line")]
    updated, offset = lh._detect_and_apply_offset(
        "vocals.wav", line_timings, lyrics_offset=None
    )

    assert offset == 0.0
    assert updated == line_timings


def test_detect_and_apply_offset_uses_second_line_after_long_interjection_gap(
    monkeypatch,
):
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda _: 27.98,
    )

    line_timings = [
        (13.42, "Yeah"),
        (26.95, "I've been tryna call"),
    ]
    updated, offset = lh._detect_and_apply_offset(
        "vocals.wav", line_timings, lyrics_offset=None
    )

    assert offset == pytest.approx(0.618, abs=0.01)
    assert updated[0][0] == pytest.approx(14.04, abs=0.01)
    assert updated[1][0] == pytest.approx(27.57, abs=0.01)


def test_distribute_word_timing_in_line():
    line = _line_with_words(["hello", "world"])

    lh._distribute_word_timing_in_line(line, line_start=0.0, next_line_start=4.0)

    assert line.words[0].start_time == pytest.approx(0.0)
    assert line.words[1].start_time > line.words[0].start_time
    assert line.words[1].end_time <= 3.95


def test_distribute_word_timing_extends_pause_heavy_long_line():
    line = _line_with_words(
        ["I", "said,", "ooh,", "I'm", "blinded", "by", "the", "lights"]
    )

    lh._distribute_word_timing_in_line(line, line_start=150.59, next_line_start=156.57)

    assert line.end_time == pytest.approx(156.42, abs=0.02)


def test_distribute_word_timing_keeps_parenthetical_interjection_tail_default_cap():
    line = _line_with_words(["The", "city's", "cold", "and", "empty", "(oh)"])

    lh._distribute_word_timing_in_line(line, line_start=106.92, next_line_start=111.65)

    assert line.end_time == pytest.approx(109.18, abs=0.02)


def test_distribute_word_timing_keeps_comma_heavy_non_interjection_line_default_cap():
    line = _line_with_words(
        [
            "Sans",
            "toi",
            "ma",
            "vie",
            "n'est",
            "qu'un",
            "decor",
            "qui",
            "brille,",
            "vide",
            "de",
            "sens",
        ]
    )

    lh._distribute_word_timing_in_line(line, line_start=92.55, next_line_start=99.9)

    assert line.end_time == pytest.approx(96.81, abs=0.02)


def test_distribute_word_timing_keeps_leading_interjection_phrase_default_cap():
    line = _line_with_words(
        ["Oh,", "when", "I'm", "like", "this,", "you're", "the", "one", "I", "trust"]
    )

    lh._distribute_word_timing_in_line(line, line_start=134.17, next_line_start=139.33)

    assert line.end_time == pytest.approx(137.52, abs=0.02)


def test_distribute_word_timing_extends_phrase_break_dense_line():
    line = _line_with_words(
        ["Maybe", "you", "can", "show", "me", "how", "to", "love,", "maybe"]
    )

    lh._distribute_word_timing_in_line(line, line_start=32.62, next_line_start=38.24)

    assert line.end_time == pytest.approx(36.42, abs=0.03)


def test_clean_text_lines_uncensors_common_starred_profanity():
    assert lh._clean_text_lines(["  sh*t  ", "What the f*ck is that"]) == [
        "shit",
        "What the fuck is that",
    ]


def test_load_lyrics_file_uncensors_plain_text_and_lrc(tmp_path: Path):
    plain = tmp_path / "lyrics.txt"
    plain.write_text("This sh*t hurts\nWhat the f*ck is that\n", encoding="utf-8")

    _, plain_timings, plain_lines = lh._load_lyrics_file(plain, filter_promos=False)
    assert plain_timings is None
    assert plain_lines == ["This shit hurts", "What the fuck is that"]

    synced = tmp_path / "lyrics.lrc"
    synced.write_text(
        "[00:01.00]This sh*t hurts\n[00:02.00]What the f*ck is that\n",
        encoding="utf-8",
    )

    _, synced_timings, synced_lines = lh._load_lyrics_file(synced, filter_promos=False)
    assert synced_timings == [
        (1.0, "This shit hurts"),
        (2.0, "What the fuck is that"),
    ]
    assert synced_lines == ["This shit hurts", "What the fuck is that"]


def test_distribute_word_timing_extends_long_line_with_trailing_parenthetical_interjection():
    line = _line_with_words(
        ["Will", "never", "let", "you", "go", "this", "time", "(ooh)"]
    )

    lh._distribute_word_timing_in_line(line, line_start=145.46, next_line_start=150.59)

    assert line.end_time > 148.65


def test_apply_timing_to_lines():
    lines = [_line_with_words(["a", "b"]), _line_with_words(["c"])]
    line_timings = [(1.0, "a b"), (3.0, "c")]

    lh._apply_timing_to_lines(lines, line_timings)

    assert lines[0].words[0].start_time == pytest.approx(1.0)
    assert lines[1].words[0].start_time == pytest.approx(3.0)


def test_romanize_lines_only_non_ascii(monkeypatch):
    line = _line_with_words(["hello", "café"])
    monkeypatch.setattr(
        "y2karaoke.core.components.lyrics.helpers.romanize_line", lambda text: "cafe"
    )

    lh._romanize_lines([line])

    assert line.words[0].text == "hello"
    assert line.words[1].text == "cafe"


def test_apply_singer_info_sets_word_and_line_singers():
    lines = [_line_with_words(["a", "b"]), _line_with_words(["c"])]
    genius_lines = [("a b", "Singer A"), ("c", "Singer B")]
    metadata = SongMetadata(singers=["Singer A", "Singer B"], is_duet=True)

    lw._apply_singer_info(lines, genius_lines, metadata)

    assert lines[0].singer is not None
    assert lines[0].words[0].singer == lines[0].singer
    assert lines[1].singer is not None
    assert lines[1].words[0].singer == lines[1].singer


def test_detect_offset_with_issues_tracks_large_delta(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda _: 15.0,
    )

    issues = []
    line_timings = [(1.0, "Line")]
    updated, offset = lw._detect_offset_with_issues(
        "vocals.wav", line_timings, lyrics_offset=None, issues=issues
    )

    assert offset == 0.0
    assert issues
    assert updated == line_timings


def test_detect_offset_with_issues_scales_auto_offset(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda _: 2.0,
    )

    issues = []
    line_timings = [(1.0, "Line")]
    updated, offset = lw._detect_offset_with_issues(
        "vocals.wav",
        line_timings,
        lyrics_offset=None,
        issues=issues,
        auto_offset_scale=0.6,
    )

    assert offset == pytest.approx(0.6)
    assert updated[0][0] == pytest.approx(1.6)


def test_detect_offset_with_issues_skips_scaling_outside_window(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda _: 2.0,
    )

    issues = []
    line_timings = [(1.0, "Line")]
    updated, offset = lw._detect_offset_with_issues(
        "vocals.wav",
        line_timings,
        lyrics_offset=None,
        issues=issues,
        auto_offset_scale=0.6,
        scaled_offset_min_abs_sec=1.2,
        scaled_offset_max_abs_sec=1.4,
    )

    assert offset == pytest.approx(1.0)
    assert updated[0][0] == pytest.approx(2.0)


def test_detect_offset_with_issues_scales_large_negative_offset_when_enabled(
    monkeypatch,
):
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda _: -0.62,
    )

    issues = []
    line_timings = [(1.0, "Line")]
    updated, offset = lw._detect_offset_with_issues(
        "vocals.wav",
        line_timings,
        lyrics_offset=None,
        issues=issues,
        auto_offset_scale=0.6,
        scaled_offset_min_abs_sec=0.9,
        scaled_offset_max_abs_sec=1.4,
        scale_large_negative_offsets=True,
    )

    assert offset == pytest.approx(-0.972)
    assert updated[0][0] == pytest.approx(0.028)


def test_detect_offset_with_issues_does_not_scale_large_negative_offset_by_default(
    monkeypatch,
):
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda _: -0.62,
    )

    issues = []
    line_timings = [(1.0, "Line")]
    updated, offset = lw._detect_offset_with_issues(
        "vocals.wav",
        line_timings,
        lyrics_offset=None,
        issues=issues,
        auto_offset_scale=0.6,
        scaled_offset_min_abs_sec=0.9,
        scaled_offset_max_abs_sec=1.4,
    )

    assert offset == pytest.approx(-1.62)
    assert updated[0][0] == pytest.approx(-0.62)


def test_detect_offset_with_issues_can_skip_moderate_negative_offset(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda _: 5.43,
    )

    issues = []
    line_timings = [(6.22, "Line")]
    updated, offset = lw._detect_offset_with_issues(
        "vocals.wav",
        line_timings,
        lyrics_offset=None,
        issues=issues,
        suppress_moderate_negative_offset=True,
    )

    assert offset == pytest.approx(0.0)
    assert updated == line_timings


def test_detect_offset_with_issues_skips_negative_offset_up_to_guard_threshold(
    monkeypatch,
):
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda _: 0.67,
    )

    issues = []
    line_timings = [(1.67, "Line")]
    updated, offset = lw._detect_offset_with_issues(
        "vocals.wav",
        line_timings,
        lyrics_offset=None,
        issues=issues,
        suppress_moderate_negative_offset=True,
    )

    assert offset == pytest.approx(0.0)
    assert updated == line_timings


def test_detect_offset_with_issues_uses_second_line_after_long_interjection_gap(
    monkeypatch,
):
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda _: 27.98,
    )

    issues = []
    line_timings = [
        (13.42, "Yeah"),
        (26.95, "I've been tryna call"),
    ]
    updated, offset = lw._detect_offset_with_issues(
        "vocals.wav",
        line_timings,
        lyrics_offset=None,
        issues=issues,
    )

    assert offset == pytest.approx(0.618, abs=0.01)
    assert issues == []
    assert updated[0][0] == pytest.approx(14.04, abs=0.01)
    assert updated[1][0] == pytest.approx(27.57, abs=0.01)


def test_detect_offset_with_issues_keeps_suspicious_positive_offset_blocked_by_default(
    monkeypatch,
):
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda _: 8.45,
    )

    issues = []
    line_timings = [(4.10, "I can breathe for the first time")]
    updated, offset = lw._detect_offset_with_issues(
        "vocals.wav",
        line_timings,
        lyrics_offset=None,
        issues=issues,
    )

    assert offset == 0.0
    assert updated == line_timings


def test_detect_offset_with_issues_can_allow_suspicious_positive_offset(
    monkeypatch,
):
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda _: 8.45,
    )

    issues = []
    line_timings = [(4.10, "I can breathe for the first time")]
    updated, offset = lw._detect_offset_with_issues(
        "vocals.wav",
        line_timings,
        lyrics_offset=None,
        issues=issues,
        allow_suspicious_positive_offset=True,
    )

    assert offset == pytest.approx(4.35)
    assert updated[0][0] == pytest.approx(8.45)


def test_map_lrc_lines_uses_whisper_pause_for_word_slots():
    line = Line(
        words=[
            Word(text="Yesterday", start_time=0.0, end_time=0.5),
            Word(text="all", start_time=0.5, end_time=1.0),
            Word(text="my", start_time=1.0, end_time=1.5),
            Word(text="troubles", start_time=1.5, end_time=2.0),
        ]
    )
    segment_words = [
        TranscriptionWord(start=5.0, end=5.4, text="Yesterday"),
        TranscriptionWord(start=7.5, end=7.7, text="all"),
        TranscriptionWord(start=7.8, end=8.0, text="my"),
        TranscriptionWord(start=8.1, end=8.5, text="troubles"),
    ]
    segment = TranscriptionSegment(
        start=5.0,
        end=8.6,
        text="Yesterday all my troubles",
        words=segment_words,
    )

    adjusted, fixes, _issues = lw._map_lrc_lines_to_whisper_segments(
        [line],
        [segment],
        language="eng-Latn",
        lrc_line_starts=[5.0, 9.0],
    )

    assert fixes == 1
    adjusted_line = adjusted[0]
    assert adjusted_line.words[0].start_time == pytest.approx(5.0, abs=0.05)
    gap_1 = adjusted_line.words[1].start_time - adjusted_line.words[0].start_time
    gap_2 = adjusted_line.words[2].start_time - adjusted_line.words[1].start_time
    gap_3 = adjusted_line.words[3].start_time - adjusted_line.words[2].start_time
    assert gap_1 > gap_2 * 1.5
    assert gap_1 > gap_3 * 1.5
    assert adjusted_line.end_time <= 9.0


def test_map_lrc_lines_falls_back_to_window_words_on_large_offset():
    line = Line(
        words=[
            Word(text="Yesterday", start_time=0.0, end_time=0.5),
            Word(text="all", start_time=0.5, end_time=1.0),
        ]
    )
    # Segment with matching text but far away in time.
    seg_early = TranscriptionSegment(
        start=0.0,
        end=1.0,
        text="Yesterday all",
        words=[
            TranscriptionWord(start=0.1, end=0.2, text="Yesterday"),
            TranscriptionWord(start=0.3, end=0.4, text="all"),
        ],
    )
    # Later segment provides window words aligned to LRC timing.
    seg_window = TranscriptionSegment(
        start=10.0,
        end=11.5,
        text="Yesterday all",
        words=[
            TranscriptionWord(start=10.0, end=10.3, text="Yesterday"),
            TranscriptionWord(start=11.0, end=11.2, text="all"),
        ],
    )

    adjusted, fixes, issues = lw._map_lrc_lines_to_whisper_segments(
        [line],
        [seg_early, seg_window],
        language="eng-Latn",
        lrc_line_starts=[10.0, 12.0],
    )

    assert fixes == 1
    assert any("window-only mapping" in issue for issue in issues)
    adjusted_line = adjusted[0]
    assert adjusted_line.words[0].start_time == pytest.approx(10.0, abs=0.05)
    gap = adjusted_line.words[1].start_time - adjusted_line.words[0].start_time
    assert gap > 0.5
