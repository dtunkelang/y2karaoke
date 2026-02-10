import pytest

from y2karaoke.core import lyrics
from y2karaoke.core import lyrics_helpers as lh
from y2karaoke.core import lyrics_whisper as lw
from y2karaoke.core.models import Line, Word, SongMetadata


def _line(text, start=0.0, end=1.0):
    return Line(words=[Word(text=text, start_time=start, end_time=end)])


def test_calculate_quality_score_applies_method_and_issues():
    report = {
        "lyrics_quality": {"quality_score": 80.0},
        "alignment_method": "whisper_hybrid",
        "issues": ["a", "b"],
    }

    score = lw._calculate_quality_score(report)

    assert score == pytest.approx(80.0)


def test_fetch_lrc_text_and_timings_uses_best_source(monkeypatch):
    class Report:
        overall_score = 91.2

    monkeypatch.setattr(
        "y2karaoke.core.timing_evaluator.select_best_source",
        lambda *a, **k: ("[00:01.00]hi", "best", Report()),
    )
    monkeypatch.setattr(lh, "parse_lrc_with_timing", lambda *a, **k: [(1.0, "hi")])

    lrc_text, timings, source = lw._fetch_lrc_text_and_timings(
        "Title",
        "Artist",
        vocals_path="vocals.wav",
        evaluate_sources=True,
    )

    assert lrc_text == "[00:01.00]hi"
    assert timings == [(1.0, "hi")]
    assert source == "best"


def test_fetch_lrc_text_and_timings_returns_none_when_no_duration_match(
    monkeypatch,
):
    monkeypatch.setattr(
        "y2karaoke.core.sync.fetch_lyrics_for_duration",
        lambda *a, **k: (None, False, "", None),
    )

    lrc_text, timings, source = lw._fetch_lrc_text_and_timings(
        "Title",
        "Artist",
        target_duration=120,
    )

    assert lrc_text is None
    assert timings is None
    assert source == ""


def test_fetch_lrc_text_and_timings_eval_sources_falls_back(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.timing_evaluator.select_best_source",
        lambda *a, **k: (None, None, None),
    )
    monkeypatch.setattr(
        "y2karaoke.core.sync.fetch_lyrics_for_duration",
        lambda *a, **k: ("[00:01.00]hi", True, "provider", 120),
    )
    monkeypatch.setattr(lh, "parse_lrc_with_timing", lambda *a, **k: [(1.0, "hi")])

    lrc_text, timings, source = lw._fetch_lrc_text_and_timings(
        "Title",
        "Artist",
        target_duration=120,
        vocals_path="vocals.wav",
        evaluate_sources=True,
    )

    assert lrc_text == "[00:01.00]hi"
    assert timings == [(1.0, "hi")]
    assert source == "provider"


def test_fetch_lrc_text_and_timings_eval_sources_needs_vocals(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.sync.fetch_lyrics_for_duration",
        lambda *a, **k: ("[00:02.00]hey", True, "provider", 200),
    )
    monkeypatch.setattr(lh, "parse_lrc_with_timing", lambda *a, **k: [(2.0, "hey")])

    lrc_text, timings, source = lw._fetch_lrc_text_and_timings(
        "Title",
        "Artist",
        target_duration=200,
        evaluate_sources=True,
        vocals_path=None,
    )

    assert lrc_text == "[00:02.00]hey"
    assert timings == [(2.0, "hey")]
    assert source == "provider"


def test_fetch_lrc_text_and_timings_filters_promos(monkeypatch):
    lrc_text = "\n".join(
        [
            "[00:02.00]Download the song at https://example.com",
            "[00:06.00]Aucun Express, je reviens encore",
            "[00:12.00]Et puis je pars demain",
        ]
    )

    monkeypatch.setattr(
        "y2karaoke.core.sync.fetch_lyrics_for_duration",
        lambda *a, **k: (lrc_text, True, "provider", 180),
    )

    lrc_text_out, timings, _source = lw._fetch_lrc_text_and_timings(
        "Aucun Express",
        "Alain Bashung",
        target_duration=180,
        filter_promos=True,
    )

    assert lrc_text_out == lrc_text
    assert [text for _ts, text in timings] == [
        "Aucun Express, je reviens encore",
        "Et puis je pars demain",
    ]

    _, timings_unfiltered, _ = lw._fetch_lrc_text_and_timings(
        "Aucun Express",
        "Alain Bashung",
        target_duration=180,
        filter_promos=False,
    )

    assert len(timings_unfiltered) == 3


def test_get_lyrics_simple_falls_back_to_genius(monkeypatch):
    monkeypatch.setattr(
        lw, "_fetch_lrc_text_and_timings", lambda *a, **k: (None, None, "")
    )

    metadata = SongMetadata(singers=["Singer"], is_duet=False)
    monkeypatch.setattr(
        "y2karaoke.core.genius.fetch_genius_lyrics_with_singers",
        lambda *a, **k: ([("hello world", "Singer")], metadata),
    )
    monkeypatch.setattr(
        lw,
        "create_lines_from_lrc",
        lambda *a, **k: [_line("hello", 0.0, 1.0)],
    )

    lines, meta = lw.get_lyrics_simple("Title", "Artist", vocals_path=None)

    assert meta == metadata
    assert lines[0].words[0].text == "hello"


def test_detect_offset_with_issues_skips_huge_delta(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda _: 100.0,
    )

    issues = []
    line_timings = [(1.0, "Line")]
    updated, offset = lw._detect_offset_with_issues(
        "vocals.wav", line_timings, lyrics_offset=None, issues=issues
    )

    assert offset == 0.0
    assert updated == line_timings
    assert any("not applied" in issue for issue in issues)
