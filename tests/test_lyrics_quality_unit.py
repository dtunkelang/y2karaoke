import pytest

from y2karaoke.core import lyrics
from y2karaoke.core.models import Line, Word


def _make_line(text, start=0.0, end=1.0):
    words = [Word(text=w, start_time=start, end_time=end) for w in text.split()]
    return Line(words=words)


def test_refine_timing_with_audio_adjusts_on_duration_mismatch(monkeypatch):
    lines = [_make_line("hello", 0.0, 0.5)]
    line_timings = [(0.0, "hello"), (2.0, "world")]

    monkeypatch.setattr("y2karaoke.core.refine.refine_word_timing", lambda l, v: l)
    monkeypatch.setattr("y2karaoke.core.sync.get_lrc_duration", lambda *_: 120)

    called = {}

    def fake_adjust(lines_in, timings, vocals_path, lrc_duration, audio_duration):
        called["args"] = (timings, lrc_duration, audio_duration)
        return lines_in

    monkeypatch.setattr(
        "y2karaoke.core.alignment.adjust_timing_for_duration_mismatch", fake_adjust
    )

    result = lyrics._refine_timing_with_audio(
        lines,
        vocals_path="vocals.wav",
        line_timings=line_timings,
        lrc_text="[00:01.00]hello",
        target_duration=200,
    )

    assert result is lines
    assert called["args"][1] == 120
    assert called["args"][2] == 200


def test_refine_timing_with_audio_no_adjust_when_close(monkeypatch):
    lines = [_make_line("hello", 0.0, 0.5)]
    line_timings = [(0.0, "hello"), (2.0, "world")]

    monkeypatch.setattr("y2karaoke.core.refine.refine_word_timing", lambda l, v: l)
    monkeypatch.setattr("y2karaoke.core.sync.get_lrc_duration", lambda *_: 100)

    monkeypatch.setattr(
        "y2karaoke.core.alignment.adjust_timing_for_duration_mismatch",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not adjust")),
    )

    result = lyrics._refine_timing_with_audio(
        lines,
        vocals_path="vocals.wav",
        line_timings=line_timings,
        lrc_text="[00:01.00]hello",
        target_duration=105,
    )
    assert result is lines


def test_apply_whisper_alignment_records_fixes(monkeypatch):
    lines = [_make_line("hello", 0.0, 0.5)]
    monkeypatch.setattr(
        "y2karaoke.core.timing_evaluator.correct_timing_with_whisper",
        lambda l, v, language=None, model_size="base", force_dtw=False: (
            l,
            ["fix1", "fix2"],
            {},
        ),
    )
    aligned, fixes, metrics = lyrics._apply_whisper_alignment(
        lines,
        "vocals.wav",
        whisper_language="en",
        whisper_model="base",
        whisper_force_dtw=False,
    )
    assert aligned is lines
    assert fixes == ["fix1", "fix2"]
    assert metrics == {}


def test_refine_timing_with_quality_sets_method(monkeypatch):
    lines = [_make_line("hello", 0.0, 0.5)]
    line_timings = [(0.0, "hello"), (2.0, "world")]

    monkeypatch.setattr("y2karaoke.core.refine.refine_word_timing", lambda l, v: l)
    monkeypatch.setattr("y2karaoke.core.sync.get_lrc_duration", lambda *_: 120)
    monkeypatch.setattr(
        "y2karaoke.core.alignment.adjust_timing_for_duration_mismatch",
        lambda *a, **k: lines,
    )

    issues = []
    refined, method = lyrics._refine_timing_with_quality(
        lines,
        vocals_path="vocals.wav",
        line_timings=line_timings,
        lrc_text="[00:01.00]hello",
        target_duration=200,
        issues=issues,
    )
    assert refined is lines
    assert method == "onset_refined"
    assert any("Duration mismatch" in issue for issue in issues)


def test_apply_whisper_with_quality_handles_error(monkeypatch):
    lines = [_make_line("hello", 0.0, 0.5)]

    def raise_error(*args, **kwargs):
        raise RuntimeError("whisper down")

    monkeypatch.setattr(lyrics, "_apply_whisper_alignment", raise_error)
    report = {"whisper_used": False, "whisper_corrections": 0, "issues": []}

    updated, report = lyrics._apply_whisper_with_quality(
        lines,
        vocals_path="vocals.wav",
        whisper_language="en",
        whisper_model="base",
        whisper_force_dtw=False,
        quality_report=report,
    )

    assert updated is lines
    assert report["whisper_used"] is False
    assert any("Whisper alignment failed" in issue for issue in report["issues"])


def test_fetch_genius_with_quality_tracking_no_lrc(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.genius.fetch_genius_lyrics_with_singers",
        lambda *a, **k: (None, None),
    )
    report = {"alignment_method": "none", "issues": [], "overall_score": 100.0}

    lines, metadata = lyrics._fetch_genius_with_quality_tracking(
        line_timings=None,
        title="Song",
        artist="Artist",
        quality_report=report,
    )

    assert lines is None
    assert metadata is None
    assert report["alignment_method"] == "genius_fallback"
    assert report["overall_score"] == 0.0
    assert any("No lyrics found" in issue for issue in report["issues"])


def test_get_lyrics_simple_whisper_only(monkeypatch):
    from y2karaoke.core import timing_evaluator as te

    word = te.TranscriptionWord(start=1.0, end=1.2, text="hello", probability=0.9)
    segment = te.TranscriptionSegment(start=1.0, end=1.4, text="hello", words=[word])

    monkeypatch.setattr(
        te, "transcribe_vocals", lambda *_a, **_k: ([segment], [word], "en")
    )

    lines, metadata = lyrics.get_lyrics_simple(
        title="Song",
        artist="Artist",
        vocals_path="vocals.wav",
        whisper_only=True,
    )

    assert metadata is not None
    assert lines
    assert lines[0].words[0].text == "hello"
    assert lines[0].words[0].start_time == 1.0
