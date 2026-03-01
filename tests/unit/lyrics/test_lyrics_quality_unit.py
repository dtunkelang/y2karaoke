import pytest

from y2karaoke.core.components.lyrics import api as lyrics
from y2karaoke.core import lyrics_whisper as lw
from y2karaoke.core.components.lyrics import helpers as lh
from y2karaoke.core.components.lyrics.lyrics_whisper_pipeline import (
    should_auto_enable_whisper,
)
from y2karaoke.core.models import Line, Word


def _make_line(text, start=0.0, end=1.0):
    words = [Word(text=w, start_time=start, end_time=end) for w in text.split()]
    return Line(words=words)


def test_should_auto_enable_whisper_requires_untimed_vocals_and_non_whisper_mode():
    assert (
        should_auto_enable_whisper(
            vocals_path="vocals.wav",
            line_timings=None,
            use_whisper=False,
            whisper_only=False,
            whisper_map_lrc=False,
        )
        is True
    )
    assert (
        should_auto_enable_whisper(
            vocals_path="vocals.wav",
            line_timings=[(1.0, "line")],
            use_whisper=False,
            whisper_only=False,
            whisper_map_lrc=False,
        )
        is False
    )


def test_refine_timing_with_audio_adjusts_on_duration_mismatch(monkeypatch):
    lines = [_make_line("hello", 0.0, 0.5)]
    line_timings = [(0.0, "hello"), (2.0, "world")]

    monkeypatch.setattr(
        "y2karaoke.core.refine.refine_word_timing",
        lambda lines_in, _vocals: lines_in,
    )
    monkeypatch.setattr(
        "y2karaoke.core.components.lyrics.sync.get_lrc_duration", lambda *_: 120
    )

    called = {}

    def fake_adjust(lines_in, timings, vocals_path, lrc_duration, audio_duration):
        called["args"] = (timings, lrc_duration, audio_duration)
        return lines_in

    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.adjust_timing_for_duration_mismatch",
        fake_adjust,
    )

    result = lh._refine_timing_with_audio(
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

    monkeypatch.setattr(
        "y2karaoke.core.refine.refine_word_timing",
        lambda lines_in, _vocals: lines_in,
    )
    monkeypatch.setattr(
        "y2karaoke.core.components.lyrics.sync.get_lrc_duration", lambda *_: 100
    )

    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.adjust_timing_for_duration_mismatch",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not adjust")),
    )

    result = lh._refine_timing_with_audio(
        lines,
        vocals_path="vocals.wav",
        line_timings=line_timings,
        lrc_text="[00:01.00]hello",
        target_duration=105,
    )
    assert result is lines


def test_apply_whisper_alignment_records_fixes(monkeypatch):
    lines = [_make_line("hello", 0.0, 0.5)]
    # Patch where it's imported in lyrics_helpers
    monkeypatch.setattr(
        "y2karaoke.core.components.whisper.whisper_integration.correct_timing_with_whisper",
        lambda *args, **kwargs: (
            args[0],
            ["fix1", "fix2"],
            {},
        ),
    )
    monkeypatch.setattr(
        "y2karaoke.core.audio_analysis.extract_audio_features",
        lambda *_: None,
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

    monkeypatch.setattr(
        "y2karaoke.core.refine.refine_word_timing",
        lambda lines_in, _vocals: lines_in,
    )
    monkeypatch.setattr(
        "y2karaoke.core.components.lyrics.sync.get_lrc_duration", lambda *_: 120
    )
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.adjust_timing_for_duration_mismatch",
        lambda *a, **k: lines,
    )

    issues = []
    refined, method = lw._refine_timing_with_quality(
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

    report = {"whisper_used": False, "whisper_corrections": 0, "issues": []}
    with lw.use_lyrics_whisper_hooks(apply_whisper_alignment_fn=raise_error):
        updated, report = lw._apply_whisper_with_quality(
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


def test_apply_whisper_with_quality_marks_no_evidence_fallback_as_lrc_only():
    lines = [_make_line("hello", 0.0, 0.5)]
    report = {"alignment_method": "onset_refined", "issues": []}

    with lw.use_lyrics_whisper_hooks(
        apply_whisper_alignment_fn=lambda *args, **kwargs: (
            lines,
            ["rolled back"],
            {"no_evidence_fallback": 1.0},
        )
    ):
        updated, report = lw._apply_whisper_with_quality(
            lines,
            vocals_path="vocals.wav",
            whisper_language="en",
            whisper_model="base",
            whisper_force_dtw=False,
            quality_report=report,
        )

    assert updated is lines
    assert report["alignment_method"] == "lrc_only"
    assert report["whisper_used"] is False
    assert report["whisper_corrections"] == 0
    assert report["dtw_metrics"]["no_evidence_fallback"] == 1.0


def test_fetch_genius_with_quality_tracking_no_lrc(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.components.lyrics.genius.fetch_genius_lyrics_with_singers",
        lambda *a, **k: (None, None),
    )
    report = {"alignment_method": "none", "issues": [], "overall_score": 100.0}

    lines, metadata = lw._fetch_genius_with_quality_tracking(
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
    from y2karaoke.core.components.whisper import whisper_integration as wi

    word = te.TranscriptionWord(start=1.0, end=1.2, text="hello", probability=0.9)
    segment = te.TranscriptionSegment(start=1.0, end=1.4, text="hello", words=[word])

    # Patch where it's imported in lyrics_whisper
    monkeypatch.setattr(
        wi, "transcribe_vocals", lambda *_a, **_k: ([segment], [word], "en", "base")
    )

    lines, metadata = lw.get_lyrics_simple(
        title="Song",
        artist="Artist",
        vocals_path="vocals.wav",
        whisper_only=True,
    )

    assert metadata is not None
    assert lines
    assert lines[0].words[0].text == "hello"
    assert lines[0].words[0].start_time == 1.0


def test_get_lyrics_simple_whisper_only_retries_large_on_low_confidence():
    from y2karaoke.core import timing_evaluator as te

    low_word = te.TranscriptionWord(start=1.0, end=1.2, text="hello", probability=0.2)
    low_seg = te.TranscriptionSegment(
        start=1.0, end=1.4, text="hello", words=[low_word]
    )
    hi_word = te.TranscriptionWord(start=1.0, end=1.3, text="better", probability=0.8)
    hi_seg = te.TranscriptionSegment(start=1.0, end=1.5, text="better", words=[hi_word])

    calls = []

    def fake_transcribe(*args, **kwargs):
        model_size = args[2]
        calls.append(model_size)
        if model_size == "base":
            return [low_seg], [low_word], "en", "base"
        return [hi_seg], [hi_word], "en", "large"

    with lw.use_lyrics_whisper_hooks(transcribe_vocals_fn=fake_transcribe):
        lines, metadata = lw.get_lyrics_simple(
            title="Song",
            artist="Artist",
            vocals_path="vocals.wav",
            whisper_only=True,
        )

    assert metadata is not None
    assert calls == ["base", "large"]
    assert lines
    assert lines[0].words[0].text == "better"


def test_get_lyrics_simple_whisper_map_lrc(monkeypatch):
    from y2karaoke.core import timing_evaluator as te
    from y2karaoke.core import genius

    lrc_lines = [_make_line("bonjour", 0.0, 0.5)]

    def fake_fetch(*_a, **_k):
        return "[00:00.00]bonjour", [(0.0, "bonjour")], "source"

    word = te.TranscriptionWord(start=2.0, end=2.4, text="bonjour", probability=0.9)
    segment = te.TranscriptionSegment(start=2.0, end=2.6, text="bonjour", words=[word])

    monkeypatch.setattr(lw, "create_lines_from_lrc", lambda *a, **k: lrc_lines)
    monkeypatch.setattr(
        genius, "fetch_genius_lyrics_with_singers", lambda *_a, **_k: (None, None)
    )
    with lw.use_lyrics_whisper_hooks(
        fetch_lrc_text_and_timings_fn=fake_fetch,
        detect_and_apply_offset_fn=lambda v, t, o: (t, 0.0),
        refine_timing_with_audio_fn=lambda lines, *_a, **_k: lines,
        transcribe_vocals_fn=lambda *_a, **_k: ([segment], [word], "fr", "base"),
        whisper_lang_to_epitran_fn=lambda *_a, **_k: "fra-Latn",
    ):
        lines, _ = lw.get_lyrics_simple(
            title="Song",
            artist="Artist",
            vocals_path="vocals.wav",
            whisper_map_lrc=True,
        )

    assert lines[0].start_time == pytest.approx(2.0)
