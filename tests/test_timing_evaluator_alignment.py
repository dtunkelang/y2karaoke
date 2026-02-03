import builtins

import y2karaoke.core.timing_evaluator as te
from y2karaoke.core.models import Line, Word


def test_align_lyrics_to_transcription_adjusts_times():
    lines = [
        Line(
            words=[
                Word(text="hello", start_time=0.0, end_time=0.5),
                Word(text="world", start_time=0.5, end_time=1.0),
            ]
        )
    ]
    transcription = [
        te.TranscriptionSegment(start=5.0, end=7.0, text="hello world", words=[])
    ]

    aligned, notes = te.align_lyrics_to_transcription(
        lines,
        transcription,
        min_similarity=0.1,
        max_time_shift=10.0,
        language="eng-Latn",
    )
    assert aligned[0].words[0].start_time == 5.0
    assert aligned[0].words[1].start_time == 6.0
    assert notes


def test_align_lyrics_to_transcription_skips_when_out_of_range():
    lines = [Line(words=[Word(text="hello", start_time=0.0, end_time=0.5)])]
    transcription = [
        te.TranscriptionSegment(start=20.0, end=21.0, text="hello", words=[])
    ]
    aligned, notes = te.align_lyrics_to_transcription(
        lines,
        transcription,
        min_similarity=0.1,
        max_time_shift=5.0,
        language="eng-Latn",
    )
    assert aligned[0].words[0].start_time == 0.0
    assert notes == []


def test_transcribe_vocals_returns_empty_without_dependency(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    segments, words, language = te.transcribe_vocals("vocals.wav")
    assert segments == []
    assert words == []
    assert language == ""


def test_text_similarity_phonetic_falls_back(monkeypatch):
    monkeypatch.setattr(te, "_get_panphon_distance", lambda: None)
    assert te._text_similarity("hello", "hello", use_phonetic=True) == 1.0


def test_get_ipa_returns_none_without_epitran(monkeypatch):
    monkeypatch.setattr(te, "_get_epitran", lambda _lang="fra-Latn": None)
    assert te._get_ipa("bonjour", language="fra-Latn") is None
