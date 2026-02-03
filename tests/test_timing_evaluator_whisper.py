import builtins

import pytest

import y2karaoke.core.timing_evaluator as te
from y2karaoke.core.models import Line, Word


def test_align_words_to_whisper_adjusts_word():
    te._ipa_cache.clear()
    lines = [
        Line(words=[Word(text="hello", start_time=0.0, end_time=0.5)])
    ]
    whisper_words = [
        te.TranscriptionWord(start=2.0, end=2.4, text="hello", probability=0.9)
    ]
    te._get_ipa = lambda *_args, **_kwargs: None
    aligned, corrections = te.align_words_to_whisper(
        lines,
        whisper_words,
        min_similarity=0.1,
        max_time_shift=5.0,
        language="eng-Latn",
    )
    assert aligned[0].words[0].start_time == 2.0
    assert corrections


def test_assess_lrc_quality_reports_good_match():
    lines = [
        Line(words=[Word(text="hello", start_time=1.0, end_time=1.5)])
    ]
    whisper_words = [
        te.TranscriptionWord(start=1.2, end=1.6, text="hello", probability=0.9)
    ]
    quality, assessments = te._assess_lrc_quality(
        lines, whisper_words, language="eng-Latn", tolerance=1.5
    )
    assert quality == 1.0
    assert assessments


def test_extract_lrc_words_returns_indices():
    lines = [
        Line(words=[Word(text="hi", start_time=0.0, end_time=0.2)]),
        Line(words=[Word(text="yo", start_time=1.0, end_time=1.2)]),
    ]
    words = te._extract_lrc_words(lines)
    assert words[0]["line_idx"] == 0
    assert words[1]["line_idx"] == 1


def test_compute_phonetic_costs_includes_matches(monkeypatch):
    lrc_words = [
        {"text": "hello", "start": 1.0},
        {"text": "world", "start": 5.0},
    ]
    whisper_words = [
        te.TranscriptionWord(start=1.2, end=1.6, text="hello", probability=0.9)
    ]

    monkeypatch.setattr(te, "_phonetic_similarity", lambda *_args, **_kwargs: 0.8)
    costs = te._compute_phonetic_costs(lrc_words, whisper_words, "eng-Latn", 0.5)
    assert costs[(0, 0)] == pytest.approx(0.2)


def test_extract_alignments_from_path_filters_by_similarity(monkeypatch):
    lrc_words = [
        {"text": "hello", "start": 1.0},
    ]
    whisper_words = [
        te.TranscriptionWord(start=1.2, end=1.6, text="hello", probability=0.9)
    ]
    monkeypatch.setattr(te, "_phonetic_similarity", lambda *_args, **_kwargs: 0.6)
    alignments = te._extract_alignments_from_path(
        [(0, 0)], lrc_words, whisper_words, "eng-Latn", 0.5
    )
    assert 0 in alignments


def test_apply_dtw_alignments_shifts_large_offsets():
    lines = [
        Line(words=[Word(text="hello", start_time=0.0, end_time=0.5)])
    ]
    lrc_words = [
        {"line_idx": 0, "word_idx": 0, "text": "hello", "start": 0.0, "end": 0.5}
    ]
    alignments = {
        0: (te.TranscriptionWord(start=2.0, end=2.4, text="hello", probability=0.9), 0.9)
    }
    aligned, corrections = te._apply_dtw_alignments(lines, lrc_words, alignments)
    assert aligned[0].words[0].start_time == 2.0
    assert corrections


def test_align_dtw_whisper_falls_back_without_fastdtw(monkeypatch):
    lines = [
        Line(words=[Word(text="hello", start_time=0.0, end_time=0.5)])
    ]
    whisper_words = [
        te.TranscriptionWord(start=1.0, end=1.4, text="hello", probability=0.9)
    ]

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "fastdtw":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    aligned, corrections = te.align_dtw_whisper(lines, whisper_words)
    assert aligned == lines
    assert corrections == []


def test_correct_timing_with_whisper_no_transcription(monkeypatch):
    monkeypatch.setattr(te, "transcribe_vocals", lambda *_args, **_kwargs: ([], [], ""))
    lines = [
        Line(words=[Word(text="hello", start_time=0.0, end_time=0.5)])
    ]
    aligned, corrections = te.correct_timing_with_whisper(
        lines, "vocals.wav", language="en"
    )
    assert aligned == lines
    assert corrections == []


def test_correct_timing_with_whisper_uses_dtw(monkeypatch):
    lines = [
        Line(words=[Word(text="hello", start_time=0.0, end_time=0.5)])
    ]
    transcription = [te.TranscriptionSegment(start=0.0, end=0.5, text="hello", words=[])]
    all_words = [te.TranscriptionWord(start=0.0, end=0.5, text="hello", probability=0.9)]

    monkeypatch.setattr(
        te, "transcribe_vocals", lambda *_args, **_kwargs: (transcription, all_words, "en")
    )
    monkeypatch.setattr(te, "_get_ipa", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(te, "_assess_lrc_quality", lambda *_args, **_kwargs: (0.2, []))
    monkeypatch.setattr(
        te,
        "align_dtw_whisper",
        lambda *_args, **_kwargs: (lines, ["dtw"]),
    )
    monkeypatch.setattr(te, "_fix_ordering_violations", lambda o, n, a: (n, a))

    aligned, corrections = te.correct_timing_with_whisper(
        lines, "vocals.wav", language="en"
    )
    assert aligned == lines
    assert corrections == ["dtw"]
