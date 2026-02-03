import pytest

from y2karaoke.core import timing_evaluator as te
from y2karaoke.core.models import Line, Word


def _make_line(text, start, end):
    words = [Word(text=w, start_time=start, end_time=end) for w in text.split()]
    return Line(words=words)


def test_find_best_whisper_segment_selects_best(monkeypatch):
    segments = [
        te.TranscriptionSegment(start=5.0, end=6.0, text="hello", words=[]),
        te.TranscriptionSegment(start=10.0, end=11.0, text="world", words=[]),
    ]

    def fake_sim(text1, text2, language):
        return 0.9 if "world" in text2 else 0.4

    monkeypatch.setattr(te, "_phonetic_similarity", fake_sim)

    seg, sim, offset = te._find_best_whisper_segment(
        line_text="world",
        line_start=9.5,
        sorted_segments=segments,
        language="eng-Latn",
        min_similarity=0.5,
    )

    assert seg == segments[1]
    assert sim == 0.9
    assert offset == pytest.approx(0.5)


def test_compute_phonetic_costs_marks_matches(monkeypatch):
    lrc_words = [
        {"text": "hello", "start": 1.0},
        {"text": "world", "start": 10.0},
    ]
    whisper_words = [
        te.TranscriptionWord(start=1.2, end=1.4, text="hello"),
        te.TranscriptionWord(start=30.0, end=30.2, text="noise"),
    ]

    monkeypatch.setattr(te, "_phonetic_similarity", lambda a, b, lang: 0.8)

    costs = te._compute_phonetic_costs(
        lrc_words, whisper_words, language="eng-Latn", min_similarity=0.5
    )

    assert costs[(0, 0)] < 1.0
    assert costs[(1, 0)] < 1.0


def test_extract_alignments_from_path(monkeypatch):
    lrc_words = [{"text": "hello"}, {"text": "world"}]
    whisper_words = [
        te.TranscriptionWord(start=1.0, end=1.1, text="hello"),
        te.TranscriptionWord(start=2.0, end=2.1, text="world"),
    ]

    monkeypatch.setattr(te, "_phonetic_similarity", lambda *a, **k: 0.6)

    alignments = te._extract_alignments_from_path(
        path=[(0, 0), (1, 1)],
        lrc_words=lrc_words,
        whisper_words=whisper_words,
        language="eng-Latn",
        min_similarity=0.5,
    )

    assert alignments[0][0] == whisper_words[0]
    assert alignments[1][0] == whisper_words[1]


def test_apply_dtw_alignments_shifts():
    lines = [_make_line("hello", 1.0, 1.5)]
    lrc_words = [
        {
            "line_idx": 0,
            "word_idx": 0,
            "text": "hello",
            "start": 1.0,
            "end": 1.5,
        }
    ]
    whisper_word = te.TranscriptionWord(start=5.0, end=5.2, text="hello")
    alignments = {0: (whisper_word, 0.9)}

    aligned, corrections = te._apply_dtw_alignments(lines, lrc_words, alignments)

    assert aligned[0].words[0].start_time == pytest.approx(5.0)
    assert corrections


def test_correct_timing_with_whisper_good_quality(monkeypatch):
    lines = [_make_line("hello", 1.0, 1.5)]
    segments = [te.TranscriptionSegment(start=1.2, end=1.6, text="hello", words=[])]
    words = [te.TranscriptionWord(start=1.2, end=1.4, text="hello")]

    monkeypatch.setattr(te, "transcribe_vocals", lambda *a, **k: (segments, words, "en"))
    monkeypatch.setattr(te, "_get_ipa", lambda *a, **k: None)
    monkeypatch.setattr(te, "_assess_lrc_quality", lambda *a, **k: (0.8, []))

    called = {"hybrid": 0}

    def fake_hybrid(*args, **kwargs):
        called["hybrid"] += 1
        return args[0], ["aligned"]

    monkeypatch.setattr(te, "align_hybrid_lrc_whisper", fake_hybrid)
    monkeypatch.setattr(te, "_fix_ordering_violations", lambda o, a, c: (a, c))

    aligned, corrections = te.correct_timing_with_whisper(lines, "vocals.wav")

    assert aligned == lines
    assert corrections == ["aligned"]
    assert called["hybrid"] == 1


def test_correct_timing_with_whisper_poor_quality(monkeypatch):
    lines = [_make_line("hello", 1.0, 1.5)]
    segments = [te.TranscriptionSegment(start=10.0, end=11.0, text="hello", words=[])]
    words = [te.TranscriptionWord(start=10.0, end=10.2, text="hello")]

    monkeypatch.setattr(te, "transcribe_vocals", lambda *a, **k: (segments, words, "en"))
    monkeypatch.setattr(te, "_get_ipa", lambda *a, **k: None)
    monkeypatch.setattr(te, "_assess_lrc_quality", lambda *a, **k: (0.2, []))

    monkeypatch.setattr(te, "align_dtw_whisper", lambda *a, **k: (lines, ["dtw"]))
    monkeypatch.setattr(te, "_fix_ordering_violations", lambda o, a, c: (a, c))

    aligned, corrections = te.correct_timing_with_whisper(lines, "vocals.wav")

    assert aligned == lines
    assert corrections == ["dtw"]


def test_correct_timing_with_whisper_no_transcription(monkeypatch):
    lines = [_make_line("hello", 1.0, 1.5)]
    monkeypatch.setattr(te, "transcribe_vocals", lambda *a, **k: ([], [], ""))

    aligned, corrections = te.correct_timing_with_whisper(lines, "vocals.wav")

    assert aligned == lines
    assert corrections == []
