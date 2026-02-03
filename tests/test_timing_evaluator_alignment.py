import pytest

from y2karaoke.core import timing_evaluator as te
from y2karaoke.core.models import Line, Word


def _make_line(text, start, end):
    words = [Word(text=w, start_time=start, end_time=end) for w in text.split()]
    return Line(words=words)


def test_align_lyrics_to_transcription_shifts_line(monkeypatch):
    lines = [_make_line("hello", 5.0, 5.5)]
    segment = te.TranscriptionSegment(start=7.0, end=7.5, text="hello", words=[])

    monkeypatch.setattr(te, "_text_similarity", lambda *a, **k: 0.9)

    aligned, alignments = te.align_lyrics_to_transcription(
        lines, [segment], min_similarity=0.4, max_time_shift=10.0
    )

    assert aligned[0].start_time == pytest.approx(7.0)
    assert alignments


def test_align_words_to_whisper_adjusts_word(monkeypatch):
    line = _make_line("hello", 1.0, 1.3)
    whisper_word = te.TranscriptionWord(start=2.0, end=2.2, text="hello")

    monkeypatch.setattr(
        te,
        "_find_best_whisper_match",
        lambda *a, **k: (whisper_word, 0, 0.9),
    )

    aligned, corrections = te.align_words_to_whisper([line], [whisper_word])

    assert aligned[0].words[0].start_time == pytest.approx(2.0)
    assert corrections


def test_assess_lrc_quality_counts_good(monkeypatch):
    line = _make_line("hi", 1.0, 1.2)
    whisper_words = [te.TranscriptionWord(start=1.4, end=1.5, text="hi")]

    monkeypatch.setattr(te, "_phonetic_similarity", lambda *a, **k: 0.6)

    quality, assessments = te._assess_lrc_quality(
        [line], whisper_words, language="eng-Latn", tolerance=0.5
    )

    assert quality == 1.0
    assert len(assessments) == 1


def test_apply_offset_to_line_shifts():
    line = _make_line("hello", 1.0, 1.5)
    shifted = te._apply_offset_to_line(line, 2.0)
    assert shifted.start_time == pytest.approx(3.0)
    assert shifted.end_time == pytest.approx(3.5)


def test_calculate_drift_correction():
    drift = te._calculate_drift_correction([0.0, 1.2, 1.1], trust_threshold=1.0)
    assert drift is not None
    assert drift > 1.0


def test_fix_ordering_violations_reverts():
    original = [_make_line("one", 1.0, 2.0), _make_line("two", 3.0, 4.0)]
    aligned = [_make_line("one", 1.0, 2.0), _make_line("two", 1.5, 2.5)]

    fixed, alignments = te._fix_ordering_violations(
        original, aligned, ["Line 2 shifted"]
    )

    assert fixed[1].start_time == pytest.approx(3.0)
    assert alignments == []


def test_align_hybrid_trusts_good_line(monkeypatch):
    line = _make_line("hello", 5.0, 5.5)
    segment = te.TranscriptionSegment(start=5.2, end=5.6, text="hello", words=[])

    monkeypatch.setattr(te, "_get_ipa", lambda *a, **k: None)
    monkeypatch.setattr(
        te, "_find_best_whisper_segment", lambda *a, **k: (segment, 0.9, 0.2)
    )

    aligned, corrections = te.align_hybrid_lrc_whisper(
        [line], [segment], [], trust_threshold=1.0, correct_threshold=1.5
    )

    assert aligned[0].start_time == pytest.approx(5.0)
    assert corrections == []


def test_align_hybrid_applies_shift(monkeypatch):
    line = _make_line("hello", 5.0, 5.5)
    segment = te.TranscriptionSegment(start=8.0, end=8.5, text="hello", words=[])

    monkeypatch.setattr(te, "_get_ipa", lambda *a, **k: None)
    monkeypatch.setattr(
        te, "_find_best_whisper_segment", lambda *a, **k: (segment, 0.9, 3.0)
    )

    aligned, corrections = te.align_hybrid_lrc_whisper(
        [line], [segment], [], trust_threshold=1.0, correct_threshold=1.5
    )

    assert aligned[0].start_time == pytest.approx(8.0)
    assert corrections
