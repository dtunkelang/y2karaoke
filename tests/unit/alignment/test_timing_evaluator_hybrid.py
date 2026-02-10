import y2karaoke.core.components.alignment.timing_evaluator as te
import y2karaoke.core.components.whisper.whisper_integration as wi
import y2karaoke.core.components.whisper.whisper_alignment_refinement as wa_ref
from y2karaoke.core.models import Line, Word


def test_apply_offset_to_line_shifts_words():
    line = Line(words=[Word(text="a", start_time=1.0, end_time=1.5)])
    shifted = te._apply_offset_to_line(line, 2.0)
    assert shifted.words[0].start_time == 3.0
    assert shifted.words[0].end_time == 3.5


def test_calculate_drift_correction_requires_consistency():
    assert te._calculate_drift_correction([0.1], 0.5) is None
    drift = te._calculate_drift_correction([-1.0, -1.2, -1.1], 0.5)
    assert drift is not None and drift < 0


def test_align_hybrid_lrc_whisper_applies_offsets(monkeypatch):
    lines = [
        Line(words=[Word(text="hello", start_time=0.0, end_time=0.5)]),
        Line(words=[Word(text="world", start_time=5.0, end_time=5.5)]),
    ]
    segments = [
        te.TranscriptionSegment(start=0.0, end=1.0, text="hello", words=[]),
        te.TranscriptionSegment(start=7.0, end=8.0, text="world", words=[]),
    ]

    def fake_find_best(line_text, line_start, _segments, _lang, _min_sim):
        if "hello" in line_text:
            return segments[0], 0.9, 0.2
        return segments[1], 0.6, 2.0

    monkeypatch.setattr(wi, "_find_best_whisper_segment", fake_find_best)

    monkeypatch.setattr(wa_ref, "_find_best_whisper_segment", fake_find_best)
    monkeypatch.setattr(wi, "_get_ipa", lambda *_args, **_kwargs: None)

    aligned, corrections = te.align_hybrid_lrc_whisper(
        lines,
        segments,
        [],
        language="eng-Latn",
        trust_threshold=0.5,
        correct_threshold=1.0,
    )
    assert aligned[0].words[0].start_time == 0.0
    assert aligned[1].words[0].start_time == 7.0
    assert corrections


def test_align_hybrid_lrc_whisper_drift_correction(monkeypatch):
    lines = [
        Line(words=[Word(text="one", start_time=0.0, end_time=0.5)]),
        Line(words=[Word(text="two", start_time=2.0, end_time=2.5)]),
        Line(words=[Word(text="three", start_time=4.0, end_time=4.5)]),
    ]
    segments = [
        te.TranscriptionSegment(start=3.0, end=3.5, text="one", words=[]),
        te.TranscriptionSegment(start=5.0, end=5.5, text="two", words=[]),
        te.TranscriptionSegment(start=6.0, end=6.5, text="three", words=[]),
    ]

    offsets = [2.5, 2.5, 1.2]

    def fake_find_best(line_text, line_start, _segments, _lang, _min_sim):
        offset = offsets.pop(0)
        return segments[0], 0.7, offset

    monkeypatch.setattr(wi, "_find_best_whisper_segment", fake_find_best)

    monkeypatch.setattr(wa_ref, "_find_best_whisper_segment", fake_find_best)
    monkeypatch.setattr(wi, "_get_ipa", lambda *_args, **_kwargs: None)

    aligned, corrections = te.align_hybrid_lrc_whisper(
        lines,
        segments,
        [],
        language="eng-Latn",
        trust_threshold=0.5,
        correct_threshold=2.0,
    )
    # Third line should be drift-corrected to ~+2.5s
    assert aligned[2].words[0].start_time == 6.5
    assert any("drift-corrected" in c for c in corrections)


def test_align_hybrid_lrc_whisper_no_match_uses_drift(monkeypatch):
    lines = [
        Line(words=[Word(text="one", start_time=0.0, end_time=0.5)]),
        Line(words=[Word(text="two", start_time=2.0, end_time=2.5)]),
        Line(words=[Word(text="three", start_time=4.0, end_time=4.5)]),
    ]
    segments = [
        te.TranscriptionSegment(start=3.0, end=3.5, text="one", words=[]),
        te.TranscriptionSegment(start=5.0, end=5.5, text="two", words=[]),
    ]

    offsets = [2.5, 2.5]

    def fake_find_best(line_text, line_start, _segments, _lang, _min_sim):
        if offsets:
            offset = offsets.pop(0)
            return segments[0], 0.7, offset
        return None, 0.0, 0.0

    monkeypatch.setattr(wi, "_find_best_whisper_segment", fake_find_best)

    monkeypatch.setattr(wa_ref, "_find_best_whisper_segment", fake_find_best)
    monkeypatch.setattr(wi, "_get_ipa", lambda *_args, **_kwargs: None)

    aligned, corrections = te.align_hybrid_lrc_whisper(
        lines,
        segments,
        [],
        language="eng-Latn",
        trust_threshold=0.5,
        correct_threshold=2.0,
    )
    # Third line should be drift-corrected using previous offsets
    assert aligned[2].words[0].start_time == 6.5
    assert any("no match" in c for c in corrections)


def test_fix_ordering_violations_reverts(monkeypatch):
    original = [
        Line(words=[Word(text="a", start_time=0.0, end_time=1.0)]),
        Line(words=[Word(text="b", start_time=2.0, end_time=3.0)]),
    ]
    aligned = [
        Line(words=[Word(text="a", start_time=0.0, end_time=1.0)]),
        Line(words=[Word(text="b", start_time=0.5, end_time=1.0)]),
    ]
    fixed_lines, fixed_alignments = te._fix_ordering_violations(
        original, aligned, ["a", "b"]
    )
    assert fixed_lines[1].start_time == 2.0
    assert len(fixed_alignments) == 1
