import sys
import types

import numpy as np

import y2karaoke.core.timing_evaluator as te
from y2karaoke.core.models import Line, Word


def test_align_dtw_whisper_uses_fastdtw(monkeypatch):
    line = Line(words=[Word(text="hello", start_time=0.0, end_time=0.5)])
    whisper_words = [
        te.TranscriptionWord(start=2.0, end=2.5, text="hello", probability=0.9)
    ]

    monkeypatch.setattr(te, "_get_ipa", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(te, "_phonetic_similarity", lambda *_a, **_k: 1.0)

    def fake_fastdtw(lrc_seq, whisper_seq, dist):
        # Exercise the distance function at least once
        _ = dist(lrc_seq[0], whisper_seq[0])
        return 0.0, [(0, 0)]

    fake_module = types.SimpleNamespace(fastdtw=fake_fastdtw)
    monkeypatch.setitem(sys.modules, "fastdtw", fake_module)

    aligned, corrections, metrics = te.align_dtw_whisper(
        [line], whisper_words, language="eng-Latn", min_similarity=0.1
    )

    assert aligned[0].words[0].start_time == 2.0
    assert corrections
    assert metrics["matched_ratio"] > 0.0


def test_print_comparison_report_formats_output(monkeypatch, capsys):
    report_a = te.TimingReport(
        source_name="source-a",
        overall_score=85.0,
        line_alignment_score=80.0,
        pause_alignment_score=90.0,
        summary="Summary A",
        issues=[
            te.TimingIssue(
                issue_type="early_line",
                line_index=0,
                lyrics_time=1.0,
                audio_time=1.2,
                delta=-0.2,
                severity="severe",
                description="Line early",
            )
        ],
    )
    report_b = te.TimingReport(
        source_name="source-b",
        overall_score=70.0,
        line_alignment_score=60.0,
        pause_alignment_score=80.0,
        summary="Summary B",
        issues=[],
    )

    monkeypatch.setattr(
        te,
        "compare_sources",
        lambda *_args, **_kwargs: {
            "source-a": report_a,
            "source-b": report_b,
        },
    )

    te.print_comparison_report("Title", "Artist", "vocals.wav")
    out = capsys.readouterr().out

    assert "Lyrics Timing Comparison: Artist - Title" in out
    assert "Recommended source: source-a" in out
    assert "[SEVERE] Line early" in out


def test_find_best_whisper_segment_picks_highest_similarity(monkeypatch):
    segments = [
        te.TranscriptionSegment(start=5.0, end=6.0, text="alpha", words=[]),
        te.TranscriptionSegment(start=5.5, end=6.5, text="beta", words=[]),
    ]

    def fake_similarity(text1, text2, *_args, **_kwargs):
        return 0.2 if text2 == "alpha" else 0.8

    monkeypatch.setattr(te, "_phonetic_similarity", fake_similarity)

    best_segment, similarity, offset = te._find_best_whisper_segment(
        "target",
        5.2,
        segments,
        language="eng-Latn",
        min_similarity=0.3,
    )

    assert best_segment is segments[1]
    assert similarity == 0.8
    assert offset == segments[1].start - 5.2
