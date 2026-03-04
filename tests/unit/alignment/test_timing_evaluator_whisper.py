import y2karaoke.core.components.alignment.timing_evaluator as te
import y2karaoke.core.components.whisper.whisper_integration as wi
import y2karaoke.core.components.whisper.whisper_dtw as w_dtw
import y2karaoke.core.components.whisper.whisper_alignment as wa
from y2karaoke.core.models import Line, Word


def _alignment_similarity_hook(fn):
    return wa.use_whisper_alignment_hooks(phonetic_similarity_fn=fn)


def test_align_words_to_whisper_adjusts_word():
    lines = [Line(words=[Word(text="hello", start_time=0.0, end_time=0.5)])]
    whisper_words = [
        te.TranscriptionWord(start=2.0, end=2.4, text="hello", probability=0.9)
    ]
    with wi.use_whisper_integration_hooks(get_ipa_fn=lambda *_args, **_kwargs: None):
        aligned, corrections = wi.align_words_to_whisper(
            lines,
            whisper_words,
            language="eng-Latn",
        )
    assert aligned[0].words[0].start_time == 2.0
    assert corrections


def test_assess_lrc_quality_reports_good_match(monkeypatch):
    lines = [Line(words=[Word(text="hello", start_time=1.0, end_time=1.5)])]
    words = [te.TranscriptionWord(start=1.1, end=1.6, text="hello", probability=0.9)]

    with w_dtw.use_whisper_dtw_hooks(
        phonetic_similarity_fn=lambda *_args, **_kwargs: 0.8
    ):
        quality, assessments = wi._assess_lrc_quality(lines, words, "eng-Latn")
    assert quality >= 0.9
    assert assessments


def test_extract_lrc_words_returns_indices():
    lines = [Line(words=[Word(text="one", start_time=0, end_time=1)])]
    lrc_words = w_dtw._extract_lrc_words_base(lines)
    assert len(lrc_words) == 1
    assert lrc_words[0]["text"] == "one"
    assert lrc_words[0]["line_idx"] == 0


def test_compute_phonetic_costs_includes_matches(monkeypatch):
    lrc_words = [{"text": "hello", "start": 1.0}]
    whisper_words = [
        te.TranscriptionWord(start=1.1, end=1.6, text="hello", probability=0.9)
    ]

    with w_dtw.use_whisper_dtw_hooks(
        phonetic_similarity_fn=lambda *_args, **_kwargs: 0.6
    ):
        costs = wi._compute_phonetic_costs(lrc_words, whisper_words, "eng-Latn", 0.5)
    assert (0, 0) in costs
    assert costs[(0, 0)] < 0.5


def test_extract_alignments_from_path_filters_by_similarity(monkeypatch):
    path = [(0, 0)]
    lrc_words = [{"text": "hello", "start": 1.0}]
    whisper_words = [
        te.TranscriptionWord(start=1.1, end=1.6, text="hullo", probability=0.9)
    ]

    with w_dtw.use_whisper_dtw_hooks(
        phonetic_similarity_fn=lambda *_args, **_kwargs: 0.1
    ):
        alignments = wi._extract_alignments_from_path(
            path, lrc_words, whisper_words, "eng-Latn", 0.5
        )
    assert len(alignments) == 0


def test_apply_dtw_alignments_shifts_large_offsets():
    lines = [Line(words=[Word(text="hello", start_time=1.0, end_time=1.5)])]
    lrc_words = [{"text": "hello", "start": 1.0, "line_idx": 0, "word_idx": 0}]
    alignments = {
        0: (
            te.TranscriptionWord(start=5.0, end=5.5, text="hello", probability=0.9),
            0.9,
        )
    }

    aligned, corrections = wi._apply_dtw_alignments(lines, lrc_words, alignments)
    assert aligned[0].words[0].start_time == 5.0
    assert corrections


def test_align_dtw_whisper_falls_back_without_fastdtw(monkeypatch):
    lines = [Line(words=[Word(text="hello", start_time=1.0, end_time=1.5)])]
    words = [te.TranscriptionWord(start=1.1, end=1.6, text="hello", probability=0.9)]

    with w_dtw.use_whisper_dtw_hooks(
        phonetic_similarity_fn=lambda *_a, **_k: 1.0,
        load_fastdtw_fn=lambda: (_ for _ in ()).throw(ImportError("missing fastdtw")),
    ):
        aligned, corrections, metrics = te.align_dtw_whisper(
            lines, words, min_similarity=0.1
        )
    assert len(aligned) == 1
    assert metrics["matched_ratio"] > 0.0
    assert metrics["line_coverage"] > 0.0


def test_correct_timing_with_whisper_no_transcription(monkeypatch):
    lines = [Line(words=[Word(text="a", start_time=0, end_time=1)])]
    with wi.use_whisper_integration_hooks(
        transcribe_vocals_fn=lambda *_: ([], [], "en", "base")
    ):
        aligned, corrections, metrics = te.correct_timing_with_whisper(
            lines, "vocals.wav"
        )
    assert aligned == lines
    assert not corrections


def test_correct_timing_with_whisper_uses_dtw(monkeypatch):
    lines = [Line(words=[Word(text="hello", start_time=10.0, end_time=11.0)])]
    words = [te.TranscriptionWord(start=20.0, end=21.0, text="hello", probability=0.9)]
    segments = [
        te.TranscriptionSegment(start=20.0, end=21.0, text="hello", words=words)
    ]

    with wi.use_whisper_integration_hooks(
        transcribe_vocals_fn=lambda *_: (segments, words, "en", "base"),
        extract_audio_features_fn=lambda *_: None,
        assess_lrc_quality_fn=lambda *_, **__: (0.1, []),
        align_dtw_whisper_with_data_fn=lambda *_, **__: (
            [Line(words=[Word(text="hello", start_time=20.0, end_time=21.0)])],
            ["dtw"],
            {"matched_ratio": 0.8, "avg_similarity": 0.8, "line_coverage": 0.8},
            [],
            {0: (words[0], 0.9)},
        ),
    ):
        aligned, corrections, metrics = te.correct_timing_with_whisper(
            lines, "vocals.wav"
        )
    assert aligned[0].start_time == 20.0
    assert "dtw" in corrections or any("DTW" in c for c in corrections)


def test_correct_timing_with_whisper_quality_good_uses_hybrid(monkeypatch):
    lines = [Line(words=[Word(text="hello", start_time=10.0, end_time=11.0)])]
    words = [te.TranscriptionWord(start=10.1, end=11.1, text="hello", probability=0.9)]
    segments = [
        te.TranscriptionSegment(start=10.0, end=11.5, text="hello", words=words)
    ]

    with wi.use_whisper_integration_hooks(
        transcribe_vocals_fn=lambda *_: (segments, words, "en", "base"),
        extract_audio_features_fn=lambda *_: None,
        assess_lrc_quality_fn=lambda *_, **__: (0.9, []),
        align_hybrid_lrc_whisper_fn=lambda *_, **__: (
            [Line(words=[Word(text="hello", start_time=10.1, end_time=11.1)])],
            ["hybrid"],
        ),
    ):
        aligned, corrections, metrics = te.correct_timing_with_whisper(
            lines, "vocals.wav"
        )
    assert "hybrid" in corrections


def test_correct_timing_with_whisper_quality_mixed_uses_hybrid(monkeypatch):
    lines = [Line(words=[Word(text="hello", start_time=10.0, end_time=11.0)])]
    words = [te.TranscriptionWord(start=10.1, end=11.1, text="hello", probability=0.9)]
    segments = [
        te.TranscriptionSegment(start=10.0, end=11.5, text="hello", words=words)
    ]

    with wi.use_whisper_integration_hooks(
        transcribe_vocals_fn=lambda *_: (segments, words, "en", "base"),
        extract_audio_features_fn=lambda *_: None,
        assess_lrc_quality_fn=lambda *_, **__: (0.5, []),
        align_hybrid_lrc_whisper_fn=lambda *_, **__: (
            [Line(words=[Word(text="hello", start_time=10.1, end_time=11.1)])],
            ["hybrid"],
        ),
    ):
        aligned, corrections, metrics = te.correct_timing_with_whisper(
            lines, "vocals.wav"
        )
    assert "hybrid" in corrections


def test_correct_timing_with_whisper_uses_dtw_retime_when_confident():
    lines = [Line(words=[Word(text="hello", start_time=10.0, end_time=11.0)])]
    words = [te.TranscriptionWord(start=20.0, end=21.0, text="hello", probability=0.9)]
    segments = [
        te.TranscriptionSegment(start=20.0, end=21.0, text="hello", words=words)
    ]

    # High confidence metrics
    metrics_high = {"matched_ratio": 0.9, "avg_similarity": 0.9, "line_coverage": 0.9}
    lrc_words = [{"line_idx": 0, "word_idx": 0, "text": "hello"}]
    align_map = {0: (words[0], 0.9)}
    with wi.use_whisper_integration_hooks(
        transcribe_vocals_fn=lambda *_: (segments, words, "en", "base"),
        extract_audio_features_fn=lambda *_: None,
        assess_lrc_quality_fn=lambda *_, **__: (0.1, []),
        align_dtw_whisper_with_data_fn=lambda *_, **__: (
            lines,
            ["dtw_data"],
            metrics_high,
            lrc_words,
            align_map,
        ),
        retime_lines_from_dtw_alignments_fn=lambda *_, **__: (
            [Line(words=[Word(text="hello", start_time=20.0, end_time=21.0)])],
            ["retimed"],
        ),
    ):
        aligned, corrections, metrics = te.correct_timing_with_whisper(
            lines, "vocals.wav"
        )
    assert "retimed" in corrections
