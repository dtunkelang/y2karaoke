from collections import defaultdict

import numpy as np

from y2karaoke.core.components.alignment.timing_models import TranscriptionWord
from y2karaoke.core.components.whisper import whisper_dtw
from y2karaoke.core.models import Line, Word


def _raise_import_error():
    raise ImportError("fastdtw unavailable")


def test_align_dtw_whisper_base_fallback_without_fastdtw():
    lines = [Line(words=[Word(text="hello", start_time=1.0, end_time=1.4)])]
    whisper_words = [
        TranscriptionWord(text="hello", start=3.0, end=3.4, probability=0.9)
    ]

    with whisper_dtw.use_whisper_dtw_hooks(
        load_fastdtw_fn=_raise_import_error,
        phonetic_similarity_fn=lambda *_a, **_k: 1.0,
        get_ipa_fn=lambda *_a, **_k: "ipa",
    ):
        aligned, _corrections = whisper_dtw.align_dtw_whisper_base(
            lines, whisper_words, language="eng-Latn"
        )

    assert aligned
    assert aligned[0].words[0].start_time == 3.0


def test_align_dtw_whisper_with_data_fallback_without_fastdtw(monkeypatch):
    lines = [Line(words=[Word(text="hello", start_time=1.0, end_time=1.4)])]
    whisper_words = [
        TranscriptionWord(text="hello", start=3.0, end=3.4, probability=0.9)
    ]

    monkeypatch.setattr(
        "y2karaoke.core.components.whisper.whisper_phonetic_dtw._compute_phonetic_costs",
        lambda *_a, **_k: defaultdict(lambda: 0.0),
    )

    with whisper_dtw.use_whisper_dtw_hooks(
        load_fastdtw_fn=_raise_import_error,
        phonetic_similarity_fn=lambda *_a, **_k: 1.0,
        get_ipa_fn=lambda *_a, **_k: "ipa",
    ):
        aligned, _corrections, metrics, _lrc_words, _alignments = (
            whisper_dtw._align_dtw_whisper_with_data(
                lines,
                whisper_words,
                language="eng-Latn",
            )
        )

    assert aligned
    assert metrics["word_coverage"] >= 1.0


def test_dtw_fallback_runtime_guard_uses_banded_window(monkeypatch):
    lrc_seq = np.zeros((700, 2), dtype=float)
    whisper_seq = np.zeros((700, 2), dtype=float)
    calls = []

    def fake_path(_lrc, _whisper, _dist, *, window=None):
        calls.append(window)
        return 0.0, [(0, 0)]

    monkeypatch.setattr(whisper_dtw, "_dtw_fallback_path", fake_path)

    whisper_dtw._dtw_fallback_with_runtime_guard(
        lrc_seq,
        whisper_seq,
        dist=lambda _a, _b: 0.0,
    )

    assert calls
    assert calls[0] is not None


def test_extract_alignments_uses_precomputed_similarity_without_recompute():
    lrc_words = [{"text": "hello", "line_idx": 0, "word_idx": 0, "start": 0.0}]
    whisper_words = [
        TranscriptionWord(text="hullo", start=1.0, end=1.2, probability=0.9)
    ]

    with whisper_dtw.use_whisper_dtw_hooks(
        phonetic_similarity_fn=lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("unexpected phonetic similarity call")
        )
    ):
        alignments = whisper_dtw._extract_alignments_from_path_base(
            path=[(0, 0)],
            lrc_words=lrc_words,
            whisper_words=whisper_words,
            language="eng-Latn",
            min_similarity=0.4,
            precomputed_similarity={(0, 0): 0.8},
        )

    assert 0 in alignments
    assert alignments[0][1] == 0.8
