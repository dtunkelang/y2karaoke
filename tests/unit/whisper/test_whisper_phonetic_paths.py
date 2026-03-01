import builtins

import pytest

from y2karaoke.core.components.whisper import whisper_phonetic_paths as paths
from y2karaoke.core.components.alignment.timing_models import TranscriptionWord


def test_build_phoneme_dtw_path_greedy_fallback_maps_each_lrc_token(monkeypatch):
    original_import = builtins.__import__

    def patched_import(name, *args, **kwargs):
        if name == "fastdtw":
            raise ImportError("fastdtw unavailable in test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", patched_import)

    lrc_phonemes = [
        {"ipa": "a"},
        {"ipa": "b"},
        {"ipa": "c"},
    ]
    whisper_phonemes = [
        {"ipa": "a"},
        {"ipa": "b"},
        {"ipa": "c"},
    ]

    def identity_similarity(ipa1, ipa2, _language):
        return 1.0 if ipa1 == ipa2 else 0.0

    path = paths._build_phoneme_dtw_path(
        lrc_phonemes=lrc_phonemes,
        whisper_phonemes=whisper_phonemes,
        language="eng-Latn",
        phoneme_similarity_fn=identity_similarity,
    )

    assert path == [(0, 0), (1, 1), (2, 2)]


def test_compute_phonetic_costs_reuses_similarity_for_repeated_pairs(monkeypatch):
    calls = {"count": 0}

    def fake_similarity(left, right, _language):
        calls["count"] += 1
        return 0.8 if left == right else 0.0

    monkeypatch.setattr(paths.phonetic_utils, "_phonetic_similarity", fake_similarity)

    lrc_words = [
        {"text": "baila", "start": 10.0},
        {"text": "baila", "start": 10.2},
    ]
    whisper_words = [
        TranscriptionWord(start=10.1, end=10.2, text="baila", probability=1.0),
        TranscriptionWord(start=10.3, end=10.4, text="baila", probability=1.0),
    ]

    costs = paths._compute_phonetic_costs(
        lrc_words=lrc_words,
        whisper_words=whisper_words,
        language="spa-Latn",
        min_similarity=0.4,
    )

    assert costs[(0, 0)] == pytest.approx(0.2)
    assert costs[(0, 1)] == pytest.approx(0.2)
    assert costs[(1, 0)] == pytest.approx(0.2)
    assert costs[(1, 1)] == pytest.approx(0.2)
    assert calls["count"] == 1
