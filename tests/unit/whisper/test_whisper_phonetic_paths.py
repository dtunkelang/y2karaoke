import builtins

from y2karaoke.core.components.whisper import whisper_phonetic_paths as paths


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
