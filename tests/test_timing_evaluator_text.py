from y2karaoke.core.timing_evaluator import (
    TranscriptionSegment,
    TranscriptionWord,
    _get_whisper_cache_path,
    _load_whisper_cache,
    _normalize_text_for_matching,
    _save_whisper_cache,
    _text_similarity,
    _text_similarity_basic,
    _whisper_lang_to_epitran,
)


def test_whisper_cache_round_trip(tmp_path):
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"data")
    cache_path = _get_whisper_cache_path(str(vocals), "base", "fr")
    assert cache_path is not None
    assert cache_path.endswith("vocals_whisper_base_fr.json")

    words = [TranscriptionWord(start=0.0, end=0.5, text="bonjour", probability=0.9)]
    segments = [TranscriptionSegment(start=0.0, end=0.6, text="bonjour", words=words)]
    _save_whisper_cache(cache_path, segments, words, "fr")

    loaded = _load_whisper_cache(cache_path)
    assert loaded is not None
    loaded_segments, loaded_words, lang = loaded
    assert lang == "fr"
    assert len(loaded_segments) == 1
    assert len(loaded_words) == 1
    assert loaded_segments[0].text == "bonjour"


def test_whisper_cache_missing_returns_none(tmp_path):
    missing = tmp_path / "missing.json"
    assert _load_whisper_cache(str(missing)) is None


def test_normalize_text_for_matching_strips_accents_and_punct():
    text = "C'est déjà l'été!"
    assert _normalize_text_for_matching(text) == "cest deja lete"


def test_text_similarity_basic_handles_empty():
    assert _text_similarity_basic("", "hello") == 0.0


def test_text_similarity_basic_matches_identical():
    assert _text_similarity_basic("hello world", "hello world") == 1.0


def test_text_similarity_no_phonetic_falls_back():
    assert _text_similarity("hello", "hello", use_phonetic=False) == 1.0


def test_whisper_lang_to_epitran_mapping():
    assert _whisper_lang_to_epitran("fr") == "fra-Latn"
    assert _whisper_lang_to_epitran("xx") == "eng-Latn"
