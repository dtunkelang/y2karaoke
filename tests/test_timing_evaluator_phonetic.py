import pytest

from y2karaoke.core import timing_evaluator as te
from y2karaoke.core import phonetic_utils as pu
from y2karaoke.core import whisper_integration as wi


def test_whisper_lang_to_epitran_mapping():
    assert pu._whisper_lang_to_epitran("fr") == "fra-Latn"
    assert pu._whisper_lang_to_epitran("ja") == "jpn-Hira"
    assert pu._whisper_lang_to_epitran("unknown") == "eng-Latn"


def test_get_ipa_uses_cache(monkeypatch):
    calls = {"count": 0}

    class FakeEpi:
        def transliterate(self, text):
            calls["count"] += 1
            return "ipa"

    monkeypatch.setattr(pu, "_get_epitran", lambda *_: FakeEpi())
    pu._ipa_cache.clear()

    assert pu._get_ipa("Hello", "eng-Latn") == "ipa"
    assert pu._get_ipa("Hello", "eng-Latn") == "ipa"
    assert calls["count"] == 1


def test_get_ipa_returns_none_without_epitran(monkeypatch):
    monkeypatch.setattr(pu, "_get_epitran", lambda *_: None)
    pu._ipa_cache.clear()
    assert pu._get_ipa("Hello", "eng-Latn") is None


def test_phonetic_similarity_falls_back(monkeypatch):
    monkeypatch.setattr(pu, "_get_panphon_distance", lambda: None)
    monkeypatch.setattr(pu, "_text_similarity_basic", lambda *args, **kwargs: 0.42)
    assert pu._phonetic_similarity("a", "b") == 0.42


def test_phonetic_similarity_with_panphon(monkeypatch):
    class FakeDistance:
        def feature_edit_distance(self, a, b):
            return 0.0

    class FakeFT:
        def ipa_segs(self, ipa):
            return ["a", "b"]

    monkeypatch.setattr(pu, "_get_panphon_distance", lambda: FakeDistance())
    monkeypatch.setattr(pu, "_get_panphon_ft", lambda: FakeFT())
    monkeypatch.setattr(pu, "_get_ipa", lambda text, language="fra-Latn": "ipa")

    assert pu._phonetic_similarity("bonjour", "bonjour") == 1.0


def test_text_similarity_basic_path():
    assert te._text_similarity("hello", "hello", use_phonetic=False) == 1.0


def test_find_best_whisper_match_picks_candidate(monkeypatch):
    word = te.TranscriptionWord(start=1.0, end=1.2, text="hello", probability=1.0)
    whisper_words = [word]

    monkeypatch.setattr(pu, "_text_similarity_basic", lambda *args, **kwargs: 0.3)
    monkeypatch.setattr(pu, "_phonetic_similarity", lambda a, b, language: 0.6)

    match, idx, sim = te._find_best_whisper_match(
        lrc_text="hello",
        lrc_start=1.1,
        sorted_whisper=whisper_words,
        used_indices=set(),
        min_similarity=0.5,
        max_time_shift=2.0,
        language="eng-Latn",
    )

    assert match is word
    assert idx == 0
    assert sim == 0.6


def test_find_best_whisper_match_respects_min_index(monkeypatch):
    early = te.TranscriptionWord(start=1.0, end=1.2, text="alpha", probability=1.0)
    later = te.TranscriptionWord(start=5.0, end=5.2, text="beta", probability=1.0)
    whisper_words = [early, later]

    monkeypatch.setattr(pu, "_text_similarity_basic", lambda *args, **kwargs: 0.3)
    monkeypatch.setattr(pu, "_phonetic_similarity", lambda a, b, language: 0.6)

    match, idx, sim = te._find_best_whisper_match(
        lrc_text="beta",
        lrc_start=5.1,
        sorted_whisper=whisper_words,
        used_indices=set(),
        min_similarity=0.5,
        max_time_shift=2.0,
        language="eng-Latn",
        min_index=1,
    )

    assert match is later
    assert idx == 1
    assert sim == 0.6


def test_normalize_text_for_phonetic_expands_contractions():
    normalized = pu._normalize_text_for_phonetic("You're here", "eng-Latn")
    assert normalized == "you are here"


def test_normalize_text_for_phonetic_keeps_non_english():
    normalized = pu._normalize_text_for_phonetic("tu es", "fra-Latn")
    assert normalized == "tu es"
