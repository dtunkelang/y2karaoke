import pytest

from y2karaoke.core import timing_evaluator as te


def test_whisper_lang_to_epitran_mapping():
    assert te._whisper_lang_to_epitran("fr") == "fra-Latn"
    assert te._whisper_lang_to_epitran("ja") == "jpn-Hira"
    assert te._whisper_lang_to_epitran("unknown") == "eng-Latn"


def test_get_ipa_uses_cache(monkeypatch):
    calls = {"count": 0}

    class FakeEpi:
        def transliterate(self, text):
            calls["count"] += 1
            return "ipa"

    monkeypatch.setattr(te, "_get_epitran", lambda *_: FakeEpi())
    te._ipa_cache.clear()

    assert te._get_ipa("Hello", "eng-Latn") == "ipa"
    assert te._get_ipa("Hello", "eng-Latn") == "ipa"
    assert calls["count"] == 1


def test_get_ipa_returns_none_without_epitran(monkeypatch):
    monkeypatch.setattr(te, "_get_epitran", lambda *_: None)
    te._ipa_cache.clear()
    assert te._get_ipa("Hello", "eng-Latn") is None


def test_phonetic_similarity_falls_back(monkeypatch):
    monkeypatch.setattr(te, "_get_panphon_distance", lambda: None)
    monkeypatch.setattr(te, "_text_similarity_basic", lambda a, b: 0.42)
    assert te._phonetic_similarity("a", "b") == 0.42


def test_phonetic_similarity_with_panphon(monkeypatch):
    class FakeDistance:
        def feature_edit_distance(self, a, b):
            return 0.0

    class FakeFT:
        def ipa_segs(self, ipa):
            return ["a", "b"]

    monkeypatch.setattr(te, "_get_panphon_distance", lambda: FakeDistance())
    monkeypatch.setattr(te, "_get_panphon_ft", lambda: FakeFT())
    monkeypatch.setattr(te, "_get_ipa", lambda text, language="fra-Latn": "ipa")

    assert te._phonetic_similarity("bonjour", "bonjour") == 1.0


def test_text_similarity_basic_path():
    assert te._text_similarity("hello", "hello", use_phonetic=False) == 1.0


def test_find_best_whisper_match_picks_candidate(monkeypatch):
    word = te.TranscriptionWord(start=1.0, end=1.2, text="hello", probability=1.0)
    whisper_words = [word]

    monkeypatch.setattr(te, "_text_similarity_basic", lambda a, b: 0.3)
    monkeypatch.setattr(te, "_phonetic_similarity", lambda a, b, language: 0.6)

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
