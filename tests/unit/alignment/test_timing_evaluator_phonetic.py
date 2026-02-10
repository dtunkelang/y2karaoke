from y2karaoke.core import timing_evaluator as te
from y2karaoke.core import phonetic_utils as pu


def test_whisper_lang_to_epitran_mapping():
    assert pu._whisper_lang_to_epitran("fr") == "fra-Latn"
    assert pu._whisper_lang_to_epitran("ja") == "jpn-Hira"
    assert pu._whisper_lang_to_epitran("unknown") == "eng-Latn"


def test_get_ipa_uses_cache():
    calls = {"count": 0}

    class FakeEpi:
        def transliterate(self, text):
            calls["count"] += 1
            return "ipa"

    pu._ipa_cache.clear()
    with pu.use_phonetic_utils_hooks(get_epitran_fn=lambda *_: FakeEpi()):
        assert pu._get_ipa("Hello", "eng-Latn") == "ipa"
        assert pu._get_ipa("Hello", "eng-Latn") == "ipa"
    assert calls["count"] == 1


def test_get_ipa_returns_none_without_epitran():
    pu._ipa_cache.clear()
    with pu.use_phonetic_utils_hooks(get_epitran_fn=lambda *_: None):
        assert pu._get_ipa("Hello", "eng-Latn") is None


def test_phonetic_similarity_falls_back():
    with pu.use_phonetic_utils_hooks(
        get_panphon_distance_fn=lambda: None,
        text_similarity_basic_fn=lambda *args, **kwargs: 0.42,
    ):
        assert pu._phonetic_similarity("a", "b") == 0.42


def test_phonetic_similarity_with_panphon():
    class FakeDistance:
        def feature_edit_distance(self, a, b):
            return 0.0

    class FakeFT:
        def ipa_segs(self, ipa):
            return ["a", "b"]

    with pu.use_phonetic_utils_hooks(
        get_panphon_distance_fn=lambda: FakeDistance(),
        get_panphon_ft_fn=lambda: FakeFT(),
        get_ipa_fn=lambda text, language="fra-Latn": "ipa",
    ):
        assert pu._phonetic_similarity("bonjour", "bonjour") == 1.0


def test_text_similarity_basic_path():
    assert te._text_similarity("hello", "hello", use_phonetic=False) == 1.0


def test_find_best_whisper_match_picks_candidate():
    word = te.TranscriptionWord(start=1.0, end=1.2, text="hello", probability=1.0)
    whisper_words = [word]

    with pu.use_phonetic_utils_hooks(
        text_similarity_basic_fn=lambda *args, **kwargs: 0.3,
        phonetic_similarity_fn=lambda a, b, language: 0.6,
    ):
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


def test_find_best_whisper_match_respects_min_index():
    early = te.TranscriptionWord(start=1.0, end=1.2, text="alpha", probability=1.0)
    later = te.TranscriptionWord(start=5.0, end=5.2, text="beta", probability=1.0)
    whisper_words = [early, later]

    with pu.use_phonetic_utils_hooks(
        text_similarity_basic_fn=lambda *args, **kwargs: 0.3,
        phonetic_similarity_fn=lambda a, b, language: 0.6,
    ):
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
