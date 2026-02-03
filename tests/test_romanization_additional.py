import builtins
import runpy

from y2karaoke.core import romanization


def test_romanize_arabic_transliteration(monkeypatch):
    monkeypatch.setattr(romanization, "ARABIC_ROMANIZER_AVAILABLE", True)

    result = romanization.romanize_arabic("سلام")

    assert result == "slam"


def test_romanize_hebrew_transliteration():
    result = romanization.romanize_hebrew("שלום")

    assert result == "shlvm"


def test_romanize_multilingual_uses_script_map(monkeypatch):
    def fake_hebrew(text):
        return "shalom"

    monkeypatch.setattr(
        romanization,
        "SCRIPT_ROMANIZER_MAP",
        [(romanization.HEBREW_RANGES, fake_hebrew)],
    )

    result = romanization.romanize_multilingual("Hello שלום")

    assert result == "Hello shalom"


def test_romanize_korean_unavailable_returns_original(monkeypatch):
    monkeypatch.setattr(romanization, "KOREAN_ROMANIZER_AVAILABLE", False)
    assert romanization.romanize_korean("안녕") == "안녕"


def test_romanize_korean_exception_returns_original(monkeypatch):
    monkeypatch.setattr(romanization, "KOREAN_ROMANIZER_AVAILABLE", True)

    class BrokenRomanizer:
        def __init__(self, *args, **kwargs):
            pass

        def romanize(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(romanization, "Romanizer", BrokenRomanizer)
    assert romanization.romanize_korean("안녕") == "안녕"


def test_romanize_chinese_unavailable_returns_original(monkeypatch):
    monkeypatch.setattr(romanization, "CHINESE_ROMANIZER_AVAILABLE", False)
    assert romanization.romanize_chinese("你好") == "你好"


def test_romanize_chinese_exception_returns_original(monkeypatch):
    monkeypatch.setattr(romanization, "CHINESE_ROMANIZER_AVAILABLE", True)

    def broken_lazy_pinyin(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(romanization, "lazy_pinyin", broken_lazy_pinyin)
    assert romanization.romanize_chinese("你好") == "你好"


def test_romanize_japanese_unavailable_returns_original(monkeypatch):
    monkeypatch.setattr(romanization, "JAPANESE_ROMANIZER_AVAILABLE", False)
    assert romanization.romanize_japanese("こんにちは") == "こんにちは"


def test_romanize_japanese_exception_returns_original(monkeypatch):
    monkeypatch.setattr(romanization, "JAPANESE_ROMANIZER_AVAILABLE", True)

    class BrokenConverter:
        def convert(self, text):
            raise RuntimeError("boom")

    monkeypatch.setattr(romanization, "_JAPANESE_CONVERTER", BrokenConverter())
    assert romanization.romanize_japanese("こんにちは") == "こんにちは"


def test_romanize_arabic_unavailable_returns_original(monkeypatch):
    monkeypatch.setattr(romanization, "ARABIC_ROMANIZER_AVAILABLE", False)
    assert romanization.romanize_arabic("سلام") == "سلام"


def test_romanize_arabic_exception_returns_original(monkeypatch):
    monkeypatch.setattr(romanization, "ARABIC_ROMANIZER_AVAILABLE", True)
    monkeypatch.setattr(romanization, "ARABIC_TO_LATIN", None)
    assert romanization.romanize_arabic("سلام") == "سلام"


def test_romanize_hebrew_exception_returns_original(monkeypatch):
    monkeypatch.setattr(romanization, "HEBREW_TO_LATIN", None)
    assert romanization.romanize_hebrew("שלום") == "שלום"


def test_romanize_multilingual_handles_romanizer_error(monkeypatch):
    def broken(text):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        romanization,
        "SCRIPT_ROMANIZER_MAP",
        [(romanization.HEBREW_RANGES, broken)],
    )

    result = romanization.romanize_multilingual("Hello שלום")

    assert result == "Hello שלום"


def test_romanize_multilingual_returns_block_when_no_map(monkeypatch):
    monkeypatch.setattr(romanization, "SCRIPT_ROMANIZER_MAP", [])
    assert romanization.romanize_multilingual("שלום") == "שלום"


def test_import_fallbacks_when_dependencies_missing(monkeypatch):
    module_path = romanization.__file__
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith(("korean_romanizer", "pypinyin", "pykakasi", "pyarabic")):
            raise ImportError("missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    romanization_globals = runpy.run_path(module_path)

    assert romanization_globals["KOREAN_ROMANIZER_AVAILABLE"] is False
    assert romanization_globals["Romanizer"] is None
    assert romanization_globals["CHINESE_ROMANIZER_AVAILABLE"] is False
    assert romanization_globals["lazy_pinyin"] is None
    assert romanization_globals["Style"] is None
    assert romanization_globals["JAPANESE_ROMANIZER_AVAILABLE"] is False
    assert romanization_globals["ARABIC_ROMANIZER_AVAILABLE"] is False
