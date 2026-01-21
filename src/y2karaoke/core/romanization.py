"""Multilingual romanization for lyrics display."""

import re

# ----------------------
# Language library imports
# ----------------------

# Korean
try:
    from korean_romanizer.romanizer import Romanizer
    KOREAN_ROMANIZER_AVAILABLE = True
except ImportError:
    KOREAN_ROMANIZER_AVAILABLE = False

# Chinese
try:
    from pypinyin import lazy_pinyin, Style
    CHINESE_ROMANIZER_AVAILABLE = True
except ImportError:
    CHINESE_ROMANIZER_AVAILABLE = False

# Japanese
try:
    from pykakasi import kakasi
    JAPANESE_ROMANIZER_AVAILABLE = True
except ImportError:
    JAPANESE_ROMANIZER_AVAILABLE = False

# Arabic
try:
    import pyarabic.araby  # noqa: F401 - imported for availability check
    ARABIC_ROMANIZER_AVAILABLE = True
except ImportError:
    ARABIC_ROMANIZER_AVAILABLE = False

HEBREW_ROMANIZER_AVAILABLE = True

# ----------------------
# Unicode ranges for script detection
# ----------------------
KOREAN_RANGES = [(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF), (0xA960, 0xA97F), (0xD7B0, 0xD7FF)]
JAPANESE_HIRAGANA = (0x3040, 0x309F)
JAPANESE_KATAKANA = (0x30A0, 0x30FF)
JAPANESE_KANJI_RANGES = [(0x3400, 0x4DBF), (0x4E00, 0x9FFF)]
CHINESE_RANGES = [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2CEAF)]
ARABIC_RANGES = [(0x0600, 0x06FF), (0x0750, 0x077F)]
HEBREW_RANGES = [(0x0590, 0x05FF)]

# ----------------------
# Transliteration tables
# ----------------------
ARABIC_TO_LATIN = {
    'ا': 'a', 'ب': 'b', 'ت': 't', 'ث': 'th', 'ج': 'j', 'ح': 'h', 'خ': 'kh',
    'د': 'd', 'ذ': 'dh', 'ر': 'r', 'ز': 'z', 'س': 's', 'ش': 'sh', 'ص': 's',
    'ض': 'd', 'ط': 't', 'ظ': 'z', 'ع': 'a', 'غ': 'gh', 'ف': 'f', 'ق': 'q',
    'ك': 'k', 'ل': 'l', 'م': 'm', 'ن': 'n', 'ه': 'h', 'و': 'w', 'ي': 'y',
    'ى': 'a', 'ة': 'h', 'ء': '', 'ئ': '', 'ؤ': 'w', 'إ': 'i', 'أ': 'a', 'آ': 'aa',
    'َ': 'a', 'ُ': 'u', 'ِ': 'i', 'ّ': '', 'ْ': ''
}

HEBREW_TO_LATIN = {
    'א': 'a', 'ב': 'b', 'ג': 'g', 'ד': 'd', 'ה': 'h', 'ו': 'v', 'ז': 'z',
    'ח': 'ch', 'ט': 't', 'י': 'y', 'כ': 'k', 'ך': 'kh', 'ל': 'l', 'מ': 'm',
    'ם': 'm', 'נ': 'n', 'ן': 'n', 'ס': 's', 'ע': 'a', 'פ': 'p', 'ף': 'f',
    'צ': 'ts', 'ץ': 'ts', 'ק': 'k', 'ר': 'r', 'ש': 'sh', 'ת': 't'
}

# ----------------------
# Language-specific romanizers
# ----------------------
_JAPANESE_CONVERTER = None


def romanize_korean(text: str) -> str:
    """Romanize Korean text using korean_romanizer."""
    if not KOREAN_ROMANIZER_AVAILABLE:
        return text
    try:
        return Romanizer(text).romanize()
    except Exception:
        return text


def romanize_chinese(text: str) -> str:
    """Romanize Chinese text using pypinyin."""
    if not CHINESE_ROMANIZER_AVAILABLE:
        return text
    try:
        return " ".join(lazy_pinyin(text, style=Style.NORMAL))
    except Exception:
        return text


def romanize_japanese(text: str) -> str:
    """Romanize Japanese text using pykakasi."""
    global _JAPANESE_CONVERTER
    if not JAPANESE_ROMANIZER_AVAILABLE:
        return text
    if _JAPANESE_CONVERTER is None:
        _JAPANESE_CONVERTER = kakasi()
    try:
        result = _JAPANESE_CONVERTER.convert(text)
        return " ".join(item["hepburn"] for item in result)
    except Exception:
        return text


def romanize_arabic(text: str) -> str:
    """Romanize Arabic text using transliteration table."""
    if not ARABIC_ROMANIZER_AVAILABLE:
        return text
    try:
        return ''.join(ARABIC_TO_LATIN.get(c, c) for c in text)
    except Exception:
        return text


def romanize_hebrew(text: str) -> str:
    """Romanize Hebrew text using transliteration table."""
    try:
        return ''.join(HEBREW_TO_LATIN.get(c, c) for c in text)
    except Exception:
        return text


# ----------------------
# Multilingual single-pass romanizer
# ----------------------
SCRIPT_ROMANIZER_MAP = [
    (KOREAN_RANGES, romanize_korean),
    ([JAPANESE_HIRAGANA, JAPANESE_KATAKANA] + JAPANESE_KANJI_RANGES, romanize_japanese),
    (CHINESE_RANGES, romanize_chinese),
    (ARABIC_RANGES, romanize_arabic),
    (HEBREW_RANGES, romanize_hebrew),
]

# Regex for blocks of scripts
KOREAN_RE = r'\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF\uA960-\uA97F\uD7B0-\uD7FF'
JAPANESE_RE = r'\u3040-\u309F\u30A0-\u30FF\u3400-\u4DBF\u4E00-\u9FFF'
CHINESE_RE = r'\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002CEAF'
ARABIC_RE = r'\u0600-\u06FF\u0750-\u077F'
HEBREW_RE = r'\u0590-\u05FF'
MULTILINGUAL_RE = re.compile(f'([{KOREAN_RE}{JAPANESE_RE}{CHINESE_RE}{ARABIC_RE}{HEBREW_RE}]+)')


def romanize_multilingual(text: str) -> str:
    """
    Romanize text containing any mixture of scripts.

    Detects script blocks and applies the appropriate romanizer to each.
    """
    def replace_block(match: re.Match) -> str:
        block = match.group()
        code = ord(block[0])
        for ranges, romanizer in SCRIPT_ROMANIZER_MAP:
            for start, end in ranges:
                if start <= code <= end:
                    try:
                        return romanizer(block)
                    except Exception:
                        return block
        return block
    romanized = MULTILINGUAL_RE.sub(replace_block, text)
    return " ".join(romanized.split())  # collapse repeated spaces


# Main entry point alias
romanize_line = romanize_multilingual
