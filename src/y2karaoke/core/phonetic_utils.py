"""Phonetic and text similarity utilities for lyrics alignment."""

import re
import unicodedata
from typing import Dict, List, Optional, Any
from difflib import SequenceMatcher

from ..utils.logging import get_logger

logger = get_logger(__name__)

_VOWEL_REGEX = re.compile("[aeiouæøɪʊɔɛɑɐəœɜɞʌɒɨɯʉøɤɚɝ]", re.IGNORECASE)

# Cache for phonetic instances (they're expensive to create)
_epitran_cache: Dict[str, Any] = {}
_panphon_distance = None
_panphon_ft = None
_ipa_cache: Dict[str, Optional[str]] = {}  # Cache for IPA transliterations
_ipa_segs_cache: Dict[str, List[str]] = {}  # Cache for IPA segments


def _whisper_lang_to_epitran(lang: str) -> str:
    """Map Whisper language code to epitran language code."""
    mapping = {
        "fr": "fra-Latn",
        "en": "eng-Latn",
        "es": "spa-Latn",
        "de": "deu-Latn",
        "it": "ita-Latn",
        "pt": "por-Latn",
        "nl": "nld-Latn",
        "pl": "pol-Latn",
        "ru": "rus-Cyrl",
        "ja": "jpn-Hira",
        "ko": "kor-Hang",
        "zh": "cmn-Hans",
    }
    return mapping.get(lang, "eng-Latn")  # Default to English


def _normalize_text_for_matching(text: str) -> str:
    """Normalize text for fuzzy matching (basic normalization)."""
    # Convert to lowercase
    text = text.lower()

    # Normalize unicode (é -> e, etc.)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _normalize_text_for_phonetic(text: str, language: str) -> str:
    """Normalize text for phonetic matching with light English heuristics."""
    base = _normalize_text_for_matching(text)
    if not base:
        return base

    if not language.startswith("eng") and not language.startswith("en"):
        return base

    tokens = base.split()
    if not tokens:
        return base

    contractions = {
        "im": ["i", "am"],
        "ive": ["i", "have"],
        "ill": ["i", "will"],
        "id": ["i", "would"],
        "youre": ["you", "are"],
        "youve": ["you", "have"],
        "youll": ["you", "will"],
        "youd": ["you", "would"],
        "theyre": ["they", "are"],
        "theyve": ["they", "have"],
        "theyll": ["they", "will"],
        "theyd": ["they", "would"],
        "were": ["we", "are"],
        "weve": ["we", "have"],
        "well": ["we", "will"],
        "wed": ["we", "would"],
        "cant": ["can", "not"],
        "wont": ["will", "not"],
        "dont": ["do", "not"],
        "didnt": ["did", "not"],
        "doesnt": ["does", "not"],
        "isnt": ["is", "not"],
        "arent": ["are", "not"],
        "wasnt": ["was", "not"],
        "werent": ["were", "not"],
        "shouldnt": ["should", "not"],
        "couldnt": ["could", "not"],
        "wouldnt": ["would", "not"],
        "theres": ["there", "is"],
        "thats": ["that", "is"],
        "whats": ["what", "is"],
        "lets": ["let", "us"],
    }

    homophones = {
        "youre": "your",
        "your": "your",
        "theyre": "their",
        "their": "their",
        "there": "their",
        "to": "to",
        "too": "to",
        "two": "to",
        "for": "for",
        "four": "for",
        "its": "its",
    }

    filler_map = {
        "mmm": "mm",
        "mm": "mm",
        "mhm": "mm",
        "hmm": "mm",
        "uh": "uh",
        "uhh": "uh",
        "um": "um",
        "umm": "um",
    }

    expanded: List[str] = []
    for token in tokens:
        if token in contractions:
            expanded.extend(contractions[token])
            continue
        token = homophones.get(token, token)
        token = filler_map.get(token, token)
        expanded.append(token)

    return " ".join(expanded)


def _consonant_skeleton(text: str) -> str:
    """Create a consonant skeleton for quick approximate matching."""
    vowels = set("aeiouy")
    return "".join(ch for ch in text if ch not in vowels)


def _get_epitran(language: str = "fra-Latn"):
    """Get or create an epitran instance for a language."""
    if language not in _epitran_cache:
        try:
            import epitran

            _epitran_cache[language] = epitran.Epitran(language)
        except ImportError:
            return None
        except Exception as e:
            logger.debug(f"Could not create epitran for {language}: {e}")
            return None
    return _epitran_cache[language]


def _get_panphon_distance():
    """Get or create a panphon distance calculator."""
    global _panphon_distance
    if _panphon_distance is None:
        try:
            import panphon.distance

            _panphon_distance = panphon.distance.Distance()
        except ImportError:
            return None
    return _panphon_distance


def _get_panphon_ft():
    """Get or create a panphon FeatureTable."""
    global _panphon_ft
    if _panphon_ft is None:
        try:
            import panphon.featuretable

            _panphon_ft = panphon.featuretable.FeatureTable()
        except ImportError:
            return None
    return _panphon_ft


def _is_vowel(ipa: str) -> bool:
    """Simple heuristic to determine if an IPA segment contains a vowel."""
    if not ipa:
        return False
    ft = _get_panphon_ft()
    if ft and hasattr(ft, "is_vowel"):
        try:
            return ft.is_vowel(ipa)
        except Exception:
            pass
    return bool(_VOWEL_REGEX.search(ipa))


def _get_ipa(text: str, language: str = "fra-Latn") -> Optional[str]:
    """Get IPA transliteration with caching."""
    norm = _normalize_text_for_phonetic(text, language)
    cache_key = f"{language}:{norm}"
    epi = _get_epitran(language)
    if epi is None:
        return None
    if cache_key in _ipa_cache:
        return _ipa_cache[cache_key]

    try:
        ipa = epi.transliterate(norm)
    except Exception as exc:
        logger.debug(f"IPA transliteration failed for '{text}': {exc}")
        _ipa_cache[cache_key] = None
        return None
    _ipa_cache[cache_key] = ipa
    return ipa


def _get_ipa_segs(ipa: str) -> List[str]:
    """Segment IPA string into phonetic segments with caching."""
    if ipa in _ipa_segs_cache:
        return _ipa_segs_cache[ipa]

    ft = _get_panphon_ft()
    if ft is None:
        return list(ipa)  # Fallback to character segments

    try:
        segs = ft.ipa_segs(ipa)
        _ipa_segs_cache[ipa] = segs
        return segs
    except Exception:
        return list(ipa)


def _text_similarity_basic(
    text1: str, text2: str, language: Optional[str] = None
) -> float:
    """Basic text similarity using SequenceMatcher."""
    if language:
        norm1 = _normalize_text_for_phonetic(text1, language)
        norm2 = _normalize_text_for_phonetic(text2, language)
    else:
        norm1 = _normalize_text_for_matching(text1)
        norm2 = _normalize_text_for_matching(text2)

    if not norm1 or not norm2:
        return 0.0

    return SequenceMatcher(None, norm1, norm2).ratio()


def _phonetic_similarity(text1: str, text2: str, language: str = "fra-Latn") -> float:
    """Calculate phonetic similarity using epitran and panphon."""
    dst = _get_panphon_distance()

    if dst is None:
        return _text_similarity_basic(text1, text2, language)

    try:
        # Get cached IPA transliterations
        ipa1 = _get_ipa(text1, language)
        ipa2 = _get_ipa(text2, language)

        if not ipa1 or not ipa2:
            return _text_similarity_basic(text1, text2, language)

        # Get cached IPA segments
        segs1 = _get_ipa_segs(ipa1)
        segs2 = _get_ipa_segs(ipa2)

        if not segs1 or not segs2:
            return _text_similarity_basic(text1, text2, language)

        # Calculate feature edit distance (weighted Levenshtein)
        fed = dst.feature_edit_distance(ipa1, ipa2)

        # Normalize by max segment count
        max_segs = max(len(segs1), len(segs2))

        # Convert normalized distance to similarity
        normalized_distance = fed / max_segs
        similarity = max(0.0, 1.0 - normalized_distance)

        norm1 = _normalize_text_for_phonetic(text1, language)
        norm2 = _normalize_text_for_phonetic(text2, language)
        if norm1 and norm2 and norm1 == norm2:
            return max(similarity, 0.98)

        basic_sim = _text_similarity_basic(text1, text2, language)
        blended = max(similarity, basic_sim * 0.9)

        if (
            (language.startswith("eng") or language.startswith("en"))
            and " " not in norm1
            and " " not in norm2
        ):
            sk1 = _consonant_skeleton(norm1)
            sk2 = _consonant_skeleton(norm2)
            if sk1 and sk2:
                sk_sim = SequenceMatcher(None, sk1, sk2).ratio()
                blended = max(blended, sk_sim * 0.95)

        return blended

    except Exception as e:
        logger.debug(f"Phonetic similarity failed: {e}")
        return _text_similarity_basic(text1, text2, language)


def _text_similarity(
    text1: str, text2: str, use_phonetic: bool = True, language: str = "fra-Latn"
) -> float:
    """Calculate similarity between two text strings."""
    if not text1 or not text2:
        return 0.0

    if use_phonetic:
        return _phonetic_similarity(text1, text2, language)
    return _text_similarity_basic(text1, text2, language)
