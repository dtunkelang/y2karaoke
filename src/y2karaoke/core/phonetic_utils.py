"""Phonetic and text similarity utilities for lyrics alignment."""

import atexit
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import unicodedata
from typing import Callable, Dict, Iterator, List, Optional, Any
from difflib import SequenceMatcher

from ..config import get_cache_dir
from ..utils.logging import get_logger

logger = get_logger(__name__)

_VOWEL_REGEX = re.compile("[aeiouæøɪʊɔɛɑɐəœɜɞʌɒɨɯʉøɤɚɝ]", re.IGNORECASE)

# Cache for phonetic instances (they're expensive to create)
_epitran_cache: Dict[str, Any] = {}
_panphon_distance = None
_panphon_ft = None
_ipa_cache: Dict[str, Optional[str]] = {}  # Cache for IPA transliterations
_ipa_segs_cache: Dict[str, List[str]] = {}  # Cache for IPA segments
_ipa_disk_cache_loaded = False
_ipa_disk_cache_dirty = 0
_IPA_DISK_FLUSH_THRESHOLD = 64
_IPA_DISK_CACHE_MAX_ENTRIES = 25000


def _ipa_disk_cache_enabled() -> bool:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    setting = os.getenv("Y2KARAOKE_PHONETIC_DISK_CACHE", "1").strip().lower()
    return setting not in {"0", "false", "no", "off"}


def _ipa_disk_cache_path() -> Path:
    return get_cache_dir() / "phonetic_ipa_cache.json"


def _load_ipa_cache_from_disk() -> None:
    global _ipa_disk_cache_loaded
    if _ipa_disk_cache_loaded or not _ipa_disk_cache_enabled():
        return
    _ipa_disk_cache_loaded = True
    path = _ipa_disk_cache_path()
    try:
        if not path.exists():
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug(f"Failed to load phonetic IPA disk cache: {exc}")
        return
    if not isinstance(raw, dict):
        return
    for key, value in raw.items():
        if isinstance(key, str) and (isinstance(value, str) or value is None):
            _ipa_cache[key] = value


def _flush_ipa_cache_to_disk(*, force: bool = False) -> None:
    global _ipa_disk_cache_dirty
    if not _ipa_disk_cache_enabled():
        return
    if not force and _ipa_disk_cache_dirty < _IPA_DISK_FLUSH_THRESHOLD:
        return
    try:
        path = _ipa_disk_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            key: value for key, value in _ipa_cache.items() if isinstance(value, str)
        }
        if len(serializable) > _IPA_DISK_CACHE_MAX_ENTRIES:
            keep = len(serializable) - _IPA_DISK_CACHE_MAX_ENTRIES
            for key in list(serializable.keys())[:keep]:
                del serializable[key]
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(serializable), encoding="utf-8")
        tmp_path.replace(path)
        _ipa_disk_cache_dirty = 0
    except Exception as exc:
        logger.debug(f"Failed to persist phonetic IPA disk cache: {exc}")


atexit.register(lambda: _flush_ipa_cache_to_disk(force=True))


@contextmanager
def use_phonetic_utils_hooks(
    *,
    get_epitran_fn: Optional[Callable[..., Any]] = None,
    get_panphon_distance_fn: Optional[Callable[..., Any]] = None,
    get_panphon_ft_fn: Optional[Callable[..., Any]] = None,
    get_ipa_fn: Optional[Callable[..., Optional[str]]] = None,
    text_similarity_basic_fn: Optional[Callable[..., float]] = None,
    phonetic_similarity_fn: Optional[Callable[..., float]] = None,
) -> Iterator[None]:
    """Temporarily override phonetic utility collaborators for tests."""
    global _get_epitran, _get_panphon_distance, _get_panphon_ft
    global _get_ipa, _text_similarity_basic, _phonetic_similarity

    prev_get_epitran = _get_epitran
    prev_get_panphon_distance = _get_panphon_distance
    prev_get_panphon_ft = _get_panphon_ft
    prev_get_ipa = _get_ipa
    prev_text_similarity_basic = _text_similarity_basic
    prev_phonetic_similarity = _phonetic_similarity

    if get_epitran_fn is not None:
        _get_epitran = get_epitran_fn
    if get_panphon_distance_fn is not None:
        _get_panphon_distance = get_panphon_distance_fn
    if get_panphon_ft_fn is not None:
        _get_panphon_ft = get_panphon_ft_fn
    if get_ipa_fn is not None:
        _get_ipa = get_ipa_fn
    if text_similarity_basic_fn is not None:
        _text_similarity_basic = text_similarity_basic_fn
    if phonetic_similarity_fn is not None:
        _phonetic_similarity = phonetic_similarity_fn
    try:
        yield
    finally:
        _get_epitran = prev_get_epitran
        _get_panphon_distance = prev_get_panphon_distance
        _get_panphon_ft = prev_get_panphon_ft
        _get_ipa = prev_get_ipa
        _text_similarity_basic = prev_text_similarity_basic
        _phonetic_similarity = prev_phonetic_similarity


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
    _load_ipa_cache_from_disk()
    norm = _normalize_text_for_phonetic(text, language)
    cache_key = f"{language}:{norm}"
    if cache_key in _ipa_cache:
        return _ipa_cache[cache_key]
    epi = _get_epitran(language)
    if epi is None:
        return None

    def _set_cache_value(key: str, value: Optional[str]) -> Optional[str]:
        global _ipa_disk_cache_dirty
        if _ipa_cache.get(key) != value:
            _ipa_cache[key] = value
            _ipa_disk_cache_dirty += 1
            _flush_ipa_cache_to_disk()
        else:
            _ipa_cache[key] = value
        return value

    def _transliterate_cached(piece: str) -> Optional[str]:
        piece_key = f"{language}:{piece}"
        if piece_key in _ipa_cache:
            return _ipa_cache[piece_key]
        try:
            piece_ipa = epi.transliterate(piece)
        except Exception as exc:
            logger.debug(f"IPA transliteration failed for '{piece}': {exc}")
            return _set_cache_value(piece_key, None)
        return _set_cache_value(piece_key, piece_ipa)

    try:
        ipa: Optional[str]
        # English transliteration often receives many repeated token combinations
        # across line/segment comparisons; token-level caching keeps subprocess
        # calls bounded by vocabulary size instead of phrase permutations.
        if (language.startswith("eng") or language.startswith("en")) and " " in norm:
            token_ipas: List[str] = []
            for token in norm.split():
                token_ipa = _transliterate_cached(token)
                if token_ipa is None:
                    return _set_cache_value(cache_key, None)
                token_ipas.append(token_ipa)
            ipa = " ".join(token_ipas)
        else:
            ipa = _transliterate_cached(norm)
            if ipa is None:
                return _set_cache_value(cache_key, None)
    except Exception as exc:
        logger.debug(f"IPA transliteration failed for '{text}': {exc}")
        return _set_cache_value(cache_key, None)
    return _set_cache_value(cache_key, ipa)


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
    norm1 = _normalize_text_for_phonetic(text1, language)
    norm2 = _normalize_text_for_phonetic(text2, language)

    if not norm1 or not norm2:
        return 0.0

    if norm1 == norm2:
        return 1.0

    basic_sim = _text_similarity_basic(text1, text2, language)
    if basic_sim <= 0.12:
        return basic_sim

    is_english = language.startswith("eng") or language.startswith("en")
    if is_english and " " not in norm1 and " " not in norm2:
        if basic_sim >= 0.93:
            return max(basic_sim, 0.95)
        if basic_sim <= 0.30 and norm1[0] != norm2[0]:
            return basic_sim

    dst = _get_panphon_distance()

    if dst is None:
        return basic_sim

    try:
        # Get cached IPA transliterations
        ipa1 = _get_ipa(text1, language)
        ipa2 = _get_ipa(text2, language)

        if not ipa1 or not ipa2:
            return basic_sim

        # Get cached IPA segments
        segs1 = _get_ipa_segs(ipa1)
        segs2 = _get_ipa_segs(ipa2)

        if not segs1 or not segs2:
            return basic_sim

        # Calculate feature edit distance (weighted Levenshtein)
        fed = dst.feature_edit_distance(ipa1, ipa2)

        # Normalize by max segment count
        max_segs = max(len(segs1), len(segs2))

        # Convert normalized distance to similarity
        normalized_distance = fed / max_segs
        similarity = max(0.0, 1.0 - normalized_distance)

        blended = max(similarity, basic_sim * 0.9)

        if is_english and " " not in norm1 and " " not in norm2:
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
