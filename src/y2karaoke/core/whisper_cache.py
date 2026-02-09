"""Caching logic for Whisper transcription results."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from .timing_models import TranscriptionSegment, TranscriptionWord

logger = logging.getLogger(__name__)

_MODEL_ORDER = ["tiny", "base", "small", "medium", "large"]


def _model_index(model_size: str) -> int:
    try:
        return _MODEL_ORDER.index(model_size)
    except ValueError:
        return -1


def _get_whisper_cache_path(
    vocals_path: str,
    model_size: str,
    language: Optional[str],
    aggressive: bool = False,
    temperature: float = 0.0,
) -> Optional[str]:
    """Get the cache file path for Whisper transcription results."""
    vocals_file = Path(vocals_path)
    if not vocals_file.exists():
        return None

    # Create cache filename with model and language info
    lang_suffix = f"_{language}" if language else "_auto"
    mode_suffix = "_aggr" if aggressive else ""
    temp_suffix = f"_temp{temperature}" if temperature > 0 else ""
    cache_name = f"{vocals_file.stem}_whisper_{model_size}{lang_suffix}{mode_suffix}{temp_suffix}.json"
    return str(vocals_file.parent / cache_name)


def _find_best_cached_whisper_model(  # noqa: C901
    vocals_path: str,
    language: Optional[str],
    aggressive: bool,
    target_model: str,
    temperature: float = 0.0,
) -> Optional[Tuple[str, str]]:
    """Return the cache path and model name for the best cached model ≥ target."""
    vocals_file = Path(vocals_path)
    if not vocals_file.exists():
        return None

    lang_token = language or "auto"
    mode_suffix = "_aggr" if aggressive else ""
    best_idx = _model_index(target_model)
    best_path = None
    best_model = None
    best_lang_exact = False  # prefer exact language matches
    best_temp_exact = False

    pattern = f"{vocals_file.stem}_whisper_*{mode_suffix}.json"
    for cache_file in vocals_file.parent.glob(pattern):
        stem = cache_file.stem
        if "_whisper_" not in stem:
            continue
        tail = stem.split("_whisper_", 1)[1]
        if tail.endswith("_aggr"):
            tail = tail[: -len("_aggr")]

        # Parse temperature from filename if present
        file_temp = 0.0
        if "_temp" in tail:
            temp_parts = tail.split("_temp")
            try:
                file_temp = float(temp_parts[1])
            except (ValueError, IndexError):
                pass
            tail = temp_parts[0]

        if "_" not in tail:
            continue
        model_part, lang_part = tail.rsplit("_", 1)

        # Language matching
        lang_exact = lang_part == lang_token
        if not lang_exact:
            if language is None:
                pass
            elif lang_part == "auto":
                pass
            else:
                continue

        # Temperature matching
        temp_exact = abs(file_temp - temperature) < 1e-6
        if not temp_exact and file_temp != 0.0:
            continue

        model_idx = _model_index(model_part)
        if model_idx < 0:
            continue

        better = False
        if not best_path:
            better = True
        else:
            if temp_exact and not best_temp_exact:
                better = True
            elif temp_exact == best_temp_exact:
                if model_idx > best_idx:
                    better = True
                elif model_idx == best_idx:
                    if lang_exact and not best_lang_exact:
                        better = True

        if better:
            best_idx = model_idx
            best_path = cache_file
            best_model = model_part
            best_lang_exact = lang_exact
            best_temp_exact = temp_exact

    if best_path and best_model:
        return str(best_path), best_model
    return None


def _load_whisper_cache(
    cache_path: str,
) -> Optional[Tuple[List[TranscriptionSegment], List[TranscriptionWord], str]]:
    """Load cached Whisper transcription if available."""
    cache_file = Path(cache_path)
    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "r") as f:
            data = json.load(f)

        segments = []
        all_words = []
        for seg_data in data.get("segments", []):
            seg_words = []
            for w_data in seg_data.get("words", []):
                tw = TranscriptionWord(
                    start=w_data["start"],
                    end=w_data["end"],
                    text=w_data["text"],
                    probability=w_data.get("probability", 1.0),
                )
                seg_words.append(tw)
                all_words.append(tw)
            segments.append(
                TranscriptionSegment(
                    start=seg_data["start"],
                    end=seg_data["end"],
                    text=seg_data["text"],
                    words=seg_words,
                )
            )

        detected_lang = data.get("language", "")
        cached_model = data.get("model_size", "unknown")
        logger.info(
            f"Loaded cached Whisper transcription ({cached_model}): "
            f"{len(segments)} segments, {len(all_words)} words"
        )
        return segments, all_words, detected_lang

    except Exception as e:
        logger.debug(f"Failed to load Whisper cache: {e}")
        return None


def _save_whisper_cache(
    cache_path: str,
    segments: List[TranscriptionSegment],
    all_words: List[TranscriptionWord],
    language: str,
    model_size: str,
    aggressive: bool,
    temperature: float = 0.0,
) -> None:
    """Save Whisper transcription to cache."""
    try:
        data = {
            "language": language,
            "model_size": model_size,
            "aggressive": aggressive,
            "temperature": temperature,
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "words": [
                        {
                            "start": w.start,
                            "end": w.end,
                            "text": w.text,
                            "probability": w.probability,
                        }
                        for w in (seg.words or [])
                    ],
                }
                for seg in segments
            ],
        }

        with open(cache_path, "w") as f:
            json.dump(data, f)

        logger.debug(f"Saved Whisper transcription to cache: {cache_path}")

    except Exception as e:
        logger.debug(f"Failed to save Whisper cache: {e}")
