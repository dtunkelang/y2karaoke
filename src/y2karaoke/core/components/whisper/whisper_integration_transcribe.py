"""Whisper transcription implementation for integration pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..alignment import timing_models
from .whisperx_compat import patch_torchaudio_for_whisperx


@dataclass
class _WhisperxWord:
    start: float
    end: float
    text: str
    probability: float


@dataclass
class _WhisperxSegment:
    start: float
    end: float
    text: str
    words: List[_WhisperxWord]


def _transcribe_with_whisperx(
    *,
    vocals_path: str,
    language: Optional[str],
    model_size: str,
    temperature: float,
    logger,
) -> Optional[Tuple[List[_WhisperxSegment], List[_WhisperxWord], str]]:
    patch_torchaudio_for_whisperx()
    try:
        import whisperx  # type: ignore
    except Exception:
        return None

    model_name = model_size if model_size != "large" else "large-v2"
    try:
        audio = whisperx.load_audio(vocals_path)
        model = whisperx.load_model(
            model_name,
            device="cpu",
            compute_type="int8",
            language=language,
        )
        transcribed = model.transcribe(audio, batch_size=4)
        detected_lang = transcribed.get("language") or (language or "en")
        align_model, metadata = whisperx.load_align_model(
            language_code=detected_lang,
            device="cpu",
        )
        aligned = whisperx.align(
            transcribed["segments"],
            align_model,
            metadata,
            audio,
            device="cpu",
            return_char_alignments=False,
        )
    except Exception as exc:
        logger.debug("whisperx fallback transcription failed: %s", exc)
        return None

    raw_segments = aligned.get("segments", [])
    segments: List[_WhisperxSegment] = []
    words: List[_WhisperxWord] = []
    for seg in raw_segments:
        seg_words: List[_WhisperxWord] = []
        for w in seg.get("words", []):
            ws = w.get("start")
            we = w.get("end")
            if not isinstance(ws, (int, float)) or not isinstance(we, (int, float)):
                continue
            token = str(w.get("word") or w.get("text") or "").strip()
            if not token:
                continue
            prob = float(w.get("score", 1.0) or 1.0)
            tw = _WhisperxWord(
                start=float(ws),
                end=float(we),
                text=token,
                probability=prob,
            )
            seg_words.append(tw)
            words.append(tw)
        if not seg_words:
            continue
        seg_start = float(seg_words[0].start)
        seg_end = float(seg_words[-1].end)
        seg_text = str(seg.get("text") or " ".join(w.text for w in seg_words)).strip()
        segments.append(
            _WhisperxSegment(
                start=seg_start,
                end=seg_end,
                text=seg_text,
                words=seg_words,
            )
        )

    if not words:
        return None

    logger.info(
        "whisperx fallback produced %d segments, %d words (language: %s)",
        len(segments),
        len(words),
        detected_lang,
    )
    return segments, words, str(detected_lang)


def _maybe_upgrade_sparse_transcription_with_whisperx(
    *,
    result: List[timing_models.TranscriptionSegment],
    all_words: List[timing_models.TranscriptionWord],
    detected_lang: str,
    vocals_path: str,
    language: Optional[str],
    model_size: str,
    temperature: float,
    logger,
) -> Tuple[
    List[timing_models.TranscriptionSegment],
    List[timing_models.TranscriptionWord],
    str,
    bool,
]:
    sparse_word_output = len(all_words) < 80 and len(result) <= 8
    if not sparse_word_output:
        return result, all_words, detected_lang, False

    wx = _transcribe_with_whisperx(
        vocals_path=vocals_path,
        language=language or detected_lang,
        model_size=model_size,
        temperature=temperature,
        logger=logger,
    )
    if wx is None:
        return result, all_words, detected_lang, False

    wx_segments, wx_words, wx_lang = wx
    if len(wx_words) < max(120, len(all_words) * 2):
        return result, all_words, detected_lang, False
    if not _should_accept_whisperx_upgrade(
        base_segments=result,
        base_words=all_words,
        upgraded_segments=wx_segments,
        upgraded_words=wx_words,
        logger=logger,
    ):
        return result, all_words, detected_lang, False

    upgraded_segments = [
        timing_models.TranscriptionSegment(
            start=s.start,
            end=s.end,
            text=s.text,
            words=[
                timing_models.TranscriptionWord(
                    start=w.start,
                    end=w.end,
                    text=w.text.strip(),
                    probability=w.probability,
                )
                for w in s.words
            ],
        )
        for s in wx_segments
    ]
    upgraded_words = [
        timing_models.TranscriptionWord(
            start=w.start,
            end=w.end,
            text=w.text.strip(),
            probability=w.probability,
        )
        for w in wx_words
    ]
    logger.info(
        "Using whisperx fallback transcript due to sparse faster-whisper output"
    )
    return upgraded_segments, upgraded_words, wx_lang, True


def _should_accept_whisperx_upgrade(
    *,
    base_segments: List[timing_models.TranscriptionSegment],
    base_words: List[timing_models.TranscriptionWord],
    upgraded_segments: List[_WhisperxSegment],
    upgraded_words: List[_WhisperxWord],
    logger,
) -> bool:
    """Guard WhisperX upgrades with basic timing-shape sanity checks."""
    if not upgraded_words:
        return False

    durations = [max(0.0, float(w.end) - float(w.start)) for w in upgraded_words]
    if not durations:
        return False
    overlaps = 0
    prev_end = None
    for w in upgraded_words:
        if prev_end is not None and float(w.start) < prev_end - 0.02:
            overlaps += 1
        prev_end = float(w.end)
    p95 = (
        statistics.quantiles(durations, n=20)[18]
        if len(durations) >= 20
        else max(durations)
    )
    median = statistics.median(durations)
    too_short = sum(1 for d in durations if d < 0.03)
    short_ratio = too_short / len(durations)

    if overlaps > max(3, int(len(upgraded_words) * 0.03)):
        logger.debug(
            "Rejected whisperx fallback: excessive overlaps (%d/%d)",
            overlaps,
            len(upgraded_words),
        )
        return False
    if p95 > 4.5 or median < 0.03 or short_ratio > 0.2:
        logger.debug(
            "Rejected whisperx fallback: bad duration shape (median=%.3f, p95=%.3f, short_ratio=%.2f)",
            median,
            p95,
            short_ratio,
        )
        return False

    base_end = max((float(w.end) for w in base_words), default=0.0)
    upgraded_end = max((float(w.end) for w in upgraded_words), default=0.0)
    if base_end > 0.0 and upgraded_end < base_end * 0.9:
        logger.debug(
            "Rejected whisperx fallback: shorter transcript span (base=%.2f, upgraded=%.2f)",
            base_end,
            upgraded_end,
        )
        return False

    if len(upgraded_segments) < max(6, len(base_segments)):
        logger.debug(
            "Rejected whisperx fallback: insufficient segment coverage (%d < %d)",
            len(upgraded_segments),
            max(6, len(base_segments)),
        )
        return False
    return True


def transcribe_vocals_impl(
    vocals_path: str,
    language: Optional[str],
    model_size: str,
    aggressive: bool,
    temperature: float,
    *,
    get_whisper_cache_path_fn: Callable[..., Optional[str]],
    find_best_cached_whisper_model_fn: Callable[..., Optional[Tuple[str, str]]],
    load_whisper_cache_fn: Callable[..., Optional[Tuple[Any, Any, str]]],
    save_whisper_cache_fn: Callable[..., None],
    load_whisper_model_class_fn: Callable[[], Any],
    logger,
) -> Tuple[
    List[timing_models.TranscriptionSegment],
    List[timing_models.TranscriptionWord],
    str,
    str,
]:
    """Transcribe vocals with disk cache and return timing models."""
    cache_path = get_whisper_cache_path_fn(
        vocals_path, model_size, language, aggressive, temperature
    )
    cached_model = model_size
    if cache_path:
        best_cached = find_best_cached_whisper_model_fn(
            vocals_path, language, aggressive, model_size, temperature
        )
        if best_cached:
            cache_path, cached_model = best_cached
        cached = load_whisper_cache_fn(cache_path)
        if cached:
            segments, all_words, detected_lang = cached
            segments, all_words, detected_lang, upgraded = (
                _maybe_upgrade_sparse_transcription_with_whisperx(
                    result=segments,
                    all_words=all_words,
                    detected_lang=detected_lang,
                    vocals_path=vocals_path,
                    language=language,
                    model_size=model_size,
                    temperature=temperature,
                    logger=logger,
                )
            )
            if upgraded:
                save_whisper_cache_fn(
                    cache_path,
                    segments,
                    all_words,
                    detected_lang,
                    cached_model,
                    aggressive,
                    temperature,
                )
            return segments, all_words, detected_lang, cached_model

    try:
        whisper_model_class = load_whisper_model_class_fn()
    except ImportError:
        logger.warning("faster-whisper not installed, cannot transcribe")
        return [], [], "", model_size

    try:
        logger.info(f"Loading Whisper model ({model_size})...")
        model = whisper_model_class(model_size, device="cpu", compute_type="int8")

        logger.info(f"Transcribing vocals{f' in {language}' if language else ''}...")
        transcribe_kwargs: Dict[str, object] = {
            "language": language,
            "word_timestamps": True,
            "vad_filter": True,
            "temperature": temperature,
        }
        if aggressive:
            transcribe_kwargs.update(
                {
                    "vad_filter": False,
                    "no_speech_threshold": 1.0,
                    "log_prob_threshold": -2.0,
                }
            )
        segments, info = model.transcribe(vocals_path, **transcribe_kwargs)

        result = []
        all_words = []
        for seg in segments:
            seg_words = []
            if seg.words:
                for w in seg.words:
                    tw = timing_models.TranscriptionWord(
                        start=w.start,
                        end=w.end,
                        text=w.word.strip(),
                        probability=w.probability,
                    )
                    seg_words.append(tw)
                    all_words.append(tw)
            result.append(
                timing_models.TranscriptionSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    words=seg_words,
                )
            )

        detected_lang = info.language
        result, all_words, detected_lang, _upgraded = (
            _maybe_upgrade_sparse_transcription_with_whisperx(
                result=result,
                all_words=all_words,
                detected_lang=detected_lang,
                vocals_path=vocals_path,
                language=language,
                model_size=model_size,
                temperature=temperature,
                logger=logger,
            )
        )
        logger.info(
            "Transcribed %d segments, %d words (language: %s)",
            len(result),
            len(all_words),
            detected_lang,
        )

        if cache_path:
            save_whisper_cache_fn(
                cache_path,
                result,
                all_words,
                detected_lang,
                model_size,
                aggressive,
                temperature,
            )

        return result, all_words, detected_lang, model_size

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return [], [], "", model_size
