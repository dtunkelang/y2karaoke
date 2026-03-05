"""Whisper transcription implementation for integration pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

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


def _normalize_whisperx_segments(
    segments: List[_WhisperxSegment],
) -> Tuple[List[_WhisperxSegment], List[_WhisperxWord]]:
    """Enforce monotonic non-overlapping WhisperX word timings."""
    normalized_segments: List[_WhisperxSegment] = []
    normalized_words: List[_WhisperxWord] = []
    prev_end: Optional[float] = None

    for seg in segments:
        seg_words: List[_WhisperxWord] = []
        for word in sorted(seg.words, key=lambda w: (float(w.start), float(w.end))):
            start = max(0.0, float(word.start))
            end = max(start + 0.01, float(word.end))
            if prev_end is not None and start < prev_end:
                shift = prev_end - start
                start += shift
                end += shift
            adjusted = _WhisperxWord(
                start=start,
                end=end,
                text=word.text,
                probability=word.probability,
            )
            seg_words.append(adjusted)
            normalized_words.append(adjusted)
            prev_end = adjusted.end
        if not seg_words:
            continue
        normalized_segments.append(
            _WhisperxSegment(
                start=seg_words[0].start,
                end=seg_words[-1].end,
                text=seg.text,
                words=seg_words,
            )
        )

    return normalized_segments, normalized_words


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
        aligned, detected_lang = _run_whisperx_alignment(
            whisperx=whisperx,
            vocals_path=vocals_path,
            language=language,
            model_name=model_name,
        )
    except Exception as exc:
        logger.debug("whisperx fallback transcription failed: %s", exc)
        return None

    segments, words = _extract_whisperx_segments_and_words(aligned.get("segments", []))

    if not words:
        return None

    logger.info(
        "whisperx fallback produced %d segments, %d words (language: %s)",
        len(segments),
        len(words),
        detected_lang,
    )
    return segments, words, str(detected_lang)


def _run_whisperx_alignment(
    *,
    whisperx: Any,
    vocals_path: str,
    language: Optional[str],
    model_name: str,
) -> tuple[Dict[str, Any], str]:
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
    return aligned, str(detected_lang)


def _extract_whisperx_word(raw_word: Dict[str, Any]) -> _WhisperxWord | None:
    ws = raw_word.get("start")
    we = raw_word.get("end")
    if not isinstance(ws, (int, float)) or not isinstance(we, (int, float)):
        return None
    token = str(raw_word.get("word") or raw_word.get("text") or "").strip()
    if not token:
        return None
    prob = float(raw_word.get("score", 1.0) or 1.0)
    return _WhisperxWord(
        start=float(ws),
        end=float(we),
        text=token,
        probability=prob,
    )


def _extract_whisperx_segments_and_words(
    raw_segments: List[Dict[str, Any]],
) -> tuple[List[_WhisperxSegment], List[_WhisperxWord]]:
    segments: List[_WhisperxSegment] = []
    words: List[_WhisperxWord] = []
    for seg in raw_segments:
        seg_words = [
            tw
            for tw in (
                _extract_whisperx_word(cast(Dict[str, Any], w))
                for w in seg.get("words", [])
            )
            if tw is not None
        ]
        if not seg_words:
            continue
        words.extend(seg_words)
        seg_text = str(seg.get("text") or " ".join(w.text for w in seg_words)).strip()
        segments.append(
            _WhisperxSegment(
                start=float(seg_words[0].start),
                end=float(seg_words[-1].end),
                text=seg_text,
                words=seg_words,
            )
        )
    return segments, words


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
    wx_segments, wx_words = _normalize_whisperx_segments(wx_segments)
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


def _load_cached_or_upgraded_transcription(
    *,
    vocals_path: str,
    language: Optional[str],
    model_size: str,
    aggressive: bool,
    temperature: float,
    cache_path: Optional[str],
    find_best_cached_whisper_model_fn: Callable[..., Optional[Tuple[str, str]]],
    load_whisper_cache_fn: Callable[..., Optional[Tuple[Any, Any, str]]],
    save_whisper_cache_fn: Callable[..., None],
    logger,
) -> Optional[Tuple[List[Any], List[Any], str, str]]:
    if not cache_path:
        return None
    cached_path = cache_path
    cached_model = model_size
    best_cached = find_best_cached_whisper_model_fn(
        vocals_path, language, aggressive, model_size, temperature
    )
    if best_cached:
        cached_path, cached_model = best_cached
    cached = load_whisper_cache_fn(cached_path)
    if not cached:
        return None
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
            cached_path,
            segments,
            all_words,
            detected_lang,
            cached_model,
            aggressive,
            temperature,
        )
    return segments, all_words, detected_lang, cached_model


@dataclass(frozen=True)
class _WhisperTranscriptionConfig:
    use_vad_filter: bool
    no_speech_threshold: float | None
    log_prob_threshold: float | None


def _default_transcription_config(*, aggressive: bool) -> _WhisperTranscriptionConfig:
    if aggressive:
        return _WhisperTranscriptionConfig(
            use_vad_filter=False,
            no_speech_threshold=1.0,
            log_prob_threshold=-2.0,
        )
    return _WhisperTranscriptionConfig(
        use_vad_filter=True,
        no_speech_threshold=None,
        log_prob_threshold=None,
    )


def _run_whisper_transcription(
    *,
    model,
    vocals_path: str,
    language: Optional[str],
    aggressive: bool,
    temperature: float,
) -> Tuple[Any, Any]:
    config = _default_transcription_config(aggressive=aggressive)
    transcribe_kwargs: Dict[str, object] = {
        "language": language,
        "word_timestamps": True,
        "vad_filter": config.use_vad_filter,
        "temperature": temperature,
    }
    if config.no_speech_threshold is not None:
        transcribe_kwargs["no_speech_threshold"] = config.no_speech_threshold
    if config.log_prob_threshold is not None:
        transcribe_kwargs["log_prob_threshold"] = config.log_prob_threshold
    return model.transcribe(vocals_path, **transcribe_kwargs)


def _convert_whisper_segments(
    segments,
) -> Tuple[
    List[timing_models.TranscriptionSegment], List[timing_models.TranscriptionWord]
]:
    result: List[timing_models.TranscriptionSegment] = []
    all_words: List[timing_models.TranscriptionWord] = []
    for seg in segments:
        seg_words: List[timing_models.TranscriptionWord] = []
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
    return result, all_words


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
    cached_result = _load_cached_or_upgraded_transcription(
        vocals_path=vocals_path,
        language=language,
        model_size=model_size,
        aggressive=aggressive,
        temperature=temperature,
        cache_path=cache_path,
        find_best_cached_whisper_model_fn=find_best_cached_whisper_model_fn,
        load_whisper_cache_fn=load_whisper_cache_fn,
        save_whisper_cache_fn=save_whisper_cache_fn,
        logger=logger,
    )
    if cached_result is not None:
        return cached_result

    try:
        whisper_model_class = load_whisper_model_class_fn()
    except ImportError:
        logger.warning("faster-whisper not installed, cannot transcribe")
        return [], [], "", model_size

    try:
        logger.info(f"Loading Whisper model ({model_size})...")
        model = whisper_model_class(model_size, device="cpu", compute_type="int8")

        logger.info(f"Transcribing vocals{f' in {language}' if language else ''}...")
        segments, info = _run_whisper_transcription(
            model=model,
            vocals_path=vocals_path,
            language=language,
            aggressive=aggressive,
            temperature=temperature,
        )
        result, all_words = _convert_whisper_segments(segments)

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
