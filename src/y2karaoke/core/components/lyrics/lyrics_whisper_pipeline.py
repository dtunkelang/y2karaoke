"""Pipeline implementations for lyrics_whisper orchestration."""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

from ...models import Line, SongMetadata

LyricsFileLoadResult = Tuple[
    Optional[str], Optional[List[Tuple[float, str]]], List[str]
]
LrcFetchResult = Tuple[Optional[str], Optional[List[Tuple[float, str]]], str]


def _average_word_probability(transcription: list) -> Optional[float]:
    probs: list[float] = []
    for seg in transcription:
        words = getattr(seg, "words", None) or []
        for word in words:
            prob = getattr(word, "probability", None)
            if isinstance(prob, (int, float)):
                probs.append(float(prob))
    if not probs:
        return None
    return sum(probs) / len(probs)


def should_auto_enable_whisper(
    *,
    vocals_path: Optional[str],
    line_timings: Optional[List[Tuple[float, str]]],
    use_whisper: bool,
    whisper_only: bool,
    whisper_map_lrc: bool,
) -> bool:
    return (
        bool(vocals_path)
        and not line_timings
        and not use_whisper
        and not whisper_only
        and not whisper_map_lrc
    )


def should_keep_lrc_timings_for_trailing_outro_padding(
    *,
    line_timings: Optional[List[Tuple[float, str]]],
    lrc_duration: Optional[int],
    target_duration: Optional[int],
    max_outro_padding_sec: float = 20.0,
    max_outro_padding_ratio: float = 0.13,
    max_outro_padding_cap_sec: float = 32.0,
    max_lrc_tail_gap_sec: float = 8.0,
    min_lyric_span_ratio: float = 0.75,
) -> bool:
    """Return True when duration mismatch likely comes from trailing instrumental outro.

    We treat positive duration deltas as likely-outro only when:
    - audio is longer than LRC,
    - extra tail is bounded (not an arbitrary long mismatch),
    - last timed lyric appears near the LRC end.
    """
    if not line_timings or not lrc_duration or not target_duration:
        return False

    duration_delta = float(target_duration) - float(lrc_duration)
    allowed_padding = max(
        float(max_outro_padding_sec),
        float(target_duration) * float(max_outro_padding_ratio),
    )
    allowed_padding = min(allowed_padding, float(max_outro_padding_cap_sec))
    if duration_delta <= 0 or duration_delta > allowed_padding:
        return False

    lyric_starts = [float(t) for t, _ in line_timings]
    first_timed_lyric_start = min(lyric_starts)
    last_timed_lyric_start = max(lyric_starts)
    lrc_tail_gap = float(lrc_duration) - float(last_timed_lyric_start)
    if not (lrc_tail_gap >= 0 and lrc_tail_gap <= max_lrc_tail_gap_sec):
        return False

    lyric_span_ratio = 0.0
    if lrc_duration > 0:
        lyric_span_ratio = max(
            0.0, last_timed_lyric_start - first_timed_lyric_start
        ) / float(lrc_duration)
    return lyric_span_ratio >= float(min_lyric_span_ratio)


def _transcribe_with_whisper_only_retry(
    *,
    vocals_path: str,
    whisper_language: Optional[str],
    whisper_model: Optional[str],
    whisper_aggressive: bool,
    whisper_temperature: float,
    transcribe_vocals_for_state_fn: Callable[..., Tuple[list, list, str, str]],
    logger,
) -> list:
    model_size = whisper_model or "base"
    transcription, _, _detected_lang, _model = transcribe_vocals_for_state_fn(
        vocals_path,
        whisper_language,
        model_size,
        whisper_aggressive,
        whisper_temperature,
    )
    avg_prob = _average_word_probability(transcription)
    if whisper_model is None and model_size == "base" and transcription:
        if avg_prob is not None and avg_prob < 0.45:
            logger.info(
                "Low-confidence Whisper-only transcription "
                "(avg_prob=%.3f); retrying with large model",
                avg_prob,
            )
            retry_transcription, _retry_words, _retry_lang, _retry_model = (
                transcribe_vocals_for_state_fn(
                    vocals_path,
                    whisper_language,
                    "large",
                    whisper_aggressive,
                    whisper_temperature,
                )
            )
            retry_avg_prob = _average_word_probability(retry_transcription)
            if retry_transcription and (
                retry_avg_prob is None
                or avg_prob is None
                or retry_avg_prob >= avg_prob + 0.05
            ):
                logger.info("Using large-model retry for Whisper-only output")
                return retry_transcription
    return transcription


def _get_whisper_only_lines(
    *,
    title: str,
    artist: str,
    vocals_path: Optional[str],
    whisper_language: Optional[str],
    whisper_model: Optional[str],
    whisper_aggressive: bool,
    whisper_temperature: float,
    romanize: bool,
    create_no_lyrics_placeholder_fn: Callable[
        ..., Tuple[List[Line], Optional[SongMetadata]]
    ],
    transcribe_vocals_for_state_fn: Callable[..., Tuple[list, list, str, str]],
    create_lines_from_whisper_fn: Callable[..., List[Line]],
    romanize_lines_fn: Callable[..., None],
    logger,
) -> Tuple[List[Line], Optional[SongMetadata]]:
    if not vocals_path:
        logger.warning("Whisper-only mode requires vocals; using placeholder lyrics")
        return create_no_lyrics_placeholder_fn(title, artist)

    transcription = _transcribe_with_whisper_only_retry(
        vocals_path=vocals_path,
        whisper_language=whisper_language,
        whisper_model=whisper_model,
        whisper_aggressive=whisper_aggressive,
        whisper_temperature=whisper_temperature,
        transcribe_vocals_for_state_fn=transcribe_vocals_for_state_fn,
        logger=logger,
    )
    if not transcription:
        logger.warning("No Whisper transcription available; using placeholder lyrics")
        return create_no_lyrics_placeholder_fn(title, artist)

    lines = create_lines_from_whisper_fn(transcription)
    whisper_metadata = SongMetadata(
        singers=[],
        is_duet=False,
        title=title,
        artist=artist,
    )
    if romanize:
        romanize_lines_fn(lines)
    logger.debug(f"Returning {len(lines)} lines from Whisper-only mode")
    return lines, whisper_metadata


def _load_preferred_lrc_source(
    *,
    title: str,
    artist: str,
    lyrics_file: Optional[Path],
    filter_promos: bool,
    target_duration: Optional[int],
    vocals_path: Optional[str],
    evaluate_sources: bool,
    offline: bool,
    load_lyrics_file_fn: Callable[..., LyricsFileLoadResult],
    fetch_lrc_text_and_timings_for_state_fn: Callable[..., LrcFetchResult],
    get_lrc_duration_fn: Callable[..., Optional[int]],
    logger,
) -> tuple[Optional[str], Optional[List[Tuple[float, str]]], List[str]]:
    file_lines: List[str] = []
    file_lrc_text: Optional[str] = None
    file_line_timings: Optional[List[Tuple[float, str]]] = None
    if lyrics_file:
        file_lrc_text, file_line_timings, file_lines = load_lyrics_file_fn(
            lyrics_file, filter_promos
        )
        if file_lines or file_lrc_text:
            logger.info(f"Using lyrics from file: {lyrics_file}")
        if file_lines and not file_lrc_text and not file_line_timings:
            logger.debug("Plain text lyrics file provided; skipping provider LRC fetch")
            return None, None, file_lines

    logger.debug(
        f"Fetching LRC lyrics... (target_duration={target_duration}, evaluate={evaluate_sources})"
    )
    lrc_text, line_timings, _source = fetch_lrc_text_and_timings_for_state_fn(
        title=title,
        artist=artist,
        target_duration=target_duration,
        vocals_path=vocals_path,
        evaluate_sources=evaluate_sources,
        filter_promos=filter_promos,
        offline=offline,
    )
    if file_lrc_text and file_line_timings:
        lrc_text = file_lrc_text
        line_timings = file_line_timings

    if target_duration and lrc_text:
        lrc_duration = get_lrc_duration_fn(lrc_text)
        if lrc_duration and abs(target_duration - lrc_duration) > 8:
            mismatch = abs(target_duration - lrc_duration)
            likely_outro_padding = should_keep_lrc_timings_for_trailing_outro_padding(
                line_timings=line_timings,
                lrc_duration=lrc_duration,
                target_duration=target_duration,
            )
            if mismatch >= 12 and not likely_outro_padding:
                logger.warning(
                    "LRC duration mismatch: keeping text but ignoring LRC timings"
                )
                line_timings = None

    return lrc_text, line_timings, file_lines


def _resolve_genius_fallback(
    *,
    title: str,
    artist: str,
    line_timings: Optional[List[Tuple[float, str]]],
    lrc_text: Optional[str],
    file_lines: List[str],
    offline: bool,
    create_no_lyrics_placeholder_fn: Callable[
        ..., Tuple[List[Line], Optional[SongMetadata]]
    ],
    fetch_genius_lyrics_with_singers_for_state_fn: Callable[
        ..., Tuple[Optional[List[Tuple[str, str]]], Optional[SongMetadata]]
    ],
    logger,
) -> tuple[
    Optional[List[Tuple[str, str]]],
    Optional[SongMetadata],
    Optional[Tuple[List[Line], Optional[SongMetadata]]],
]:
    genius_lines: Optional[List[Tuple[str, str]]] = None
    metadata: Optional[SongMetadata] = None
    if not line_timings and not lrc_text and not file_lines:
        if offline:
            logger.warning("Offline mode: no cached lyrics available")
            return None, None, create_no_lyrics_placeholder_fn(title, artist)
        logger.debug("No LRC found, fetching lyrics from Genius...")
        genius_lines, metadata = fetch_genius_lyrics_with_singers_for_state_fn(
            title, artist
        )
        if not genius_lines:
            logger.warning("No lyrics found from any source, using placeholder")
            return None, None, create_no_lyrics_placeholder_fn(title, artist)
        return genius_lines, metadata, None

    genius_lines, metadata = fetch_genius_lyrics_with_singers_for_state_fn(
        title, artist
    )
    return genius_lines, metadata, None


def _build_lines_from_lrc_source(
    *,
    lrc_text: Optional[str],
    file_lines: List[str],
    line_timings: Optional[List[Tuple[float, str]]],
    title: str,
    artist: str,
    filter_promos: bool,
    vocals_path: Optional[str],
    target_duration: Optional[int],
    create_lines_from_lrc_timings_fn: Callable[..., List[Line]],
    create_lines_from_lrc_fn: Callable[..., List[Line]],
    apply_timing_to_lines_fn: Callable[..., None],
    extract_text_lines_from_lrc_fn: Callable[..., List[str]],
    create_lines_from_plain_text_fn: Callable[..., List[Line]],
    refine_timing_with_audio_for_state_fn: Callable[..., List[Line]],
) -> tuple[List[Line], bool]:
    has_lrc_timing = bool(line_timings)
    if line_timings and file_lines:
        lines = create_lines_from_lrc_timings_fn(line_timings, file_lines)
    elif line_timings and lrc_text:
        lines = create_lines_from_lrc_fn(
            lrc_text,
            romanize=False,
            title=title,
            artist=artist,
            filter_promos=filter_promos,
        )
        apply_timing_to_lines_fn(lines, line_timings)
    else:
        text_lines = file_lines or extract_text_lines_from_lrc_fn(lrc_text or "")
        lines = create_lines_from_plain_text_fn(text_lines)
        if target_duration:
            from .helpers import _spread_lines_across_target_duration

            lines = _spread_lines_across_target_duration(lines, target_duration)

    if vocals_path and line_timings and len(line_timings) > 1:
        lines = refine_timing_with_audio_for_state_fn(
            lines, vocals_path, line_timings, lrc_text or "", target_duration
        )
    return lines, has_lrc_timing


def _apply_audio_alignment_or_mapping(
    *,
    lines: List[Line],
    vocals_path: Optional[str],
    use_whisper: bool,
    whisper_map_lrc: bool,
    whisper_map_lrc_dtw: bool,
    whisper_language: Optional[str],
    whisper_model: Optional[str],
    whisper_force_dtw: bool,
    whisper_aggressive: bool,
    whisper_temperature: float,
    lenient_vocal_activity_threshold: float,
    lenient_activity_bonus: float,
    low_word_confidence_threshold: float,
    line_timings: Optional[List[Tuple[float, str]]],
    has_lrc_timing: bool,
    apply_whisper_alignment_for_state_fn: Callable[
        ..., Tuple[List[Line], List[str], dict]
    ],
    align_lrc_text_to_whisper_timings_for_state_fn: Callable[
        ..., Tuple[List[Line], list, dict]
    ],
    transcribe_vocals_for_state_fn: Callable[..., Tuple[list, list, str, str]],
    whisper_lang_to_epitran_for_state_fn: Callable[[str], str],
    map_lrc_lines_to_whisper_segments_fn: Callable[
        ..., Tuple[List[Line], int, List[str]]
    ],
    logger,
) -> List[Line]:
    if not vocals_path:
        return lines

    if use_whisper:
        return _apply_whisper_audio_alignment(
            lines=lines,
            vocals_path=vocals_path,
            whisper_language=whisper_language,
            whisper_model=whisper_model,
            whisper_force_dtw=whisper_force_dtw,
            whisper_aggressive=whisper_aggressive,
            whisper_temperature=whisper_temperature,
            has_lrc_timing=has_lrc_timing,
            lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
            lenient_activity_bonus=lenient_activity_bonus,
            low_word_confidence_threshold=low_word_confidence_threshold,
            apply_whisper_alignment_for_state_fn=apply_whisper_alignment_for_state_fn,
            logger=logger,
        )

    if not whisper_map_lrc:
        return lines

    try:
        if whisper_map_lrc_dtw:
            return _apply_dtw_lrc_mapping(
                lines=lines,
                vocals_path=vocals_path,
                whisper_language=whisper_language,
                whisper_model=whisper_model,
                whisper_aggressive=whisper_aggressive,
                whisper_temperature=whisper_temperature,
                lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
                lenient_activity_bonus=lenient_activity_bonus,
                low_word_confidence_threshold=low_word_confidence_threshold,
                align_lrc_text_to_whisper_timings_for_state_fn=align_lrc_text_to_whisper_timings_for_state_fn,
                logger=logger,
            )

        return _apply_segment_lrc_mapping(
            lines=lines,
            vocals_path=vocals_path,
            whisper_language=whisper_language,
            whisper_model=whisper_model,
            whisper_aggressive=whisper_aggressive,
            whisper_temperature=whisper_temperature,
            line_timings=line_timings,
            transcribe_vocals_for_state_fn=transcribe_vocals_for_state_fn,
            whisper_lang_to_epitran_for_state_fn=whisper_lang_to_epitran_for_state_fn,
            map_lrc_lines_to_whisper_segments_fn=map_lrc_lines_to_whisper_segments_fn,
            logger=logger,
        )
    except Exception as e:
        logger.warning(f"Whisper LRC mapping failed: {e}")
    return lines


def _apply_whisper_audio_alignment(
    *,
    lines: List[Line],
    vocals_path: str,
    whisper_language: Optional[str],
    whisper_model: Optional[str],
    whisper_force_dtw: bool,
    whisper_aggressive: bool,
    whisper_temperature: float,
    has_lrc_timing: bool,
    lenient_vocal_activity_threshold: float,
    lenient_activity_bonus: float,
    low_word_confidence_threshold: float,
    apply_whisper_alignment_for_state_fn: Callable[
        ..., Tuple[List[Line], List[str], dict]
    ],
    logger,
) -> List[Line]:
    try:
        lines, _, _ = apply_whisper_alignment_for_state_fn(
            lines,
            vocals_path,
            whisper_language,
            whisper_model,
            whisper_force_dtw,
            whisper_aggressive,
            whisper_temperature=whisper_temperature,
            prefer_whisper_timing_map=not has_lrc_timing,
            lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
            lenient_activity_bonus=lenient_activity_bonus,
            low_word_confidence_threshold=low_word_confidence_threshold,
        )
    except Exception as e:
        logger.warning(f"Whisper alignment failed: {e}")
    return lines


def _apply_dtw_lrc_mapping(
    *,
    lines: List[Line],
    vocals_path: str,
    whisper_language: Optional[str],
    whisper_model: Optional[str],
    whisper_aggressive: bool,
    whisper_temperature: float,
    lenient_vocal_activity_threshold: float,
    lenient_activity_bonus: float,
    low_word_confidence_threshold: float,
    align_lrc_text_to_whisper_timings_for_state_fn: Callable[
        ..., Tuple[List[Line], list, dict]
    ],
    logger,
) -> List[Line]:
    model_size = whisper_model or "large"
    lines, alignments, metrics = align_lrc_text_to_whisper_timings_for_state_fn(
        lines,
        vocals_path,
        language=whisper_language,
        model_size=model_size,
        aggressive=whisper_aggressive,
        temperature=whisper_temperature,
        lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
        lenient_activity_bonus=lenient_activity_bonus,
        low_word_confidence_threshold=low_word_confidence_threshold,
    )
    logger.info(f"DTW-mapped {len(alignments)} LRC line(s) onto Whisper timing")
    if metrics:
        logger.debug(f"DTW metrics: {metrics}")
    return lines


def _apply_segment_lrc_mapping(
    *,
    lines: List[Line],
    vocals_path: str,
    whisper_language: Optional[str],
    whisper_model: Optional[str],
    whisper_aggressive: bool,
    whisper_temperature: float,
    line_timings: Optional[List[Tuple[float, str]]],
    transcribe_vocals_for_state_fn: Callable[..., Tuple[list, list, str, str]],
    whisper_lang_to_epitran_for_state_fn: Callable[[str], str],
    map_lrc_lines_to_whisper_segments_fn: Callable[
        ..., Tuple[List[Line], int, List[str]]
    ],
    logger,
) -> List[Line]:
    model_size = whisper_model or "large"
    transcription, _, detected_lang, _model = transcribe_vocals_for_state_fn(
        vocals_path,
        whisper_language,
        model_size,
        whisper_aggressive,
        whisper_temperature,
    )
    if not transcription:
        return lines

    lang = whisper_lang_to_epitran_for_state_fn(detected_lang)
    lines, mapped, issues = map_lrc_lines_to_whisper_segments_fn(
        lines, transcription, lang, lrc_line_starts=None
    )
    if mapped:
        logger.info(f"Mapped {mapped} LRC line(s) onto Whisper timing")
    for issue in issues:
        logger.debug(issue)
    return lines


def _build_lines_from_genius_fallback(
    *,
    genius_lines: Optional[List[Tuple[str, str]]],
    title: str,
    artist: str,
    romanize: bool,
    filter_promos: bool,
    create_lines_from_lrc_fn: Callable[..., List[Line]],
) -> List[Line]:
    if genius_lines:
        text_lines = [text for text, _ in genius_lines if text.strip()]
        lrc_text = "\n".join(text_lines)
    else:
        lrc_text = ""
    return create_lines_from_lrc_fn(
        lrc_text,
        romanize=romanize,
        title=title,
        artist=artist,
        filter_promos=filter_promos,
    )


def get_lyrics_simple_impl(  # noqa: C901
    title: str,
    artist: str,
    vocals_path: Optional[str],
    lyrics_offset: Optional[float],
    romanize: bool,
    filter_promos: bool,
    target_duration: Optional[int],
    evaluate_sources: bool,
    use_whisper: bool,
    whisper_only: bool,
    whisper_map_lrc: bool,
    whisper_map_lrc_dtw: bool,
    lyrics_file: Optional[Path],
    whisper_language: Optional[str],
    whisper_model: Optional[str],
    whisper_force_dtw: bool,
    whisper_aggressive: bool,
    whisper_temperature: float,
    lenient_vocal_activity_threshold: float,
    lenient_activity_bonus: float,
    low_word_confidence_threshold: float,
    offline: bool,
    *,
    create_no_lyrics_placeholder_fn: Callable[
        ..., Tuple[List[Line], Optional[SongMetadata]]
    ],
    transcribe_vocals_for_state_fn: Callable[..., Tuple[list, list, str, str]],
    create_lines_from_whisper_fn: Callable[..., List[Line]],
    romanize_lines_fn: Callable[..., None],
    load_lyrics_file_fn: Callable[..., LyricsFileLoadResult],
    fetch_lrc_text_and_timings_for_state_fn: Callable[..., LrcFetchResult],
    get_lrc_duration_fn: Callable[..., Optional[int]],
    fetch_genius_lyrics_with_singers_for_state_fn: Callable[
        ..., Tuple[Optional[List[Tuple[str, str]]], Optional[SongMetadata]]
    ],
    detect_and_apply_offset_for_state_fn: Callable[
        ..., Tuple[List[Tuple[float, str]], float]
    ],
    create_lines_from_lrc_timings_fn: Callable[..., List[Line]],
    create_lines_from_lrc_fn: Callable[..., List[Line]],
    apply_timing_to_lines_fn: Callable[..., None],
    extract_text_lines_from_lrc_fn: Callable[..., List[str]],
    create_lines_from_plain_text_fn: Callable[..., List[Line]],
    refine_timing_with_audio_for_state_fn: Callable[..., List[Line]],
    apply_whisper_alignment_for_state_fn: Callable[
        ..., Tuple[List[Line], List[str], dict]
    ],
    align_lrc_text_to_whisper_timings_for_state_fn: Callable[
        ..., Tuple[List[Line], list, dict]
    ],
    whisper_lang_to_epitran_for_state_fn: Callable[[str], str],
    map_lrc_lines_to_whisper_segments_fn: Callable[
        ..., Tuple[List[Line], int, List[str]]
    ],
    apply_singer_info_fn: Callable[..., None],
    logger,
) -> Tuple[List[Line], Optional[SongMetadata]]:
    """Simplified lyrics pipeline favoring LRC over Genius."""
    if whisper_only:
        return _get_whisper_only_lines(
            title=title,
            artist=artist,
            vocals_path=vocals_path,
            whisper_language=whisper_language,
            whisper_model=whisper_model,
            whisper_aggressive=whisper_aggressive,
            whisper_temperature=whisper_temperature,
            romanize=romanize,
            create_no_lyrics_placeholder_fn=create_no_lyrics_placeholder_fn,
            transcribe_vocals_for_state_fn=transcribe_vocals_for_state_fn,
            create_lines_from_whisper_fn=create_lines_from_whisper_fn,
            romanize_lines_fn=romanize_lines_fn,
            logger=logger,
        )

    lrc_text, line_timings, file_lines = _load_preferred_lrc_source(
        title=title,
        artist=artist,
        lyrics_file=lyrics_file,
        filter_promos=filter_promos,
        target_duration=target_duration,
        vocals_path=vocals_path,
        evaluate_sources=evaluate_sources,
        offline=offline,
        load_lyrics_file_fn=load_lyrics_file_fn,
        fetch_lrc_text_and_timings_for_state_fn=fetch_lrc_text_and_timings_for_state_fn,
        get_lrc_duration_fn=get_lrc_duration_fn,
        logger=logger,
    )
    genius_lines, metadata, early_result = _resolve_genius_fallback(
        title=title,
        artist=artist,
        line_timings=line_timings,
        lrc_text=lrc_text,
        file_lines=file_lines,
        offline=offline,
        create_no_lyrics_placeholder_fn=create_no_lyrics_placeholder_fn,
        fetch_genius_lyrics_with_singers_for_state_fn=fetch_genius_lyrics_with_singers_for_state_fn,
        logger=logger,
    )
    if early_result is not None:
        return early_result

    if vocals_path and line_timings:
        line_timings, _ = detect_and_apply_offset_for_state_fn(
            vocals_path, line_timings, lyrics_offset
        )

    if should_auto_enable_whisper(
        vocals_path=vocals_path,
        line_timings=line_timings,
        use_whisper=use_whisper,
        whisper_only=whisper_only,
        whisper_map_lrc=whisper_map_lrc,
    ):
        use_whisper = True
        logger.info(
            "No reliable line-level timings available; auto-enabling Whisper alignment"
        )

    if lrc_text or file_lines:
        lines, has_lrc_timing = _build_lines_from_lrc_source(
            lrc_text=lrc_text,
            file_lines=file_lines,
            line_timings=line_timings,
            title=title,
            artist=artist,
            filter_promos=filter_promos,
            vocals_path=vocals_path,
            target_duration=target_duration,
            create_lines_from_lrc_timings_fn=create_lines_from_lrc_timings_fn,
            create_lines_from_lrc_fn=create_lines_from_lrc_fn,
            apply_timing_to_lines_fn=apply_timing_to_lines_fn,
            extract_text_lines_from_lrc_fn=extract_text_lines_from_lrc_fn,
            create_lines_from_plain_text_fn=create_lines_from_plain_text_fn,
            refine_timing_with_audio_for_state_fn=refine_timing_with_audio_for_state_fn,
        )
        lines = _apply_audio_alignment_or_mapping(
            lines=lines,
            vocals_path=vocals_path,
            use_whisper=use_whisper,
            whisper_map_lrc=whisper_map_lrc,
            whisper_map_lrc_dtw=whisper_map_lrc_dtw,
            whisper_language=whisper_language,
            whisper_model=whisper_model,
            whisper_force_dtw=whisper_force_dtw,
            whisper_aggressive=whisper_aggressive,
            whisper_temperature=whisper_temperature,
            lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
            lenient_activity_bonus=lenient_activity_bonus,
            low_word_confidence_threshold=low_word_confidence_threshold,
            line_timings=line_timings,
            has_lrc_timing=has_lrc_timing,
            apply_whisper_alignment_for_state_fn=apply_whisper_alignment_for_state_fn,
            align_lrc_text_to_whisper_timings_for_state_fn=align_lrc_text_to_whisper_timings_for_state_fn,
            transcribe_vocals_for_state_fn=transcribe_vocals_for_state_fn,
            whisper_lang_to_epitran_for_state_fn=whisper_lang_to_epitran_for_state_fn,
            map_lrc_lines_to_whisper_segments_fn=map_lrc_lines_to_whisper_segments_fn,
            logger=logger,
        )
    else:
        lines = _build_lines_from_genius_fallback(
            genius_lines=genius_lines,
            title=title,
            artist=artist,
            romanize=romanize,
            filter_promos=filter_promos,
            create_lines_from_lrc_fn=create_lines_from_lrc_fn,
        )

    if romanize:
        romanize_lines_fn(lines)

    if metadata and metadata.is_duet and genius_lines:
        apply_singer_info_fn(lines, genius_lines, metadata)

    logger.debug(f"Returning {len(lines)} lines")
    return lines, metadata
