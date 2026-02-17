"""Pipeline implementations for lyrics_whisper orchestration."""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

from ...models import Line, SongMetadata

LyricsFileLoadResult = Tuple[
    Optional[str], Optional[List[Tuple[float, str]]], List[str]
]
LrcFetchResult = Tuple[Optional[str], Optional[List[Tuple[float, str]]], str]


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
        if not vocals_path:
            logger.warning(
                "Whisper-only mode requires vocals; using placeholder lyrics"
            )
            return create_no_lyrics_placeholder_fn(title, artist)
        model_size = whisper_model or "base"
        transcription, _, detected_lang, _model = transcribe_vocals_for_state_fn(
            vocals_path,
            whisper_language,
            model_size,
            whisper_aggressive,
            whisper_temperature,
        )
        if not transcription:
            logger.warning(
                "No Whisper transcription available; using placeholder lyrics"
            )
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

    file_lines: List[str] = []
    file_lrc_text: Optional[str] = None
    file_line_timings: Optional[List[Tuple[float, str]]] = None
    if lyrics_file:
        file_lrc_text, file_line_timings, file_lines = load_lyrics_file_fn(
            lyrics_file, filter_promos
        )
        if file_lines or file_lrc_text:
            logger.info(f"Using lyrics from file: {lyrics_file}")

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
            logger.warning(
                "LRC duration mismatch: keeping text but ignoring LRC timings"
            )
            line_timings = None

    genius_lines: Optional[List[Tuple[str, str]]] = None
    metadata: Optional[SongMetadata] = None
    if not line_timings and not lrc_text and not file_lines:
        if offline:
            logger.warning("Offline mode: no cached lyrics available")
            return create_no_lyrics_placeholder_fn(title, artist)
        logger.debug("No LRC found, fetching lyrics from Genius...")
        genius_lines, metadata = fetch_genius_lyrics_with_singers_for_state_fn(
            title, artist
        )
        if not genius_lines:
            logger.warning("No lyrics found from any source, using placeholder")
            return create_no_lyrics_placeholder_fn(title, artist)
    else:
        genius_lines, metadata = fetch_genius_lyrics_with_singers_for_state_fn(
            title, artist
        )

    if vocals_path and line_timings:
        line_timings, _ = detect_and_apply_offset_for_state_fn(
            vocals_path, line_timings, lyrics_offset
        )

    if (
        vocals_path
        and not line_timings
        and not use_whisper
        and not whisper_only
        and not whisper_map_lrc
    ):
        use_whisper = True
        logger.info(
            "No reliable line-level timings available; auto-enabling Whisper alignment"
        )

    has_lrc_timing = bool(line_timings)
    if lrc_text or file_lines:
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

        if vocals_path and line_timings and len(line_timings) > 1:
            lines = refine_timing_with_audio_for_state_fn(
                lines, vocals_path, line_timings, lrc_text or "", target_duration
            )

        if vocals_path and use_whisper:
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
        elif vocals_path and whisper_map_lrc:
            try:
                if whisper_map_lrc_dtw:
                    model_size = whisper_model or "large"
                    lines, alignments, metrics = (
                        align_lrc_text_to_whisper_timings_for_state_fn(
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
                    )
                    logger.info(
                        f"DTW-mapped {len(alignments)} LRC line(s) onto Whisper timing"
                    )
                    if metrics:
                        logger.debug(f"DTW metrics: {metrics}")
                else:
                    model_size = whisper_model or "large"
                    transcription, _, detected_lang, _model = (
                        transcribe_vocals_for_state_fn(
                            vocals_path,
                            whisper_language,
                            model_size,
                            whisper_aggressive,
                            whisper_temperature,
                        )
                    )
                    if transcription:
                        lang = whisper_lang_to_epitran_for_state_fn(detected_lang)
                        lrc_starts = (
                            [ts for ts, _ in line_timings]
                            if line_timings and not whisper_map_lrc
                            else None
                        )
                        lines, mapped, issues = map_lrc_lines_to_whisper_segments_fn(
                            lines, transcription, lang, lrc_line_starts=lrc_starts
                        )
                        if mapped:
                            logger.info(
                                f"Mapped {mapped} LRC line(s) onto Whisper timing"
                            )
                        for issue in issues:
                            logger.debug(issue)
            except Exception as e:
                logger.warning(f"Whisper LRC mapping failed: {e}")
    else:
        if genius_lines:
            text_lines = [text for text, _ in genius_lines if text.strip()]
            lrc_text = "\n".join(text_lines)
        else:
            lrc_text = ""
        lines = create_lines_from_lrc_fn(
            lrc_text,
            romanize=romanize,
            title=title,
            artist=artist,
            filter_promos=filter_promos,
        )

    if romanize:
        romanize_lines_fn(lines)

    if metadata and metadata.is_duet and genius_lines:
        apply_singer_info_fn(lines, genius_lines, metadata)

    logger.debug(f"Returning {len(lines)} lines")
    return lines, metadata
