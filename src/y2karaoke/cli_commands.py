"""Execution helpers for CLI commands."""

import sys
from dataclasses import replace
from pathlib import Path

import click

from .config import get_cache_dir
from .core.karaoke import KaraokeGenerator
from .exceptions import Y2KaraokeError
from .pipeline.identify import TrackIdentifier
from .utils.validation import (
    validate_key_shift,
    validate_offset,
    validate_output_path,
    validate_tempo,
    validate_youtube_url,
)


def _resolve_input_query(
    *,
    logger,
    url_or_query,
    artist,
    title,
    resolve_url_or_query_fn,
):
    if not url_or_query:
        resolved = resolve_url_or_query_fn(url_or_query, artist, title)
        logger.info(f"No url_or_query provided; using title for search: {resolved}")
        return resolved
    return resolve_url_or_query_fn(url_or_query, artist, title)


def _identify_track_info(
    *,
    logger,
    identifier,
    url_or_query,
    artist,
    title,
    offline,
    identify_track_fn,
    identify_track_offline_fn,
):
    if not offline:
        return identify_track_fn(logger, identifier, url_or_query, artist, title)
    if not url_or_query.startswith("http"):
        raise click.BadParameter(
            "Offline mode requires a YouTube URL (search is not available)."
        )
    return identify_track_offline_fn(logger, identifier, url_or_query, artist, title)


def _build_generate_kwargs(
    *,
    url,
    output_path,
    offset,
    key,
    tempo,
    audio_start,
    audio_duration,
    title,
    artist,
    track_info,
    lyrics_offset,
    backgrounds,
    force,
    video_settings,
    url_or_query,
    target_duration,
    evaluate_lyrics,
    whisper,
    whisper_only,
    whisper_map_lrc,
    whisper_map_lrc_dtw,
    lyrics_file,
    drop_lrc_line_timings,
    whisper_language,
    whisper_model,
    whisper_force_dtw,
    whisper_aggressive,
    whisper_temperature,
    lenient_vocal_activity_threshold,
    lenient_activity_bonus,
    low_word_confidence_threshold,
    outro_line,
    offline,
    filter_promos,
    effective_shorten_breaks,
    max_break,
    debug_audio,
    no_render,
    skip_separation,
    timing_report,
):
    return {
        "url": url,
        "output_path": output_path,
        "offset": offset,
        "key_shift": key,
        "tempo_multiplier": tempo,
        "audio_start": audio_start,
        "audio_duration": audio_duration,
        "lyrics_title": title or track_info.title,
        "lyrics_artist": artist or track_info.artist,
        "lyrics_offset": lyrics_offset,
        "use_backgrounds": backgrounds,
        "force_reprocess": force,
        "video_settings": video_settings,
        "original_prompt": url_or_query,
        "target_duration": target_duration,
        "evaluate_lyrics_sources": evaluate_lyrics,
        "use_whisper": whisper,
        "whisper_only": whisper_only,
        "whisper_map_lrc": whisper_map_lrc,
        "whisper_map_lrc_dtw": whisper_map_lrc_dtw,
        "lyrics_file": lyrics_file,
        "drop_lrc_line_timings": drop_lrc_line_timings,
        "whisper_language": whisper_language,
        "whisper_model": whisper_model,
        "whisper_force_dtw": whisper_force_dtw,
        "whisper_aggressive": whisper_aggressive,
        "whisper_temperature": whisper_temperature,
        "lenient_vocal_activity_threshold": lenient_vocal_activity_threshold,
        "lenient_activity_bonus": lenient_activity_bonus,
        "low_word_confidence_threshold": low_word_confidence_threshold,
        "outro_line": outro_line,
        "offline": offline,
        "filter_promos": filter_promos,
        "shorten_breaks": effective_shorten_breaks,
        "max_break_duration": max_break,
        "debug_audio": debug_audio,
        "skip_render": no_render,
        "skip_separation": skip_separation,
        "timing_report_path": timing_report,
    }


def _validate_generate_probe_options(
    *,
    audio_start: float,
    audio_duration: float | None,
    skip_separation: bool,
    no_render: bool,
    shorten_breaks: bool,
) -> None:
    if audio_start < 0:
        raise click.BadParameter("--audio-start must be non-negative")
    if audio_duration is not None and audio_duration <= 0:
        raise click.BadParameter("--audio-duration must be positive")
    if skip_separation and not no_render:
        raise click.BadParameter("--skip-separation requires --no-render")
    if skip_separation and shorten_breaks:
        raise click.BadParameter(
            "--skip-separation cannot be combined with --shorten-breaks"
        )


def run_generate_command(
    *,
    ctx,
    logger,
    url_or_query,
    output,
    offset,
    key,
    tempo,
    audio_start,
    audio_duration,
    title,
    artist,
    lyrics_offset,
    backgrounds,
    force,
    keep_files,
    work_dir,
    offline,
    resolution,
    fps,
    font_size,
    no_progress,
    no_render,
    timing_report,
    evaluate_lyrics,
    whisper,
    whisper_only,
    whisper_map_lrc,
    whisper_map_lrc_dtw,
    lyrics_file,
    drop_lrc_line_timings,
    whisper_language,
    whisper_model,
    whisper_force_dtw,
    whisper_aggressive,
    whisper_temperature,
    lenient_vocal_activity_threshold,
    lenient_activity_bonus,
    low_word_confidence_threshold,
    filter_promos,
    outro_line,
    shorten_breaks,
    max_break,
    debug_audio,
    skip_separation,
    resolve_url_or_query_fn,
    identify_track_fn,
    identify_track_offline_fn,
    build_video_settings_fn,
    resolve_shorten_breaks_fn,
    log_quality_summary_fn,
    karaoke_generator_cls=KaraokeGenerator,
):
    """Execute the `generate` command implementation."""
    try:
        if whisper_map_lrc_dtw:
            whisper_map_lrc = True
        url_or_query = _resolve_input_query(
            logger=logger,
            url_or_query=url_or_query,
            artist=artist,
            title=title,
            resolve_url_or_query_fn=resolve_url_or_query_fn,
        )

        identifier = TrackIdentifier()
        track_info = _identify_track_info(
            logger=logger,
            identifier=identifier,
            url_or_query=url_or_query,
            artist=artist,
            title=title,
            offline=offline,
            identify_track_fn=identify_track_fn,
            identify_track_offline_fn=identify_track_offline_fn,
        )

        if title or artist:
            track_info = replace(
                track_info,
                title=title or track_info.title,
                artist=artist or track_info.artist,
            )

        url = validate_youtube_url(track_info.youtube_url)
        key = validate_key_shift(key)
        tempo = validate_tempo(tempo)
        offset = validate_offset(offset)

        _validate_generate_probe_options(
            audio_start=audio_start,
            audio_duration=audio_duration,
            skip_separation=skip_separation,
            no_render=no_render,
            shorten_breaks=shorten_breaks,
        )

        video_settings = build_video_settings_fn(
            resolution, fps, font_size, no_progress
        )

        cache_dir = Path(work_dir) if work_dir else get_cache_dir()
        generator = karaoke_generator_cls(cache_dir=cache_dir)

        output_path = validate_output_path(output) if output else None
        effective_shorten_breaks = resolve_shorten_breaks_fn(
            logger, shorten_breaks, track_info
        )

        target_duration: int | None
        if audio_duration is not None and audio_duration > 0:
            target_duration = max(1, int(round(audio_duration)))
        else:
            target_duration = track_info.duration if track_info.duration > 0 else None

        result = generator.generate(
            **_build_generate_kwargs(
                url=url,
                output_path=output_path,
                offset=offset,
                key=key,
                tempo=tempo,
                audio_start=audio_start,
                audio_duration=audio_duration,
                title=title,
                artist=artist,
                track_info=track_info,
                lyrics_offset=lyrics_offset,
                backgrounds=backgrounds,
                force=force,
                video_settings=video_settings,
                url_or_query=url_or_query,
                target_duration=target_duration,
                evaluate_lyrics=evaluate_lyrics,
                whisper=whisper,
                whisper_only=whisper_only,
                whisper_map_lrc=whisper_map_lrc,
                whisper_map_lrc_dtw=whisper_map_lrc_dtw,
                lyrics_file=lyrics_file,
                drop_lrc_line_timings=drop_lrc_line_timings,
                whisper_language=whisper_language,
                whisper_model=whisper_model,
                whisper_force_dtw=whisper_force_dtw,
                whisper_aggressive=whisper_aggressive,
                whisper_temperature=whisper_temperature,
                lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
                lenient_activity_bonus=lenient_activity_bonus,
                low_word_confidence_threshold=low_word_confidence_threshold,
                outro_line=outro_line,
                offline=offline,
                filter_promos=filter_promos,
                effective_shorten_breaks=effective_shorten_breaks,
                max_break=max_break,
                debug_audio=debug_audio,
                no_render=no_render,
                skip_separation=skip_separation,
                timing_report=timing_report,
            )
        )

        if result.get("rendered", True):
            logger.info(f"✅ Karaoke video generated: {result['output_path']}")
        else:
            logger.info("✅ Karaoke pipeline complete (render skipped)")
        log_quality_summary_fn(logger, result)

        if not keep_files:
            generator.cleanup_temp_files()

    except Y2KaraokeError as e:
        logger.error(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if ctx.obj.get("verbose"):
            import traceback

            traceback.print_exc()
        sys.exit(1)


def run_evaluate_timing_command(
    *,
    ctx,
    logger,
    url_or_query,
    title,
    artist,
    work_dir,
    resolve_url_or_query_fn,
):
    """Execute the `evaluate-timing` command implementation."""
    try:
        from .pipeline.audio import YouTubeDownloader, separate_vocals
        from .pipeline.alignment import print_comparison_report
        from .pipeline.identify import TrackIdentifier

        if not url_or_query:
            resolved = resolve_url_or_query_fn(url_or_query, artist, title)
            logger.info(f"No url_or_query provided; using title for search: {resolved}")
            url_or_query = resolved
        else:
            url_or_query = resolve_url_or_query_fn(url_or_query, artist, title)

        identifier = TrackIdentifier()
        if url_or_query.startswith("http"):
            track_info = identifier.identify_from_url(url_or_query)
        else:
            track_info = identifier.identify_from_search(url_or_query)

        effective_title = title or track_info.title
        effective_artist = artist or track_info.artist
        logger.info(f"Evaluating: {effective_artist} - {effective_title}")

        cache_dir = Path(work_dir) if work_dir else get_cache_dir()
        downloader = YouTubeDownloader(cache_dir=cache_dir)
        result = downloader.download_audio(track_info.youtube_url)
        audio_path = result["audio_path"]
        logger.info(f"Downloaded audio: {audio_path}")

        sep_result = separate_vocals(audio_path, output_dir=str(cache_dir))
        vocals_path = sep_result["vocals_path"]
        logger.info(f"Separated vocals: {vocals_path}")

        print_comparison_report(effective_title, effective_artist, vocals_path)

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        if ctx.obj.get("verbose"):
            import traceback

            traceback.print_exc()
        sys.exit(1)
