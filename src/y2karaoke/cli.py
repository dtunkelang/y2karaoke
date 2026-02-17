"""Command-line interface using Click."""

from pathlib import Path

import click

from . import __version__
from .config import get_cache_dir, parse_resolution as parse_resolution_value
from .core.karaoke import KaraokeGenerator

from .pipeline.identify import TrackInfo
from .cli_commands import run_evaluate_timing_command, run_generate_command
from .utils.logging import setup_logging

# --- module-level helpers ---


def parse_resolution(resolution: str) -> tuple[int, int]:
    """Parse resolution string like '1920x1080' or '1080p' into (width, height)."""
    return parse_resolution_value(resolution)


def identify_track(logger, identifier, url_or_query, artist, title):
    if url_or_query.startswith("http"):
        logger.info(f"Identifying track from URL: {url_or_query}")
        track_info = identifier.identify_from_url(
            url_or_query,
            artist_hint=artist,
            title_hint=title,
        )
    else:
        logger.info(f"Identifying track from search: {url_or_query}")
        track_info = identifier.identify_from_search(url_or_query)
        logger.info(f"Found: {track_info.youtube_url}")
    logger.info(
        f"Identified: {track_info.artist} - {track_info.title} (source: {track_info.source})"
    )
    return track_info


def identify_track_offline(logger, identifier, url, artist, title):
    cached = identifier.get_cached_youtube_metadata(url)
    if not cached:
        raise click.BadParameter(
            "Offline mode requires cached audio for the YouTube URL."
        )
    cached_title, cached_uploader, duration = cached
    final_title = title or cached_title or "Unknown"
    final_artist = artist or cached_uploader or "Unknown"
    if duration <= 0:
        logger.warning("Cached audio duration is unknown; proceeding without it.")
    track_info = TrackInfo(
        artist=final_artist,
        title=final_title,
        duration=duration,
        youtube_url=url,
        youtube_duration=duration,
        source="cache",
        identification_quality=60.0,
        quality_issues=["offline cache used"],
        fallback_used=True,
    )
    logger.info(
        f"Offline: using cached metadata for {track_info.artist} - {track_info.title}"
    )
    return track_info


def resolve_url_or_query(url_or_query, artist, title):
    if url_or_query:
        return url_or_query
    if not title:
        raise click.BadParameter(
            "Missing URL or search query. Provide url_or_query or --title."
        )
    if artist:
        return f"{artist} - {title}"
    return title


def build_video_settings(resolution, fps, font_size, no_progress):
    settings = {}
    if resolution:
        width, height = parse_resolution(resolution)
        settings["width"] = width
        settings["height"] = height
    if fps:
        settings["fps"] = fps
    if font_size:
        settings["font_size"] = font_size
    settings["show_progress"] = not no_progress
    return settings or None


def resolve_shorten_breaks(logger, shorten_breaks, track_info):
    if shorten_breaks and not track_info.lrc_validated:
        logger.warning(
            "⚠️  LRC duration doesn't match audio - disabling break shortening"
        )
        logger.warning(
            "   Break shortening requires matching LRC timing to work correctly"
        )
        return False
    return shorten_breaks


def log_quality_summary(logger, result):
    quality_score = result.get("quality_score", 0)
    quality_level = result.get("quality_level", "unknown")
    quality_issues = result.get("quality_issues", [])

    if quality_score >= 80:
        indicator = "🟢"
    elif quality_score >= 50:
        indicator = "🟡"
    else:
        indicator = "🔴"

    logger.info(f"{indicator} Quality: {quality_score:.0f}/100 ({quality_level})")
    if result.get("lyrics_source"):
        logger.info(f"   Lyrics source: {result['lyrics_source']}")
    if result.get("alignment_method"):
        logger.info(f"   Alignment: {result['alignment_method']}")
    if quality_issues and quality_score < 80:
        logger.info("   Issues:")
        for issue in quality_issues[:3]:
            logger.info(f"     - {issue}")
    if quality_score < 50:
        logger.warning("   Consider using --whisper for better alignment")


@click.group()
@click.version_option(version=__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--log-file", type=click.Path(), help="Log to file")
@click.pass_context
def cli(ctx, verbose, log_file):
    """Y2Karaoke - Generate karaoke videos from YouTube URLs."""
    ctx.ensure_object(dict)
    logger = setup_logging(
        level="DEBUG" if verbose else "INFO",
        log_file=Path(log_file) if log_file else None,
        verbose=verbose,
    )
    ctx.obj["logger"] = logger


@cli.command()
def doctor():
    """Check system health and dependencies."""
    from .core.health import SystemDoctor
    import sys

    doc = SystemDoctor()
    results = doc.check_all()

    click.echo("System Health Check:")
    click.echo("-" * 40)

    all_good = True
    for res in results:
        status_icon = "✅" if res.is_installed else ("❌" if res.critical else "⚠️")
        click.echo(f"{status_icon} {res.name}")
        if res.version:
            click.echo(f"   Version: {res.version}")
        if not res.is_installed and res.message:
            click.echo(f"   Note: {res.message}")
        if not res.is_installed and res.critical:
            all_good = False

    click.echo("-" * 40)
    if all_good:
        click.echo("✨ System is ready for karaoke generation!")
    else:
        click.echo("❌ Critical dependencies missing. Please install them to proceed.")
        sys.exit(1)


@cli.command()
@click.argument("url_or_query", required=False)
@click.option("-o", "--output", help="Output video path")
@click.option(
    "--offset",
    type=float,
    default=0.0,
    help="Timing offset in seconds (negative = earlier, positive = later)",
)
@click.option(
    "--key", type=int, default=0, help="Shift key by N semitones (-12 to +12)"
)
@click.option(
    "--tempo",
    type=float,
    default=1.0,
    help="Tempo multiplier (0.5 = half speed, 2.0 = double)",
)
@click.option(
    "--audio-start",
    type=float,
    default=0.0,
    help="Start audio processing from this many seconds (skip intro)",
)
@click.option("--title", help="Override song title for lyrics search")
@click.option("--artist", help="Override artist for lyrics search")
@click.option(
    "--lyrics-offset",
    type=float,
    default=None,
    help="Manual lyrics timing offset in seconds",
)
@click.option(
    "--backgrounds",
    is_flag=True,
    help="Use video backgrounds from original YouTube video",
)
@click.option(
    "--force", is_flag=True, help="Force re-download and re-process cached files"
)
@click.option("--keep-files", is_flag=True, help="Keep intermediate files")
@click.option(
    "--work-dir", type=click.Path(), help="Working directory for intermediate files"
)
@click.option(
    "--offline",
    is_flag=True,
    help="Run using cached media only (no network). Requires a cached YouTube URL.",
)
@click.option(
    "--resolution",
    type=str,
    default=None,
    help="Video resolution (e.g., '1920x1080', '720p', '1080p', '4k')",
)
@click.option("--fps", type=int, default=None, help="Video frame rate (default: 30)")
@click.option(
    "--font-size", type=int, default=None, help="Font size for lyrics (default: 72)"
)
@click.option(
    "--no-progress", is_flag=True, help="Disable progress bar during rendering"
)
@click.option(
    "--no-render",
    is_flag=True,
    help="Skip video rendering (run timing/lyrics pipeline only)",
)
@click.option(
    "--timing-report",
    type=click.Path(),
    default=None,
    help="Write timing report JSON to this path",
)
@click.option(
    "--evaluate-lyrics",
    is_flag=True,
    help="Compare all lyrics sources and select best based on timing",
)
@click.option(
    "--filter-promos/--no-filter-promos",
    default=True,
    help="Filter obvious promo/CTA lines at the start of LRC lyrics",
)
@click.option(
    "--whisper",
    is_flag=True,
    help="Use Whisper transcription to improve lyrics timing alignment",
)
@click.option(
    "--whisper-only",
    is_flag=True,
    help="Generate lyrics directly from Whisper transcription (no LRC/Genius)",
)
@click.option(
    "--whisper-map-lrc",
    is_flag=True,
    help="Map LRC lyrics onto Whisper timing without shifting segments",
)
@click.option(
    "--whisper-map-lrc-dtw",
    is_flag=True,
    help="Map LRC lyrics onto Whisper timing using phonetic DTW",
)
@click.option(
    "--lyrics-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Use lyrics from a local text or .lrc file as the text source",
)
@click.option(
    "--whisper-language",
    type=str,
    default=None,
    help='Language code for Whisper (e.g., "fr", "en"). Auto-detected if not specified.',
)
@click.option(
    "--whisper-model",
    type=click.Choice(["tiny", "base", "small", "medium", "large"]),
    default=None,
    help="Whisper model size (default: large for alignment paths)",
)
@click.option(
    "--whisper-force-dtw",
    is_flag=True,
    help="Force DTW-based Whisper alignment even if LRC quality seems acceptable",
)
@click.option(
    "--whisper-aggressive",
    is_flag=True,
    help="Use more aggressive Whisper settings to reduce dropped words",
)
@click.option(
    "--whisper-temperature",
    type=float,
    default=0.0,
    help="Temperature for Whisper transcription (0.0 for greedy, up to 1.0 for more diverse results)",
)
@click.option(
    "--lenient-vocal-activity-threshold",
    type=float,
    default=0.3,
    help="Threshold for vocal activity to trigger leniency in DTW alignment.",
)
@click.option(
    "--lenient-activity-bonus",
    type=float,
    default=0.4,
    help="Bonus applied to phonetic cost during DTW alignment under lenient conditions.",
)
@click.option(
    "--low-word-confidence-threshold",
    type=float,
    default=0.5,
    help="Whisper word probability threshold to trigger leniency in DTW alignment.",
)
@click.option(
    "--outro-line",
    type=str,
    default=None,
    help='Append a final lyric line at the end of the song (e.g., "Mm mm mm mm mm mm mm")',
)
@click.option(
    "--shorten-breaks",
    is_flag=True,
    help="Shorten long instrumental breaks for better karaoke flow",
)
@click.option(
    "--max-break",
    type=float,
    default=30.0,
    help="Maximum instrumental break duration in seconds (default: 30)",
)
@click.option(
    "--debug-audio",
    type=click.Choice(["instrumental", "vocals", "original"]),
    default="instrumental",
    help="Audio track to use (default: instrumental)",
)
@click.pass_context
def generate(
    ctx,
    url_or_query,
    output,
    offset,
    key,
    tempo,
    audio_start,
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
):
    logger = ctx.obj["logger"]
    run_generate_command(
        ctx=ctx,
        logger=logger,
        url_or_query=url_or_query,
        output=output,
        offset=offset,
        key=key,
        tempo=tempo,
        audio_start=audio_start,
        title=title,
        artist=artist,
        lyrics_offset=lyrics_offset,
        backgrounds=backgrounds,
        force=force,
        keep_files=keep_files,
        work_dir=work_dir,
        offline=offline,
        resolution=resolution,
        fps=fps,
        font_size=font_size,
        no_progress=no_progress,
        no_render=no_render,
        timing_report=timing_report,
        evaluate_lyrics=evaluate_lyrics,
        whisper=whisper,
        whisper_only=whisper_only,
        whisper_map_lrc=whisper_map_lrc,
        whisper_map_lrc_dtw=whisper_map_lrc_dtw,
        lyrics_file=lyrics_file,
        whisper_language=whisper_language,
        whisper_model=whisper_model,
        whisper_force_dtw=whisper_force_dtw,
        whisper_aggressive=whisper_aggressive,
        whisper_temperature=whisper_temperature,
        lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
        lenient_activity_bonus=lenient_activity_bonus,
        low_word_confidence_threshold=low_word_confidence_threshold,
        filter_promos=filter_promos,
        outro_line=outro_line,
        shorten_breaks=shorten_breaks,
        max_break=max_break,
        debug_audio=debug_audio,
        resolve_url_or_query_fn=resolve_url_or_query,
        identify_track_fn=identify_track,
        identify_track_offline_fn=identify_track_offline,
        build_video_settings_fn=build_video_settings,
        resolve_shorten_breaks_fn=resolve_shorten_breaks,
        log_quality_summary_fn=log_quality_summary,
        karaoke_generator_cls=KaraokeGenerator,
    )


@cli.group()
def cache():
    """Cache management commands."""
    pass


@cache.command()
@click.option("--cache-dir", type=click.Path(), help="Cache directory")
def stats(cache_dir):
    """Show cache statistics."""
    from .utils.cache import CacheManager

    cache_path = Path(cache_dir) if cache_dir else get_cache_dir()
    manager = CacheManager(cache_path)
    stats = manager.get_cache_stats()
    click.echo(f"Cache Directory: {stats['cache_dir']}")
    click.echo(f"Total Size: {stats['total_size_gb']:.2f} GB")
    click.echo(f"Files: {stats['file_count']}")
    click.echo(f"Videos: {stats['video_count']}")


@cache.command()
@click.option("--cache-dir", type=click.Path(), help="Cache directory")
@click.option("--days", type=int, default=30, help="Remove files older than N days")
@click.confirmation_option(prompt="Are you sure you want to cleanup cache?")
def cleanup(cache_dir, days):
    """Clean up old cache files."""
    from .utils.cache import CacheManager

    cache_path = Path(cache_dir) if cache_dir else get_cache_dir()
    manager = CacheManager(cache_path)
    manager.cleanup_old_files(days)
    click.echo("✅ Cache cleanup completed")


@cache.command()
@click.argument("video_id")
@click.option("--cache-dir", type=click.Path(), help="Cache directory")
@click.confirmation_option(prompt="Are you sure you want to clear this video cache?")
def clear(video_id, cache_dir):
    """Clear cache for specific video."""
    from .utils.cache import CacheManager

    cache_path = Path(cache_dir) if cache_dir else get_cache_dir()
    manager = CacheManager(cache_path)
    manager.clear_video_cache(video_id)
    click.echo(f"✅ Cleared cache for video {video_id}")


@cli.command()
@click.argument("url_or_query", required=False)
@click.option("--title", help="Override song title for lyrics search")
@click.option("--artist", help="Override artist for lyrics search")
@click.option("--work-dir", type=click.Path(), help="Working directory")
@click.pass_context
def evaluate_timing(ctx, url_or_query, title, artist, work_dir):
    """Evaluate lyrics timing quality against audio analysis.

    Compares timing from all available lyrics sources (lyriq, Musixmatch,
    Lrclib, NetEase, etc.) against detected vocal onsets and pauses in
    the audio to identify the most accurate source.
    """
    logger = ctx.obj["logger"]
    run_evaluate_timing_command(
        ctx=ctx,
        logger=logger,
        url_or_query=url_or_query,
        title=title,
        artist=artist,
        work_dir=work_dir,
        resolve_url_or_query_fn=resolve_url_or_query,
    )


if __name__ == "__main__":
    cli()
