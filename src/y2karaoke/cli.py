"""Command-line interface using Click."""

import sys
from pathlib import Path

import click

from . import __version__
from .config import get_cache_dir
from .exceptions import Y2KaraokeError
from .core.karaoke import KaraokeGenerator
from .core.track_identifier import TrackIdentifier
from .utils.logging import setup_logging
from .utils.validation import (
    validate_youtube_url,
    validate_key_shift,
    validate_tempo,
    validate_offset,
    validate_output_path,
)

# --- module-level helpers ---


def parse_resolution(resolution: str) -> tuple[int, int]:
    """Parse resolution string like '1920x1080' into (width, height)."""
    try:
        width_str, height_str = resolution.lower().split("x")
        width, height = int(width_str), int(height_str)
        return width, height
    except Exception:
        raise ValueError(
            f"Invalid resolution format: {resolution}. Expected 'WIDTHxHEIGHT'."
        )


def identify_track(logger, identifier, url_or_query, lyrics_artist, lyrics_title):
    if url_or_query.startswith("http"):
        logger.info(f"Identifying track from URL: {url_or_query}")
        track_info = identifier.identify_from_url(
            url_or_query,
            artist_hint=lyrics_artist,
            title_hint=lyrics_title,
        )
    else:
        logger.info(f"Identifying track from search: {url_or_query}")
        track_info = identifier.identify_from_search(url_or_query)
        logger.info(f"Found: {track_info.youtube_url}")
    logger.info(
        f"Identified: {track_info.artist} - {track_info.title} (source: {track_info.source})"
    )
    return track_info


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
@click.argument("url_or_query")
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
@click.option("--lyrics-title", help="Override song title for lyrics search")
@click.option("--lyrics-artist", help="Override artist for lyrics search")
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
    "--evaluate-lyrics",
    is_flag=True,
    help="Compare all lyrics sources and select best based on timing",
)
@click.option(
    "--whisper",
    is_flag=True,
    help="Use Whisper transcription to improve lyrics timing alignment",
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
    default="base",
    help="Whisper model size (default: base)",
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
    lyrics_title,
    lyrics_artist,
    lyrics_offset,
    backgrounds,
    force,
    keep_files,
    work_dir,
    resolution,
    fps,
    font_size,
    no_progress,
    evaluate_lyrics,
    whisper,
    whisper_language,
    whisper_model,
    shorten_breaks,
    max_break,
    debug_audio,
):
    logger = ctx.obj["logger"]

    try:
        identifier = TrackIdentifier()
        track_info = identify_track(
            logger, identifier, url_or_query, lyrics_artist, lyrics_title
        )

        url = validate_youtube_url(track_info.youtube_url)
        key = validate_key_shift(key)
        tempo = validate_tempo(tempo)
        offset = validate_offset(offset)

        if audio_start < 0:
            raise click.BadParameter("--audio-start must be non-negative")

        video_settings = build_video_settings(resolution, fps, font_size, no_progress)

        cache_dir = Path(work_dir) if work_dir else get_cache_dir()
        generator = KaraokeGenerator(cache_dir=cache_dir)

        output_path = validate_output_path(output) if output else None
        effective_shorten_breaks = resolve_shorten_breaks(
            logger, shorten_breaks, track_info
        )

        result = generator.generate(
            url=url,
            output_path=output_path,
            offset=offset,
            key_shift=key,
            tempo_multiplier=tempo,
            audio_start=audio_start,
            lyrics_title=lyrics_title or track_info.title,
            lyrics_artist=lyrics_artist or track_info.artist,
            lyrics_offset=lyrics_offset,
            use_backgrounds=backgrounds,
            force_reprocess=force,
            video_settings=video_settings,
            original_prompt=url_or_query,
            target_duration=track_info.duration,
            evaluate_lyrics_sources=evaluate_lyrics,
            use_whisper=whisper,
            whisper_language=whisper_language,
            whisper_model=whisper_model,
            shorten_breaks=effective_shorten_breaks,
            max_break_duration=max_break,
            debug_audio=debug_audio,
        )

        logger.info(f"✅ Karaoke video generated: {result['output_path']}")
        log_quality_summary(logger, result)

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
@click.argument("url_or_query")
@click.option("--lyrics-title", help="Override song title for lyrics search")
@click.option("--lyrics-artist", help="Override artist for lyrics search")
@click.option("--work-dir", type=click.Path(), help="Working directory")
@click.option("--force", is_flag=True, help="Force re-download cached files")
@click.pass_context
def evaluate_timing(ctx, url_or_query, lyrics_title, lyrics_artist, work_dir, force):
    """Evaluate lyrics timing quality against audio analysis.

    Compares timing from all available lyrics sources (lyriq, Musixmatch,
    Lrclib, NetEase, etc.) against detected vocal onsets and pauses in
    the audio to identify the most accurate source.
    """
    logger = ctx.obj["logger"]

    try:
        from .core.track_identifier import TrackIdentifier
        from .core.downloader import YouTubeDownloader
        from .core.separator import separate_vocals
        from .core.timing_evaluator import print_comparison_report

        # Identify track
        identifier = TrackIdentifier()
        if url_or_query.startswith("http"):
            track_info = identifier.identify_from_url(url_or_query)
        else:
            track_info = identifier.identify_from_search(url_or_query)

        effective_title = lyrics_title or track_info.title
        effective_artist = lyrics_artist or track_info.artist

        logger.info(f"Evaluating: {effective_artist} - {effective_title}")

        # Download audio
        cache_dir = Path(work_dir) if work_dir else get_cache_dir()
        downloader = YouTubeDownloader(cache_dir=cache_dir)
        result = downloader.download_audio(track_info.youtube_url)
        audio_path = result["audio_path"]
        logger.info(f"Downloaded audio: {audio_path}")

        # Separate vocals
        sep_result = separate_vocals(audio_path, output_dir=str(cache_dir))
        vocals_path = sep_result["vocals_path"]
        logger.info(f"Separated vocals: {vocals_path}")

        # Run comparison
        print_comparison_report(effective_title, effective_artist, vocals_path)

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        if ctx.obj.get("verbose"):
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    cli()
