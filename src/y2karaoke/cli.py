"""Command-line interface using Click."""

import sys
from pathlib import Path
from typing import Optional

import click

from . import __version__
from .config import get_cache_dir, parse_resolution, RESOLUTION_PRESETS
from .exceptions import Y2KaraokeError
from .utils.cache import CacheManager
from .utils.logging import setup_logging, get_logger
from .utils.validation import (
    validate_youtube_url, validate_key_shift, validate_tempo, 
    validate_offset, validate_output_path
)


def search_youtube(query: str) -> Optional[str]:
    """Search YouTube and return the first video URL."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        raise Y2KaraokeError("requests and beautifulsoup4 required for YouTube search")
    
    search_query = query.replace(" ", "+")
    search_url = f"https://www.youtube.com/results?search_query={search_query}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Find video ID in the page
        # YouTube embeds video data in the page HTML
        import re
        video_id_match = re.search(r'"videoId":"([^"]+)"', response.text)
        
        if video_id_match:
            video_id = video_id_match.group(1)
            return f"https://www.youtube.com/watch?v={video_id}"
        
        return None
        
    except (requests.exceptions.RequestException, TimeoutError) as e:
        raise Y2KaraokeError(f"YouTube search failed: {e}")


@click.group()
@click.version_option(version=__version__)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--log-file', type=click.Path(), help='Log to file')
@click.pass_context
def cli(ctx, verbose, log_file):
    """Y2Karaoke - Generate karaoke videos from YouTube URLs."""
    ctx.ensure_object(dict)
    
    # Setup logging
    log_path = Path(log_file) if log_file else None
    logger = setup_logging(
        level="DEBUG" if verbose else "INFO",
        log_file=log_path,
        verbose=verbose
    )
    
    ctx.obj['logger'] = logger

@cli.command()
@click.argument('url_or_query')
@click.option('-o', '--output', help='Output video path')
@click.option('--offset', type=float, default=0.0, 
              help='Timing offset in seconds (negative = earlier, positive = later)')
@click.option('--key', type=int, default=0, 
              help='Shift key by N semitones (-12 to +12)')
@click.option('--tempo', type=float, default=1.0,
              help='Tempo multiplier (0.5 = half speed, 2.0 = double)')
@click.option('--audio-start', type=float, default=0.0,
              help='Start audio processing from this many seconds (skip intro)')
@click.option('--lyrics-title', help='Override song title for lyrics search')
@click.option('--lyrics-artist', help='Override artist for lyrics search')
@click.option('--backgrounds', is_flag=True, 
              help='Use video backgrounds from original YouTube video')
@click.option('--upload', is_flag=True, 
              help='Upload to YouTube as unlisted video')
@click.option('--no-upload', is_flag=True,
              help='Skip upload prompt (for batch mode)')
@click.option('--force', is_flag=True,
              help='Force re-download and re-process cached files')
@click.option('--keep-files', is_flag=True,
              help='Keep intermediate files')
@click.option('--work-dir', type=click.Path(),
              help='Working directory for intermediate files')
@click.option('--resolution', type=str, default=None,
              help=f"Video resolution (e.g., '1920x1080', '720p', '1080p', '4k'). Default: 1080p")
@click.option('--fps', type=int, default=None,
              help='Video frame rate (default: 30)')
@click.option('--font-size', type=int, default=None,
              help='Font size for lyrics (default: 72)')
@click.option('--no-progress', is_flag=True,
              help='Disable progress bar during rendering')
@click.pass_context
def generate(ctx, url_or_query, output, offset, key, tempo, audio_start,
             lyrics_title, lyrics_artist, backgrounds, upload, no_upload,
             force, keep_files, work_dir, resolution, fps, font_size, no_progress):
    """Generate karaoke video from YouTube URL or search query."""
    logger = ctx.obj['logger']

    try:
        # Check if input is a URL or search query
        url = url_or_query
        if not url_or_query.startswith('http'):
            # It's a search query - search YouTube
            logger.info(f"Searching YouTube for: {url_or_query}")
            url = search_youtube(url_or_query)
            if not url:
                raise Y2KaraokeError(f"No YouTube results found for: {url_or_query}")
            logger.info(f"Found: {url}")

        # Validate inputs
        url = validate_youtube_url(url)
        key = validate_key_shift(key)
        tempo = validate_tempo(tempo)
        offset = validate_offset(offset)

        if audio_start < 0:
            raise click.BadParameter("--audio-start must be non-negative")

        # Parse and validate video settings
        video_settings = {}
        if resolution:
            try:
                width, height = parse_resolution(resolution)
                video_settings['width'] = width
                video_settings['height'] = height
                logger.info(f"Using resolution: {width}x{height}")
            except ValueError as e:
                raise click.BadParameter(str(e))

        if fps:
            if fps < 1 or fps > 120:
                raise click.BadParameter("FPS must be between 1 and 120")
            video_settings['fps'] = fps

        if font_size:
            if font_size < 10 or font_size > 200:
                raise click.BadParameter("Font size must be between 10 and 200")
            video_settings['font_size'] = font_size

        video_settings['show_progress'] = not no_progress
        
        # Setup generator
        cache_dir = Path(work_dir) if work_dir else get_cache_dir()
        
        # Try to import and use full generator
        try:
            from .core.karaoke import KaraokeGenerator
            
            generator = KaraokeGenerator(cache_dir=cache_dir)
            
            # Generate output path if not provided
            if not output:
                output = None
            else:
                output = validate_output_path(output)
            
            # Generate karaoke video
            result = generator.generate(
                url=url,
                output_path=output,
                offset=offset,
                key_shift=key,
                tempo_multiplier=tempo,
                audio_start=audio_start,
                lyrics_title=lyrics_title,
                lyrics_artist=lyrics_artist,
                use_backgrounds=backgrounds,
                force_reprocess=force,
                video_settings=video_settings if video_settings else None,
            )
            
            logger.info(f"✅ Karaoke video generated: {result['output_path']}")
            
            # Handle upload - only prompt if neither flag is set
            should_upload = upload
            if not should_upload and not no_upload:
                should_upload = False  # Don't prompt by default
            
            if should_upload:
                upload_result = generator.upload_video(
                    result['output_path'],
                    result['title'],
                    result['artist']
                )
                logger.info(f"🎥 Uploaded to YouTube: {upload_result['url']}")
            
            # Cleanup if requested
            if not keep_files:
                generator.cleanup_temp_files()
                
        except ImportError as e:
            # Fallback to basic functionality
            logger.warning(f"Full processing not available: {e}")
            logger.info("🚧 Some dependencies missing - showing basic validation only")
            logger.info(f"✅ Valid YouTube URL: {url}")
            logger.info(f"📁 Cache directory: {cache_dir}")
            logger.info("💡 Install all dependencies for full functionality")
            
    except Y2KaraokeError as e:
        logger.error(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)

@cli.group()
def cache():
    """Cache management commands."""
    pass

@cache.command()
@click.option('--cache-dir', type=click.Path(), help='Cache directory')
def stats(cache_dir):
    """Show cache statistics."""
    cache_path = Path(cache_dir) if cache_dir else get_cache_dir()
    manager = CacheManager(cache_path)
    
    stats = manager.get_cache_stats()
    
    click.echo(f"Cache Directory: {stats['cache_dir']}")
    click.echo(f"Total Size: {stats['total_size_gb']:.2f} GB")
    click.echo(f"Files: {stats['file_count']}")
    click.echo(f"Videos: {stats['video_count']}")

@cache.command()
@click.option('--cache-dir', type=click.Path(), help='Cache directory')
@click.option('--days', type=int, default=30, help='Remove files older than N days')
@click.confirmation_option(prompt='Are you sure you want to cleanup cache?')
def cleanup(cache_dir, days):
    """Clean up old cache files."""
    cache_path = Path(cache_dir) if cache_dir else get_cache_dir()
    manager = CacheManager(cache_path)
    
    manager.cleanup_old_files(days)
    click.echo(f"✅ Cache cleanup completed")

@cache.command()
@click.argument('video_id')
@click.option('--cache-dir', type=click.Path(), help='Cache directory')
@click.confirmation_option(prompt='Are you sure you want to clear this video cache?')
def clear(video_id, cache_dir):
    """Clear cache for specific video."""
    cache_path = Path(cache_dir) if cache_dir else get_cache_dir()
    manager = CacheManager(cache_path)
    
    manager.clear_video_cache(video_id)
    click.echo(f"✅ Cleared cache for video {video_id}")

if __name__ == '__main__':
    cli()
