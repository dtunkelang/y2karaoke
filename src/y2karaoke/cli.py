"""Command-line interface using Click."""

import sys
from pathlib import Path
from typing import Optional

import click

from . import __version__
from .config import get_cache_dir, parse_resolution
from .exceptions import Y2KaraokeError
from .core.karaoke import KaraokeGenerator
from .core.downloader import extract_video_id
from .utils.logging import setup_logging
from .utils.validation import (
    validate_youtube_url, validate_key_shift, validate_tempo,
    validate_offset, validate_output_path
)


def search_youtube(query: str) -> Optional[str]:
    """Search YouTube and return the best matching video URL.

    Filters out live, extended, and remix versions unless those terms
    are explicitly in the search query.
    """
    try:
        import requests
    except ImportError:
        raise Y2KaraokeError("requests required for YouTube search")

    search_query = query.replace(" ", "+")
    search_url = f"https://www.youtube.com/results?search_query={search_query}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()

        import re

        # Extract video data from YouTube's initial data JSON
        # Find all video entries with their titles
        candidates = []

        # Pattern to find video renderer objects with videoId and title
        # YouTube embeds this data in a JSON structure
        video_pattern = re.compile(
            r'"videoRenderer":\s*\{[^}]*"videoId":\s*"([^"]+)"[^}]*"title":\s*\{"runs":\s*\[\{"text":\s*"([^"]+)"',
            re.DOTALL
        )

        for match in video_pattern.finditer(response.text):
            video_id = match.group(1)
            title = match.group(2)
            candidates.append((video_id, title))

        # Fallback: just get video IDs if title extraction fails
        if not candidates:
            for match in re.finditer(r'"videoId":"([^"]+)"', response.text):
                video_id = match.group(1)
                # Avoid duplicates and short IDs (which may be playlist IDs)
                if len(video_id) == 11 and video_id not in [c[0] for c in candidates]:
                    candidates.append((video_id, ""))

        if not candidates:
            return None

        # Check if query contains filter terms
        query_lower = query.lower()
        filter_terms = ["live", "extended", "remix"]
        query_has_filter_term = any(term in query_lower for term in filter_terms)

        # If query doesn't have filter terms, deprioritize videos with those terms
        if not query_has_filter_term:
            preferred = []
            fallback = []

            for video_id, title in candidates:
                title_lower = title.lower()
                has_filter_term = any(term in title_lower for term in filter_terms)

                if has_filter_term:
                    fallback.append((video_id, title))
                else:
                    preferred.append((video_id, title))

            # Use preferred list if available, otherwise fall back
            candidates = preferred if preferred else fallback

        if candidates:
            return f"https://www.youtube.com/watch?v={candidates[0][0]}"
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
    logger = setup_logging(
        level="DEBUG" if verbose else "INFO",
        log_file=Path(log_file) if log_file else None,
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
@click.option('--lyrics-offset', type=float, default=None,
              help='Manual lyrics timing offset in seconds')
@click.option('--backgrounds', is_flag=True,
              help='Use video backgrounds from original YouTube video')
@click.option('--force', is_flag=True,
              help='Force re-download and re-process cached files')
@click.option('--keep-files', is_flag=True,
              help='Keep intermediate files')
@click.option('--work-dir', type=click.Path(),
              help='Working directory for intermediate files')
@click.option('--resolution', type=str, default=None,
              help="Video resolution (e.g., '1920x1080', '720p', '1080p', '4k')")
@click.option('--fps', type=int, default=None,
              help='Video frame rate (default: 30)')
@click.option('--font-size', type=int, default=None,
              help='Font size for lyrics (default: 72)')
@click.option('--no-progress', is_flag=True,
              help='Disable progress bar during rendering')
@click.pass_context
def generate(ctx, url_or_query, output, offset, key, tempo, audio_start,
             lyrics_title, lyrics_artist, lyrics_offset, backgrounds,
             force, keep_files, work_dir, resolution, fps, font_size, no_progress):
    """Generate karaoke video from YouTube URL or search query."""
    logger = ctx.obj['logger']

    try:
        # Determine YouTube URL
        url = url_or_query
        if not url_or_query.startswith('http'):
            logger.info(f"Searching YouTube for: {url_or_query}")
            url = search_youtube(url_or_query)
            if not url:
                raise Y2KaraokeError(f"No YouTube results found for: {url_or_query}")
            logger.info(f"Found: {url}")

        url = validate_youtube_url(url)
        key = validate_key_shift(key)
        tempo = validate_tempo(tempo)
        offset = validate_offset(offset)

        if audio_start < 0:
            raise click.BadParameter("--audio-start must be non-negative")

        # Video settings
        video_settings = {}
        if resolution:
            width, height = parse_resolution(resolution)
            video_settings['width'] = width
            video_settings['height'] = height
        if fps:
            video_settings['fps'] = fps
        if font_size:
            video_settings['font_size'] = font_size
        video_settings['show_progress'] = not no_progress

        cache_dir = Path(work_dir) if work_dir else get_cache_dir()
        generator = KaraokeGenerator(cache_dir=cache_dir)

        # Extract video ID
        video_id = extract_video_id(url)

        # Generate output path if not provided
        if output:
            output_path = validate_output_path(output)
        else:
            output_path = None

        # Run generation
        result = generator.generate(
            url=url,
            output_path=output_path,
            offset=offset,
            key_shift=key,
            tempo_multiplier=tempo,
            audio_start=audio_start,
            lyrics_title=lyrics_title,
            lyrics_artist=lyrics_artist,
            lyrics_offset=lyrics_offset,
            use_backgrounds=backgrounds,
            force_reprocess=force,
            video_settings=video_settings if video_settings else None,
            original_prompt=url_or_query
        )

        logger.info(f"✅ Karaoke video generated: {result['output_path']}")

        # Cleanup
        if not keep_files:
            generator.cleanup_temp_files()

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
    from .utils.cache import CacheManager
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
    from .utils.cache import CacheManager
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
    from .utils.cache import CacheManager
    cache_path = Path(cache_dir) if cache_dir else get_cache_dir()
    manager = CacheManager(cache_path)
    manager.clear_video_cache(video_id)
    click.echo(f"✅ Cleared cache for video {video_id}")


if __name__ == '__main__':
    cli()
