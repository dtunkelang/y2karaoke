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


def _get_track_duration_from_musicbrainz(query: str) -> tuple[Optional[int], Optional[str], Optional[str]]:
    """Query MusicBrainz to get canonical track duration, artist, and title.

    Tries to parse artist and title from query, then finds the most
    common duration for that track.

    Returns:
        Tuple of (duration_seconds, artist, title). Any may be None if not found.
    """
    try:
        import musicbrainzngs
        from collections import Counter

        musicbrainzngs.set_useragent('y2karaoke', '1.0', 'https://github.com/dtunkelang/y2karaoke')

        query_clean = query.strip().lower()

        # Try to extract artist and title from query
        # Common patterns: "artist - title", "artist title"
        artist_hint = None
        title_hint = query_clean

        if ' - ' in query_clean:
            parts = query_clean.split(' - ', 1)
            artist_hint = parts[0].strip()
            title_hint = parts[1].strip()

        # Search with artist filter if we have a hint
        if artist_hint:
            results = musicbrainzngs.search_recordings(
                recording=title_hint, artist=artist_hint, limit=15
            )
        else:
            # Try using query words as potential artist
            results = musicbrainzngs.search_recordings(recording=query_clean, limit=15)

        recordings = results.get('recording-list', [])

        if not recordings:
            return None, None, None

        # Collect durations and track info from matching recordings
        matches = []  # List of (duration, artist, title)
        query_words = set(query_clean.split())

        for rec in recordings:
            length = rec.get('length')
            if not length:
                continue

            # Check if recording matches query (artist name contains query words)
            artist_credits = rec.get('artist-credit', [])
            artists = [a['artist']['name'] for a in artist_credits if 'artist' in a]
            artist_str_lower = ' '.join(a.lower() for a in artists)

            # Require at least one query word in artist name for relevance
            if any(word in artist_str_lower for word in query_words):
                artist_name = ' & '.join(artists) if artists else None
                title = rec.get('title')
                matches.append((int(length) // 1000, artist_name, title))

        # If no artist matches, try a second search with query words as artist
        if not matches and not artist_hint:
            # Use first word(s) as potential artist
            words = query_clean.split()
            if len(words) >= 2:
                # Try first word as artist, rest as title
                results2 = musicbrainzngs.search_recordings(
                    recording=' '.join(words[1:]),
                    artist=words[0],
                    limit=15
                )
                for rec in results2.get('recording-list', []):
                    length = rec.get('length')
                    if length:
                        artist_credits = rec.get('artist-credit', [])
                        artists = [a['artist']['name'] for a in artist_credits if 'artist' in a]
                        artist_name = ' & '.join(artists) if artists else None
                        title = rec.get('title')
                        matches.append((int(length) // 1000, artist_name, title))

        if not matches:
            return None, None, None

        # Find most common duration (mode) to handle different versions
        durations = [m[0] for m in matches]
        rounded = [d // 5 * 5 for d in durations]
        most_common_rounded = Counter(rounded).most_common(1)[0][0]

        # Find the match with duration closest to the most common rounded value
        best_match = min(matches, key=lambda m: abs(m[0] - most_common_rounded))
        return best_match

    except Exception:
        return None, None, None


def _extract_youtube_candidates(response_text: str) -> list:
    """Extract video candidates with titles and durations from YouTube response."""
    import re

    candidates = []

    # Pattern to find video renderer objects with videoId and title
    video_pattern = re.compile(
        r'"videoRenderer":\{"videoId":"([^"]{11})".{0,800}?"title":\{"runs":\[\{"text":"([^"]+)"',
        re.DOTALL
    )

    for match in video_pattern.finditer(response_text):
        video_id = match.group(1)
        title = match.group(2)

        # Find duration for this video
        duration_pattern = rf'"videoRenderer":\{{"videoId":"{video_id}".{{0,2000}}?"simpleText":"(\d+:\d+(?::\d+)?)"'
        duration_match = re.search(duration_pattern, response_text, re.DOTALL)

        duration_sec = None
        if duration_match:
            time_str = duration_match.group(1)
            parts = time_str.split(':')
            if len(parts) == 2:
                duration_sec = int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                duration_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

        candidates.append({
            'video_id': video_id,
            'title': title,
            'duration': duration_sec
        })

    return candidates


def search_youtube(query: str) -> dict[str, Optional[str]]:
    """Search YouTube and return the best matching video URL with metadata.

    Uses MusicBrainz to find the canonical track duration, then selects
    the YouTube video with the closest duration. This helps avoid live,
    extended, and remix versions which typically have different lengths.

    Returns:
        Dict with 'url', 'artist', 'title' keys. 'url' is None if not found.
        'artist' and 'title' are set if MusicBrainz was used for matching.
    """
    try:
        import requests
    except ImportError:
        raise Y2KaraokeError("requests required for YouTube search")

    # Get canonical track info from MusicBrainz
    target_duration, mb_artist, mb_title = _get_track_duration_from_musicbrainz(query)
    used_musicbrainz = target_duration is not None

    search_query = query.replace(" ", "+")
    search_url = f"https://www.youtube.com/results?search_query={search_query}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()

        candidates = _extract_youtube_candidates(response.text)

        if not candidates:
            return {'url': None, 'artist': None, 'title': None}

        # If we have a target duration, rank by closest match
        if target_duration:
            # Filter to candidates with duration info
            with_duration = [c for c in candidates if c['duration'] is not None]

            if with_duration:
                # Sort by duration difference from target
                with_duration.sort(key=lambda c: abs(c['duration'] - target_duration))
                best = with_duration[0]

                # Only use duration match if it's reasonably close (within 30 seconds)
                if abs(best['duration'] - target_duration) <= 30:
                    return {
                        'url': f"https://www.youtube.com/watch?v={best['video_id']}",
                        'artist': mb_artist,
                        'title': mb_title,
                    }

        # Fallback: filter by title keywords (live, extended, remix)
        query_lower = query.lower()
        filter_terms = ["live", "extended", "remix"]
        query_has_filter_term = any(term in query_lower for term in filter_terms)

        if not query_has_filter_term:
            preferred = [c for c in candidates if not any(term in c['title'].lower() for term in filter_terms)]
            if preferred:
                candidates = preferred

        if candidates:
            # Didn't use MusicBrainz duration match, so don't return artist/title
            return {
                'url': f"https://www.youtube.com/watch?v={candidates[0]['video_id']}",
                'artist': None,
                'title': None,
            }
        return {'url': None, 'artist': None, 'title': None}

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
        # Determine YouTube URL and metadata
        url = url_or_query
        search_artist = None
        search_title = None
        if not url_or_query.startswith('http'):
            logger.info(f"Searching YouTube for: {url_or_query}")
            search_result = search_youtube(url_or_query)
            url = search_result['url']
            search_artist = search_result['artist']
            search_title = search_result['title']
            if not url:
                raise Y2KaraokeError(f"No YouTube results found for: {url_or_query}")
            logger.info(f"Found: {url}")
            if search_artist and search_title:
                logger.info(f"MusicBrainz match: {search_artist} - {search_title}")

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

        # Use MusicBrainz results if user didn't provide explicit values
        effective_lyrics_title = lyrics_title or search_title
        effective_lyrics_artist = lyrics_artist or search_artist

        # Run generation
        result = generator.generate(
            url=url,
            output_path=output_path,
            offset=offset,
            key_shift=key,
            tempo_multiplier=tempo,
            audio_start=audio_start,
            lyrics_title=effective_lyrics_title,
            lyrics_artist=effective_lyrics_artist,
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
