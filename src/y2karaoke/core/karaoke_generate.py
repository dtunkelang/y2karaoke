"""Generate pipeline implementation for :class:`KaraokeGenerator`."""

from __future__ import annotations

from pathlib import Path
from time import time
from typing import Any, Dict, Optional

from ..utils.logging import get_logger
from ..utils.validation import fix_line_order, validate_line_order

logger = get_logger(__name__)


def generate_karaoke(
    generator,
    *,
    url: str,
    output_path: Optional[Path] = None,
    offset: float = 0.0,
    key_shift: int = 0,
    tempo_multiplier: float = 1.0,
    audio_start: float = 0.0,
    lyrics_title: Optional[str] = None,
    lyrics_artist: Optional[str] = None,
    lyrics_offset: Optional[float] = None,
    use_backgrounds: bool = False,
    force_reprocess: bool = False,
    video_settings: Optional[Dict[str, Any]] = None,
    original_prompt: Optional[str] = None,
    target_duration: Optional[int] = None,
    evaluate_lyrics_sources: bool = False,
    use_whisper: bool = False,
    whisper_only: bool = False,
    whisper_map_lrc: bool = False,
    whisper_map_lrc_dtw: bool = False,
    lyrics_file: Optional[Path] = None,
    drop_lrc_line_timings: bool = False,
    whisper_language: Optional[str] = None,
    whisper_model: Optional[str] = None,
    whisper_force_dtw: bool = False,
    whisper_aggressive: bool = False,
    whisper_temperature: float = 0.0,
    lenient_vocal_activity_threshold: float = 0.3,
    lenient_activity_bonus: float = 0.4,
    low_word_confidence_threshold: float = 0.5,
    outro_line: Optional[str] = None,
    offline: bool = False,
    filter_promos: bool = True,
    shorten_breaks: bool = False,
    max_break_duration: float = 30.0,
    debug_audio: str = "instrumental",
    skip_render: bool = False,
    timing_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full karaoke generation pipeline."""
    from ..pipeline.audio import extract_video_id

    generator._original_prompt = original_prompt
    total_start = time()

    video_id = extract_video_id(url)
    logger.info(f"Video ID: {video_id}")

    generator.cache_manager.auto_cleanup()

    audio_result, video_path, separation_result = generator._prepare_media(
        url,
        video_id,
        audio_start,
        use_backgrounds,
        force_reprocess,
        offline,
    )

    final_title, final_artist = generator._resolve_final_metadata(
        audio_result, lyrics_title, lyrics_artist
    )

    lyrics_result = generator._get_lyrics(
        final_title,
        final_artist,
        separation_result["vocals_path"],
        video_id,
        force_reprocess,
        lyrics_offset=lyrics_offset,
        target_duration=target_duration,
        evaluate_sources=evaluate_lyrics_sources,
        use_whisper=use_whisper,
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
        offline=offline,
        filter_promos=filter_promos,
    )

    processed_audio, break_edits = generator._process_audio_track(
        debug_audio,
        separation_result,
        audio_result,
        key_shift,
        tempo_multiplier,
        video_id,
        force_reprocess,
        shorten_breaks,
        max_break_duration,
    )

    if outro_line:
        generator._append_outro_line(
            lines=lyrics_result["lines"],
            outro_line=outro_line,
            audio_path=processed_audio,
        )

    scaled_lines = generator._scale_lyrics_timing(
        lyrics_result["lines"], tempo_multiplier
    )
    scaled_lines = generator._apply_break_edits(scaled_lines, break_edits)
    splash_min_start = 0.5 if whisper_map_lrc_dtw else 3.5
    scaled_lines = generator._apply_splash_offset(
        scaled_lines, min_start=splash_min_start
    )
    scaled_lines = fix_line_order(scaled_lines)
    validate_line_order(scaled_lines)

    if timing_report_path:
        generator._write_timing_report(
            scaled_lines,
            timing_report_path,
            final_title,
            final_artist,
            lyrics_result,
            video_id=video_id,
        )

    output_path = output_path or generator._build_output_path(final_title)
    background_segments = generator._build_background_segments(
        use_backgrounds, video_path, scaled_lines, processed_audio
    )

    if not skip_render:
        generator._render_video(
            lines=scaled_lines,
            audio_path=processed_audio,
            output_path=output_path,
            title=final_title,
            artist=final_artist,
            timing_offset=offset,
            background_segments=background_segments,
            song_metadata=lyrics_result.get("metadata"),
            video_settings=video_settings,
        )
    else:
        logger.info("Skipping video rendering (--no-render)")

    total_time = time() - total_start

    quality_score, quality_issues, quality_level, quality_emoji = (
        generator._summarize_quality(lyrics_result)
    )
    lyrics_quality = lyrics_result.get("quality", {})

    logger.info(
        f"{quality_emoji} Karaoke generation complete: {output_path} ({total_time:.1f}s)"
    )
    logger.info(f"   Quality: {quality_score:.0f}/100 ({quality_level} confidence)")
    if quality_issues:
        for issue in quality_issues[:3]:
            logger.info(f"   - {issue}")

    return {
        "output_path": str(output_path),
        "title": final_title,
        "artist": final_artist,
        "video_id": video_id,
        "rendered": not skip_render,
        "quality_score": quality_score,
        "quality_level": quality_level,
        "quality_issues": quality_issues,
        "lyrics_source": lyrics_quality.get("source", ""),
        "alignment_method": lyrics_quality.get("alignment_method", ""),
    }
