"""Main karaoke generator orchestrating all components."""

from pathlib import Path
from typing import Dict, List, Optional, Any
from time import time
import musicbrainzngs

from ..config import get_cache_dir
from ..utils.cache import CacheManager
from ..utils.logging import get_logger
from ..utils.validation import sanitize_filename, validate_line_order
from .downloader import YouTubeDownloader, extract_video_id
from .separator import AudioSeparator
from .audio_effects import AudioProcessor
from .audio_utils import trim_audio_if_needed, apply_audio_effects, separate_vocals

logger = get_logger(__name__)

# Initialize MusicBrainz
musicbrainzngs.set_useragent(
    "y2karaoke", "1.0", "https://github.com/dtunkelang/y2karaoke"
)

STOP_WORDS = {"the", "a", "an", "&", "and", "of", "with", "in", "+"}


class KaraokeGenerator:
    """Main class orchestrating karaoke video generation."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or get_cache_dir()
        self.cache_manager = CacheManager(self.cache_dir)
        self.downloader = YouTubeDownloader(self.cache_dir)
        self.separator = AudioSeparator()
        self.audio_processor = AudioProcessor()
        self._temp_files: List[str] = []
        self._original_prompt: Optional[str] = None

    # ------------------------
    # Main generate method
    # ------------------------
    def generate(
        self,
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
        whisper_language: Optional[str] = None,
        whisper_model: str = "base",
        whisper_force_dtw: bool = False,
        filter_promos: bool = True,
        shorten_breaks: bool = False,
        max_break_duration: float = 30.0,
        debug_audio: str = "instrumental",
        skip_render: bool = False,
        timing_report_path: Optional[str] = None,
    ) -> Dict[str, Any]:

        self._original_prompt = original_prompt
        total_start = time()

        video_id = extract_video_id(url)
        logger.info(f"Video ID: {video_id}")

        self.cache_manager.auto_cleanup()

        # Step 1: Download audio
        audio_result = self._download_audio(video_id, url, force_reprocess)

        # Step 2: Download video if backgrounds requested
        video_path = None
        if use_backgrounds:
            video_result = self._download_video(video_id, url, force_reprocess)
            video_path = video_result["video_path"]

        # Step 3: Trim audio if needed
        effective_audio_path = trim_audio_if_needed(
            audio_result["audio_path"],
            audio_start,
            video_id,
            self.cache_manager,
            force=force_reprocess,
        )

        # Step 4: Separate vocals
        separation_result = separate_vocals(
            effective_audio_path,
            video_id,
            self.separator,
            self.cache_manager,
            force=force_reprocess,
        )

        # Step 5: Determine final artist/title for lyrics
        # Trust the upstream identification from CLI; only use download metadata as fallback
        final_artist = lyrics_artist if lyrics_artist else audio_result["artist"]
        final_title = lyrics_title if lyrics_title else audio_result["title"]

        # Step 6: Fetch lyrics (with duration validation if available)
        lyrics_result = self._get_lyrics(
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
            whisper_language=whisper_language,
            whisper_model=whisper_model,
            whisper_force_dtw=whisper_force_dtw,
            filter_promos=filter_promos,
        )

        # Step 7: Select and process audio track based on debug_audio setting
        if debug_audio == "vocals":
            base_audio_path = separation_result["vocals_path"]
            logger.info("🔊 Using vocals track (debug mode)")
        elif debug_audio == "original":
            base_audio_path = audio_result["audio_path"]
            logger.info("🔊 Using original track with vocals (debug mode)")
        else:
            base_audio_path = separation_result["instrumental_path"]

        processed_audio = apply_audio_effects(
            base_audio_path,
            key_shift,
            tempo_multiplier,
            video_id,
            self.cache_manager,
            self.audio_processor,
            force=force_reprocess,
            cache_suffix=f"_{debug_audio}" if debug_audio != "instrumental" else "",
        )

        # Step 7b: Optionally shorten long instrumental breaks
        # Break detection uses vocals, beat alignment uses instrumental for consistent cuts
        break_edits = []
        if shorten_breaks:
            processed_audio, break_edits = self._shorten_breaks(
                processed_audio,
                separation_result["vocals_path"],
                separation_result[
                    "instrumental_path"
                ],  # Always use instrumental for beat alignment
                video_id,
                max_break_duration,
                force=force_reprocess,
                cache_suffix=f"_{debug_audio}" if debug_audio != "instrumental" else "",
            )

        # Step 8: Scale lyrics timing
        scaled_lines = self._scale_lyrics_timing(
            lyrics_result["lines"], tempo_multiplier
        )

        # Step 8b: Adjust lyrics timing for shortened breaks
        if break_edits:
            from .break_shortener import adjust_lyrics_timing

            scaled_lines = adjust_lyrics_timing(scaled_lines, break_edits)

        # Step 9: Ensure lyrics start after splash
        if scaled_lines and scaled_lines[0].start_time < 3.5:
            splash_offset = 3.5 - scaled_lines[0].start_time
            from ..core.lyrics import Line, Word

            offset_lines = []
            for line in scaled_lines:
                offset_words = [
                    Word(
                        text=w.text,
                        start_time=w.start_time + splash_offset,
                        end_time=w.end_time + splash_offset,
                        singer=w.singer,
                    )
                    for w in line.words
                ]
                offset_lines.append(Line(words=offset_words, singer=line.singer))
            scaled_lines = offset_lines

        validate_line_order(scaled_lines)

        if timing_report_path:
            self._write_timing_report(
                scaled_lines,
                timing_report_path,
                final_title,
                final_artist,
                lyrics_result,
                video_id=video_id,
            )

        # Step 10: Generate output path
        if output_path is None:
            safe_title = sanitize_filename(final_title)
            output_path = Path.cwd() / f"{safe_title}_karaoke.mp4"

        # Step 11: Create background segments
        background_segments = None
        if use_backgrounds and video_path:
            background_segments = self._create_background_segments(
                video_path, scaled_lines, processed_audio
            )

        # Step 12: Render video
        if not skip_render:
            self._render_video(
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

        # Build quality report
        lyrics_quality = lyrics_result.get("quality", {})
        quality_score = lyrics_quality.get("overall_score", 50.0)
        quality_issues = lyrics_quality.get("issues", [])

        # Log quality summary
        if quality_score >= 80:
            quality_emoji = "✅"
            quality_level = "high"
        elif quality_score >= 50:
            quality_emoji = "⚠️"
            quality_level = "medium"
        else:
            quality_emoji = "❌"
            quality_level = "low"

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

    # ------------------------
    # Helper methods
    # ------------------------
    def _write_timing_report(
        self,
        lines: List["Line"],
        report_path: str,
        title: str,
        artist: str,
        lyrics_result: Dict[str, Any],
        video_id: Optional[str] = None,
    ) -> None:
        """Write a JSON timing report for downstream inspection."""
        import json

        report = {
            "title": title,
            "artist": artist,
            "lyrics_source": lyrics_result.get("quality", {}).get("source", ""),
            "alignment_method": lyrics_result.get("quality", {}).get(
                "alignment_method", ""
            ),
            "whisper_requested": lyrics_result.get("quality", {}).get(
                "whisper_requested", False
            ),
            "whisper_force_dtw": lyrics_result.get("quality", {}).get(
                "whisper_force_dtw", False
            ),
            "whisper_used": lyrics_result.get("quality", {}).get("whisper_used", False),
            "whisper_corrections": lyrics_result.get("quality", {}).get(
                "whisper_corrections", 0
            ),
            "issues": lyrics_result.get("quality", {}).get("issues", []),
            "dtw_metrics": lyrics_result.get("quality", {}).get("dtw_metrics", {}),
            "line_count": len(lines),
            "lines": [
                {
                    "index": idx + 1,
                    "start": round(line.start_time, 2),
                    "end": round(line.end_time, 2),
                    "text": line.text,
                }
                for idx, line in enumerate(lines)
                if line.words
            ],
        }

        if video_id:
            try:
                from .timing_evaluator import _phonetic_similarity

                cache_dir = self.cache_manager.get_video_cache_dir(video_id)
                whisper_files = list(cache_dir.glob("*_whisper_*.json"))
                if whisper_files:
                    whisper_data = json.loads(
                        whisper_files[0].read_text(encoding="utf-8")
                    )
                    segments = whisper_data.get("segments", whisper_data)
                    report["whisper_segments"] = [
                        {
                            "start": round(seg.get("start", 0.0), 2),
                            "end": round(seg.get("end", 0.0), 2),
                            "text": seg.get("text", ""),
                        }
                        for seg in segments[:50]
                    ]
                    for line in report["lines"]:
                        nearest_start = None
                        nearest_end = None
                        best_start_delta = None
                        best_end_delta = None
                        prior_seg = None
                        prior_late = None
                        for seg in segments:
                            s_start = seg.get("start", 0.0)
                            s_end = seg.get("end", 0.0)
                            start_delta = abs(s_start - line["start"])
                            end_delta = abs(s_end - line["start"])
                            late_by = line["start"] - s_end
                            if 0 <= late_by <= 15.0:
                                if prior_late is None or late_by < prior_late:
                                    prior_late = late_by
                                    prior_seg = seg
                            if (
                                best_start_delta is None
                                or start_delta < best_start_delta
                            ):
                                best_start_delta = start_delta
                                nearest_start = seg
                            if best_end_delta is None or end_delta < best_end_delta:
                                best_end_delta = end_delta
                                nearest_end = seg
                        best_seg = None
                        best_sim = 0.0
                        for seg in segments:
                            if abs(seg.get("start", 0.0) - line["start"]) > 15.0:
                                continue
                            sim = 0.0
                            try:
                                sim = _phonetic_similarity(
                                    line["text"],
                                    seg.get("text", ""),
                                    "fra-Latn",
                                )
                            except Exception:
                                sim = 0.0
                            if sim > best_sim:
                                best_sim = sim
                                best_seg = seg
                        if nearest_start:
                            line["nearest_segment_start"] = round(
                                nearest_start.get("start", 0.0), 2
                            )
                            line["nearest_segment_start_end"] = round(
                                nearest_start.get("end", 0.0), 2
                            )
                            line["nearest_segment_start_text"] = nearest_start.get(
                                "text", ""
                            )
                        if nearest_end:
                            line["nearest_segment_end"] = round(
                                nearest_end.get("end", 0.0), 2
                            )
                            line["nearest_segment_end_start"] = round(
                                nearest_end.get("start", 0.0), 2
                            )
                            line["nearest_segment_end_text"] = nearest_end.get(
                                "text", ""
                            )
                        if best_seg:
                            line["best_segment_start"] = round(
                                best_seg.get("start", 0.0), 2
                            )
                            line["best_segment_end"] = round(
                                best_seg.get("end", 0.0), 2
                            )
                            line["best_segment_text"] = best_seg.get("text", "")
                        if prior_seg is not None:
                            line["prior_segment_start"] = round(
                                prior_seg.get("start", 0.0), 2
                            )
                            line["prior_segment_end"] = round(
                                prior_seg.get("end", 0.0), 2
                            )
                            line["prior_segment_late_by"] = round(
                                line["start"] - prior_seg.get("end", 0.0), 2
                            )
            except Exception:
                pass

        path = Path(report_path)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"Wrote timing report to {path}")

    def _download_audio(self, video_id: str, url: str, force: bool) -> Dict[str, str]:
        metadata = self.cache_manager.load_metadata(video_id)
        if metadata and not force:
            audio_files = self.cache_manager.find_files(video_id, "*.wav")
            # Filter out separated stems (audio-separator uses parentheses like "(Vocals)")
            separated_stems = ["vocals", "bass", "drums", "other", "instrumental"]
            original_audio = [
                f
                for f in audio_files
                if not any(stem in f.name.lower() for stem in separated_stems)
            ]
            if original_audio:
                logger.info("📁 Using cached audio")
                return {
                    "audio_path": str(original_audio[0]),
                    "title": metadata["title"],
                    "artist": metadata["artist"],
                }

        logger.info("📥 Downloading audio...")

        cache_dir = self.cache_manager.get_video_cache_dir(video_id)
        result = self.downloader.download_audio(url, cache_dir)
        self.cache_manager.save_metadata(
            video_id, {"title": result["title"], "artist": result["artist"]}
        )
        return result

    def _download_video(self, video_id: str, url: str, force: bool) -> Dict[str, str]:
        logger.info("📹 Downloading video...")
        if not force:
            video_files = self.cache_manager.find_files(video_id, "*_video.*")
            if video_files:
                logger.info("📁 Using cached video")
                return {"video_path": str(video_files[0])}
        cache_dir = self.cache_manager.get_video_cache_dir(video_id)
        return self.downloader.download_video(url, cache_dir)

    def _get_lyrics(
        self,
        title: str,
        artist: str,
        vocals_path: str,
        video_id: str,
        force: bool,
        lyrics_offset: Optional[float] = None,
        target_duration: Optional[int] = None,
        evaluate_sources: bool = False,
        use_whisper: bool = False,
        whisper_only: bool = False,
        whisper_map_lrc: bool = False,
        whisper_language: Optional[str] = None,
        whisper_model: str = "base",
        whisper_force_dtw: bool = False,
        filter_promos: bool = True,
    ) -> Dict[str, Any]:
        logger.info("📝 Fetching lyrics...")
        from ..core.lyrics import get_lyrics_with_quality

        lines, metadata, quality_report = get_lyrics_with_quality(
            title=title,
            artist=artist,
            vocals_path=vocals_path,
            lyrics_offset=lyrics_offset,
            romanize=True,
            filter_promos=filter_promos,
            target_duration=target_duration,
            evaluate_sources=evaluate_sources,
            use_whisper=use_whisper,
            whisper_only=whisper_only,
            whisper_map_lrc=whisper_map_lrc,
            whisper_language=whisper_language,
            whisper_model=whisper_model,
            whisper_force_dtw=whisper_force_dtw,
        )
        return {"lines": lines, "metadata": metadata, "quality": quality_report}

    def _scale_lyrics_timing(self, lines, tempo_multiplier: float):
        if tempo_multiplier == 1.0:
            return lines
        logger.info(f"⏱️ Scaling lyrics timing for {tempo_multiplier:.2f}x tempo")
        from ..core.lyrics import Line, Word

        scaled_lines = []
        for line in lines:
            scaled_words = [
                Word(
                    text=w.text,
                    start_time=w.start_time / tempo_multiplier,
                    end_time=w.end_time / tempo_multiplier,
                    singer=w.singer,
                )
                for w in line.words
            ]
            scaled_lines.append(Line(words=scaled_words, singer=line.singer))
        return scaled_lines

    def _shorten_breaks(
        self,
        audio_path: str,
        vocals_path: str,
        instrumental_path: str,
        video_id: str,
        max_break_duration: float,
        force: bool = False,
        cache_suffix: str = "",
    ):
        """Shorten long instrumental breaks in the given audio track.

        Break detection always uses vocals, and beat alignment always uses instrumental
        to ensure consistent cuts across different audio tracks.
        """
        import json
        from .break_shortener import shorten_instrumental_breaks, BreakEdit

        # Check cache (both audio and edits)
        shortened_name = f"shortened_breaks_{max_break_duration:.0f}s{cache_suffix}.wav"
        edits_name = f"shortened_breaks_{max_break_duration:.0f}s_edits.json"  # Edits are same for all tracks

        if not force and self.cache_manager.file_exists(video_id, shortened_name):
            # Try to load cached edits
            edits_path = self.cache_manager.get_file_path(video_id, edits_name)
            if edits_path.exists():
                try:
                    with open(edits_path) as f:
                        edits_data = json.load(f)
                    edits = [
                        BreakEdit(
                            original_start=e["original_start"],
                            original_end=e["original_end"],
                            new_end=e["new_end"],
                            time_removed=e["time_removed"],
                            cut_start=e.get("cut_start", 0.0),
                        )
                        for e in edits_data
                    ]
                    logger.info(
                        f"📁 Using cached shortened audio ({len(edits)} break edits)"
                    )
                    return (
                        str(self.cache_manager.get_file_path(video_id, shortened_name)),
                        edits,
                    )
                except Exception as e:
                    logger.debug(f"Could not load cached edits: {e}")
            else:
                logger.info("📁 Using cached shortened audio (no edits)")
                return (
                    str(self.cache_manager.get_file_path(video_id, shortened_name)),
                    [],
                )

        logger.info(
            f"✂️ Shortening instrumental breaks longer than {max_break_duration:.0f}s..."
        )
        output_path = self.cache_manager.get_file_path(video_id, shortened_name)

        shortened_path, edits = shorten_instrumental_breaks(
            audio_path,
            vocals_path,
            str(output_path),
            max_break_duration=max_break_duration,
            beat_reference_path=instrumental_path,
        )

        # Cache the edits for future runs (only once, not per audio track)
        if edits:
            edits_path = self.cache_manager.get_file_path(video_id, edits_name)
            if not edits_path.exists():
                edits_data = [
                    {
                        "original_start": e.original_start,
                        "original_end": e.original_end,
                        "new_end": e.new_end,
                        "time_removed": e.time_removed,
                        "cut_start": e.cut_start,
                    }
                    for e in edits
                ]
                with open(edits_path, "w") as f:
                    json.dump(edits_data, f)

        return shortened_path, edits

    def _create_background_segments(self, video_path: str, lines, audio_path: str):
        logger.info("🎨 Creating background segments...")
        from ..core.backgrounds import BackgroundProcessor
        from moviepy import AudioFileClip

        with AudioFileClip(audio_path) as clip:
            duration = clip.duration
        processor = BackgroundProcessor()
        return processor.create_background_segments(video_path, lines, duration)

    def _render_video(self, video_settings: Optional[Dict[str, Any]] = None, **kwargs):
        logger.info("🎬 Rendering karaoke video...")
        from .video_writer import render_karaoke_video

        if video_settings:
            kwargs.update(video_settings)
        render_karaoke_video(**kwargs)

    def cleanup_temp_files(self):
        for temp_file in self._temp_files:
            try:
                Path(temp_file).unlink(missing_ok=True)
            except Exception:
                pass
        self._temp_files.clear()
