"""Main karaoke generator orchestrating all components."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Any, Tuple, TYPE_CHECKING
import musicbrainzngs

from ..config import get_cache_dir
from ..exceptions import Y2KaraokeError
from ..utils.cache import CacheManager
from ..utils.logging import get_logger
from ..utils.validation import sanitize_filename
from . import karaoke_utils
from . import karaoke_timing_report
from .karaoke_audio_helpers import append_outro_line_impl, shorten_breaks_impl
from .karaoke_generate import generate_karaoke
from ..pipeline.audio import (
    YouTubeDownloader,
    AudioSeparator,
    AudioProcessor,
    trim_audio_if_needed,
    apply_audio_effects,
    separate_vocals_cached as separate_vocals,
)

logger = get_logger(__name__)


if TYPE_CHECKING:
    from .models import Line

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

    @contextmanager
    def use_test_hooks(
        self,
        *,
        download_audio_fn=None,
        download_video_fn=None,
        get_lyrics_fn=None,
        scale_lyrics_timing_fn=None,
        shorten_breaks_fn=None,
        create_background_segments_fn=None,
        render_video_fn=None,
    ) -> Iterator[None]:
        """Temporarily override internal collaborators for tests."""
        overrides = {
            "_download_audio": download_audio_fn,
            "_download_video": download_video_fn,
            "_get_lyrics": get_lyrics_fn,
            "_scale_lyrics_timing": scale_lyrics_timing_fn,
            "_shorten_breaks": shorten_breaks_fn,
            "_create_background_segments": create_background_segments_fn,
            "_render_video": render_video_fn,
        }
        previous = {
            name: getattr(self, name)
            for name, value in overrides.items()
            if value is not None
        }
        for name, value in overrides.items():
            if value is not None:
                setattr(self, name, value)

        try:
            yield
        finally:
            for name, value in previous.items():
                setattr(self, name, value)

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
        return generate_karaoke(
            self,
            url=url,
            output_path=output_path,
            offset=offset,
            key_shift=key_shift,
            tempo_multiplier=tempo_multiplier,
            audio_start=audio_start,
            lyrics_title=lyrics_title,
            lyrics_artist=lyrics_artist,
            lyrics_offset=lyrics_offset,
            target_duration=target_duration,
            evaluate_lyrics_sources=evaluate_lyrics_sources,
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
            outro_line=outro_line,
            offline=offline,
            filter_promos=filter_promos,
            use_backgrounds=use_backgrounds,
            force_reprocess=force_reprocess,
            video_settings=video_settings,
            original_prompt=original_prompt,
            shorten_breaks=shorten_breaks,
            max_break_duration=max_break_duration,
            debug_audio=debug_audio,
            skip_render=skip_render,
            timing_report_path=timing_report_path,
        )

    # ------------------------
    # Helper methods
    # ------------------------
    def _write_timing_report(
        self,
        lines: List[Line],
        report_path: str,
        title: str,
        artist: str,
        lyrics_result: Dict[str, Any],
        video_id: Optional[str] = None,
    ) -> None:
        """Write a JSON timing report for downstream inspection."""
        karaoke_timing_report.write_timing_report(
            cache_manager=self.cache_manager,
            lines=lines,
            report_path=report_path,
            title=title,
            artist=artist,
            lyrics_result=lyrics_result,
            video_id=video_id,
        )

    def _download_audio(
        self, video_id: str, url: str, force: bool, offline: bool = False
    ) -> Dict[str, str]:
        metadata = self.cache_manager.load_metadata(video_id)
        audio_files = self.cache_manager.find_files(video_id, "*.wav")
        # Filter out separated stems (audio-separator uses parentheses like "(Vocals)")
        separated_stems = ["vocals", "bass", "drums", "other", "instrumental"]
        original_audio = [
            f
            for f in audio_files
            if not any(stem in f.name.lower() for stem in separated_stems)
        ]
        metadata_title = metadata["title"] if metadata else "Unknown"
        metadata_artist = metadata["artist"] if metadata else "Unknown"

        if metadata and not force and original_audio:
            logger.info("📁 Using cached audio")
            return {
                "audio_path": str(original_audio[0]),
                "title": metadata_title,
                "artist": metadata_artist,
            }
        if offline and original_audio:
            logger.warning("📁 Using cached audio in offline mode")
            return {
                "audio_path": str(original_audio[0]),
                "title": metadata_title,
                "artist": metadata_artist,
            }

        if offline:
            raise Y2KaraokeError(
                "Offline mode requires cached audio. Run once online first."
            )

        logger.info("📥 Downloading audio...")

        cache_dir = self.cache_manager.get_video_cache_dir(video_id)
        result = self.downloader.download_audio(url, cache_dir)
        self.cache_manager.save_metadata(
            video_id, {"title": result["title"], "artist": result["artist"]}
        )
        return result

    def _download_video(
        self, video_id: str, url: str, force: bool, offline: bool = False
    ) -> Dict[str, str]:
        logger.info("📹 Downloading video...")
        if not force:
            video_files = self.cache_manager.find_files(video_id, "*_video.*")
            if video_files:
                logger.info("📁 Using cached video")
                return {"video_path": str(video_files[0])}
        if offline:
            raise Y2KaraokeError("Offline mode requires cached video for backgrounds.")
        cache_dir = self.cache_manager.get_video_cache_dir(video_id)
        return self.downloader.download_video(url, cache_dir)

    def _prepare_media(
        self,
        url: str,
        video_id: str,
        audio_start: float,
        use_backgrounds: bool,
        force: bool,
        offline: bool,
    ) -> Tuple[Dict[str, str], Optional[str], Dict[str, str]]:
        audio_result = self._download_audio(video_id, url, force, offline)

        video_path = None
        if use_backgrounds:
            video_result = self._download_video(video_id, url, force, offline)
            video_path = video_result["video_path"]

        effective_audio_path = trim_audio_if_needed(
            audio_result["audio_path"],
            audio_start,
            video_id,
            self.cache_manager,
            force=force,
        )

        separation_result = separate_vocals(
            effective_audio_path,
            video_id,
            self.separator,
            self.cache_manager,
            force=force,
        )

        return audio_result, video_path, separation_result

    def _resolve_final_metadata(
        self,
        audio_result: Dict[str, str],
        lyrics_title: Optional[str],
        lyrics_artist: Optional[str],
    ) -> Tuple[str, str]:
        final_artist = lyrics_artist if lyrics_artist else audio_result["artist"]
        final_title = lyrics_title if lyrics_title else audio_result["title"]
        return final_title, final_artist

    def _process_audio_track(
        self,
        debug_audio: str,
        separation_result: Dict[str, str],
        audio_result: Dict[str, str],
        key_shift: int,
        tempo_multiplier: float,
        video_id: str,
        force: bool,
        shorten_breaks: bool,
        max_break_duration: float,
    ) -> Tuple[str, List[Any]]:
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
            force=force,
            cache_suffix=f"_{debug_audio}" if debug_audio != "instrumental" else "",
        )

        break_edits: List[Any] = []
        if shorten_breaks:
            processed_audio, break_edits = self._shorten_breaks(
                processed_audio,
                separation_result["vocals_path"],
                separation_result["instrumental_path"],
                video_id,
                max_break_duration,
                force=force,
                cache_suffix=f"_{debug_audio}" if debug_audio != "instrumental" else "",
            )

        return processed_audio, break_edits

    def _apply_break_edits(self, lines, break_edits):
        if not break_edits:
            return lines
        from .break_shortener import adjust_lyrics_timing

        return adjust_lyrics_timing(lines, break_edits)

    def _apply_splash_offset(self, lines, min_start: float = 3.5):
        return karaoke_utils.apply_splash_offset(lines, min_start=min_start)

    def _append_outro_line(
        self,
        lines: List[Line],
        outro_line: str,
        audio_path: str,
        min_tail: float = 0.5,
    ) -> None:
        append_outro_line_impl(
            lines,
            outro_line,
            audio_path,
            min_tail=min_tail,
        )

    def _build_output_path(self, title: str) -> Path:
        safe_title = sanitize_filename(title)
        return Path.cwd() / f"{safe_title}_karaoke.mp4"

    def _build_background_segments(
        self,
        use_backgrounds: bool,
        video_path: Optional[str],
        lines,
        processed_audio: str,
    ):
        if use_backgrounds and video_path:
            return self._create_background_segments(video_path, lines, processed_audio)
        return None

    def _summarize_quality(
        self, lyrics_result: Dict[str, Any]
    ) -> Tuple[float, List[str], str, str]:
        return karaoke_utils.summarize_quality(lyrics_result)

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
        offline: bool = False,
        filter_promos: bool = True,
    ) -> Dict[str, Any]:
        logger.info("📝 Fetching lyrics...")
        from ..pipeline.lyrics import get_lyrics_with_quality

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
        )
        return {"lines": lines, "metadata": metadata, "quality": quality_report}

    def _scale_lyrics_timing(self, lines, tempo_multiplier: float):
        if tempo_multiplier != 1.0:
            logger.info(f"⏱️ Scaling lyrics timing for {tempo_multiplier:.2f}x tempo")
        return karaoke_utils.scale_lyrics_timing(lines, tempo_multiplier)

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
        """Shorten long instrumental breaks in the given audio track."""
        return shorten_breaks_impl(
            audio_path,
            vocals_path,
            instrumental_path,
            video_id,
            max_break_duration,
            cache_manager=self.cache_manager,
            force=force,
            cache_suffix=cache_suffix,
        )

    def _create_background_segments(self, video_path: str, lines, audio_path: str):
        logger.info("🎨 Creating background segments...")
        from ..pipeline.render import BackgroundProcessor
        from moviepy import AudioFileClip

        with AudioFileClip(audio_path) as clip:
            duration = clip.duration
        processor = BackgroundProcessor()
        return processor.create_background_segments(video_path, lines, duration)

    def _render_video(self, video_settings: Optional[Dict[str, Any]] = None, **kwargs):
        logger.info("🎬 Rendering karaoke video...")
        from ..pipeline.render import render_karaoke_video

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
