"""Main karaoke generator orchestrating all components."""

from pathlib import Path
from typing import Dict, Optional, Any
from time import time
import musicbrainzngs

from ..config import get_cache_dir
from ..utils.cache import CacheManager
from ..utils.logging import get_logger
from ..utils.validation import sanitize_filename
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
        self._temp_files = []
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
        whisper_language: Optional[str] = None,
        whisper_model: str = "base",
        shorten_breaks: bool = False,
        max_break_duration: float = 20.0,
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
            whisper_language=whisper_language,
            whisper_model=whisper_model,
        )

        # Step 7: Apply audio effects to instrumental (vocals removed)
        processed_instrumental = apply_audio_effects(
            separation_result["instrumental_path"],
            key_shift,
            tempo_multiplier,
            video_id,
            self.cache_manager,
            self.audio_processor,
            force=force_reprocess,
        )

        # Step 7b: Optionally shorten long instrumental breaks
        break_edits = []
        if shorten_breaks:
            processed_instrumental, break_edits = self._shorten_breaks(
                processed_instrumental,
                separation_result["vocals_path"],
                video_id,
                max_break_duration,
                force=force_reprocess,
            )

        # Step 8: Scale lyrics timing
        scaled_lines = self._scale_lyrics_timing(lyrics_result["lines"], tempo_multiplier)

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

        # Step 10: Generate output path
        if output_path is None:
            safe_title = sanitize_filename(final_title)
            output_path = Path.cwd() / f"{safe_title}_karaoke.mp4"

        # Step 11: Create background segments
        background_segments = None
        if use_backgrounds and video_path:
            background_segments = self._create_background_segments(video_path, scaled_lines, processed_instrumental)

        # Step 12: Render video
        self._render_video(
            lines=scaled_lines,
            audio_path=processed_instrumental,
            output_path=output_path,
            title=final_title,
            artist=final_artist,
            timing_offset=offset,
            background_segments=background_segments,
            song_metadata=lyrics_result.get("metadata"),
            video_settings=video_settings,
        )

        total_time = time() - total_start
        logger.info(f"✅ Karaoke generation complete: {output_path} ({total_time:.1f}s)")

        return {"output_path": str(output_path), "title": final_title, "artist": final_artist, "video_id": video_id}

    # ------------------------
    # Helper methods
    # ------------------------
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
                return {"audio_path": str(original_audio[0]), "title": metadata["title"], "artist": metadata["artist"]}

        logger.info("📥 Downloading audio...")

        cache_dir = self.cache_manager.get_video_cache_dir(video_id)
        result = self.downloader.download_audio(url, cache_dir)
        self.cache_manager.save_metadata(video_id, {"title": result["title"], "artist": result["artist"]})
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

    def _get_lyrics(self, title: str, artist: str, vocals_path: str, video_id: str, force: bool, lyrics_offset: Optional[float] = None, target_duration: Optional[int] = None, evaluate_sources: bool = False, use_whisper: bool = False, whisper_language: Optional[str] = None, whisper_model: str = "base") -> Dict[str, Any]:
        logger.info("📝 Fetching lyrics...")
        from ..core.lyrics import get_lyrics_simple
        lines, metadata = get_lyrics_simple(
            title=title, artist=artist, vocals_path=vocals_path, lyrics_offset=lyrics_offset, romanize=True, target_duration=target_duration, evaluate_sources=evaluate_sources, use_whisper=use_whisper, whisper_language=whisper_language, whisper_model=whisper_model
        )
        return {"lines": lines, "metadata": metadata}

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

    def _shorten_breaks(self, instrumental_path: str, vocals_path: str, video_id: str, max_break_duration: float, force: bool = False):
        """Shorten long instrumental breaks."""
        import json
        from .break_shortener import shorten_instrumental_breaks, BreakEdit

        # Check cache (both audio and edits)
        shortened_name = f"shortened_breaks_{max_break_duration:.0f}s.wav"
        edits_name = f"shortened_breaks_{max_break_duration:.0f}s_edits.json"

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
                        )
                        for e in edits_data
                    ]
                    logger.info(f"📁 Using cached shortened audio ({len(edits)} break edits)")
                    return str(self.cache_manager.get_file_path(video_id, shortened_name)), edits
                except Exception as e:
                    logger.debug(f"Could not load cached edits: {e}")
            else:
                logger.info("📁 Using cached shortened audio (no edits)")
                return str(self.cache_manager.get_file_path(video_id, shortened_name)), []

        logger.info(f"✂️ Shortening instrumental breaks longer than {max_break_duration:.0f}s...")
        output_path = self.cache_manager.get_file_path(video_id, shortened_name)

        shortened_path, edits = shorten_instrumental_breaks(
            instrumental_path,
            vocals_path,
            str(output_path),
            max_break_duration=max_break_duration,
        )

        # Cache the edits for future runs
        if edits:
            edits_path = self.cache_manager.get_file_path(video_id, edits_name)
            edits_data = [
                {
                    "original_start": e.original_start,
                    "original_end": e.original_end,
                    "new_end": e.new_end,
                    "time_removed": e.time_removed,
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
