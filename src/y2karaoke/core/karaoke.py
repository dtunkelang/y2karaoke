"""Main karaoke generator orchestrating all components."""

from pathlib import Path
from typing import Dict, Optional, Any
import string
import re
from time import time
from difflib import SequenceMatcher
import musicbrainzngs

from ..config import get_cache_dir
from ..core.downloader import YouTubeDownloader, extract_video_id
from ..core.separator import AudioSeparator
from ..core.audio_effects import AudioProcessor
from ..exceptions import Y2KaraokeError
from ..utils.cache import CacheManager
from ..utils.logging import get_logger
from ..utils.validation import sanitize_filename
from ..core.title_resolver import resolve_artist_title_from_youtube


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
    ) -> Dict[str, Any]:
        """Generate karaoke video from YouTube URL."""
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
        effective_audio_path = self._trim_audio_if_needed(
            audio_result["audio_path"], audio_start, video_id, force_reprocess
        )

        # Step 4: Separate vocals
        separation_result = self._separate_vocals(effective_audio_path, video_id, force_reprocess)

        # Step 5: Determine final artist/title for lyrics
        if original_prompt and not lyrics_title and not lyrics_artist:
            if url.startswith("http"):
                try:
                    youtube_title = self.downloader.get_video_title(url)
                except Exception as e:
                    logger.warning(f"Failed to fetch YouTube title: {e}")
                    youtube_title = original_prompt
            else:
                youtube_title = original_prompt

            # Get YouTube uploader/channel as artist hint
            youtube_artist_hint = self.downloader.get_video_uploader(url)

            # Use the robust candidate resolver with hint
            final_artist, final_title = resolve_artist_title_from_youtube(
                youtube_title,
                youtube_artist=youtube_artist_hint,
                fallback_artist=audio_result['artist'],
                fallback_title=audio_result['title']
            )
        else:
            final_artist = lyrics_artist if lyrics_artist else audio_result['artist']
            final_title = lyrics_title if lyrics_title else audio_result['title']

        # Step 6: Fetch lyrics
        lyrics_result = self._get_lyrics(
            final_title,
            final_artist,
            separation_result["vocals_path"],
            video_id,
            force_reprocess,
            lyrics_offset=lyrics_offset,
        )
                
        # Step 7: Apply audio effects
        processed_instrumental = self._apply_audio_effects(
            separation_result["instrumental_path"], key_shift, tempo_multiplier, video_id, force_reprocess
        )

        # Step 8: Scale lyrics timing
        scaled_lines = self._scale_lyrics_timing(lyrics_result["lines"], tempo_multiplier)

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
        logger.info("📥 Downloading audio...")
        metadata = self.cache_manager.load_metadata(video_id)
        if metadata and not force:
            audio_files = self.cache_manager.find_files(video_id, "*.wav")
            original_audio = [
                f for f in audio_files
                if not any(stem in f.name for stem in ["_Vocals", "_Bass", "_Drums", "_Other", "_instrumental"])
            ]
            if original_audio:
                logger.info("📁 Using cached audio")
                return {"audio_path": str(original_audio[0]), "title": metadata["title"], "artist": metadata["artist"]}

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

    def _trim_audio_if_needed(self, audio_path: str, start_time: float, video_id: str, force: bool) -> str:
        if start_time <= 0:
            return audio_path
        logger.info(f"✂️ Trimming audio from {start_time:.2f}s")
        trimmed_name = f"trimmed_from_{start_time:.2f}s.wav"
        if not force and self.cache_manager.file_exists(video_id, trimmed_name):
            logger.info("📁 Using cached trimmed audio")
            return str(self.cache_manager.get_file_path(video_id, trimmed_name))
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(audio_path)
        start_ms = int(start_time * 1000)
        if start_ms >= len(audio):
            logger.warning("Start time beyond audio length, using original")
            return audio_path
        trimmed = audio[start_ms:]
        trimmed_path = self.cache_manager.get_file_path(video_id, trimmed_name)
        trimmed.export(str(trimmed_path), format="wav")
        return str(trimmed_path)

    def _separate_vocals(self, audio_path: str, video_id: str, force: bool) -> Dict[str, str]:
        logger.info("🎵 Separating vocals...")
        audio_filename = Path(audio_path).name.lower()
        if any(marker in audio_filename for marker in ["vocals", "instrumental", "drums", "bass", "other"]):
            cache_dir = self.cache_manager.get_video_cache_dir(video_id)
            vocals_files = list(Path(cache_dir).glob("*[Vv]ocals*.wav"))
            instrumental_files = list(Path(cache_dir).glob("*instrumental*.wav"))
            if vocals_files and instrumental_files:
                return {"vocals_path": str(vocals_files[0]), "instrumental_path": str(instrumental_files[0])}
            raise RuntimeError(f"Found separated file but missing vocals/instrumental: {audio_path}")
        if not force:
            vocals_files = self.cache_manager.find_files(video_id, "*vocals*.wav")
            instrumental_files = self.cache_manager.find_files(video_id, "*instrumental*.wav")
            if vocals_files and instrumental_files:
                logger.info("📁 Using cached separation")
                return {"vocals_path": str(vocals_files[0]), "instrumental_path": str(instrumental_files[0])}
        cache_dir = self.cache_manager.get_video_cache_dir(video_id)
        return self.separator.separate_vocals(audio_path, str(cache_dir))

    def _get_lyrics(self, title: str, artist: str, vocals_path: str, video_id: str, force: bool, lyrics_offset: Optional[float] = None) -> Dict[str, Any]:
        logger.info("📝 Fetching lyrics...")
        from ..core.lyrics import get_lyrics_simple
        lines, metadata = get_lyrics_simple(title=title, artist=artist, vocals_path=vocals_path, lyrics_offset=lyrics_offset, romanize=True)
        return {"lines": lines, "metadata": metadata}

    def _apply_audio_effects(self, audio_path: str, key_shift: int, tempo: float, video_id: str, force: bool) -> str:
        if key_shift == 0 and tempo == 1.0:
            return audio_path
        logger.info(f"🎛️ Applying effects: key={key_shift:+d}, tempo={tempo:.2f}x")
        effects_name = f"instrumental_key{key_shift:+d}_tempo{tempo:.2f}.wav"
        if not force and self.cache_manager.file_exists(video_id, effects_name):
            logger.info("📁 Using cached processed audio")
            return str(self.cache_manager.get_file_path(video_id, effects_name))
        output_path = self.cache_manager.get_file_path(video_id, effects_name)
        return self.audio_processor.process_audio(audio_path, str(output_path), key_shift, tempo)

    def _scale_lyrics_timing(self, lines, tempo_multiplier: float):
        if tempo_multiplier == 1.0:
            return lines
        logger.info(f"⏱️ Scaling lyrics timing for {tempo_multiplier:.2f}x tempo")
        from ..core.lyrics import Line, Word
        scaled_lines = []
        for line in lines:
            scaled_words = [Word(text=w.text, start_time=w.start_time / tempo_multiplier, end_time=w.end_time / tempo_multiplier, singer=w.singer) for w in line.words]
            scaled_lines.append(Line(words=scaled_words, singer=line.singer))
        return scaled_lines

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
        from ..core.renderer import render_karaoke_video
        if video_settings:
            kwargs.update(video_settings)
        render_karaoke_video(**kwargs)

    def upload_video(self, video_path: str, title: str, artist: str) -> Dict[str, str]:
        logger.info("📤 Uploading to YouTube...")
        from ..core.uploader import YouTubeUploader
        uploader = YouTubeUploader()
        return uploader.upload_video(video_path, title, artist)

    def cleanup_temp_files(self):
        for temp_file in self._temp_files:
            try:
                Path(temp_file).unlink(missing_ok=True)
            except Exception:
                pass
        self._temp_files.clear()
