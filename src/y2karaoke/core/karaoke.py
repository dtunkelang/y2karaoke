"""Main karaoke generator orchestrating all components."""

import tempfile
from pathlib import Path
from typing import Dict, Optional, Any

from ..config import get_cache_dir
from ..core.downloader import YouTubeDownloader, extract_video_id
from ..core.separator import AudioSeparator
from ..core.audio_effects import AudioProcessor
from ..exceptions import Y2KaraokeError
from ..utils.cache import CacheManager
from ..utils.logging import get_logger
from ..utils.validation import sanitize_filename

logger = get_logger(__name__)

class KaraokeGenerator:
    """Main class orchestrating karaoke video generation."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or get_cache_dir()
        self.cache_manager = CacheManager(self.cache_dir)
        self.downloader = YouTubeDownloader(self.cache_dir)
        self.separator = AudioSeparator()
        self.audio_processor = AudioProcessor()
        self._temp_files = []
    
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
        use_backgrounds: bool = False,
        force_reprocess: bool = False,
    ) -> Dict[str, Any]:
        """Generate karaoke video from YouTube URL."""
        import time
        
        total_start = time.time()
        
        print("\n" + "=" * 60)
        print("🎤 KARAOKE VIDEO GENERATION")
        print("=" * 60)
        logger.info(f"URL: {url}")
        
        # Extract video ID for caching
        video_id = extract_video_id(url)
        logger.info(f"Video ID: {video_id}")
        print(f"Video ID: {video_id}\n")
        
        # Auto-cleanup cache if needed
        self.cache_manager.auto_cleanup()
        
        try:
            # Step 1: Download audio
            print("[1/6] Downloading audio...")
            step_start = time.time()
            audio_result = self._download_audio(video_id, url, force_reprocess)
            print(f"  ✓ Completed in {time.time() - step_start:.1f}s\n")
            
            # Step 2: Download video if backgrounds requested
            video_path = None
            if use_backgrounds:
                print("[2/6] Downloading video for backgrounds...")
                step_start = time.time()
                video_result = self._download_video(video_id, url, force_reprocess)
                video_path = video_result['video_path']
                print(f"  ✓ Completed in {time.time() - step_start:.1f}s\n")
            
            # Step 3: Trim audio if needed
            if audio_start > 0:
                print(f"[3/6] Trimming audio (skip first {audio_start}s)...")
                step_start = time.time()
            effective_audio_path = self._trim_audio_if_needed(
                audio_result['audio_path'], audio_start, video_id, force_reprocess
            )
            if audio_start > 0:
                print(f"  ✓ Completed in {time.time() - step_start:.1f}s\n")
            
            # Step 4: Separate vocals
            print("[4/6] Separating vocals from instrumental...")
            step_start = time.time()
            separation_result = self._separate_vocals(
                effective_audio_path, video_id, force_reprocess
            )
            print(f"  ✓ Completed in {time.time() - step_start:.1f}s\n")
            
            # Step 5: Get lyrics
            print("[5/6] Fetching lyrics and timing...")
            step_start = time.time()
            lyrics_result = self._get_lyrics(
                lyrics_title or audio_result['title'],
                lyrics_artist or audio_result['artist'],
                separation_result['vocals_path'],
                video_id,
                force_reprocess
            )
            print(f"  ✓ Completed in {time.time() - step_start:.1f}s")
            print(f"  Found {len(lyrics_result['lines'])} lines of lyrics\n")
            
            # Step 6: Apply audio effects
            if key_shift != 0 or tempo_multiplier != 1.0:
                print(f"[6/6] Applying audio effects (key: {key_shift:+d}, tempo: {tempo_multiplier:.2f}x)...")
                step_start = time.time()
            processed_instrumental = self._apply_audio_effects(
                separation_result['instrumental_path'],
                key_shift,
                tempo_multiplier,
                video_id,
                force_reprocess
            )
            if key_shift != 0 or tempo_multiplier != 1.0:
                print(f"  ✓ Completed in {time.time() - step_start:.1f}s\n")
            
            # Step 7: Scale lyrics timing for tempo changes
            scaled_lines = self._scale_lyrics_timing(
                lyrics_result['lines'], tempo_multiplier
            )
            
            # Step 8: Generate output path if not provided
            if output_path is None:
                safe_title = sanitize_filename(audio_result['title'])
                output_path = Path.cwd() / f"{safe_title}_karaoke.mp4"
            
            # Step 9: Create background segments if requested
            background_segments = None
            if use_backgrounds and video_path:
                print("Creating background segments...")
                step_start = time.time()
                background_segments = self._create_background_segments(
                    video_path, scaled_lines, processed_instrumental
                )
                print(f"  ✓ Completed in {time.time() - step_start:.1f}s\n")
            
            # Step 10: Render karaoke video
            self._render_video(
                lines=scaled_lines,
                audio_path=processed_instrumental,
                output_path=output_path,
                title=lyrics_title or audio_result['title'],
                artist=lyrics_artist or audio_result['artist'],
                timing_offset=offset,
                background_segments=background_segments,
                song_metadata=lyrics_result.get('metadata'),
            )
            
            total_time = time.time() - total_start
            print(f"\n{'=' * 60}")
            print(f"✅ KARAOKE GENERATION COMPLETE")
            print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
            print(f"Output: {output_path}")
            print(f"{'=' * 60}\n")
            
            logger.info("✅ Karaoke generation completed successfully")
            
            return {
                'output_path': str(output_path),
                'title': audio_result['title'],
                'artist': audio_result['artist'],
                'video_id': video_id,
            }
            
        except Exception as e:
            logger.error(f"❌ Karaoke generation failed: {e}")
            raise Y2KaraokeError(f"Generation failed: {e}") from e
    
    def _download_audio(self, video_id: str, url: str, force: bool) -> Dict[str, str]:
        """Download audio with caching."""
        logger.info("📥 Downloading audio...")
        
        # Check cache first
        metadata = self.cache_manager.load_metadata(video_id)
        if metadata and not force:
            audio_files = self.cache_manager.find_files(video_id, "*.wav")
            # Filter out stems
            original_audio = [
                f for f in audio_files 
                if not any(stem in f.name for stem in ['_Vocals', '_Bass', '_Drums', '_Other', '_instrumental'])
            ]
            
            if original_audio:
                logger.info("📁 Using cached audio")
                return {
                    'audio_path': str(original_audio[0]),
                    'title': metadata['title'],
                    'artist': metadata['artist'],
                }
        
        # Download fresh
        cache_dir = self.cache_manager.get_video_cache_dir(video_id)
        result = self.downloader.download_audio(url, cache_dir)
        
        # Save metadata
        self.cache_manager.save_metadata(video_id, {
            'title': result['title'],
            'artist': result['artist'],
        })
        
        return result
    
    def _download_video(self, video_id: str, url: str, force: bool) -> Dict[str, str]:
        """Download video with caching."""
        logger.info("📹 Downloading video...")
        
        if not force:
            video_files = self.cache_manager.find_files(video_id, "*_video.*")
            if video_files:
                logger.info("📁 Using cached video")
                return {'video_path': str(video_files[0])}
        
        cache_dir = self.cache_manager.get_video_cache_dir(video_id)
        return self.downloader.download_video(url, cache_dir)
    
    def _trim_audio_if_needed(
        self, audio_path: str, start_time: float, video_id: str, force: bool
    ) -> str:
        """Trim audio start if requested."""
        if start_time <= 0:
            return audio_path
        
        logger.info(f"✂️ Trimming audio from {start_time:.2f}s")
        
        # Check cache
        trimmed_name = f"trimmed_from_{start_time:.2f}s.wav"
        if not force and self.cache_manager.file_exists(video_id, trimmed_name):
            logger.info("📁 Using cached trimmed audio")
            return str(self.cache_manager.get_file_path(video_id, trimmed_name))
        
        # Trim audio
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
        """Separate vocals with caching."""
        logger.info("🎵 Separating vocals...")
        
        # Never try to separate an already-separated file
        audio_filename = Path(audio_path).name.lower()
        if any(marker in audio_filename for marker in ['vocals', 'instrumental', 'drums', 'bass', 'other']):
            # This is already a separated file, find the original separation
            cache_dir = self.cache_manager.get_video_cache_dir(video_id)
            vocals_files = list(Path(cache_dir).glob("*[Vv]ocals*.wav"))
            instrumental_files = list(Path(cache_dir).glob("*instrumental*.wav"))
            
            if vocals_files and instrumental_files:
                return {
                    'vocals_path': str(vocals_files[0]),
                    'instrumental_path': str(instrumental_files[0]),
                }
            raise RuntimeError(f"Found separated file but missing vocals/instrumental: {audio_path}")
        
        if not force:
            vocals_files = self.cache_manager.find_files(video_id, "*vocals*.wav")
            instrumental_files = self.cache_manager.find_files(video_id, "*instrumental*.wav")
            
            if vocals_files and instrumental_files:
                logger.info("📁 Using cached separation")
                return {
                    'vocals_path': str(vocals_files[0]),
                    'instrumental_path': str(instrumental_files[0]),
                }
        
        cache_dir = self.cache_manager.get_video_cache_dir(video_id)
        return self.separator.separate_vocals(audio_path, str(cache_dir))
    
    def _get_lyrics(
        self, title: str, artist: str, vocals_path: str, video_id: str, force: bool
    ) -> Dict[str, Any]:
        """Get lyrics with caching."""
        logger.info("📝 Fetching lyrics...")
        
        # Import here to avoid circular imports
        from ..core.lyrics import LyricsProcessor
        
        processor = LyricsProcessor()
        cache_dir = self.cache_manager.get_video_cache_dir(video_id)
        
        return processor.get_lyrics(title, artist, vocals_path, str(cache_dir), force)
    
    def _apply_audio_effects(
        self, audio_path: str, key_shift: int, tempo: float, video_id: str, force: bool
    ) -> str:
        """Apply audio effects with caching."""
        if key_shift == 0 and tempo == 1.0:
            return audio_path
        
        logger.info(f"🎛️ Applying effects: key={key_shift:+d}, tempo={tempo:.2f}x")
        
        # Check cache
        effects_name = f"instrumental_key{key_shift:+d}_tempo{tempo:.2f}.wav"
        if not force and self.cache_manager.file_exists(video_id, effects_name):
            logger.info("📁 Using cached processed audio")
            return str(self.cache_manager.get_file_path(video_id, effects_name))
        
        # Process audio
        output_path = self.cache_manager.get_file_path(video_id, effects_name)
        return self.audio_processor.process_audio(
            audio_path, str(output_path), key_shift, tempo
        )
    
    def _scale_lyrics_timing(self, lines, tempo_multiplier: float):
        """Scale lyrics timing for tempo changes."""
        if tempo_multiplier == 1.0:
            return lines
        
        logger.info(f"⏱️ Scaling lyrics timing for {tempo_multiplier:.2f}x tempo")
        
        # Import here to avoid circular imports
        from ..core.lyrics import Line, Word
        
        scaled_lines = []
        for line in lines:
            scaled_words = [
                Word(
                    text=word.text,
                    start_time=word.start_time / tempo_multiplier,
                    end_time=word.end_time / tempo_multiplier,
                )
                for word in line.words
            ]
            scaled_lines.append(
                Line(
                    words=scaled_words,
                    start_time=line.start_time / tempo_multiplier,
                    end_time=line.end_time / tempo_multiplier,
                )
            )
        return scaled_lines
    
    def _create_background_segments(self, video_path: str, lines, audio_path: str):
        """Create background segments from video."""
        logger.info("🎨 Creating background segments...")
        
        # Import here to avoid circular imports
        from ..core.backgrounds import BackgroundProcessor
        from moviepy import AudioFileClip
        
        # Get audio duration
        with AudioFileClip(audio_path) as clip:
            duration = clip.duration
        
        processor = BackgroundProcessor()
        return processor.create_background_segments(video_path, lines, duration)
    
    def _render_video(self, **kwargs):
        """Render the final karaoke video."""
        logger.info("🎬 Rendering karaoke video...")
        
        # Import here to avoid circular imports
        from ..core.renderer import render_karaoke_video
        
        # Use the standalone function (not the class method)
        render_karaoke_video(**kwargs)
    
    def upload_video(self, video_path: str, title: str, artist: str) -> Dict[str, str]:
        """Upload video to YouTube."""
        logger.info("📤 Uploading to YouTube...")
        
        # Import here to avoid circular imports
        from ..core.uploader import YouTubeUploader
        
        uploader = YouTubeUploader()
        return uploader.upload_video(video_path, title, artist)
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        for temp_file in self._temp_files:
            try:
                Path(temp_file).unlink(missing_ok=True)
            except Exception:
                pass
        self._temp_files.clear()
