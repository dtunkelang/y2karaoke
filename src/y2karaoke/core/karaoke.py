"""Main karaoke generator orchestrating all components."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from time import time
import musicbrainzngs

from ..config import get_cache_dir
from ..exceptions import Y2KaraokeError
from ..utils.cache import CacheManager
from ..utils.logging import get_logger
from ..utils.validation import fix_line_order, sanitize_filename, validate_line_order
from .downloader import YouTubeDownloader, extract_video_id
from .separator import AudioSeparator
from .audio_effects import AudioProcessor
from .audio_utils import trim_audio_if_needed, apply_audio_effects, separate_vocals
from .models import compute_word_slots

logger = get_logger(__name__)


def _normalize_word_text(raw: str) -> str:
    """Normalize a word for comparison (lowercase alpha + apostrophes)."""
    normalized = "".join(ch for ch in (raw or "").lower() if ch.isalpha() or ch == "'")
    return normalized


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

        self._original_prompt = original_prompt
        total_start = time()

        video_id = extract_video_id(url)
        logger.info(f"Video ID: {video_id}")

        self.cache_manager.auto_cleanup()

        audio_result, video_path, separation_result = self._prepare_media(
            url,
            video_id,
            audio_start,
            use_backgrounds,
            force_reprocess,
            offline,
        )

        final_title, final_artist = self._resolve_final_metadata(
            audio_result, lyrics_title, lyrics_artist
        )

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
            whisper_map_lrc_dtw=whisper_map_lrc_dtw,
            lyrics_file=lyrics_file,
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

        processed_audio, break_edits = self._process_audio_track(
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
            self._append_outro_line(
                lines=lyrics_result["lines"],
                outro_line=outro_line,
                audio_path=processed_audio,
            )

        scaled_lines = self._scale_lyrics_timing(
            lyrics_result["lines"], tempo_multiplier
        )
        scaled_lines = self._apply_break_edits(scaled_lines, break_edits)
        scaled_lines = self._apply_splash_offset(scaled_lines, min_start=3.5)
        scaled_lines = fix_line_order(scaled_lines)
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

        output_path = output_path or self._build_output_path(final_title)
        background_segments = self._build_background_segments(
            use_backgrounds, video_path, scaled_lines, processed_audio
        )

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

        quality_score, quality_issues, quality_level, quality_emoji = (
            self._summarize_quality(lyrics_result)
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

    # ------------------------
    # Helper methods
    # ------------------------
    def _write_timing_report(  # noqa: C901
        self,
        lines: List[Line],
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
                    "words": [
                        {
                            "text": w.text,
                            "start": round(w.start_time, 3),
                            "end": round(w.end_time, 3),
                        }
                        for w in line.words
                    ],
                    "word_slots": [
                        round(slot, 3)
                        for slot in compute_word_slots(line.words, line.end_time)
                    ],
                    "word_spoken": [
                        round(w.end_time - w.start_time, 3) for w in line.words
                    ],
                }
                for idx, line in enumerate(lines)
                if line.words
            ],
        }

        dtw_metrics = lyrics_result.get("quality", {}).get("dtw_metrics", {})
        if dtw_metrics:
            report["dtw_word_coverage"] = round(
                dtw_metrics.get("word_coverage", 0.0), 3
            )
            report["dtw_line_coverage"] = round(
                dtw_metrics.get("line_coverage", 0.0), 3
            )

        if video_id:
            try:
                from .phonetic_utils import _phonetic_similarity

                cache_dir = self.cache_manager.get_video_cache_dir(video_id)
                whisper_files = list(cache_dir.glob("*_whisper_*.json"))
                if whisper_files:
                    whisper_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
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
                    all_words = []
                    for seg in segments:
                        for w in seg.get("words", []) or []:
                            if w.get("start") is None:
                                continue
                            all_words.append(w)
                all_words.sort(key=lambda w: w.get("start", 0.0))
                report["whisper_word_count"] = len(all_words)
                report["whisper_window_low_conf_threshold"] = 0.5
                for idx, line in enumerate(report["lines"]):
                    next_start = (
                        report["lines"][idx + 1]["start"]
                        if idx + 1 < len(report["lines"])
                        else line["end"] + 2.0
                    )
                    window_start = line["start"] - 1.0
                    window_words = [
                        w
                        for w in all_words
                        if window_start <= w.get("start", 0.0) < next_start
                    ]
                    probs = [
                        w.get("probability")
                        for w in window_words
                        if w.get("probability") is not None
                    ]
                    low_conf = sum(
                        1
                        for w in window_words
                        if w.get("probability") is not None
                        and w.get("probability") < 0.5
                    )
                    line["whisper_window_start"] = round(window_start, 2)
                    line["whisper_window_end"] = round(next_start, 2)
                    line["whisper_window_word_count"] = len(window_words)
                    line["whisper_window_low_conf_count"] = low_conf
                    line["whisper_window_avg_prob"] = (
                        round(sum(probs) / len(probs), 3) if probs else None
                    )
                    line["whisper_window_words"] = [
                        {
                            "text": w.get("text", ""),
                            "start": round(w.get("start", 0.0), 2),
                            "end": round(w.get("end", 0.0), 2),
                            "probability": (
                                round(w.get("probability", 0.0), 3)
                                if w.get("probability") is not None
                                else None
                            ),
                        }
                        for w in window_words
                    ]
                    line_delta = None
                    if line.get("words") and window_words:
                        first_line_word = _normalize_word_text(line["words"][0]["text"])
                        target_start = None
                        for w in window_words:
                            if (
                                _normalize_word_text(w.get("text", ""))
                                == first_line_word
                            ):
                                target_start = w.get("start")
                                break
                        if target_start is not None:
                            delta = target_start - line["start"]
                            line_delta = round(delta, 3)
                            if delta > 0:
                                line["start"] = round(line["start"] + delta, 2)
                                line["end"] = round(line["end"] + delta, 2)
                                for word_entry in line["words"]:
                                    word_entry["start"] = round(
                                        word_entry["start"] + delta, 3
                                    )
                                    word_entry["end"] = round(
                                        word_entry["end"] + delta, 3
                                    )
                    line["whisper_line_start_delta"] = line_delta
                low_conf_lines: List[Dict[str, Any]] = []
                for line_entry in report["lines"]:
                    avg_prob = line_entry.get("whisper_window_avg_prob")
                    low_conf_count = line_entry.get("whisper_window_low_conf_count", 0)
                    total_words = line_entry.get("whisper_window_word_count", 0)
                    low_conf_ratio = (
                        (low_conf_count / total_words) if total_words else 0.0
                    )
                    if avg_prob is not None and (
                        avg_prob < 0.35 or low_conf_ratio >= 0.5
                    ):
                        low_conf_lines.append(
                            {
                                "index": line_entry["index"],
                                "text": line_entry["text"],
                                "whisper_window_avg_prob": avg_prob,
                                "low_conf_ratio": round(low_conf_ratio, 2),
                            }
                        )
                report["low_confidence_lines"] = low_conf_lines
                if low_conf_lines:
                    quality = lyrics_result.get("quality")
                    if quality is not None:
                        issues = quality.setdefault("issues", [])
                        issue_msg = (
                            f"{len(low_conf_lines)} line(s) had low Whisper confidence"
                        )
                        if issue_msg not in issues:
                            issues.append(issue_msg)
                last_used_segment_idx: Dict[str, int] = {}
                for line in report["lines"]:
                    text_norm = (
                        line.get("text", "").strip().lower() if line.get("text") else ""
                    )
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
                        if best_start_delta is None or start_delta < best_start_delta:
                            best_start_delta = start_delta
                            nearest_start = seg
                        if best_end_delta is None or end_delta < best_end_delta:
                            best_end_delta = end_delta
                            nearest_end = seg
                    best_seg = None
                    best_delta = None
                    best_sim = -1.0
                    best_seg_idx = None
                    prev_segment_idx = last_used_segment_idx.get(text_norm)
                prev_line_end = (
                    report["lines"][idx - 1]["end"] if idx > 0 else line["start"] - 0.01
                )
                for seg_idx, seg in enumerate(segments):
                    if prev_segment_idx is not None and seg_idx <= prev_segment_idx:
                        continue
                    seg_start = seg.get("start", 0.0)
                    if seg_start < line["start"] - 0.6:
                        continue
                    if seg_start < prev_line_end - 0.15:
                        continue
                    delta = abs(seg_start - line["start"])
                    if delta > 15.0:
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
                        if (
                            best_delta is None
                            or delta < best_delta
                            or (delta == best_delta and sim > best_sim)
                        ):
                            best_delta = delta
                            best_sim = sim
                            best_seg = seg
                            best_seg_idx = seg_idx
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
                        line["nearest_segment_end_text"] = nearest_end.get("text", "")
                    if best_seg:
                        line["best_segment_start"] = round(
                            best_seg.get("start", 0.0), 2
                        )
                        line["best_segment_end"] = round(best_seg.get("end", 0.0), 2)
                        line["best_segment_text"] = best_seg.get("text", "")
                        if text_norm:
                            last_used_segment_idx[text_norm] = best_seg_idx or 0
                    if prior_seg is not None:
                        line["prior_segment_start"] = round(
                            prior_seg.get("start", 0.0), 2
                        )
                        line["prior_segment_end"] = round(prior_seg.get("end", 0.0), 2)
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
        if not lines or lines[0].start_time >= min_start:
            return lines
        splash_offset = min_start - lines[0].start_time
        from ..core.lyrics import Line, Word

        offset_lines = []
        for line in lines:
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
        return offset_lines

    def _append_outro_line(
        self,
        lines: List[Line],
        outro_line: str,
        audio_path: str,
        min_tail: float = 0.5,
    ) -> None:
        """Append a final lyric line near the end of the audio."""
        if not lines or not outro_line.strip():
            return

        from moviepy import AudioFileClip
        from ..config import OUTRO_DELAY
        from ..core.lyrics import Line, Word

        with AudioFileClip(audio_path) as clip:
            audio_duration = clip.duration

        last_end = lines[-1].end_time
        end_time = max(last_end + min_tail, audio_duration - OUTRO_DELAY)
        duration = min(3.0, max(1.5, end_time - last_end))
        start_time = max(last_end + min_tail, end_time - duration)
        if end_time <= start_time + 0.2:
            return

        tokens = [t for t in outro_line.strip().split() if t]
        if not tokens:
            return
        spacing = duration / len(tokens)
        words = []
        for i, token in enumerate(tokens):
            start = start_time + i * spacing
            end = start + spacing * 0.9
            words.append(Word(text=token, start_time=start, end_time=end))
        lines.append(Line(words=words))

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
        lyrics_quality = lyrics_result.get("quality", {})
        quality_score = lyrics_quality.get("overall_score", 50.0)
        quality_issues = lyrics_quality.get("issues", [])

        if quality_score >= 80:
            quality_emoji = "✅"
            quality_level = "high"
        elif quality_score >= 50:
            quality_emoji = "⚠️"
            quality_level = "medium"
        else:
            quality_emoji = "❌"
            quality_level = "low"

        return quality_score, quality_issues, quality_level, quality_emoji

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
            whisper_map_lrc_dtw=whisper_map_lrc_dtw,
            lyrics_file=lyrics_file,
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
