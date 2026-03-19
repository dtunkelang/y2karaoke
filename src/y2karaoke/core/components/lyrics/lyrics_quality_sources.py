"""Source-resolution helpers for lyrics quality workflows."""

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from .lyrics_whisper_pipeline import should_keep_lrc_timings_for_trailing_outro_padding
from ...models import Line, SongMetadata
from ..alignment.alignment_policy import decide_lrc_timing_trust

if TYPE_CHECKING:
    from .lyrics_whisper import LyricsWhisperHooks
    from .runtime_config import LyricsRuntimeConfig


def _should_suppress_disagreement_negative_offset(issues_list: list[str]) -> bool:
    return not any("Ignoring provider LRC timestamps" in issue for issue in issues_list)


def _format_lrc_timestamp(seconds: float) -> str:
    total_centiseconds = max(0, int(round(float(seconds) * 100)))
    minutes, centiseconds = divmod(total_centiseconds, 6000)
    secs, hundredths = divmod(centiseconds, 100)
    return f"[{minutes:02d}:{secs:02d}.{hundredths:02d}]"


def _rebase_line_timings_to_audio_window(
    line_timings: List[Tuple[float, str]],
    *,
    audio_start: float,
    target_duration: Optional[int],
    trailing_line_estimate_sec: float = 5.0,
) -> List[Tuple[float, str]]:
    if not line_timings or audio_start <= 0.0 or not target_duration:
        return line_timings

    clip_start = float(audio_start)
    clip_end = clip_start + float(target_duration)
    rebased: List[Tuple[float, str]] = []
    for index, (line_start, text) in enumerate(line_timings):
        next_start = (
            float(line_timings[index + 1][0])
            if index + 1 < len(line_timings)
            else float(line_start) + float(trailing_line_estimate_sec)
        )
        if next_start <= clip_start or float(line_start) >= clip_end:
            continue
        rebased.append((max(0.0, float(line_start) - clip_start), text))
    return rebased or line_timings


def _build_lrc_text_from_line_timings(
    line_timings: List[Tuple[float, str]],
) -> Optional[str]:
    if not line_timings:
        return None
    return "\n".join(
        f"{_format_lrc_timestamp(timestamp)}{text}"
        for timestamp, text in line_timings
        if text.strip()
    )


def _apply_lrc_timing_trust_policy(
    *,
    line_timings: Optional[List[Tuple[float, str]]],
    lrc_text: Optional[str],
    target_duration: Optional[int],
    drop_lrc_line_timings: bool,
    vocals_path: Optional[str],
    use_whisper: bool,
    whisper_map_lrc: bool,
    issues_list: List[str],
    quality_report: dict,
) -> Optional[List[Tuple[float, str]]]:
    if drop_lrc_line_timings and line_timings:
        issues_list.append(
            "Configured to ignore provider LRC line timings; deriving timing from audio"
        )
        quality_report["lrc_timing_trust"] = "dropped_configured"
        return None

    if target_duration and lrc_text:
        from .sync import get_lrc_duration

        lrc_duration = get_lrc_duration(lrc_text)
        if lrc_duration and abs(target_duration - lrc_duration) > 8:
            lrc_duration_mismatch_sec = abs(target_duration - lrc_duration)
            issues_list.append(
                f"LRC duration mismatch: LRC={lrc_duration}s vs audio={target_duration}s"
            )
            quality_report["lrc_timing_trust"] = "degraded_duration_mismatch"
            can_recover_with_audio_alignment = bool(vocals_path) and bool(
                use_whisper or whisper_map_lrc
            )
            likely_outro_padding = should_keep_lrc_timings_for_trailing_outro_padding(
                line_timings=line_timings,
                lrc_duration=lrc_duration,
                target_duration=target_duration,
            )
            if can_recover_with_audio_alignment and lrc_duration_mismatch_sec >= 12.0:
                decision = decide_lrc_timing_trust(
                    lrc_duration_mismatch_sec=lrc_duration_mismatch_sec,
                    can_recover_with_audio_alignment=can_recover_with_audio_alignment,
                    likely_outro_padding=likely_outro_padding,
                )
                quality_report["lrc_timing_trust"] = decision.mode
                if decision.keep_lrc_timings:
                    if decision.mode == "degraded_outro_padding":
                        issues_list.append(
                            "Detected likely trailing instrumental outro padding; "
                            "keeping provider LRC timestamps"
                        )
                    return line_timings
                issues_list.append(
                    "Ignoring provider LRC timestamps due to severe duration mismatch; "
                    "using audio/Whisper timing alignment instead"
                )
                return None
    return line_timings


def _resolve_lrc_inputs(
    *,
    title: str,
    artist: str,
    lyrics_file: Optional[Path],
    audio_start: float,
    filter_promos: bool,
    target_duration: Optional[int],
    vocals_path: Optional[str],
    evaluate_sources: bool,
    offline: bool,
    quality_report: dict,
    issues_list: List[str],
    drop_lrc_line_timings: bool,
    use_whisper: bool,
    whisper_map_lrc: bool,
    hooks: Optional["LyricsWhisperHooks"],
    runtime_config: Optional["LyricsRuntimeConfig"],
) -> tuple[
    Optional[str],
    Optional[List[Tuple[float, str]]],
    str,
    List[str],
]:
    from .helpers import _load_lyrics_file
    from .lyrics_whisper import _fetch_lrc_text_and_timings_for_state

    file_lines: List[str] = []
    file_lrc_text: Optional[str] = None
    file_line_timings: Optional[List[Tuple[float, str]]] = None
    if lyrics_file:
        file_lrc_text, file_line_timings, file_lines = _load_lyrics_file(
            lyrics_file, filter_promos
        )
        if file_lrc_text or file_lines:
            quality_report["source"] = f"lyrics_file:{lyrics_file}"
        if file_lines and not file_lrc_text and not file_line_timings:
            quality_report.update(
                {
                    "lyrics_source_audio_scoring_used": False,
                    "lyrics_source_disagreement_flagged": False,
                    "lyrics_source_disagreement_reasons": [],
                    "lyrics_source_candidate_count": 0,
                    "lyrics_source_comparable_candidate_count": 0,
                    "lyrics_source_selection_mode": "lyrics_file_plain_text",
                    "lyrics_source_routing_skip_reason": "lyrics_file_plain_text",
                }
            )
            return None, None, str(quality_report["source"]), file_lines

    routing_diagnostics = {
        "lyrics_source_audio_scoring_used": False,
        "lyrics_source_disagreement_flagged": False,
        "lyrics_source_disagreement_reasons": [],
        "lyrics_source_candidate_count": 0,
        "lyrics_source_comparable_candidate_count": 0,
        "lyrics_source_selection_mode": "default",
        "lyrics_source_routing_skip_reason": "none",
    }
    lrc_text, line_timings, source = _fetch_lrc_text_and_timings_for_state(
        title=title,
        artist=artist,
        target_duration=target_duration,
        vocals_path=vocals_path,
        evaluate_sources=evaluate_sources,
        filter_promos=filter_promos,
        offline=offline,
        routing_diagnostics=routing_diagnostics,
        hooks=hooks,
        runtime_config=runtime_config,
    )
    quality_report.update(routing_diagnostics)
    if routing_diagnostics["lyrics_source_disagreement_flagged"]:
        raw_reasons = routing_diagnostics["lyrics_source_disagreement_reasons"]
        reasons = raw_reasons if isinstance(raw_reasons, list) else []
        reason_text = ", ".join(str(item) for item in reasons) if reasons else "unknown"
        issues_list.append(
            "Lyrics source disagreement triggered routing: " f"{reason_text}"
        )
    if file_lrc_text and file_line_timings:
        lrc_text = file_lrc_text
        line_timings = file_line_timings
        source = "lyrics_file_lrc"
    elif line_timings and audio_start > 0.0 and target_duration:
        rebased_line_timings = _rebase_line_timings_to_audio_window(
            line_timings,
            audio_start=audio_start,
            target_duration=target_duration,
        )
        if rebased_line_timings != line_timings:
            line_timings = rebased_line_timings
            lrc_text = _build_lrc_text_from_line_timings(line_timings)
            quality_report["lyrics_source_selection_mode"] = "provider_clip_window"
            quality_report["lyrics_source_routing_skip_reason"] = "provider_clip_window"
    if not quality_report["source"]:
        quality_report["source"] = source

    line_timings = _apply_lrc_timing_trust_policy(
        line_timings=line_timings,
        lrc_text=lrc_text,
        target_duration=target_duration,
        drop_lrc_line_timings=drop_lrc_line_timings,
        vocals_path=vocals_path,
        use_whisper=use_whisper,
        whisper_map_lrc=whisper_map_lrc,
        issues_list=issues_list,
        quality_report=quality_report,
    )
    return lrc_text, line_timings, source, file_lines


def _resolve_genius_lines_and_metadata(
    *,
    lrc_text: Optional[str],
    file_lines: List[str],
    line_timings: Optional[List[Tuple[float, str]]],
    offline: bool,
    title: str,
    artist: str,
    quality_report: dict,
) -> tuple[
    Optional[List[Tuple[str, str]]],
    Optional[SongMetadata],
    Optional[Tuple[List[Line], Optional[SongMetadata]]],
]:
    from .helpers import _create_no_lyrics_placeholder
    from .genius import fetch_genius_lyrics_with_singers
    from . import lyrics_whisper as lw

    if (lrc_text or file_lines) and not line_timings:
        if offline:
            return None, None, None
        genius_lines, metadata = fetch_genius_lyrics_with_singers(title, artist)
        return genius_lines, metadata, None

    if offline:
        return None, None, None
    genius_lines, metadata = lw._fetch_genius_with_quality_tracking(
        line_timings, title, artist, quality_report
    )
    if genius_lines is None and not line_timings:
        lines, meta = _create_no_lyrics_placeholder(title, artist)
        return genius_lines, metadata, (lines, meta)
    return genius_lines, metadata, None


def _build_quality_report(
    *,
    use_whisper: bool,
    whisper_only: bool,
    whisper_map_lrc: bool,
    whisper_force_dtw: bool,
) -> dict:
    return {
        "lyrics_quality": {},
        "alignment_method": "none",
        "lrc_timing_trust": "normal",
        "whisper_used": False,
        "whisper_auto_enabled": False,
        "whisper_corrections": 0,
        "whisper_requested": use_whisper or whisper_only or whisper_map_lrc,
        "whisper_force_dtw": whisper_force_dtw,
        "total_lines": 0,
        "overall_score": 0.0,
        "issues": [],
        "source": "",
        "lyrics_source_audio_scoring_used": False,
        "lyrics_source_disagreement_flagged": False,
        "lyrics_source_disagreement_reasons": [],
        "lyrics_source_candidate_count": 0,
        "lyrics_source_comparable_candidate_count": 0,
        "lyrics_source_selection_mode": "default",
        "lyrics_source_routing_skip_reason": "none",
        "pre_whisper_lines": [],
        "pre_whisper_line_count": 0,
    }


def _detect_line_timing_offset(
    *,
    vocals_path: Optional[str],
    line_timings: Optional[List[Tuple[float, str]]],
    lyrics_offset: Optional[float],
    quality_report: dict,
    issues_list: List[str],
) -> Optional[List[Tuple[float, str]]]:
    from .lyrics_whisper import _detect_offset_with_issues

    if not (vocals_path and line_timings):
        return line_timings

    auto_offset_scale = 1.0
    scaled_offset_min_abs_sec = 0.0
    scaled_offset_max_abs_sec = float("inf")
    scale_large_negative_offsets = False
    allow_suspicious_positive_offset = False
    suppress_moderate_negative_offset = False
    if (
        quality_report.get("lyrics_source_audio_scoring_used")
        and quality_report.get("lyrics_source_selection_mode")
        == "audio_scored_disagreement"
    ):
        auto_offset_scale = 0.6
        scaled_offset_min_abs_sec = 0.9
        scaled_offset_max_abs_sec = 1.4
        scale_large_negative_offsets = True
        lyrics_quality_raw = quality_report.get("lyrics_quality")
        lyrics_quality = (
            lyrics_quality_raw if isinstance(lyrics_quality_raw, dict) else {}
        )
        allow_suspicious_positive_offset = bool(
            lyrics_quality.get("duration_match", False)
        )
        suppress_moderate_negative_offset = (
            _should_suppress_disagreement_negative_offset(issues_list)
        )
    detected_timings, _ = _detect_offset_with_issues(
        vocals_path,
        line_timings,
        lyrics_offset,
        issues_list,
        auto_offset_scale=auto_offset_scale,
        scaled_offset_min_abs_sec=scaled_offset_min_abs_sec,
        scaled_offset_max_abs_sec=scaled_offset_max_abs_sec,
        scale_large_negative_offsets=scale_large_negative_offsets,
        allow_suspicious_positive_offset=allow_suspicious_positive_offset,
        suppress_moderate_negative_offset=suppress_moderate_negative_offset,
    )
    return detected_timings


def _fetch_genius_with_quality_tracking_impl(
    line_timings: Optional[List[Tuple[float, str]]],
    title: str,
    artist: str,
    quality_report: dict,
) -> Tuple[Optional[List[Tuple[str, str]]], Optional[SongMetadata]]:
    from .genius import fetch_genius_lyrics_with_singers

    if not line_timings:
        genius_lines, metadata = fetch_genius_lyrics_with_singers(title, artist)
        quality_report["alignment_method"] = "genius_fallback"
        quality_report["issues"].append("No synced LRC found, using Genius text")
        if not genius_lines:
            quality_report["issues"].append("No lyrics found from any source")
            quality_report["overall_score"] = 0.0
            return None, None
        return genius_lines, metadata
    return fetch_genius_lyrics_with_singers(title, artist)


def _fetch_genius_with_quality_tracking(
    line_timings: Optional[List[Tuple[float, str]]],
    title: str,
    artist: str,
    quality_report: dict,
) -> Tuple[Optional[List[Tuple[str, str]]], Optional[SongMetadata]]:
    return _fetch_genius_with_quality_tracking_impl(
        line_timings, title, artist, quality_report
    )
