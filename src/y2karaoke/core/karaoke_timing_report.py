"""Timing report generation helpers for karaoke output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..utils.logging import get_logger
from .models import compute_word_slots

if TYPE_CHECKING:
    from .models import Line

logger = get_logger(__name__)


def _normalize_word_text(raw: str) -> str:
    """Normalize a word for comparison (lowercase alpha + apostrophes)."""
    return "".join(ch for ch in (raw or "").lower() if ch.isalpha() or ch == "'")


def _build_base_report(
    lines: List["Line"],
    title: str,
    artist: str,
    lyrics_result: Dict[str, Any],
) -> Dict[str, Any]:
    quality = lyrics_result.get("quality", {})
    report: Dict[str, Any] = {
        "title": title,
        "artist": artist,
        "lyrics_source": quality.get("source", ""),
        "alignment_method": quality.get("alignment_method", ""),
        "whisper_requested": quality.get("whisper_requested", False),
        "whisper_force_dtw": quality.get("whisper_force_dtw", False),
        "whisper_used": quality.get("whisper_used", False),
        "whisper_corrections": quality.get("whisper_corrections", 0),
        "issues": quality.get("issues", []),
        "dtw_metrics": quality.get("dtw_metrics", {}),
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
    dtw_metrics = quality.get("dtw_metrics", {})
    if dtw_metrics:
        report["dtw_word_coverage"] = round(dtw_metrics.get("word_coverage", 0.0), 3)
        report["dtw_line_coverage"] = round(dtw_metrics.get("line_coverage", 0.0), 3)
    return report


def _load_whisper_segments(
    cache_manager: Any, video_id: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cache_dir = cache_manager.get_video_cache_dir(video_id)
    whisper_files = list(cache_dir.glob("*_whisper_*.json"))
    if not whisper_files:
        return [], []
    whisper_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    whisper_data = json.loads(whisper_files[0].read_text(encoding="utf-8"))
    segments = whisper_data.get("segments", whisper_data)
    if not isinstance(segments, list):
        return [], []
    words: List[Dict[str, Any]] = []
    for seg in segments:
        for word in seg.get("words", []) or []:
            if word.get("start") is None:
                continue
            words.append(word)
    words.sort(key=lambda w: w.get("start", 0.0))
    return segments, words


def _adjust_lines_from_whisper_word_windows(
    report: Dict[str, Any],
    all_words: List[Dict[str, Any]],
) -> None:
    report["whisper_word_count"] = len(all_words)
    report["whisper_window_low_conf_threshold"] = 0.5
    lines = report.get("lines", [])
    for idx, line in enumerate(lines):
        next_start = (
            lines[idx + 1]["start"] if idx + 1 < len(lines) else line["end"] + 2.0
        )
        window_start = line["start"] - 1.0
        window_words = [
            w for w in all_words if window_start <= w.get("start", 0.0) < next_start
        ]
        probs = [
            float(prob)
            for w in window_words
            for prob in [w.get("probability")]
            if prob is not None
        ]
        low_conf = sum(
            1
            for w in window_words
            for prob in [w.get("probability")]
            if prob is not None and float(prob) < 0.5
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
                if _normalize_word_text(w.get("text", "")) == first_line_word:
                    target_start = w.get("start")
                    break
            if target_start is not None:
                delta = target_start - line["start"]
                line_delta = round(delta, 3)
                if delta > 0:
                    line["start"] = round(line["start"] + delta, 2)
                    line["end"] = round(line["end"] + delta, 2)
                    for word_entry in line["words"]:
                        word_entry["start"] = round(word_entry["start"] + delta, 3)
                        word_entry["end"] = round(word_entry["end"] + delta, 3)
        line["whisper_line_start_delta"] = line_delta


def _attach_low_confidence_summary(
    report: Dict[str, Any], lyrics_result: Dict[str, Any]
) -> None:
    low_conf_lines: List[Dict[str, Any]] = []
    for line_entry in report.get("lines", []):
        avg_prob = line_entry.get("whisper_window_avg_prob")
        low_conf_count = line_entry.get("whisper_window_low_conf_count", 0)
        total_words = line_entry.get("whisper_window_word_count", 0)
        low_conf_ratio = (low_conf_count / total_words) if total_words else 0.0
        if avg_prob is not None and (avg_prob < 0.35 or low_conf_ratio >= 0.5):
            low_conf_lines.append(
                {
                    "index": line_entry["index"],
                    "text": line_entry["text"],
                    "whisper_window_avg_prob": avg_prob,
                    "low_conf_ratio": round(low_conf_ratio, 2),
                }
            )
    report["low_confidence_lines"] = low_conf_lines
    if not low_conf_lines:
        return
    quality = lyrics_result.get("quality")
    if quality is None:
        return
    issues = quality.setdefault("issues", [])
    issue_msg = f"{len(low_conf_lines)} line(s) had low Whisper confidence"
    if issue_msg not in issues:
        issues.append(issue_msg)


def _attach_nearest_segments(
    report: Dict[str, Any], segments: List[Dict[str, Any]]
) -> None:
    for line in report.get("lines", []):
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
            if 0 <= late_by <= 15.0 and (prior_late is None or late_by < prior_late):
                prior_late = late_by
                prior_seg = seg
            if best_start_delta is None or start_delta < best_start_delta:
                best_start_delta = start_delta
                nearest_start = seg
            if best_end_delta is None or end_delta < best_end_delta:
                best_end_delta = end_delta
                nearest_end = seg

        if nearest_start:
            line["nearest_segment_start"] = round(nearest_start.get("start", 0.0), 2)
            line["nearest_segment_start_end"] = round(nearest_start.get("end", 0.0), 2)
            line["nearest_segment_start_text"] = nearest_start.get("text", "")
        if nearest_end:
            line["nearest_segment_end"] = round(nearest_end.get("end", 0.0), 2)
            line["nearest_segment_end_start"] = round(nearest_end.get("start", 0.0), 2)
            line["nearest_segment_end_text"] = nearest_end.get("text", "")
        if prior_seg is not None:
            line["prior_segment_start"] = round(prior_seg.get("start", 0.0), 2)
            line["prior_segment_end"] = round(prior_seg.get("end", 0.0), 2)
            line["prior_segment_late_by"] = round(
                line["start"] - prior_seg.get("end", 0.0), 2
            )


def write_timing_report(
    cache_manager: Any,
    lines: List["Line"],
    report_path: str,
    title: str,
    artist: str,
    lyrics_result: Dict[str, Any],
    video_id: Optional[str] = None,
) -> None:
    """Write a JSON timing report for downstream inspection."""
    report = _build_base_report(lines, title, artist, lyrics_result)
    if video_id:
        try:
            segments, all_words = _load_whisper_segments(cache_manager, video_id)
            if segments:
                report["whisper_segments"] = [
                    {
                        "start": round(seg.get("start", 0.0), 2),
                        "end": round(seg.get("end", 0.0), 2),
                        "text": seg.get("text", ""),
                    }
                    for seg in segments[:50]
                ]
                _adjust_lines_from_whisper_word_windows(report, all_words)
                _attach_low_confidence_summary(report, lyrics_result)
                _attach_nearest_segments(report, segments)
        except Exception:
            pass

    path = Path(report_path)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Wrote timing report to {path}")
