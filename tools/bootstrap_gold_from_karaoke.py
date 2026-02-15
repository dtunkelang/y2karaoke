#!/usr/bin/env python3
"""
Find a karaoke video and bootstrap/refine word-level gold timings using computer vision.

This tool uses PaddleOCR for text recognition and dynamic ROI detection to extract
high-precision word timings from karaoke videos. It anchors these visual cues
to Ground Truth line-starts provided by an LRC file or existing Gold JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# Timing precision constants
SNAP_SECONDS = 0.05
MIN_WORD_DURATION = 0.05
_LRC_TS_RE = re.compile(r"\[(\d+):([0-5]?\d(?:\.\d{1,3})?)\]")
_OCR_ENGINE = None


def _np():
    import numpy as np  # type: ignore

    return np


def _cv2():
    import cv2  # type: ignore

    return cv2


def _get_ocr():
    """Lazy initialize PaddleOCR engine."""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from paddleocr import PaddleOCR  # type: ignore

        # PaddleOCR emits a lot of logs to stdout; we suppress them for CLI clarity.
        _OCR_ENGINE = PaddleOCR(lang="en")
    return _OCR_ENGINE


def _yt_dlp_bin() -> str:
    """Locate the yt-dlp binary in the current environment."""
    cwd = Path.cwd()
    candidates = [
        str(Path(sys.executable).resolve().parent / "yt-dlp"),
        str(cwd / "venv" / "bin" / "yt-dlp"),
        str(cwd / ".venv" / "bin" / "yt-dlp"),
        shutil.which("yt-dlp"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise RuntimeError(
        "yt-dlp not found. Please install it in your virtual environment."
    )


@dataclass
class KaraokeCandidate:
    video_id: str
    url: str
    title: str
    channel: str
    view_count: int
    duration: float
    score: float


@dataclass
class TargetLine:
    line_index: int
    start: float
    end: float | None
    text: str
    words: list[str]


def _snap(value: float) -> float:
    """Snap a timestamp to the nearest grid increment (default 0.05s)."""
    return round(round(float(value) / SNAP_SECONDS) * SNAP_SECONDS, 3)


def _slug(text: str) -> str:
    """Create a filesystem-friendly version of a string."""
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")


def _normalize_text(text: str) -> str:
    """Normalize text for fuzzy comparison."""
    return re.sub(r"[^a-z0-9 ]+", "", (text or "").lower()).strip()


def _text_similarity(a: str, b: str) -> float:
    """Calculate fuzzy similarity between two strings."""
    na, nb = _normalize_text(a), _normalize_text(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def parse_lrc_lines(lrc_text: str) -> list[TargetLine]:
    """Parse standard .lrc format into TargetLine objects."""
    lines: list[TargetLine] = []
    for raw in lrc_text.splitlines():
        timestamps = list(_LRC_TS_RE.finditer(raw))
        if not timestamps:
            continue
        lyric = raw[timestamps[-1].end() :].strip()
        if not lyric:
            continue
        words = lyric.split()
        for ts in timestamps:
            minute, sec = int(ts.group(1)), float(ts.group(2))
            start = _snap(minute * 60 + sec)
            lines.append(
                TargetLine(
                    line_index=len(lines) + 1,
                    start=start,
                    end=None,
                    text=lyric,
                    words=words,
                )
            )
    lines.sort(key=lambda ln: ln.start)
    for idx, ln in enumerate(lines, start=1):
        ln.line_index = idx
    return lines


def _load_target_lines(
    *, artist: str, title: str, gold_in: Path | None, lrc_in: Path | None
) -> tuple[dict[str, Any], list[TargetLine], str]:
    """Load line-start constraints from either a Gold JSON or an LRC file."""
    if gold_in:
        raw = json.loads(gold_in.read_text(encoding="utf-8"))
        base_doc = raw
        target_lines = []
        for i, line in enumerate(raw.get("lines", [])):
            words = [str(w.get("text", "")).strip() for w in line.get("words", [])]
            words = [w for w in words if w]
            if not words:
                continue
            target_lines.append(
                TargetLine(
                    line_index=int(line.get("line_index", i + 1)),
                    start=float(line.get("start", 0.0)),
                    end=float(line["end"]) if line.get("end") else None,
                    text=" ".join(words),
                    words=words,
                )
            )
    else:
        base_doc = {
            "schema_version": "1.0",
            "title": title,
            "artist": artist,
            "audio_path": "",
            "source_timing_path": "",
            "lines": [],
        }
        target_lines = []

    source_timing_path = ""
    if lrc_in:
        lrc_lines = parse_lrc_lines(lrc_in.read_text(encoding="utf-8"))
        source_timing_path = str(lrc_in.resolve())
        if target_lines and len(lrc_lines) == len(target_lines):
            for i in range(len(target_lines)):
                target_lines[i].start = lrc_lines[i].start
        else:
            target_lines = lrc_lines
    elif gold_in:
        source_timing_path = str(gold_in.resolve())

    return base_doc, target_lines, source_timing_path


def score_candidate(entry: dict[str, Any]) -> float:
    """Score a YouTube search result based on its likelihood of being a good karaoke video."""
    title = str(entry.get("title") or "")
    channel = str(entry.get("channel") or entry.get("uploader") or "")
    view_count = int(entry.get("view_count") or 0)
    duration = float(entry.get("duration") or 0.0)
    t = title.lower()
    c = channel.lower()
    score = 0.0
    if "karaoke" in t:
        score += 80
    if "instrumental" in t:
        score += 12
    if "lyrics" in t:
        score += 8
    if "version" in t:
        score += 8
    if "sing king" in c:
        score += 12
    if "karafun" in c:
        score += 10
    if duration and 110 <= duration <= 420:
        score += 10
    elif duration:
        score -= 20
    score += min(view_count / 300000.0, 40)
    return score


def find_karaoke_candidates(
    artist: str,
    title: str,
    *,
    max_candidates: int = 20,
) -> list[KaraokeCandidate]:
    """Search YouTube for potential karaoke video matches."""
    query = f"{artist} {title} karaoke lyrics"
    ytdlp = _yt_dlp_bin()
    cmd = [
        ytdlp,
        f"ytsearch{max_candidates}:{query}",
        "--flat-playlist",
        "--dump-single-json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp search failed: {proc.stderr.strip()}")
    data = json.loads(proc.stdout)
    entries = data.get("entries") or []
    out: list[KaraokeCandidate] = []
    for entry in entries:
        video_id = str(entry.get("id") or "")
        if not video_id:
            continue
        out.append(
            KaraokeCandidate(
                video_id=video_id,
                url=f"https://www.youtube.com/watch?v={video_id}",
                title=str(entry.get("title") or ""),
                channel=str(entry.get("channel") or entry.get("uploader") or ""),
                view_count=int(entry.get("view_count") or 0),
                duration=float(entry.get("duration") or 0.0),
                score=score_candidate(entry),
            )
        )
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def download_karaoke_video(url: str, *, out_dir: Path) -> Path:
    """Download the best available MP4 from YouTube."""
    out_dir.mkdir(parents=True, exist_ok=True)
    video_id = url.split("=")[-1] if "=" in url else url.split("/")[-1]

    # Check if we already have a file for this video ID
    existing = list(out_dir.glob(f"{video_id}.*"))
    if existing:
        print(f"Using existing video: {existing[0].name}")
        return existing[0]

    ytdlp = _yt_dlp_bin()
    out_tpl = str((out_dir / "%(id)s.%(ext)s").resolve())
    subprocess.run(
        [
            ytdlp,
            "--no-playlist",
            "--format",
            "best[height<=1080][ext=mp4]/best[ext=mp4]/best",
            "--output",
            out_tpl,
            url,
        ],
        capture_output=True,
        check=True,
    )
    mp4_paths = sorted(
        out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not mp4_paths:
        raise RuntimeError("Failed to download video")
    return mp4_paths[0]


def _get_cache_key(video_path: Path, prefix: str, **kwargs) -> str:
    """Generate a unique filename for caching results based on parameters."""
    h = hashlib.md5(
        f"{video_path.name}_{json.dumps(kwargs, sort_keys=True)}".encode()
    ).hexdigest()
    return f"{prefix}_{h}.json"


def detect_lyric_roi(
    video_path: Path, work_dir: Path, sample_fps: float = 1.0
) -> tuple[int, int, int, int]:
    """Identify the bounding box where lyrics appear by sampling the video."""
    cache_path = work_dir / _get_cache_key(video_path, "roi", fps=sample_fps)
    if cache_path.exists():
        print(f"Loading cached ROI from {cache_path.name}...")
        return tuple(json.loads(cache_path.read_text()))

    print(f"Detecting lyric ROI (DBNet @ {sample_fps} fps)...")
    cv2, np = _cv2(), _np()
    ocr = _get_ocr()
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / src_fps
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    all_boxes = []
    mid = duration / 2
    # Sample a 30s window from the middle where lyrics are guaranteed to be active.
    for t in np.arange(max(0, mid - 15), min(duration, mid + 15), 1.0 / sample_fps):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        res = ocr.predict(frame)
        if res and "rec_boxes" in res[0]:
            for box in res[0]["rec_boxes"]:
                np_box = np.array(box).reshape(-1, 2)
                x, y = int(min(np_box[:, 0])), int(min(np_box[:, 1]))
                w, h = int(max(np_box[:, 0]) - x), int(max(np_box[:, 1]) - y)
                # Filter out tiny detections or edges (logo/watermark avoidance)
                if 0.1 * width < x + w / 2 < 0.9 * width and h > 10:
                    all_boxes.append((x, y, x + w, y + h))
    cap.release()

    if not all_boxes:
        roi = (int(width * 0.1), int(height * 0.4), int(width * 0.8), int(height * 0.5))
    else:
        all_boxes = np.array(all_boxes)
        x1, y1 = int(np.percentile(all_boxes[:, 0], 5)), int(
            np.percentile(all_boxes[:, 1], 5)
        )
        x2, y2 = int(np.percentile(all_boxes[:, 2], 95)), int(
            np.percentile(all_boxes[:, 3], 95)
        )
        # Add margin for safety
        roi = (
            max(0, x1 - 10),
            max(0, y1 - 10),
            min(width, x2 - x1 + 20),
            min(height, y2 - y1 + 20),
        )

    print(f"  Detected Lyric ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    cache_path.write_text(json.dumps(roi))
    return roi


def _cluster_colors(pixel_samples: list[Any]) -> tuple[Any, Any]:
    """Cluster pixel samples into two dominant colors."""
    cv2, np = _cv2(), _np()
    if len(pixel_samples) < 10:
        return np.array([255, 255, 255]), np.array([0, 0, 255])  # Fallback to White/Red

    samples = np.float32(pixel_samples)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(
        samples, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    return centers[0], centers[1]


def _classify_word_state(roi: Any, color_unselected: Any, color_selected: Any) -> str:
    """Determine if a word is highlighted based on its internal pixel colors."""
    np = _np()
    if roi.size == 0:
        return "unknown"
    max_c = np.max(roi, axis=2)
    pixels = roi[max_c > 80]  # Ignore dark background
    if len(pixels) < 5:
        return "unknown"
    d_un = np.linalg.norm(pixels - color_unselected, axis=1)
    d_sel = np.linalg.norm(pixels - color_selected, axis=1)

    threshold = 45
    c_un, c_sel = np.sum(d_un < threshold), np.sum(d_sel < threshold)
    total = c_un + c_sel
    if total < 5:
        return "unknown"

    r_sel = c_sel / total
    if r_sel > 0.7:
        return "selected"
    if r_sel > 0.15:
        return "mixed"
    return "unselected"


def _infer_lyric_colors(
    video_path: Path, roi_rect: tuple[int, int, int, int]
) -> tuple[Any, Any, list[Any]]:
    """Automatically discover the unselected and selected lyric colors."""
    cv2, np = _cv2(), _np()
    cap = cv2.VideoCapture(str(video_path))
    rx, ry, rw, rh = roi_rect
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    samples = []
    # Sample throughout the middle section of the song
    for t in np.arange(duration * 0.2, duration * 0.8, 5.0):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok:
            samples.extend(frame[ry : ry + rh, rx : rx + rw][::50].reshape(-1, 3))
    cap.release()

    if not samples:
        return np.array([255, 255, 255]), np.array([0, 255, 0]), []

    _, labels, centers = cv2.kmeans(
        np.float32(samples),
        6,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )
    counts = np.bincount(labels.flatten())
    idx = np.argsort(counts)[::-1]
    bg = [centers[i] for i in idx[:2]]  # Top 2 are usually background
    rem = idx[2:]
    c_un = centers[
        rem[np.argmax([np.sum(centers[i]) for i in rem])]
    ]  # Brightest is unselected
    c_sel = centers[
        rem[np.argmax([np.std(centers[i]) for i in rem])]
    ]  # Most colorful is highlight
    return c_un, c_sel, bg


def _collect_raw_frames(
    video_path: Path,
    start: float,
    end: float,
    fps: float,
    c_un: Any,
    c_sel: Any,
    roi_rect: tuple[int, int, int, int],
) -> list[dict[str, Any]]:
    """Perform OCR on a specific time range and extract word states."""
    cv2, np = _cv2(), _np()
    ocr = _get_ocr()
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, start - 0.1) * 1000.0)
    step = max(int(round(src_fps / fps)), 1)
    rx, ry, rw, rh = roi_rect
    raw = []

    while True:
        ok, frame = cap.read()
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if not ok or t > end + 0.2:
            break
        if int(round(t * src_fps)) % step != 0:
            continue

        roi = frame[ry : ry + rh, rx : rx + rw]
        res = ocr.predict(roi)
        if res and "rec_texts" in res[0]:
            words = []
            for txt, box in zip(res[0]["rec_texts"], res[0]["rec_boxes"]):
                np_box = np.array(box).reshape(-1, 2)
                x, y = int(min(np_box[:, 0])), int(min(np_box[:, 1]))
                bw, bh = int(max(np_box[:, 0]) - x), int(max(np_box[:, 1]) - y)
                state = _classify_word_state(roi[y : y + bh, x : x + bw], c_un, c_sel)
                if state != "unknown":
                    words.append({"text": txt, "color": state, "x": x, "y": y})
            if words:
                words.sort(key=lambda w: (w["y"] // 30, w["x"]))  # Sort by line then X
                raw.append(
                    {
                        "time": t,
                        "timestamp": f"{int(t // 60):02d}:{t % 60:05.2f}",
                        "words": words,
                    }
                )
    cap.release()
    return raw


def _get_global_visual_sequence(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert raw frame detections into a clean sequence of highlight events."""
    seq, last_k = [], None
    for f in raw:
        high = [w for w in f["words"] if w["color"] in ("mixed", "selected")]
        if not high:
            continue
        # The 'leader' is the furthest word currently highlighted
        leader = max(high, key=lambda w: (w["y"], w["x"]))
        k = (leader["text"], leader["y"], leader["x"])
        if k != last_k:
            seq.append(
                {"text": leader["text"], "start": f["time"], "end": f["time"] + 0.3}
            )
            last_k = k
    return seq


def _detect_first_highlight_time(
    video_path: Path,
    roi_rect: tuple[int, int, int, int],
    c_un: Any,
    c_sel: Any,
    work_dir: Path,
    target_lines: list[TargetLine],
) -> float:
    """Scan the beginning of the video to find a highlight matching one of the first few lines."""
    cache_path = work_dir / _get_cache_key(video_path, "v_start_v2")
    if cache_path.exists():
        return float(json.loads(cache_path.read_text())["v_start"])

    print("Probing video for start anchor (matching first 3 lines)...")
    cv2, np = _cv2(), _np()
    ocr = _get_ocr()
    cap = cv2.VideoCapture(str(video_path))
    rx, ry, rw, rh = roi_rect

    # Sample first 60 seconds at 2 FPS
    for t in np.arange(0.0, 60.0, 0.5):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break

        roi = frame[ry : ry + rh, rx : rx + rw]
        res = ocr.predict(roi)
        if res and "rec_texts" in res[0]:
            for txt, box in zip(res[0]["rec_texts"], res[0]["rec_boxes"]):
                np_box = np.array(box).reshape(-1, 2)
                x, y = int(min(np_box[:, 0])), int(min(np_box[:, 1]))
                bw, bh = int(max(np_box[:, 0]) - x), int(max(np_box[:, 1]) - y)
                state = _classify_word_state(roi[y : y + bh, x : x + bw], c_un, c_sel)

                if state in ("selected", "mixed"):
                    # Check if this text matches any of the first 3 lines
                    for i in range(min(3, len(target_lines))):
                        sim = _text_similarity(txt, target_lines[i].words[0])
                        if sim > 0.7:
                            v_time = float(t)
                            lrc_time = target_lines[i].start
                            # v_start is the video time of the FIRST LRC line
                            v_start = v_time - (lrc_time - target_lines[0].start)
                            print(
                                f"  Anchor matched line {i+1} ('{txt}') at {v_time:.2f}s "
                                f"(LRC: {lrc_time:.2f}s) => Video Start: {v_start:.2f}s"
                            )
                            cap.release()
                            cache_path.write_text(
                                json.dumps({"v_start": float(v_start)})
                            )
                            return float(v_start)

    cap.release()
    print("  WARNING: No anchor found in first 60s. Defaulting to 0.0s.")
    return 0.0


def _collect_tokens_with_cache(
    video_path: Path,
    target_lines: list[TargetLine],
    global_offset: float,
    visual_fps: float,
    c_un: Any,
    c_sel: Any,
    roi_rect: tuple[int, int, int, int],
    work_dir: Path,
) -> list[dict[str, Any]]:
    """Extract and cache visual tokens from video windows."""
    cache_path = work_dir / _get_cache_key(
        video_path, "tokens", fps=visual_fps, roi=roi_rect
    )

    if cache_path.exists():
        print(f"Loading cached tokens from {cache_path.name}...")
        return json.loads(cache_path.read_text())

    print(f"Extracting tokens from line windows ({visual_fps} fps)...")
    global_tokens = []
    for i, ln in enumerate(target_lines):
        # The window in the VIDEO timebase
        # If ln.start is in official time, we search at official_time - global_offset
        v_start = max(0, ln.start - global_offset - 5.0)
        v_end = (
            target_lines[i + 1].start - global_offset
            if i + 1 < len(target_lines)
            else ln.start - global_offset + 10.0
        ) + 2.0

        print(
            f"  Line {i+1}/{len(target_lines)}: {ln.text[:30]}... (video: {v_start:.1f}s - {v_end:.1f}s)"
        )
        global_tokens.extend(
            _collect_raw_frames(
                video_path, v_start, v_end, visual_fps, c_un, c_sel, roi_rect
            )
        )
    cache_path.write_text(json.dumps(global_tokens))
    return global_tokens


def _align_words_to_visuals(
    all_t_words: list[str],
    word_map: list[int],
    target_lines: list[TargetLine],
    visual_seq: list[dict[str, Any]],
) -> list[tuple[float, float] | None]:
    """Perform monotonic alignment of target words to visual highlight events."""
    matched: list[tuple[float, float] | None] = [None] * len(all_t_words)
    v_cur = 0
    for t_idx, t_word in enumerate(all_t_words):
        best_v, best_s = -1, 0.0
        # Search window for matching visual cues
        for v_idx in range(v_cur, min(v_cur + 300, len(visual_seq))):
            # Be very permissive: allow matching visual cues within a wide window
            dt = visual_seq[v_idx]["start"] - target_lines[word_map[t_idx]].start
            if dt > 60.0:
                break
            if dt < -60.0:
                continue

            s = _text_similarity(t_word, visual_seq[v_idx]["text"])
            if s > 0.75 and s > best_s:
                best_s, best_v = s, v_idx
            if s > 0.95:
                break
        if best_v != -1:
            matched[t_idx] = (visual_seq[best_v]["start"], visual_seq[best_v]["end"])
            v_cur = best_v + 1
    return matched


def _balance_word_durations(words_out: list[dict[str, Any]]) -> None:
    """Ensure tiny words are scaled up and subsequent words shifted."""
    if len(words_out) < 2:
        return
    durs = [w["end"] - w["start"] for w in words_out]
    valid_durs = [d for d in durs if d > MIN_WORD_DURATION + 0.01]
    if valid_durs and any(d <= MIN_WORD_DURATION + 0.01 for d in durs):
        min_v = min(valid_durs)
        for j, w in enumerate(words_out):
            if (w["end"] - w["start"]) <= MIN_WORD_DURATION + 0.01:
                w["end"] = _snap(w["start"] + min_v)
                for k in range(j + 1, len(words_out)):
                    sh = max(0, words_out[k - 1]["end"] - words_out[k]["start"])
                    if sh > 0:
                        words_out[k]["start"] = _snap(words_out[k]["start"] + sh)
                        words_out[k]["end"] = _snap(words_out[k]["end"] + sh)


def _apply_safety_limits(
    words_out: list[dict[str, Any]], l_s: float, l_lim: float
) -> None:
    """Scale words if they exceed line limit and apply monotonicity safety."""
    if words_out[-1]["end"] > l_lim:
        sc = (l_lim - l_s) / (words_out[-1]["end"] - l_s)
        curr = l_s
        for w in words_out:
            d = (w["end"] - w["start"]) * sc
            w["start"], w["end"] = _snap(curr), _snap(curr + d)
            curr += d

    for wi, w in enumerate(words_out):
        w["start"] = max(0.0, _snap(w["start"]))
        if wi > 0:
            w["start"] = max(w["start"], words_out[wi - 1]["end"])
        w["end"] = max(w["start"] + MIN_WORD_DURATION, _snap(w["end"]))


def _fit_words_to_line(
    ln: TargetLine,
    l_s: float,
    l_lim: float,
    l_m: list[tuple[float, float] | None],
) -> list[dict[str, Any]]:
    """Fit word timings for a single line based on visual evidence or fallback."""
    anchor_wi = next((j for j, m in enumerate(l_m) if m), None)
    fitted: list[tuple[float, float]] = []
    if anchor_wi is not None:
        anchor_val = l_m[anchor_wi]
        assert anchor_val is not None
        curr_v = anchor_val[0]
        # Backtrack to find un-anchored words at the start of line
        for j in range(anchor_wi - 1, -1, -1):
            curr_v -= 0.4
        off = l_s - curr_v
        for m in l_m:
            fitted.append((_snap(m[0] + off), _snap(m[1] + off)) if m else (0.0, 0.0))
    else:
        # Uniform fallback for lines with no visual evidence
        span = min(len(ln.words) * 0.4, l_lim - l_s - 0.1)
        full_len = sum(len(w) for w in ln.words) + 1e-6
        cur_rat = 0.0
        for w in ln.words:
            fitted.append(
                (
                    _snap(l_s + span * cur_rat),
                    _snap(l_s + span * (cur_rat + len(w) / full_len)),
                )
            )
            cur_rat += len(w) / full_len

    words_out: list[dict[str, Any]] = []
    for wi, (text, (s, e)) in enumerate(zip(ln.words, fitted), 1):
        if wi == 1:
            s = l_s
        if words_out:
            s = float(max(s, words_out[-1]["end"]))
        e = max(e if e > 0 else s + 0.3, s + MIN_WORD_DURATION)
        words_out.append(
            {"word_index": wi, "text": text, "start": _snap(s), "end": _snap(e)}
        )

    _balance_word_durations(words_out)
    _apply_safety_limits(words_out, l_s, l_lim)
    return words_out


def build_gold_from_visual_karaoke(
    *,
    base_doc: dict[str, Any],
    target_lines: list[TargetLine],
    video_path: Path,
    c_un: Any,
    c_sel: Any,
    roi_rect: tuple[int, int, int, int],
    visual_fps: float,
    work_dir: Path,
    global_offset: float = 0.0,
) -> dict[str, Any]:
    """Core logic to align LRC lyrics with visual evidence from OCR."""
    global_tokens = _collect_tokens_with_cache(
        video_path=video_path,
        target_lines=target_lines,
        global_offset=global_offset,
        visual_fps=visual_fps,
        c_un=c_un,
        c_sel=c_sel,
        roi_rect=roi_rect,
        work_dir=work_dir,
    )

    # All visual evidence is in VIDEO time. Convert to OFFICIAL time.
    if global_offset != 0.0:
        print(
            f"Mapping visual cues to official audio time (offset {global_offset:+.3f}s)"
        )
        for f in global_tokens:
            f["time"] += global_offset

    visual_seq = _get_global_visual_sequence(global_tokens)
    print(f"Video produced {len(visual_seq)} highlight events.")

    all_t_words, word_map = [], []
    for i, ln in enumerate(target_lines):
        for word in ln.words:
            all_t_words.append(word)
            word_map.append(i)

    # Global Monotonic Alignment (all in official audio time now)
    matched = _align_words_to_visuals(all_t_words, word_map, target_lines, visual_seq)

    match_count = sum(1 for m in matched if m)
    print(f"Alignment: Matched {match_count}/{len(all_t_words)} words to visual cues.")

    lines_out, p_end = [], 0.0
    for i, ln in enumerate(target_lines):
        l_s = max(_snap(ln.start), _snap(p_end + 0.05))
        l_lim = _snap(
            target_lines[i + 1].start if i + 1 < len(target_lines) else ln.start + 10.0
        )
        l_m = [matched[j] for j, m_idx in enumerate(word_map) if m_idx == i]

        words_out = _fit_words_to_line(ln, l_s, l_lim, l_m)

        lines_out.append(
            {
                "line_index": i + 1,
                "text": ln.text,
                "start": words_out[0]["start"],
                "end": words_out[-1]["end"],
                "words": words_out,
            }
        )
        p_end = words_out[-1]["end"]

    out = dict(base_doc)
    out.update({"lines": lines_out, "karaoke_timing_method": "paddleocr_roi_alignment"})
    return out

    out = dict(base_doc)
    out.update({"lines": lines_out, "karaoke_timing_method": "paddleocr_roi_alignment"})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artist", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--gold-in", type=Path)
    parser.add_argument("--lrc-in", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-url")
    parser.add_argument("--visual-fps", type=float, default=1.0)
    parser.add_argument(
        "--work-dir", type=Path, default=Path(".cache/karaoke_bootstrap")
    )
    args = parser.parse_args()

    base_doc, target_lines, _ = _load_target_lines(
        artist=args.artist, title=args.title, gold_in=args.gold_in, lrc_in=args.lrc_in
    )

    song_dir = args.work_dir / _slug(args.artist) / _slug(args.title)
    video_path = download_karaoke_video(args.candidate_url, out_dir=song_dir / "video")

    global_offset = 0.0
    roi_rect = detect_lyric_roi(video_path, song_dir)
    c_un, c_sel, _ = _infer_lyric_colors(video_path, roi_rect)

    # Detect the very first highlight to anchor the entire video timeline to the LRC
    v_start_time = _detect_first_highlight_time(
        video_path, roi_rect, c_un, c_sel, song_dir, target_lines
    )
    lrc_start_time = target_lines[0].start
    global_offset = lrc_start_time - v_start_time
    print(
        f"Global Anchor: Video={v_start_time:.2f}s, LRC={lrc_start_time:.2f}s => Offset={global_offset:+.3f}s"
    )

    # Manual overrides for problematic high-glare videos
    if "LEdBLhABQRs" in str(video_path):
        import numpy as np

        c_un, c_sel = np.array([243, 249, 245]), np.array([114, 180, 26])

    doc = build_gold_from_visual_karaoke(
        base_doc=base_doc,
        target_lines=target_lines,
        video_path=video_path,
        c_un=c_un,
        c_sel=c_sel,
        roi_rect=roi_rect,
        visual_fps=args.visual_fps,
        work_dir=song_dir,
        global_offset=global_offset,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote gold JSON: {args.output}")


if __name__ == "__main__":
    main()
