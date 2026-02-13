#!/usr/bin/env python3
"""Find a karaoke video and bootstrap/refine word-level gold timings.

Default pipeline is visual-only:
1. Find karaoke-style YouTube candidate for (artist, title) unless URL given.
2. Download karaoke video locally.
3. Estimate a global offset from LRC/gold line starts vs frame-activity peaks.
4. Infer per-line word timings from highlight progress in the video frames.
5. Write gold JSON compatible with tools/gold_timing_editor.py.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

SNAP_SECONDS = 0.05
MIN_WORD_DURATION = 0.05
_LRC_TS_RE = re.compile(r"\[(\d+):([0-5]?\d(?:\.\d{1,3})?)\]")


def _np():
    import numpy as np  # type: ignore

    return np


def _cv2():
    import cv2  # type: ignore

    return cv2


def _pytesseract():
    import pytesseract  # type: ignore

    return pytesseract


def _yt_dlp_bin() -> str:
    cwd = Path.cwd()
    candidates = [
        str(Path(sys.executable).resolve().parent / "yt-dlp"),
        str(cwd / "venv" / "bin" / "yt-dlp"),
        str(cwd / ".venv" / "bin" / "yt-dlp"),
        shutil.which("yt-dlp"),
        shutil.which("ytdlp"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError(
        "yt-dlp not found. Install it in current env (e.g., ./venv/bin/pip install yt-dlp)."
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
    return round(round(float(value) / SNAP_SECONDS) * SNAP_SECONDS, 3)


def _slug(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (text or "").lower()).strip()


def _text_similarity(a: str, b: str) -> float:
    na = _normalize_text(a)
    nb = _normalize_text(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def parse_lrc_lines(lrc_text: str) -> list[TargetLine]:
    lines: list[TargetLine] = []
    for raw in lrc_text.splitlines():
        timestamps = list(_LRC_TS_RE.finditer(raw))
        if not timestamps:
            continue
        lyric = raw[timestamps[-1].end() :].strip()
        if not lyric:
            continue
        words = lyric.split()
        if not words:
            continue
        for ts in timestamps:
            minute = int(ts.group(1))
            sec = float(ts.group(2))
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
    *,
    artist: str,
    title: str,
    gold_in: Path | None,
    lrc_in: Path | None,
) -> tuple[dict[str, Any], list[TargetLine], str]:
    if gold_in:
        raw = json.loads(gold_in.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("lines"), list):
            raise ValueError(f"Invalid gold file: {gold_in}")
        base_doc = raw
        gold_lines = raw["lines"]
        target_lines: list[TargetLine] = []
        for line in gold_lines:
            words = [str(w.get("text", "")).strip() for w in line.get("words", [])]
            words = [w for w in words if w]
            if not words:
                continue
            target_lines.append(
                TargetLine(
                    line_index=int(line.get("line_index", len(target_lines) + 1)),
                    start=float(line.get("start", 0.0)),
                    end=float(line["end"]) if line.get("end") is not None else None,
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
        if target_lines:
            if len(lrc_lines) != len(target_lines):
                raise ValueError(
                    f"LRC line count ({len(lrc_lines)}) != target line count ({len(target_lines)})"
                )
            for i in range(len(target_lines)):
                target_lines[i].start = lrc_lines[i].start
        else:
            target_lines = lrc_lines
    elif gold_in:
        source_timing_path = str(gold_in.resolve())

    if not target_lines:
        raise ValueError("No target lines found; provide --gold-in and/or --lrc-in")

    return base_doc, target_lines, source_timing_path


def score_candidate(entry: dict[str, Any]) -> float:
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


def _run_or_raise(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({' '.join(cmd)}): {proc.stderr.strip()}")


def download_karaoke_video(url: str, *, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ytdlp = _yt_dlp_bin()
    out_tpl = str((out_dir / "%(id)s.%(ext)s").resolve())
    _run_or_raise(
        [
            ytdlp,
            "--extractor-args",
            "youtube:player_client=android,web_creator",
            "--no-playlist",
            "--format",
            "mp4/best[ext=mp4]/best",
            "--output",
            out_tpl,
            url,
        ]
    )
    mp4_paths = sorted(
        out_dir.glob("*.mp4"), key=lambda p: (p.stat().st_mtime, p.name), reverse=True
    )
    if not mp4_paths:
        raise RuntimeError("Failed to download karaoke video")
    return mp4_paths[0]


def _get_fg_pixels(roi: Any) -> Any:
    np = _np()
    if roi.size == 0:
        return np.array([])
    max_c = np.max(roi, axis=2)
    mask = max_c > 80  # Threshold to ignore black background
    return roi[mask]


def _cluster_colors(pixel_samples: list[Any]) -> tuple[Any, Any]:
    cv2 = _cv2()
    np = _np()
    if len(pixel_samples) < 10:
        return np.array([255, 255, 255]), np.array([0, 0, 255])  # Fallback to White/Red

    samples = np.float32(pixel_samples)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(
        samples, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    return centers[0], centers[1]


def _classify_word_state(roi: Any, color_unselected: Any, color_selected: Any) -> str:
    np = _np()
    pixels = _get_fg_pixels(roi)
    if len(pixels) < 5:
        return "unknown"

    d_un = np.linalg.norm(pixels - color_unselected, axis=1)
    d_sel = np.linalg.norm(pixels - color_selected, axis=1)

    threshold = 60
    count_un = np.sum(d_un < threshold)
    count_sel = np.sum(d_sel < threshold)

    total = count_un + count_sel
    if total < 5:
        return "unknown"

    ratio_un = count_un / total
    ratio_sel = count_sel / total

    if ratio_un > 0.85:
        return "unselected"
    if ratio_sel > 0.85:
        return "selected"
    if ratio_un > 0.1 and ratio_sel > 0.1:
        return "mixed"
    return "unselected" if ratio_un > ratio_sel else "selected"


def _infer_lyric_colors(
    video_path: Path,
    *,
    sample_interval_sec: float = 2.0,
) -> tuple[Any, Any]:
    cv2 = _cv2()
    np = _np()
    pytesseract = _pytesseract()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for color inference: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    y_min, y_max = height * 0.15, height * 0.85
    x_min, x_max = width * 0.20, width * 0.80
    min_h = height * 0.02

    pixel_samples = []
    duration = frame_count / fps
    for t_sec in np.arange(0, duration, sample_interval_sec):
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break

        max_c = np.max(frame, axis=2)
        processed = cv2.bitwise_not(max_c)
        data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)

        for i in range(len(data["text"])):
            if data["level"][i] != 5 or not str(data["text"][i]).strip():
                continue
            x, y = data["left"][i], data["top"][i]
            w, h = data["width"][i], data["height"][i]
            cx, cy = x + w / 2, y + h / 2
            if y_min < cy < y_max and h > min_h and x_min < cx < x_max:
                pixels = _get_fg_pixels(frame[y : y + h, x : x + w])
                if len(pixels) > 20:
                    pixel_samples.extend(pixels[::5])
    cap.release()

    if not pixel_samples:
        return np.array([220, 220, 220]), np.array([30, 30, 180])

    c1, c2 = _cluster_colors(pixel_samples)
    if np.sum(c1) > np.sum(c2):
        return c1, c2
    return c2, c1


def _interp_cross_time(
    times: list[float], progress: list[float], threshold: float
) -> float | None:
    if not times or not progress:
        return None
    threshold = min(max(threshold, 0.0), 1.0)
    if progress[0] >= threshold:
        return times[0]
    for i in range(1, len(progress)):
        p0 = progress[i - 1]
        p1 = progress[i]
        if p1 < threshold:
            continue
        t0 = times[i - 1]
        t1 = times[i]
        if p1 <= p0 + 1e-9:
            return t1
        frac = (threshold - p0) / (p1 - p0)
        frac = min(max(frac, 0.0), 1.0)
        return t0 + ((t1 - t0) * frac)
    return None


def _word_min_duration(word: str) -> float:
    alnum_len = len(re.sub(r"[^a-z0-9]", "", word.lower()))
    return max(MIN_WORD_DURATION, 0.10 + (0.03 * min(alnum_len, 8)))


def _fill_missing_starts(
    starts: list[float | None],
    words: list[str],
    line_start: float,
    line_end: float,
) -> list[float]:
    n = len(words)
    if n == 0:
        return []
    span = max(line_end - line_start, MIN_WORD_DURATION)
    default_starts = [line_start + (span * s) for s, _ in _word_ratios(words)]
    out: list[float | None] = list(starts)
    known = [i for i, v in enumerate(out) if v is not None]
    if not known:
        out = list(default_starts)
    else:
        for i in range(n):
            if out[i] is not None:
                continue
            prev_idx = max((k for k in known if k < i), default=None)
            next_idx = min((k for k in known if k > i), default=None)
            if prev_idx is not None and next_idx is not None:
                left = float(out[prev_idx])  # type: ignore[arg-type]
                right = float(out[next_idx])  # type: ignore[arg-type]
                frac = (i - prev_idx) / max(next_idx - prev_idx, 1)
                out[i] = left + ((right - left) * frac)
            elif prev_idx is not None:
                prev = float(out[prev_idx])  # type: ignore[arg-type]
                out[i] = prev + (default_starts[i] - default_starts[prev_idx])
            elif next_idx is not None:
                nxt = float(out[next_idx])  # type: ignore[arg-type]
                out[i] = nxt - (default_starts[next_idx] - default_starts[i])
            else:
                out[i] = default_starts[i]

    min_gap = 0.02
    numeric = [
        min(max(v, line_start), line_end) for v in out if v is not None
    ]  # type: ignore[arg-type]
    for i in range(1, n):
        numeric[i] = max(numeric[i], numeric[i - 1] + min_gap)
    for i in range(n - 2, -1, -1):
        numeric[i] = min(numeric[i], numeric[i + 1] - min_gap)
    numeric[0] = max(line_start, numeric[0])
    for i in range(1, n):
        numeric[i] = max(numeric[i], numeric[i - 1] + min_gap)
    numeric[-1] = min(line_end, numeric[-1])
    for i in range(n - 2, -1, -1):
        numeric[i] = min(numeric[i], numeric[i + 1] - min_gap)
    return [min(max(v, line_start), line_end) for v in numeric]


def _word_ratios(words: list[str]) -> list[tuple[float, float]]:
    if not words:
        return []
    full = " ".join(words)
    total = max(len(full), 1)
    ratios: list[tuple[float, float]] = []
    cursor = 0
    for i, word in enumerate(words):
        start = cursor / total
        cursor += len(word)
        end = cursor / total
        ratios.append((start, end))
        if i + 1 < len(words):
            cursor += 1
    return ratios


def _audio_line_end(target_lines: list[TargetLine], idx: int) -> float:
    line = target_lines[idx]
    if line.end is not None and line.end > line.start:
        return line.end
    if idx + 1 < len(target_lines):
        return target_lines[idx + 1].start
    return line.start + max(4.0, len(line.words) * 0.7)


def _ocr_text_from_frame(frame: Any) -> str:
    cv2 = _cv2()
    pytesseract = _pytesseract()
    h, _ = frame.shape[:2]
    roi = frame[int(h * 0.45) : int(h * 0.95), :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    txt = pytesseract.image_to_string(
        bw,
        config="--oem 1 --psm 6",
    )
    return txt or ""


def detect_line_video_windows(
    target_lines: list[TargetLine],
    video_path: Path,
    *,
    sample_fps: float = 1.5,
) -> list[tuple[float, float]]:
    """Estimate per-line video windows from OCR text matches with order constraints."""
    cv2 = _cv2()
    np = _np()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0) / max(src_fps, 1e-6)
    step = max(int(round(src_fps / max(sample_fps, 0.2))), 1)

    times: list[float] = []
    text_presence: list[float] = []
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % step != 0:
            continue
        t = idx / src_fps
        txt = _ocr_text_from_frame(frame)
        alpha_count = sum(1 for ch in txt if ch.isalpha())
        times.append(t)
        text_presence.append(float(alpha_count))
    cap.release()

    if not times:
        # fallback uniform windows
        if not target_lines:
            return []
        span = max(duration / max(len(target_lines), 1), 0.5)
        return [
            (i * span, min(duration, (i + 1) * span)) for i in range(len(target_lines))
        ]

    audio_start = target_lines[0].start if target_lines else 0.0
    audio_end = (
        _audio_line_end(target_lines, len(target_lines) - 1) if target_lines else 0.0
    )
    audio_span = max(audio_end - audio_start, 1.0)

    # Detect lyric-active span by OCR text presence (intro/outro handling).
    arr = np.asarray(text_presence, dtype=float)
    if arr.size >= 7:
        kernel = np.ones(7, dtype=float) / 7.0
        smooth = np.convolve(arr, kernel, mode="same")
    else:
        smooth = arr
    threshold = max(4.0, float(np.percentile(smooth, 60.0)))
    active_idx = [i for i, v in enumerate(smooth) if float(v) >= threshold]
    if active_idx:
        v_start = times[active_idx[0]]
        v_end = times[active_idx[-1]]
    else:
        v_start = 0.0
        v_end = duration
    if v_end <= v_start + 2.0:
        v_start = 0.0
        v_end = duration
    video_span = max(v_end - v_start, 1.0)

    windows: list[tuple[float, float]] = []
    prev_end = 0.0
    for i, line in enumerate(target_lines):
        a0 = line.start
        a1 = _audio_line_end(target_lines, i)
        r0 = max(0.0, min(1.0, (a0 - audio_start) / audio_span))
        r1 = max(r0, min(1.0, (a1 - audio_start) / audio_span))
        s = v_start + (video_span * r0)
        e = v_start + (video_span * r1)
        s = max(s, prev_end)
        e = max(e, s + 0.1)
        windows.append((max(0.0, s), min(duration, e)))
        prev_end = windows[-1][1]
    return windows


def _collect_raw_frames(
    video_path: Path,
    line_start: float,
    line_end: float,
    fps: float,
    c_un: Any,
    c_sel: Any,
) -> list[dict[str, Any]]:
    cv2 = _cv2()
    np = _np()
    pytesseract = _pytesseract()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(int(round(src_fps / max(fps, 0.5))), 1)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(line_start - 0.1, 0.0) * 1000.0)

    y_min, y_max = height * 0.15, height * 0.85
    x_min, x_max = width * 0.20, width * 0.80
    min_h = height * 0.02

    raw_frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t > line_end + 0.2:
            break

        frame_idx = int(round(t * src_fps))
        if frame_idx % step != 0:
            continue

        max_c = np.max(frame, axis=2)
        processed = cv2.bitwise_not(max_c)
        data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)

        words_in_frame = []
        for i in range(len(data["text"])):
            if data["level"][i] != 5:
                continue
            text = str(data["text"][i]).strip()
            if not text:
                continue
            x, y = data["left"][i], data["top"][i]
            w, h = data["width"][i], data["height"][i]
            cx, cy = x + w / 2, y + h / 2
            if y_min < cy < y_max and h > min_h and x_min < cx < x_max:
                color_state = _classify_word_state(
                    frame[y : y + h, x : x + w], c_un, c_sel
                )
                if color_state != "unknown":
                    words_in_frame.append(
                        {"text": text, "color": color_state, "x": x, "y": y}
                    )
        if words_in_frame:
            words_in_frame.sort(key=lambda w: (w["y"] // 30, w["x"]))
            timestamp = f"{int(t // 60):02d}:{t % 60:05.2f}"
            raw_frames.append(
                {"time": t, "timestamp": timestamp, "words": words_in_frame}
            )
    cap.release()
    return raw_frames


def _track_block_transitions(raw_frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active_block: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for f in raw_frames:
        words = f["words"]
        t_str = f["timestamp"]
        current_texts = [w["text"] for w in words]
        active_texts = [w["text"] for w in active_block]

        is_same_block = len(current_texts) == len(active_texts)
        if is_same_block:
            for ct, at in zip(current_texts, active_texts):
                if _text_similarity(ct, at) < 0.7:
                    is_same_block = False
                    break

        if not is_same_block:
            for w in active_block:
                if w["start"] is not None:
                    candidates.append(w)
            active_block = []
            for w in words:
                active_block.append(
                    {
                        "text": w["text"],
                        "x": w["x"],
                        "y": w["y"],
                        "start": None,
                        "end": None,
                        "last_color": w["color"],
                    }
                )
        else:
            for i in range(len(words)):
                cur_color = words[i]["color"]
                prev_color = active_block[i]["last_color"]
                if prev_color == "unselected" and cur_color in ("mixed", "selected"):
                    active_block[i]["start"] = t_str
                if cur_color == "selected" and prev_color != "selected":
                    active_block[i]["end"] = t_str
                    if active_block[i]["start"] is None:
                        active_block[i]["start"] = t_str
                active_block[i]["x"] = words[i]["x"]
                active_block[i]["y"] = words[i]["y"]
                active_block[i]["last_color"] = cur_color

    for w in active_block:
        if w["start"] is not None:
            candidates.append(w)
    return candidates


def _extract_line_word_candidates_v2(
    video_path: Path,
    line: TargetLine,
    *,
    line_start: float,
    line_end: float,
    fps: float,
    c_un: Any,
    c_sel: Any,
) -> list[tuple[float, float] | None]:
    if line_end <= line_start + 0.05 or not line.words:
        return [None for _ in line.words]

    raw_frames = _collect_raw_frames(video_path, line_start, line_end, fps, c_un, c_sel)
    if not raw_frames:
        return [None for _ in line.words]

    candidates = _track_block_transitions(raw_frames)

    def _ts_to_float(ts: str) -> float:
        m, s = ts.split(":")
        return int(m) * 60 + float(s)

    merged = []
    if candidates:
        candidates.sort(
            key=lambda x: (x["y"] // 30, x["x"] // 50, _ts_to_float(x["start"]))
        )
        curr = candidates[0]
        for i in range(1, len(candidates)):
            nxt = candidates[i]
            if (
                nxt["text"] == curr["text"]
                and abs(nxt["x"] - curr["x"]) < 60
                and abs(nxt["y"] - curr["y"]) < 40
            ):
                if nxt["end"] and (not curr["end"] or nxt["end"] > curr["end"]):
                    curr["end"] = nxt["end"]
            else:
                merged.append(curr)
                curr = nxt
        merged.append(curr)

    results: list[tuple[float, float] | None] = [None] * len(line.words)
    target_norm = [_normalize_text(w) for w in line.words]
    for m in merged:
        m_norm = _normalize_text(m["text"])
        for i, t_norm in enumerate(target_norm):
            if results[i] is None and _text_similarity(m_norm, t_norm) > 0.8:
                s = _ts_to_float(m["start"])
                e = _ts_to_float(m["end"]) if m["end"] else s + 0.3
                results[i] = (s, e)
                break
    return results


def _line_limit(target_lines: list[TargetLine], idx: int) -> float:
    return _audio_line_end(target_lines, idx)


def _enforce_line_capacity(
    words_count: int, line_start: float, line_end: float
) -> None:
    cap = max(0.0, line_end - line_start)
    need = words_count * MIN_WORD_DURATION
    if need > cap + 1e-9:
        print(
            f"WARNING: Line capacity too short ({cap:.2f}s) for {words_count} words at "
            f"{MIN_WORD_DURATION:.2f}s minimum duration. Timings will be crowded."
        )


def _fit_line_word_times(
    line: TargetLine,
    candidate_times: list[tuple[float, float] | None],
    line_end_limit: float,
) -> list[tuple[float, float]]:
    n = len(line.words)
    _enforce_line_capacity(n, line.start, line_end_limit)
    out: list[tuple[float, float]] = []
    cursor = line.start

    for i in range(n):
        cand = candidate_times[i]
        remaining = n - i
        latest_start = line_end_limit - (remaining * MIN_WORD_DURATION)
        if cand is None:
            slot = (line_end_limit - cursor) / max(remaining, 1)
            dur = max(MIN_WORD_DURATION, min(0.7, slot * 0.85))
            start = cursor
            end = start + dur
        else:
            cstart, cend = cand
            start = min(max(cursor, cstart), latest_start)
            end = max(start + MIN_WORD_DURATION, cend)

        end = min(end, line_end_limit - ((remaining - 1) * MIN_WORD_DURATION))
        if end < start + MIN_WORD_DURATION:
            end = start + MIN_WORD_DURATION
        out.append((_snap(start), _snap(end)))
        cursor = end

    out[0] = (_snap(line.start), max(_snap(line.start + MIN_WORD_DURATION), out[0][1]))

    fixed: list[tuple[float, float]] = []
    prev_end = line.start
    for i, (start, end) in enumerate(out):
        remaining = n - i
        max_start = line_end_limit - (remaining * MIN_WORD_DURATION)
        start = min(max(start, prev_end), max_start)
        end = max(end, start + MIN_WORD_DURATION)
        max_end = line_end_limit - ((remaining - 1) * MIN_WORD_DURATION)
        end = min(end, max_end)
        fixed.append((_snap(start), _snap(end)))
        prev_end = end

    line_span = max(line_end_limit - line.start, MIN_WORD_DURATION)
    used_span = max(fixed[-1][1] - fixed[0][0], MIN_WORD_DURATION)
    target_span = line_span * 0.85
    if used_span < line_span * 0.65:
        scale = target_span / used_span
        stretched: list[tuple[float, float]] = []
        for s, e in fixed:
            ns = line.start + ((s - line.start) * scale)
            ne = line.start + ((e - line.start) * scale)
            stretched.append((ns, ne))
        fixed = []
        prev_end = line.start
        for i, (s, e) in enumerate(stretched):
            remaining = n - i
            max_start = line_end_limit - (remaining * MIN_WORD_DURATION)
            s = min(max(s, prev_end), max_start)
            e = max(e, s + MIN_WORD_DURATION)
            max_end = line_end_limit - ((remaining - 1) * MIN_WORD_DURATION)
            e = min(e, max_end)
            fixed.append((_snap(s), _snap(e)))
            prev_end = e
    return fixed


def build_gold_from_visual_karaoke(
    *,
    base_doc: dict[str, Any],
    target_lines: list[TargetLine],
    video_path: Path,
    source_timing_path: str,
    video_url: str,
    line_video_windows: list[tuple[float, float]],
    visual_fps: float,
    c_un: Any,
    c_sel: Any,
) -> dict[str, Any]:
    if len(line_video_windows) != len(target_lines):
        raise ValueError("line_video_windows length must match target_lines length")

    lines_out: list[dict[str, Any]] = []
    prev_end_global = 0.0

    for i, line in enumerate(target_lines):
        line_start = _snap(line.start)
        line_end = _snap(_line_limit(target_lines, i))
        if line_end <= line_start:
            raise ValueError(f"Non-positive line span at line {i + 1}")

        video_line_start, video_line_end = line_video_windows[i]
        if video_line_end <= video_line_start + 0.1:
            video_line_end = video_line_start + 0.1
        cands = _extract_line_word_candidates_v2(
            video_path,
            line,
            line_start=video_line_start,
            line_end=video_line_end,
            fps=visual_fps,
            c_un=c_un,
            c_sel=c_sel,
        )
        # Convert candidate timings from video line window to audio line window
        # while keeping line starts anchored to LRC/gold timings.
        video_span = max(video_line_end - video_line_start, 1e-6)
        audio_span = max(line_end - line_start, MIN_WORD_DURATION)
        cands_audio: list[tuple[float, float] | None] = []
        for cand in cands:
            if cand is None:
                cands_audio.append(None)
                continue
            vs, ve = cand
            fs = max(0.0, min(1.0, (vs - video_line_start) / video_span))
            fe = max(fs, min(1.0, (ve - video_line_start) / video_span))
            as_ = line_start + (audio_span * fs)
            ae = line_start + (audio_span * fe)
            cands_audio.append((as_, ae))
        fitted = _fit_line_word_times(line, cands_audio, line_end)

        words_out: list[dict[str, Any]] = []
        for wi, (text, (start, end)) in enumerate(zip(line.words, fitted), start=1):
            start = max(start, prev_end_global)
            end = max(end, start + MIN_WORD_DURATION)
            words_out.append(
                {
                    "word_index": wi,
                    "text": text,
                    "start": _snap(start),
                    "end": _snap(end),
                }
            )
            prev_end_global = words_out[-1]["end"]

        lines_out.append(
            {
                "line_index": i + 1,
                "text": " ".join(line.words),
                "start": words_out[0]["start"],
                "end": words_out[-1]["end"],
                "words": words_out,
            }
        )

    out = dict(base_doc)
    out["schema_version"] = "1.0"
    out["source_timing_path"] = source_timing_path
    out["audio_path"] = str(base_doc.get("audio_path", "") or "")
    out["karaoke_source_url"] = video_url
    out["karaoke_video_path"] = str(video_path)
    out["karaoke_line_windows"] = [
        {"line_index": i + 1, "video_start": round(s, 3), "video_end": round(e, 3)}
        for i, (s, e) in enumerate(line_video_windows)
    ]
    out["karaoke_timing_method"] = "visual_highlight_progress"
    out["lines"] = lines_out
    return out


def _choose_candidate(
    *,
    artist: str,
    title: str,
    explicit_url: str | None,
    max_candidates: int,
    print_candidates: bool,
) -> KaraokeCandidate:
    if explicit_url:
        video_id = explicit_url.split("v=")[-1][:11]
        return KaraokeCandidate(
            video_id=video_id,
            url=explicit_url,
            title="explicit",
            channel="",
            view_count=0,
            duration=0.0,
            score=0.0,
        )
    candidates = find_karaoke_candidates(artist, title, max_candidates=max_candidates)
    if not candidates:
        raise RuntimeError("No karaoke candidates found")
    if print_candidates:
        for idx, c in enumerate(candidates[:8], start=1):
            print(
                f"{idx}. score={c.score:.1f} views={c.view_count} "
                f"[{c.channel}] {c.title} -> {c.url}"
            )
    return candidates[0]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap/refine word-level gold timings from karaoke video highlights"
    )
    parser.add_argument("--artist", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--gold-in", type=Path, help="Existing gold JSON to refine")
    parser.add_argument("--lrc-in", type=Path, help="LRC with line-start constraints")
    parser.add_argument("--output", type=Path, required=True, help="Output gold JSON")
    parser.add_argument(
        "--work-dir", type=Path, default=Path(".cache/karaoke_bootstrap")
    )
    parser.add_argument(
        "--candidate-url", help="Use this karaoke URL instead of search"
    )
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--show-candidates", action="store_true")
    parser.add_argument("--visual-fps", type=float, default=20.0)
    parser.add_argument("--ocr-sample-fps", type=float, default=1.5)
    parser.add_argument(
        "--dry-run-search",
        action="store_true",
        help="Only find/print karaoke candidates and exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    candidate = _choose_candidate(
        artist=args.artist,
        title=args.title,
        explicit_url=args.candidate_url,
        max_candidates=args.max_candidates,
        print_candidates=args.show_candidates,
    )
    print(f"Selected karaoke: {candidate.url}")
    if args.dry_run_search:
        return 0

    base_doc, target_lines, source_timing_path = _load_target_lines(
        artist=args.artist,
        title=args.title,
        gold_in=args.gold_in,
        lrc_in=args.lrc_in,
    )

    song_dir = (
        args.work_dir / f"{_slug(args.artist)}-{_slug(args.title)}" / candidate.video_id
    )
    video_path = download_karaoke_video(candidate.url, out_dir=song_dir)
    print(f"Downloaded video: {video_path}")

    line_video_windows = detect_line_video_windows(
        target_lines,
        video_path,
        sample_fps=args.ocr_sample_fps,
    )
    print(
        f"Detected line windows from video OCR: {len(line_video_windows)} / "
        f"{len(target_lines)}"
    )

    c_un, c_sel = _infer_lyric_colors(video_path)
    print(f"Inferred lyric colors (BGR): {c_un.astype(int)}, {c_sel.astype(int)}")

    doc = build_gold_from_visual_karaoke(
        base_doc=base_doc,
        target_lines=target_lines,
        video_path=video_path,
        source_timing_path=source_timing_path,
        video_url=candidate.url,
        line_video_windows=line_video_windows,
        visual_fps=args.visual_fps,
        c_un=c_un,
        c_sel=c_sel,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote gold JSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
