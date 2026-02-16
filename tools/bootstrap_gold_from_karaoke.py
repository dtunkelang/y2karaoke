#!/usr/bin/env python3
"""
Find a karaoke video and bootstrap/refine word-level gold timings using computer vision.
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

import cv2
import numpy as np

# Timing precision constants
SNAP_SECONDS = 0.05
MIN_WORD_DURATION = 0.05
_LRC_TS_RE = re.compile(r"\[(\d+):([0-5]?\d(?:\.\d{1,3})?)\]")
_OCR_ENGINE = None


def _np():
    return np


def _cv2():
    return cv2


def _get_ocr():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        import platform

        if platform.system() == "Darwin" and platform.machine() == "arm64":
            try:
                _OCR_ENGINE = VisionOCR()
                print("Using Apple Vision OCR.")
            except Exception as e:
                print(
                    f"Failed to initialize Apple Vision OCR: {e}. Falling back to PaddleOCR."
                )
        if _OCR_ENGINE is None:
            from paddleocr import PaddleOCR

            _OCR_ENGINE = PaddleOCR(use_textline_orientation=True, lang="en")
            print("Using PaddleOCR.")
    return _OCR_ENGINE


class VisionOCR:
    def __init__(self):
        import Vision
        from Quartz import CIImage
        import objc

        self.Vision = Vision
        self.CIImage = CIImage
        self.objc = objc

    def predict(self, frame_nd):
        from Quartz import CIImage, kCIFormatRGBA8
        import Vision
        import cv2

        frame_rgba = cv2.cvtColor(frame_nd, cv2.COLOR_BGR2RGBA)
        h, w = frame_rgba.shape[:2]
        bytes_data = frame_rgba.tobytes()
        ci_image = CIImage.imageWithBitmapData_bytesPerRow_size_format_colorSpace_(
            bytes_data, w * 4, (w, h), kCIFormatRGBA8, None
        )
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)
        handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
            ci_image, None
        )
        success, error = handler.performRequests_error_([request], None)
        if not success:
            return []
        results = request.results()
        rec_texts, rec_boxes, rec_scores = [], [], []
        for result in results:
            candidates = result.topCandidates_(1)
            if not candidates:
                continue
            top_candidate = candidates[0]
            full_text = top_candidate.string()
            words = full_text.split()
            current_pos = 0
            for word_text in words:
                word_start_idx = full_text.find(word_text, current_pos)
                if word_start_idx == -1:
                    continue
                current_pos = word_start_idx + len(word_text)
                try:
                    range_obj = (word_start_idx, len(word_text))
                    box_result_tuple = top_candidate.boundingBoxForRange_error_(
                        range_obj, None
                    )
                    if not box_result_tuple or not box_result_tuple[0]:
                        continue

                    first_char_range = (word_start_idx, 1)
                    char_box_result = top_candidate.boundingBoxForRange_error_(
                        first_char_range, None
                    )
                    char_box = None
                    if char_box_result and char_box_result[0]:
                        cobs = char_box_result[0]
                        cbbox = cobs.boundingBox()
                        char_box = [
                            cbbox.origin.x * w,
                            (1.0 - cbbox.origin.y - cbbox.size.height) * h,
                            cbbox.size.width * w,
                            cbbox.size.height * h,
                        ]

                    bbox = box_result_tuple[0].boundingBox()
                    px_x, px_y, px_w, px_h = (
                        bbox.origin.x * w,
                        (1.0 - bbox.origin.y - bbox.size.height) * h,
                        bbox.size.width * w,
                        bbox.size.height * h,
                    )
                    box = [
                        [px_x, px_y],
                        [px_x + px_w, px_y],
                        [px_x + px_w, px_y + px_h],
                        [px_x, px_y + px_h],
                    ]
                    rec_texts.append(word_text)
                    rec_boxes.append({"word": box, "first_char": char_box})
                    rec_scores.append(top_candidate.confidence())
                except Exception:
                    continue
        return [
            {"rec_texts": rec_texts, "rec_boxes": rec_boxes, "rec_scores": rec_scores}
        ]


@dataclass
class TargetLine:
    line_index: int
    start: float
    end: float | None
    text: str
    words: list[str]
    y: float
    word_starts: list[float | None] | None = None
    word_ends: list[float | None] | None = None
    word_rois: list[tuple[int, int, int, int]] | None = None
    char_rois: list[tuple[int, int, int, int] | None] | None = None


def _cluster_colors(pixel_samples: list[np.ndarray], k: int = 2) -> list[np.ndarray]:
    if not pixel_samples:
        return [np.array([255, 255, 255]), np.array([0, 0, 0])]
    data = np.stack(pixel_samples).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    # Use dummy labels array to match cv2.kmeans signature correctly for mypy
    _, _, centers = cv2.kmeans(data, k, None, criteria, 10, flags)  # type: ignore
    return [c for c in centers]


def _infer_lyric_colors(
    video_path: Path, roi_rect: tuple[int, int, int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample text colors to find Unselected and Selected (highlight) prototypes."""
    ocr = _get_ocr()
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / src_fps
    rx, ry, rw, rh = roi_rect
    pixel_samples = []

    # Sample middle of the song
    for t in np.arange(duration * 0.3, duration * 0.7, 5.0):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        roi = frame[ry : ry + rh, rx : rx + rw]
        res = ocr.predict(roi)
        if res and res[0]:
            items = res[0]
            boxes = (
                items["rec_boxes"]
                if isinstance(items, dict)
                else [it[0] for it in items]
            )
            for box in boxes:
                nb = np.array(box["word"] if isinstance(box, dict) else box).reshape(
                    -1, 2
                )
                x, y = int(min(nb[:, 0])), int(min(nb[:, 1]))
                bw, bh = int(max(nb[:, 0]) - x), int(max(nb[:, 1]) - y)
                if bw > 4 and bh > 4:
                    word_roi = roi[y : y + bh, x : x + bw]
                    # Sample center pixels to avoid borders
                    cx, cy = bw // 2, bh // 2
                    pixel_samples.append(word_roi[cy, cx])

    cap.release()
    centers = _cluster_colors(pixel_samples, k=2)
    centers.sort(key=lambda c: np.mean(c), reverse=True)  # Brighter first
    return centers[0], centers[1], np.array([0, 0, 0])


def _classify_word_state(
    word_roi: np.ndarray, c_un: np.ndarray, c_sel: np.ndarray
) -> tuple[str, float]:
    """Determine if a word is unselected, selected, or mixed."""
    if word_roi.size == 0:
        return "unknown", 0.0
    pixels = word_roi.reshape(-1, 3).astype(np.float32)
    dist_un = np.linalg.norm(pixels - c_un, axis=1)
    dist_sel = np.linalg.norm(pixels - c_sel, axis=1)

    is_sel = dist_sel < dist_un
    ratio = np.mean(is_sel)

    if ratio > 0.8:
        return "selected", float(ratio)
    if ratio > 0.2:
        return "mixed", float(ratio)
    return "unselected", float(ratio)


def _snap(value: float) -> float:
    return round(round(float(value) / SNAP_SECONDS) * SNAP_SECONDS, 3)


def _slug(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.lower().replace("-", " ")
    return re.sub(r"[^a-z0-9' ]+", "", t).strip()


def _text_similarity(a: str, b: str) -> float:
    na, nb = _normalize_text(a), _normalize_text(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def canon_punct(text: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFKC", text)
    trans = {
        "’": "'",
        "‘": "'",
        "´": "'",
        "`": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
    }
    text = "".join(trans.get(ch, ch) for ch in text)
    return " ".join(text.split())


def _spell_correct(text: str) -> str:
    if not text:
        return text
    try:
        from AppKit import NSSpellChecker

        checker = NSSpellChecker.sharedSpellChecker()
        words = text.split()
        corrected = []
        for w in words:
            if len(w) < 3:
                corrected.append(w)
                continue
            missed = checker.checkSpellingOfString_startingAt_(w, 0)
            if missed.length > 0:
                guesses = checker.guessesForWordRange_inString_language_inSpellDocumentWithTag_(
                    missed, w, "en", 0
                )
                if guesses:
                    corrected.append(guesses[0])
                    continue
            corrected.append(w)
        return " ".join(corrected)
    except Exception:
        return text


def normalize_ocr_line(text: str) -> str:
    text = canon_punct(text)
    if not text:
        return text
    if text.lower().startswith("have "):
        text = "I " + text
    text = text.replace("problei", "problem")
    toks = text.split()
    out: list[str] = []
    contractions = {"'ll", "'re", "'ve", "'m", "'d"}
    confusable_i = {"1", "|", "!"}
    for i, tok in enumerate(toks):
        prev_tok = out[-1] if out else ""
        next_tok = toks[i + 1] if i + 1 < len(toks) else ""
        if tok in contractions and prev_tok:
            out[-1] = prev_tok + tok
            continue
        if tok in confusable_i:
            if any(c.isalpha() for c in prev_tok) or any(c.isalpha() for c in next_tok):
                tok = "I"
        out.append(tok)
    return _spell_correct(" ".join(out))


def _yt_dlp_bin() -> str:
    candidates = [
        str(Path(sys.executable).resolve().parent / "yt-dlp"),
        shutil.which("yt-dlp"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise RuntimeError("yt-dlp not found")


def download_karaoke_video(
    url: str, *, out_dir: Path, extract_audio: bool = False
) -> tuple[Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    video_id = url.split("=")[-1] if "=" in url else url.split("/")[-1]
    ytdlp = _yt_dlp_bin()
    out_tpl = str((out_dir / "%(id)s.%(ext)s").resolve())
    cmd = [
        ytdlp,
        "--no-playlist",
        "--format",
        "best[height<=1080][ext=mp4]/best",
        "--output",
        out_tpl,
    ]
    if extract_audio:
        cmd.extend(["--extract-audio", "--audio-format", "wav", "--keep-video"])
    cmd.append(url)
    subprocess.run(cmd, capture_output=True, check=True)
    v_paths = list(out_dir.glob(f"{video_id}.mp4"))
    if not v_paths:
        v_paths = list(out_dir.glob("*.mp4"))
    v_path = v_paths[0]
    a_path = out_dir / f"{video_id}.wav"
    return v_path, a_path if a_path.exists() else None


def detect_lyric_roi(
    video_path: Path, work_dir: Path, sample_fps: float = 1.0
) -> tuple[int, int, int, int]:
    ocr = _get_ocr()
    cv2, np = _cv2(), _np()
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / src_fps
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    all_boxes = []
    print(f"Detecting ROI over {duration:.1f}s video...")
    mid = duration / 2
    for t in np.arange(max(0, mid - 15), min(duration, mid + 15), 1.0 / sample_fps):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        res = ocr.predict(frame)
        if res and res[0]:
            items = res[0]
            if isinstance(items, dict):
                for b_data in items["rec_boxes"]:
                    try:
                        nb = np.array(b_data["word"]).reshape(-1, 2)
                        all_boxes.append(
                            (min(nb[:, 0]), min(nb[:, 1]), max(nb[:, 0]), max(nb[:, 1]))
                        )
                    except Exception:
                        continue
            else:
                for item in items[0] if isinstance(items[0], list) else items:
                    try:
                        box = item[0]
                        nb = np.array(box).reshape(-1, 2)
                        all_boxes.append(
                            (min(nb[:, 0]), min(nb[:, 1]), max(nb[:, 0]), max(nb[:, 1]))
                        )
                    except Exception:
                        continue
    cap.release()
    roi = (
        int(width * 0.02),
        int(height * 0.1),
        int(width * 0.96),
        int(height * 0.8),
    )
    print(f"Detected Extremely Generous ROI: {roi}")
    return roi


def _collect_raw_frames(
    video_path: Path,
    start: float,
    end: float,
    fps: float,
    roi_rect: tuple[int, int, int, int],
) -> list[dict[str, Any]]:
    cv2, np = _cv2(), _np()
    ocr = _get_ocr()
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(src_fps / fps)), 1)
    rx, ry, rw, rh = roi_rect
    raw = []
    print(f"Sampling frames at {fps} FPS...")
    while True:
        ok, frame = cap.read()
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if not ok or t > end + 0.2:
            break
        if int(round(t * src_fps)) % step != 0:
            continue
        if int(t) % 10 == 0:
            print(f"  Progress: {t:.1f}s...")
        roi = frame[ry : ry + rh, rx : rx + rw]
        res = ocr.predict(roi)
        if res and res[0]:
            words = []
            items = res[0]
            if isinstance(items, dict):
                for txt, b_data in zip(items["rec_texts"], items["rec_boxes"]):
                    word_box = b_data["word"]
                    nb = np.array(word_box).reshape(-1, 2)
                    x, y = int(min(nb[:, 0])), int(min(nb[:, 1]))
                    bw, bh = int(max(nb[:, 0]) - x), int(max(nb[:, 1]) - y)
                    words.append({"text": txt, "x": x, "y": y, "w": bw, "h": bh})
            if words:
                raw.append({"time": t, "words": words})
    cap.release()
    return raw


def _word_fill_mask(roi_bgr: np.ndarray, c_bg: np.ndarray) -> np.ndarray:
    cv2, np = _cv2(), _np()
    dist_bg = np.linalg.norm(roi_bgr - c_bg, axis=2)
    mask = (dist_bg > 35).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    return mask


def _refine_word_timings_at_high_fps(  # noqa: C901
    video_path: Path,
    target_lines: list[TargetLine],
    roi_rect: tuple[int, int, int, int],
) -> None:
    cv2, np = _cv2(), _np()
    cap = cv2.VideoCapture(str(video_path))
    rx, ry, rw, rh = roi_rect
    print("Refining timings with Departure-Onset detection...")

    for i, ln in enumerate(target_lines):
        if not ln.word_rois:
            continue
        # Window: independent per line
        v_start, v_end = max(0.0, ln.start - 1.0), (ln.end or ln.start) + 1.0
        cap.set(cv2.CAP_PROP_POS_MSEC, v_start * 1000.0)
        line_frames = []
        while True:
            ok, frame = cap.read()
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not ok or t > v_end:
                break
            line_frames.append((t, frame[ry : ry + rh, rx : rx + rw]))

        if len(line_frames) < 20:
            continue
        c_bg_line = np.mean(
            [np.mean(f[1], axis=(0, 1)) for f in line_frames[:10]], axis=0
        )

        new_starts: list[float | None] = []
        new_ends: list[float | None] = []
        word_rois = ln.word_rois
        assert word_rois is not None
        for wi in range(len(ln.words)):
            wx, wy, ww, wh = word_rois[wi]
            # 1. Identify TEXT-ONLY frames
            word_vals = []
            for t, roi in line_frames:
                if wy + wh <= roi.shape[0] and wx + ww <= roi.shape[1]:
                    word_roi = roi[wy : wy + wh, wx : wx + ww]
                    mask = _word_fill_mask(word_roi, c_bg_line)
                    if np.sum(mask > 0) > 30:  # Glyph is present
                        lab = cv2.cvtColor(word_roi, cv2.COLOR_BGR2LAB).astype(
                            np.float32
                        )
                        word_vals.append(
                            {
                                "t": t,
                                "mask": mask,
                                "lab": lab,
                                "avg": lab[mask.astype(bool)].mean(axis=0),
                            }
                        )

            s, e = None, None
            if len(word_vals) > 10:
                l_vals = np.array([v["avg"][0] for v in word_vals])
                l_smooth = np.convolve(
                    l_vals,
                    np.ones(min(10, len(l_vals))) / min(10, len(l_vals)),
                    mode="same",
                )
                idx_peak = int(np.argmax(l_smooth))
                c_initial = word_vals[idx_peak]["avg"]
                idx_valley = idx_peak + int(np.argmin(l_smooth[idx_peak:]))
                c_final = word_vals[idx_valley]["avg"]

                if np.linalg.norm(c_final - c_initial) > 2.0:
                    times, dists_in = [], []
                    for v in word_vals:
                        times.append(v["t"])
                        dists_in.append(np.linalg.norm(v["avg"] - c_initial))

                    # 2. Departure search: find exact frame where color starts moving
                    stable_range = dists_in[max(0, idx_peak - 5) : idx_peak + 5]
                    noise_floor = float(
                        np.mean(stable_range) + 2 * np.std(stable_range)
                        if stable_range
                        else 1.0
                    )

                    for j in range(idx_peak, len(times)):
                        if s is None and dists_in[j] > noise_floor:
                            if j + 3 < len(times) and all(
                                dists_in[j + k] > dists_in[j + k - 1]
                                for k in range(1, 4)
                            ):
                                s = times[j]
                        if s is not None and e is None:
                            if np.linalg.norm(
                                word_vals[j]["avg"] - c_final
                            ) < np.linalg.norm(word_vals[j]["avg"] - c_initial):
                                e = times[j]
                                break
            new_starts.append(s)
            new_ends.append(e)
        ln.word_starts, ln.word_ends = new_starts, new_ends
    cap.release()


def reconstruct_lyrics_from_visuals(  # noqa: C901
    raw_frames: list[dict[str, Any]], visual_fps: float
) -> list[TargetLine]:
    on_screen: dict[str, dict[str, Any]] = {}
    committed = []
    for frame in raw_frames:
        words = frame.get("words", [])
        current_norms = set()
        if words:
            words.sort(key=lambda w: w["y"])
            lines_in_frame = []
            if words:
                curr = [words[0]]
                for i in range(1, len(words)):
                    if words[i]["y"] - curr[-1]["y"] < 20:
                        curr.append(words[i])
                    else:
                        lines_in_frame.append(curr)
                        curr = [words[i]]
                lines_in_frame.append(curr)

            for ln_w in lines_in_frame:
                ln_w.sort(key=lambda w: w["x"])
                txt = normalize_ocr_line(" ".join([w["text"] for w in ln_w]))
                if not txt:
                    continue
                y_pos = int(sum(w["y"] for w in ln_w) / len(ln_w))
                norm = f"y{y_pos // 30}_{_normalize_text(txt)}"
                current_norms.add(norm)
                if norm in on_screen:
                    on_screen[norm]["last"] = frame["time"]
                else:
                    on_screen[norm] = {
                        "text": txt,
                        "words": [w["text"] for w in ln_w],
                        "first": frame["time"],
                        "last": frame["time"],
                        "y": y_pos,
                        "w_rois": [(w["x"], w["y"], w["w"], w["h"]) for w in ln_w],
                    }
        for nt in list(on_screen.keys()):
            if nt not in current_norms and frame["time"] - on_screen[nt]["last"] > 1.0:
                committed.append(on_screen.pop(nt))
    for ent in on_screen.values():
        committed.append(ent)

    unique: list[dict[str, Any]] = []
    for ent in committed:
        # Deduplication: Only merge if text is similar AND it's at the same Y level AND appearing very close in time
        is_dup = False
        for u in unique:
            if _text_similarity(ent["text"], u["text"]) > 0.9:
                if abs(ent["y"] - u["y"]) < 20 and abs(ent["first"] - u["first"]) < 2.0:
                    is_dup = True
                    break
        if not is_dup:
            unique.append(ent)

    unique.sort(key=lambda x: (round(float(x["first"]) / 2.0) * 2.0, x["y"]))

    out: list[TargetLine] = []
    for i, ent in enumerate(unique):
        s = _snap(float(ent["first"]))
        if i + 1 < len(unique):
            nxt_s = _snap(float(unique[i + 1]["first"]))
            e = nxt_s if (nxt_s - s < 3.0) else _snap(float(ent["last"]) + 2.0)
        else:
            e = _snap(float(ent["last"]) + 2.0)
        out.append(
            TargetLine(
                i + 1,
                s,
                e,
                ent["text"],
                ent["words"],
                ent["y"],
                None,
                None,
                ent["w_rois"],
                None,
            )
        )
    return out


def main():  # noqa: C901
    cv2, np = _cv2(), _np()
    p = argparse.ArgumentParser()
    p.add_argument("--artist")
    p.add_argument("--title")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidate-url")
    p.add_argument("--visual-fps", type=float, default=2.0)
    p.add_argument("--strict-sequential", action="store_true")
    args = p.parse_args()
    song_dir = (
        Path(".cache/karaoke_bootstrap")
        / _slug(args.artist or "unk")
        / _slug(args.title or "unk")
    )
    v_path, a_path = download_karaoke_video(
        args.candidate_url, out_dir=song_dir / "video", extract_audio=True
    )
    roi = detect_lyric_roi(v_path, song_dir)
    raw = _collect_raw_frames(v_path, 0, 210, args.visual_fps, roi)
    t_lines = reconstruct_lyrics_from_visuals(raw, args.visual_fps)

    print("Re-anchoring lines...")
    cap = cv2.VideoCapture(str(v_path))
    rx, ry, rw, rh = roi
    for ln in t_lines:
        if not ln.word_rois:
            continue
        word_roi = ln.word_rois[0]
        v_bg_s = max(0.0, ln.start - 3.0)
        cap.set(cv2.CAP_PROP_POS_MSEC, v_bg_s * 1000.0)
        bg_s = []
        for _ in range(15):
            ok, f = cap.read()
            if ok:
                bg_s.append(
                    np.mean(
                        f[
                            ry + word_roi[1] : ry + word_roi[1] + word_roi[3],
                            rx + word_roi[0] : rx + word_roi[0] + word_roi[2],
                        ],
                        axis=(0, 1),
                    )
                )
        if not bg_s:
            continue
        c_bg = np.mean(bg_s, axis=0)
        found_t = None
        for _ in range(120):
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            ok, f = cap.read()
            if not ok:
                break
            if (
                np.linalg.norm(
                    np.mean(
                        f[
                            ry + word_roi[1] : ry + word_roi[1] + word_roi[3],
                            rx + word_roi[0] : rx + word_roi[0] + word_roi[2],
                        ],
                        axis=(0, 1),
                    )
                    - c_bg
                )
                > 30.0
            ):
                found_t = t
                break
        if found_t is not None:
            ln.start = found_t

    _refine_word_timings_at_high_fps(v_path, t_lines, roi)

    lines_out: list[dict[str, Any]] = []
    prev_line_end = 5.0
    for idx, ln in enumerate(t_lines):
        if ln.start < 7.0 and not ln.word_starts:
            continue
        if _normalize_text(ln.text) in [
            _normalize_text(args.title),
            _normalize_text(args.artist),
        ]:
            continue
        w_out: list[dict[str, Any]] = []
        n_words = len(ln.words)
        l_start = max(ln.start, prev_line_end)

        if not ln.word_starts or all(s is None for s in ln.word_starts):
            duration = max((ln.end or (l_start + 1.0)) - l_start, 1.0)
            step = duration / n_words
            for j, txt in enumerate(ln.words):
                ws = l_start + j * step
                we = ws + step
                w_out.append(
                    {
                        "word_index": j + 1,
                        "text": txt,
                        "start": _snap(ws),
                        "end": _snap(we),
                    }
                )
        else:
            # Type-safe access to word_starts/ends since we checked ln.word_starts above
            word_starts: list[float | None] = ln.word_starts
            word_ends: list[float | None] = ln.word_ends or [None] * n_words
            vs = [j for j, s in enumerate(word_starts) if s is not None]
            out_s: list[float] = []
            out_e: list[float] = []
            for j in range(n_words):
                ws_val = word_starts[j]
                if ws_val is not None:
                    out_s.append(ws_val)
                    out_e.append(word_ends[j] or ws_val + 0.1)
                else:
                    prev_v = max([idx for idx in vs if idx < j], default=-1)
                    next_v = min([idx for idx in vs if idx > j], default=-1)
                    if prev_v == -1:
                        base = ln.start
                        first_vs_val = word_starts[vs[0]]
                        assert first_vs_val is not None
                        next_t = first_vs_val
                        step = max(0.1, (next_t - base) / (vs[0] + 1))
                        out_s.append(next_t - (vs[0] - j + 1) * step)
                    elif next_v == -1:
                        base = out_e[prev_v]
                        out_s.append(base + (j - prev_v) * 0.3)
                    else:
                        frac = (j - prev_v) / (next_v - prev_v)
                        next_vs_val = word_starts[next_v]
                        assert next_vs_val is not None
                        out_s.append(
                            out_e[prev_v] + frac * (next_vs_val - out_e[prev_v])
                        )
                    out_e.append(out_s[-1] + 0.1)
            for j in range(n_words):
                if j == 0:
                    out_s[j] = max(out_s[j], prev_line_end)
                else:
                    out_s[j] = max(out_s[j], out_e[j - 1] + 0.05)
                out_e[j] = min(max(out_e[j], out_s[j] + 0.1), out_s[j] + 0.8)
                w_out.append(
                    {
                        "word_index": j + 1,
                        "text": ln.words[j],
                        "start": _snap(out_s[j]),
                        "end": _snap(out_e[j]),
                    }
                )

        line_start, line_end = w_out[0]["start"], w_out[-1]["end"]
        prev_line_end = line_end
        lines_out.append(
            {
                "line_index": idx + 1,  # type: ignore
                "text": ln.text,
                "start": line_start,
                "end": line_end,
                "words": w_out,
                "y": ln.y,
                "word_rois": ln.word_rois,
                "char_rois": [],
            }
        )

    for i, line_dict_any in enumerate(lines_out):
        line_dict_any["line_index"] = i + 1

    res = {
        "schema_version": "1.0",
        "title": args.title,
        "artist": args.artist,
        "audio_path": str(a_path.resolve()) if a_path else "",
        "lines": lines_out,
    }
    args.output.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
