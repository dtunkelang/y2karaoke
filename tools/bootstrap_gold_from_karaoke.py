#!/usr/bin/env python3
"""
Bootstrap and refine word-level gold timings using computer vision.

This tool acts as a CLI wrapper around the `y2karaoke.vision` and `y2karaoke.core` libraries.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, List

# Add src to path if running from tools/
src_path = Path(__file__).resolve().parents[1] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from y2karaoke.utils.logging import setup_logging, get_logger
from y2karaoke.core.components.audio.downloader import YouTubeDownloader
from y2karaoke.core.refine_visual import (
    reconstruct_lyrics_from_visuals,
    refine_word_timings_at_high_fps,
    _snap,  # Reuse snap from refine_visual or redefine locally
)
from y2karaoke.core.text_utils import make_slug, normalize_text_basic
from y2karaoke.vision.ocr import get_ocr_engine
from y2karaoke.vision.roi import detect_lyric_roi

try:
    import cv2
    import numpy as np
except ImportError:
    print("Error: OpenCV and Numpy are required. Please install them.")
    sys.exit(1)

logger = get_logger(__name__)


def _collect_raw_frames(
    video_path: Path,
    start: float,
    end: float,
    fps: float,
    roi_rect: tuple[int, int, int, int],
) -> list[dict]:
    ocr = get_ocr_engine()
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(src_fps / fps)), 1)
    rx, ry, rw, rh = roi_rect
    raw = []
    
    logger.info(f"Sampling frames at {fps} FPS...")
    while True:
        ok, frame = cap.read()
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if not ok or t > end + 0.2:
            break
        
        # Skip frames to match target FPS
        if int(round(t * src_fps)) % step != 0:
            continue
            
        roi = frame[ry : ry + rh, rx : rx + rw]
        res = ocr.predict(roi)
        
        if res and res[0]:
            items = res[0]
            rec_texts = items.get("rec_texts", [])
            rec_boxes = items.get("rec_boxes", [])
            
            words = []
            for txt, box_data in zip(rec_texts, rec_boxes):
                if isinstance(box_data, dict):
                    points = box_data["word"]
                else:
                    points = box_data
                    
                nb = np.array(points).reshape(-1, 2)
                x, y = int(min(nb[:, 0])), int(min(nb[:, 1]))
                bw, bh = int(max(nb[:, 0]) - x), int(max(nb[:, 1]) - y)
                words.append({"text": txt, "x": x, "y": y, "w": bw, "h": bh})
            
            if words:
                raw.append({"time": t, "words": words})
                
    cap.release()
    return raw


def main():
    setup_logging(verbose=True)
    p = argparse.ArgumentParser()
    p.add_argument("--artist")
    p.add_argument("--title")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidate-url")
    p.add_argument("--visual-fps", type=float, default=2.0)
    p.add_argument("--strict-sequential", action="store_true") # Kept for API compatibility
    args = p.parse_args()

    slug_artist = make_slug(args.artist or "unk")
    slug_title = make_slug(args.title or "unk")
    song_dir = Path(".cache/karaoke_bootstrap") / slug_artist / slug_title
    
    # 1. Download Video
    downloader = YouTubeDownloader(cache_dir=song_dir.parent)
    logger.info(f"Downloading video from {args.candidate_url}...")
    try:
        vid_info = downloader.download_video(args.candidate_url, output_dir=song_dir / "video")
        v_path = Path(vid_info["video_path"])
        aud_info = downloader.download_audio(args.candidate_url, output_dir=song_dir / "video")
        a_path = Path(aud_info["audio_path"])
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)

    # 2. Detect ROI
    roi = detect_lyric_roi(v_path, sample_fps=1.0)
    
    # 3. Initial coarse scan
    cap = cv2.VideoCapture(str(v_path))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    
    raw_frames = _collect_raw_frames(v_path, 0, duration, args.visual_fps, roi)
    
    # 4. Reconstruct Lines
    t_lines = reconstruct_lyrics_from_visuals(raw_frames, args.visual_fps)
    logger.info(f"Reconstructed {len(t_lines)} initial lines.")

    # 5. Refine Timings (High FPS)
    # Re-anchoring logic removed as it's implicit in the high-FPS refinement
    refine_word_timings_at_high_fps(v_path, t_lines, roi)

    # 6. Output Generation
    lines_out: List[dict[str, Any]] = []
    prev_line_end = 5.0
    
    for idx, ln in enumerate(t_lines):
        # Filter metadata
        if ln.start < 7.0 and (not ln.word_starts or all(s is None for s in ln.word_starts)):
            continue
        
        # Filter title/artist matches
        norm_txt = normalize_text_basic(ln.text)
        if norm_txt in [normalize_text_basic(args.title or ""), normalize_text_basic(args.artist or "")]:
            continue

        w_out: List[dict[str, Any]] = []
        n_words = len(ln.words)
        l_start = max(ln.start, prev_line_end)

        # Fallback if refinement failed completely for a line
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
            # We have refined timings
            # Ensure types are happy
            word_starts = ln.word_starts
            word_ends = ln.word_ends or [None] * n_words
            
            vs = [j for j, s in enumerate(word_starts) if s is not None]
            out_s: List[float] = []
            out_e: List[float] = []
            
            for j in range(n_words):
                ws_val = word_starts[j]
                if ws_val is not None:
                    out_s.append(ws_val)
                    out_e.append(word_ends[j] or ws_val + 0.1)
                else:
                    # Interpolation logic for missing words
                    prev_v = max([idx for idx in vs if idx < j], default=-1)
                    next_v = min([idx for idx in vs if idx > j], default=-1)
                    
                    if prev_v == -1:
                        # Before first detected word
                        base = ln.start
                        first_vs_val = word_starts[vs[0]] if vs else base + 1.0
                        assert first_vs_val is not None
                        next_t = first_vs_val
                        # Back-project
                        step = max(0.1, (next_t - base) / (len(vs) + 1 if vs else 2))
                        out_s.append(max(base, next_t - (vs[0] - j + 1) * step if vs else base + j * 0.5))
                    elif next_v == -1:
                        # After last detected word
                        base = out_e[prev_v]
                        out_s.append(base + (j - prev_v) * 0.3)
                    else:
                        # Between two detected words
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
                
                # Cap duration
                out_e[j] = min(max(out_e[j], out_s[j] + 0.1), out_s[j] + 0.8)
                
                w_out.append(
                    {
                        "word_index": j + 1,
                        "text": ln.words[j],
                        "start": _snap(out_s[j]),
                        "end": _snap(out_e[j]),
                    }
                )

        line_start = w_out[0]["start"]
        line_end = w_out[-1]["end"]
        prev_line_end = line_end
        
        lines_out.append(
            {
                "line_index": idx + 1, # Placeholder, updated below
                "text": ln.text,
                "start": line_start,
                "end": line_end,
                "words": w_out,
                "y": ln.y,
                "word_rois": ln.word_rois,
                "char_rois": [],
            }
        )

    # Renumber lines sequentially
    for i, line_dict in enumerate(lines_out):
        line_dict["line_index"] = i + 1  # type: ignore[assignment]

    res = {
        "schema_version": "1.0",
        "title": args.title,
        "artist": args.artist,
        "audio_path": str(a_path.resolve()) if a_path else "",
        "lines": lines_out,
    }
    
    args.output.write_text(json.dumps(res, indent=2))
    logger.info(f"Saved refined gold timings to {args.output}")


if __name__ == "__main__":
    main()
