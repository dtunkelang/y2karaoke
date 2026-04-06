#!/usr/bin/env python3
"""Deterministic helper for curated clip audio and editor setup."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
import wave
from pathlib import Path
from urllib.parse import urlencode, urlparse

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "benchmarks" / "curated_clip_songs.yaml"
CLIP_GOLD_ROOT = (
    REPO_ROOT / "benchmarks" / "clip_gold_candidate" / "20260312T_curated_clips"
)
DEFAULT_GOLD_ROOT = REPO_ROOT / "benchmarks" / "gold_set_candidate" / "20260305T231015Z"
EDITOR_BASE_URL = "http://127.0.0.1:8765/"
EDITOR_READY_TIMEOUT_SEC = 5.0
EDITOR_READY_POLL_SEC = 0.1
CLIP_DURATION_TOLERANCE_SEC = 0.5


def _song_float(song: dict[str, object], key: str, default: float = 0.0) -> float:
    value = song.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _slugify(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _base_slug(artist: str, title: str, youtube_id: str) -> str:
    slug = _slugify(f"{artist}-{title}")
    return slug or youtube_id


def _song_slug(song: dict[str, object]) -> str:
    base = _base_slug(
        str(song["artist"]),
        str(song["title"]),
        str(song["youtube_id"]),
    )
    clip_id = str(song.get("clip_id") or "").strip()
    if not clip_id:
        return base
    clip_slug = _slugify(clip_id)
    return f"{base}-{clip_slug}" if clip_slug else base


def _load_manifest(path: Path) -> list[dict[str, object]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    songs = raw.get("songs", [])
    if not isinstance(songs, list):
        raise ValueError("Manifest songs must be a list")
    return [song for song in songs if isinstance(song, dict)]


def _match_song(
    songs: list[dict[str, object]], match: str
) -> tuple[int, dict[str, object]]:
    lowered = match.lower()
    hits: list[tuple[int, dict[str, object]]] = []
    for index, song in enumerate(songs, start=1):
        haystack = " ".join(
            [
                str(song.get("artist", "")),
                str(song.get("title", "")),
                str(song.get("clip_id", "")),
            ]
        ).lower()
        if lowered in haystack:
            hits.append((index, song))
    if not hits:
        raise ValueError(f"No curated clip matched {match!r}")
    if len(hits) > 1:
        labels = [
            f"{i}: {s['artist']} - {s['title']} [{s.get('clip_id', '')}]"
            for i, s in hits
        ]
        raise ValueError(f"Multiple curated clips matched {match!r}: {labels}")
    return hits[0]


def _load_stale_gold_tool():
    module_path = REPO_ROOT / "tools" / "report_stale_curated_gold.py"
    spec = importlib.util.spec_from_file_location(
        "report_stale_curated_gold", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load stale-gold helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _match_stale_song(
    songs: list[dict[str, object]],
    *,
    stale_index: int,
    tolerance_sec: float,
) -> tuple[int, dict[str, object], Path | None]:
    module = _load_stale_gold_tool()
    entries = module.collect_stale_gold_entries(
        manifest_path=MANIFEST_PATH,
        gold_root=CLIP_GOLD_ROOT,
        tolerance_sec=tolerance_sec,
    )
    if not entries:
        raise ValueError("No stale curated gold entries found")
    if stale_index < 1 or stale_index > len(entries):
        raise ValueError(
            "stale-index "
            f"{stale_index} is out of range for {len(entries)} stale entries"
        )
    entry = entries[stale_index - 1]
    artist = str(entry["artist"]).strip().lower()
    title = str(entry["title"]).strip().lower()
    clip_id = str(entry["clip_id"]).strip().lower()
    for index, song in enumerate(songs, start=1):
        if (
            str(song.get("artist", "")).strip().lower() == artist
            and str(song.get("title", "")).strip().lower() == title
            and str(song.get("clip_id", "")).strip().lower() == clip_id
        ):
            gold_path = entry.get("gold_path")
            if isinstance(gold_path, Path):
                return index, song, gold_path
            if isinstance(gold_path, str) and gold_path:
                return index, song, Path(gold_path)
            return index, song, None
    raise ValueError(
        "Stale curated gold entry not found in manifest: "
        f"{entry['artist']} - {entry['title']} [{entry['clip_id']}]"
    )


def _gold_path(index: int, song: dict[str, object]) -> Path:
    slug = _song_slug(song)
    indexed = CLIP_GOLD_ROOT / f"{index:02d}_{slug}.gold.json"
    if indexed.exists():
        return indexed
    matches = sorted(CLIP_GOLD_ROOT.glob(f"*_{slug}.gold.json"))
    if matches:
        return matches[0]
    fallback = CLIP_GOLD_ROOT / f"{slug}.gold.json"
    if fallback.exists():
        return fallback
    bootstrapped = _bootstrap_missing_gold(index=index, slug=slug)
    if bootstrapped is not None:
        return bootstrapped
    raise FileNotFoundError(f"No gold file found for curated clip {slug}")


def _bootstrap_missing_gold(index: int, slug: str) -> Path | None:
    indexed = DEFAULT_GOLD_ROOT / f"{index:02d}_{slug}.gold.json"
    if indexed.exists():
        return _copy_bootstrapped_gold(index=index, slug=slug, source=indexed)
    matches = sorted(DEFAULT_GOLD_ROOT.glob(f"*_{slug}.gold.json"))
    if matches:
        return _copy_bootstrapped_gold(index=index, slug=slug, source=matches[0])
    return None


def _copy_bootstrapped_gold(*, index: int, slug: str, source: Path) -> Path:
    target = CLIP_GOLD_ROOT / f"{index:02d}_{slug}.gold.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def _cache_dir(song: dict[str, object]) -> Path:
    return Path.home() / ".cache" / "karaoke" / str(song["youtube_id"])


def _canonical_trimmed_clip_path(song: dict[str, object]) -> Path:
    start = _song_float(song, "audio_start_sec")
    duration = _song_float(song, "clip_duration_sec")
    if duration <= 0:
        raise ValueError("clip_duration_sec must be set for canonical clip audio")
    return _cache_dir(song) / f"trimmed_from_{start:.2f}s_for_{duration:.2f}s.wav"


def _source_audio_candidates(song: dict[str, object]) -> list[Path]:
    cache_dir = _cache_dir(song)
    title = str(song["title"])
    start = _song_float(song, "audio_start_sec")
    candidates = [
        cache_dir / f"{title}.wav",
        cache_dir / f"trimmed_from_{start:.2f}s.wav",
    ]
    if cache_dir.exists():
        for path in sorted(cache_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".wav", ".m4a", ".mp3", ".webm", ".mp4"}:
                continue
            if "trimmed_from_" in path.name:
                continue
            if path not in candidates:
                candidates.append(path)
    return candidates


def _wav_duration_sec(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
    except (FileNotFoundError, OSError, wave.Error):
        return None
    if frame_rate <= 0:
        return None
    return frame_count / frame_rate


def _source_trim_offset_sec(source: Path, song: dict[str, object]) -> float:
    start = _song_float(song, "audio_start_sec")
    trimmed_name = f"trimmed_from_{start:.2f}s"
    if source.stem == trimmed_name:
        return 0.0
    return start


def _ensure_clip_audio(song: dict[str, object]) -> Path:
    clip_path = _canonical_trimmed_clip_path(song)
    duration = _song_float(song, "clip_duration_sec")
    if clip_path.exists() and clip_path.stat().st_size > 44:
        clip_duration = _wav_duration_sec(clip_path)
        if (
            clip_duration is not None
            and abs(clip_duration - duration) <= CLIP_DURATION_TOLERANCE_SEC
        ):
            return clip_path

    source = next(
        (path for path in _source_audio_candidates(song) if path.exists()), None
    )
    if source is None:
        raise FileNotFoundError(
            f"No source audio found in {_cache_dir(song)}; expected one of "
            f"{[str(path) for path in _source_audio_candidates(song)]}"
        )

    source_offset = _source_trim_offset_sec(source, song)
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_clip_path = clip_path.with_suffix(".tmp.wav")
    if tmp_clip_path.exists():
        tmp_clip_path.unlink()
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{source_offset:g}",
        "-t",
        f"{duration:g}",
        "-i",
        str(source),
    ]
    if source.suffix.lower() == ".wav":
        ffmpeg_cmd.extend(["-c", "copy"])
    else:
        ffmpeg_cmd.extend(["-vn", "-acodec", "pcm_s16le", "-f", "wav"])
    ffmpeg_cmd.append(str(tmp_clip_path))
    subprocess.run(
        ffmpeg_cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not tmp_clip_path.exists() or tmp_clip_path.stat().st_size <= 44:
        raise RuntimeError(f"Failed to generate valid clip audio at {tmp_clip_path}")
    generated_duration = _wav_duration_sec(tmp_clip_path)
    if (
        generated_duration is None
        or abs(generated_duration - duration) > CLIP_DURATION_TOLERANCE_SEC
    ):
        tmp_clip_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Generated clip duration "
            f"{generated_duration!r}s does not match requested {duration:.2f}s"
        )
    tmp_clip_path.replace(clip_path)
    return clip_path


def _update_gold_audio_path(gold_path: Path, audio_path: Path) -> None:
    doc = json.loads(gold_path.read_text(encoding="utf-8"))
    doc["audio_path"] = str(audio_path)
    lines = doc.get("lines", [])
    if isinstance(lines, list):
        repaired_lines = []
        for line_index, line in enumerate(lines, start=1):
            if not isinstance(line, dict):
                continue
            text = str(line.get("text", "")).strip()
            start = float(line.get("start", 0.0) or 0.0)
            end = float(line.get("end", start) or start)
            words = line.get("words", [])
            if not isinstance(words, list) or not words:
                tokens = [token for token in text.split() if token]
                if tokens and end > start:
                    step = (end - start) / len(tokens)
                    rebuilt_words = []
                    for word_index, token in enumerate(tokens, start=1):
                        word_start = start + (word_index - 1) * step
                        word_end = (
                            end
                            if word_index == len(tokens)
                            else start + word_index * step
                        )
                        rebuilt_words.append(
                            {
                                "word_index": word_index,
                                "text": token,
                                "start": round(word_start, 3),
                                "end": round(word_end, 3),
                            }
                        )
                    line = {
                        "line_index": line_index,
                        "text": text,
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "words": rebuilt_words,
                    }
            repaired_lines.append(line)
        doc["lines"] = repaired_lines
    gold_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _editor_url(gold_path: Path, audio_path: Path) -> str:
    query = urlencode(
        {
            "timing": str(gold_path),
            "audio": str(audio_path),
            "save": str(gold_path),
        }
    )
    return f"{EDITOR_BASE_URL}?{query}"


def _editor_host_port() -> tuple[str, int]:
    parsed = urlparse(EDITOR_BASE_URL)
    return parsed.hostname or "127.0.0.1", parsed.port or 80


def _is_editor_reachable(host: str, port: int, timeout_sec: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except OSError:
        return False


def _is_editor_healthy(timeout_sec: float = 0.5) -> bool:
    try:
        with urllib.request.urlopen(EDITOR_BASE_URL, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8", errors="ignore")
            return response.status == 200 and "Gold Timing Editor" in body
    except Exception:
        return False


def _start_editor_server() -> subprocess.Popen[bytes]:
    host, port = _editor_host_port()
    return subprocess.Popen(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "gold_timing_editor.py"),
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=str(REPO_ROOT),
        env=os.environ.copy(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _ensure_editor_server() -> None:
    host, port = _editor_host_port()
    if _is_editor_healthy():
        return

    if _is_editor_reachable(host, port) and not _is_editor_healthy():
        raise RuntimeError(
            f"Port {host}:{port} is already in use, "
            "but it is not serving the gold timing editor"
        )

    proc = _start_editor_server()
    deadline = time.monotonic() + EDITOR_READY_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if _is_editor_healthy():
            return
        if proc.poll() is not None:
            raise RuntimeError(
                f"Gold timing editor exited before becoming ready on {host}:{port}"
            )
        time.sleep(EDITOR_READY_POLL_SEC)

    raise RuntimeError(f"Gold timing editor did not become ready on {host}:{port}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match", help="Substring match for artist/title/clip_id")
    parser.add_argument(
        "--stale-index",
        type=int,
        default=None,
        help="Open the Nth stale curated gold entry from report_stale_curated_gold.py",
    )
    parser.add_argument(
        "--stale-tolerance-sec",
        type=float,
        default=0.5,
        help="Tolerance used when selecting stale curated gold entries",
    )
    parser.add_argument(
        "--open-editor",
        action="store_true",
        help="Open the resolved curated clip in the local gold editor",
    )
    args = parser.parse_args()
    if bool(args.match) == bool(args.stale_index is not None):
        parser.error("Provide exactly one of --match or --stale-index")

    songs = _load_manifest(MANIFEST_PATH)
    try:
        if args.stale_index is not None:
            index, song, preferred_gold_path = _match_stale_song(
                songs,
                stale_index=args.stale_index,
                tolerance_sec=args.stale_tolerance_sec,
            )
        else:
            index, song = _match_song(songs, args.match)
            preferred_gold_path = None
    except ValueError as exc:
        parser.exit(1, f"curated_clip_helper: {exc}\n")
    gold_path = preferred_gold_path or _gold_path(index, song)
    audio_path = _ensure_clip_audio(song)
    _update_gold_audio_path(gold_path, audio_path)
    url = _editor_url(gold_path, audio_path)

    print(
        json.dumps(
            {
                "gold_path": str(gold_path),
                "audio_path": str(audio_path),
                "editor_url": url,
            },
            indent=2,
        )
    )
    if args.open_editor:
        _ensure_editor_server()
        webbrowser.open(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
