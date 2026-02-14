#!/usr/bin/env python3
"""Local web editor for word-level gold lyric timings."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

EDITOR_DIR = Path(__file__).resolve().parent / "gold_timing_editor"
SNAP_SECONDS = 0.05


@dataclass
class ValidationError(Exception):
    message: str


def _round_snap(value: float) -> float:
    return round(round(float(value) / SNAP_SECONDS) * SNAP_SECONDS, 3)


def _validate_and_normalize_gold(doc: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(doc, dict):
        raise ValidationError("Document must be an object")
    lines = doc.get("lines", [])
    if not isinstance(lines, list) or not lines:
        raise ValidationError("Missing lines")
    normalized_lines: list[dict[str, Any]] = []
    for i, line in enumerate(lines, start=1):
        words = line.get("words", [])
        nw = []
        for j, word in enumerate(words, start=1):
            text = str(word.get("text", "")).strip()
            start, end = _round_snap(float(word["start"])), _round_snap(
                float(word["end"])
            )
            nw.append({"word_index": j, "text": text, "start": start, "end": end})
        if not nw:
            continue
        normalized_lines.append(
            {
                "line_index": i,
                "text": " ".join(w["text"] for w in nw),
                "start": nw[0]["start"],
                "end": nw[-1]["end"],
                "words": nw,
            }
        )
    return {
        "schema_version": "1.0",
        "title": str(doc.get("title", "")),
        "artist": str(doc.get("artist", "")),
        "audio_path": str(doc.get("audio_path", "")),
        "lines": normalized_lines,
    }


def _load_document(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("lines"), list):
        return _validate_and_normalize_gold(raw)
    raise ValidationError("Unsupported file format")


def _load_document(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("lines"), list):
        return _validate_and_normalize_gold(raw)
    raise ValidationError("Unsupported file format")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(EDITOR_DIR), **kwargs)

    def _serve_absolute_audio(self, path: Path) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        file_size = path.stat().st_size
        suffix = path.suffix.lower()
        ctype = (
            "audio/wav"
            if suffix == ".wav"
            else (
                "audio/mpeg"
                if suffix == ".mp3"
                else (
                    "audio/mp4"
                    if suffix in (".m4a", ".mp4")
                    else "application/octet-stream"
                )
            )
        )

        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            try:
                ranges = range_header.split("=")[1].split("-")
                start = int(ranges[0]) if ranges[0] else 0
                end = int(ranges[1]) if len(ranges) > 1 and ranges[1] else file_size - 1
            except Exception:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return

            if start >= file_size or end >= file_size:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return

            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with path.open("rb") as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = f.read(min(remaining, 65536))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        else:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with path.open("rb") as f:
                shutil.copyfileobj(f, self.wfile)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/audio":
            path_str = unquote(parse_qs(parsed.query).get("path", [""])[0])
            real_path = Path(path_str).expanduser().resolve()
            self._serve_absolute_audio(real_path)
        else:
            super().do_GET()

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/api/load":
                path = Path(body["path"]).expanduser().resolve()
                doc = json.loads(path.read_text(encoding="utf-8"))
                self._send_json(
                    HTTPStatus.OK, {"document": _validate_and_normalize_gold(doc)}
                )
            elif self.path == "/api/save":
                path = Path(body["path"]).expanduser().resolve()
                doc = _validate_and_normalize_gold(body["document"])
                path.write_text(json.dumps(doc, indent=2) + "\n")
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "saved": str(path),
                        "word_count": sum(len(line["words"]) for line in doc["lines"]),
                    },
                )
        except Exception as e:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)})

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print(f"Serving editor at http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
