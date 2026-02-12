#!/usr/bin/env python3
"""Local web editor for word-level gold lyric timings.

Runs a tiny HTTP server that:
- serves a static UI
- loads/saves JSON timing files by local path
- streams local audio files for browser playback and seeking
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

EDITOR_DIR = Path(__file__).resolve().parent / "gold_timing_editor"
SNAP_SECONDS = 0.1


@dataclass
class ValidationError(Exception):
    message: str


def _round_snap(value: float) -> float:
    return round(round(float(value) / SNAP_SECONDS) * SNAP_SECONDS, 3)


def _validate_and_normalize_gold(doc: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(doc, dict):
        raise ValidationError("Document must be an object")

    lines = doc.get("lines")
    if not isinstance(lines, list) or not lines:
        raise ValidationError("Document must include a non-empty 'lines' array")

    normalized_lines: list[dict[str, Any]] = []
    prev_end = 0.0

    for i, line in enumerate(lines, start=1):
        if not isinstance(line, dict):
            raise ValidationError(f"Line {i} must be an object")

        words = line.get("words")
        if not isinstance(words, list) or not words:
            raise ValidationError(f"Line {i} must include a non-empty 'words' array")

        normalized_words: list[dict[str, Any]] = []
        for j, word in enumerate(words, start=1):
            if not isinstance(word, dict):
                raise ValidationError(f"Line {i}, word {j} must be an object")
            text = str(word.get("text", "")).strip()
            if not text:
                raise ValidationError(f"Line {i}, word {j} text is required")

            try:
                start = _round_snap(float(word["start"]))
                end = _round_snap(float(word["end"]))
            except (KeyError, TypeError, ValueError):
                raise ValidationError(
                    f"Line {i}, word {j} must include numeric start/end"
                ) from None

            if start < 0:
                raise ValidationError(f"Line {i}, word {j} start cannot be negative")
            if end < start:
                raise ValidationError(f"Line {i}, word {j} end must be >= start")
            if start < prev_end:
                raise ValidationError(
                    f"Line {i}, word {j} overlaps previous word ({start} < {prev_end})"
                )

            normalized_word = {
                "word_index": j,
                "text": text,
                "start": start,
                "end": end,
            }
            normalized_words.append(normalized_word)
            prev_end = end

        normalized_lines.append(
            {
                "line_index": i,
                "text": " ".join(w["text"] for w in normalized_words),
                "start": normalized_words[0]["start"],
                "end": normalized_words[-1]["end"],
                "words": normalized_words,
            }
        )

    return {
        "schema_version": "1.0",
        "title": str(doc.get("title", "")).strip(),
        "artist": str(doc.get("artist", "")).strip(),
        "audio_path": str(doc.get("audio_path", "")).strip(),
        "source_timing_path": str(doc.get("source_timing_path", "")).strip(),
        "lines": normalized_lines,
    }


def _from_timing_report(report: dict[str, Any]) -> dict[str, Any]:
    lines = report.get("lines")
    if not isinstance(lines, list) or not lines:
        raise ValidationError("Timing report has no lines")

    normalized_lines: list[dict[str, Any]] = []
    prev_end = 0.0

    for i, line in enumerate(lines, start=1):
        raw_words = line.get("words", [])
        if not isinstance(raw_words, list) or not raw_words:
            continue

        words: list[dict[str, Any]] = []
        for j, word in enumerate(raw_words, start=1):
            text = str(word.get("text", "")).strip()
            if not text:
                continue
            start = _round_snap(float(word.get("start", 0.0)))
            end = _round_snap(float(word.get("end", start)))
            if end < start:
                end = start
            if start < prev_end:
                start = prev_end
            if end < start:
                end = start
            words.append(
                {
                    "word_index": j,
                    "text": text,
                    "start": start,
                    "end": end,
                }
            )
            prev_end = end

        if not words:
            continue

        normalized_lines.append(
            {
                "line_index": len(normalized_lines) + 1,
                "text": " ".join(w["text"] for w in words),
                "start": words[0]["start"],
                "end": words[-1]["end"],
                "words": words,
            }
        )

    if not normalized_lines:
        raise ValidationError("Timing report has no usable word timings")

    return {
        "schema_version": "1.0",
        "title": str(report.get("title", "")).strip(),
        "artist": str(report.get("artist", "")).strip(),
        "audio_path": "",
        "source_timing_path": "",
        "lines": normalized_lines,
    }


def _load_document(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(raw, dict) and isinstance(raw.get("lines"), list):
        first_line = raw["lines"][0] if raw["lines"] else None
        if (
            isinstance(first_line, dict)
            and isinstance(first_line.get("words"), list)
            and first_line["words"]
            and "start" in first_line["words"][0]
            and "end" in first_line["words"][0]
        ):
            if raw.get("schema_version"):
                return _validate_and_normalize_gold(raw)
            return _from_timing_report(raw)

    raise ValidationError("Unsupported file format")


class Handler(BaseHTTPRequestHandler):
    server_version = "GoldTimingEditor/1.0"

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValidationError("Invalid Content-Length")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            raise ValidationError("Request body must be valid JSON") from None
        if not isinstance(data, dict):
            raise ValidationError("Request body must be a JSON object")
        return data

    def _serve_static(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        ctype, _ = mimetypes.guess_type(str(file_path))
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _safe_path(self, raw_path: str) -> Path:
        if not raw_path:
            raise ValidationError("Path is required")
        return Path(raw_path).expanduser().resolve()

    def _serve_audio(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        file_size = path.stat().st_size
        ctype, _ = mimetypes.guess_type(str(path))
        ctype = ctype or "audio/mpeg"

        range_header = self.headers.get("Range")
        if not range_header:
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(data)
            return

        if not range_header.startswith("bytes="):
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return

        start_s, _, end_s = range_header[6:].partition("-")
        try:
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else file_size - 1
        except ValueError:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return

        if start < 0 or end >= file_size or start > end:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return

        with path.open("rb") as fh:
            fh.seek(start)
            data = fh.read(end - start + 1)

        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_static(EDITOR_DIR / "index.html")
            return
        if parsed.path == "/app.js":
            self._serve_static(EDITOR_DIR / "app.js")
            return
        if parsed.path == "/styles.css":
            self._serve_static(EDITOR_DIR / "styles.css")
            return
        if parsed.path == "/api/audio":
            query = parse_qs(parsed.query)
            raw_path = unquote((query.get("path") or [""])[0])
            try:
                path = self._safe_path(raw_path)
            except ValidationError as exc:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": exc.message})
                return
            self._serve_audio(path)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._read_json()
            if self.path == "/api/load":
                path = self._safe_path(str(body.get("path", "")))
                doc = _load_document(path)
                doc["source_timing_path"] = str(path)
                self._write_json(HTTPStatus.OK, {"document": doc})
                return

            if self.path == "/api/save":
                path = self._safe_path(str(body.get("path", "")))
                doc = _validate_and_normalize_gold(body.get("document", {}))
                if not path.parent.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "saved": str(path),
                        "line_count": len(doc["lines"]),
                        "word_count": sum(len(line["words"]) for line in doc["lines"]),
                    },
                )
                return

            self.send_error(HTTPStatus.NOT_FOUND)
        except ValidationError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": exc.message})
        except FileNotFoundError:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "File not found"})
        except PermissionError:
            self._write_json(HTTPStatus.FORBIDDEN, {"error": "Permission denied"})
        except json.JSONDecodeError:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON file"})
        except Exception as exc:  # pragma: no cover
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"Unexpected error: {exc}"},
            )

    def log_message(self, fmt: str, *args: object) -> None:
        if os.environ.get("Y2K_GOLD_EDITOR_QUIET") == "1":
            return
        super().log_message(fmt, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local gold timing editor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if not EDITOR_DIR.exists():
        raise SystemExit(f"Missing UI directory: {EDITOR_DIR}")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Gold timing editor listening at {url}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
