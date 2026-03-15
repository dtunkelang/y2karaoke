#!/usr/bin/env python3
"""Run the benchmark song suite and emit an aggregated timing-quality report."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import unicodedata
from urllib.parse import parse_qs, urlparse
from typing import Any, Callable, Iterable, cast

import yaml  # type: ignore[import-untyped]

from y2karaoke.core.audio_analysis import extract_audio_features
from y2karaoke.core.components.whisper.whisper_alignment_line_helpers import (
    first_onset_after,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "benchmark_songs.yaml"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"
DEFAULT_GOLD_ROOT = REPO_ROOT / "benchmarks" / "gold_set_candidate" / "20260305T231015Z"
DEFAULT_CLIP_GOLD_ROOT = (
    REPO_ROOT / "benchmarks" / "clip_gold_candidate" / "20260312T_curated_clips"
)
_AGREEMENT_FILLER_TOKENS = {
    "ah",
    "aah",
    "eh",
    "ha",
    "hey",
    "la",
    "na",
    "oh",
    "ooh",
    "uh",
    "um",
    "woo",
    "woah",
    "whoa",
    "whoo",
    "ya",
    "yeah",
    "yo",
}

_AGREEMENT_LEAD_IN_TOKENS = {
    "a",
    "ce",
    "dans",
    "de",
    "des",
    "dont",
    "du",
    "et",
    "j",
    "je",
    "la",
    "le",
    "les",
    "ma",
    "mes",
    "mon",
    "oh",
    "pour",
    "sans",
    "sur",
    "ta",
    "tes",
    "ton",
    "un",
    "une",
}

_AGREEMENT_TRAILING_FILLER_TOKENS = {
    "ah",
    "ay",
    "eh",
    "oh",
    "ooh",
    "uh",
    "woo",
}
_GOLD_SOFTENED_TAG_TOKENS = {
    "chris",
    "jedi",
    "jeje",
    "omega",
}
_OPTIONAL_HOOK_BOUNDARY_PHRASES: tuple[tuple[str, ...], ...] = (
    ("come", "on"),
    ("hot", "damn"),
)
_AGREEMENT_CONTRACTION_SPECIALS: dict[str, tuple[str, ...]] = {
    "can't": ("can", "not"),
    "cannot": ("can", "not"),
    "won't": ("will", "not"),
    "shan't": ("shall", "not"),
    "ain't": ("is", "not"),
    "i'm": ("i", "am"),
    "it's": ("it", "is"),
    "you're": ("you", "are"),
    "we're": ("we", "are"),
    "they're": ("they", "are"),
    "that's": ("that", "is"),
    "there's": ("there", "is"),
    "what's": ("what", "is"),
    "let's": ("let", "us"),
}
_AGREEMENT_COLLOQUIAL_SPECIALS: dict[str, tuple[str, ...]] = {
    "gonna": ("going", "to"),
    "wanna": ("want", "to"),
    "gotta": ("got", "to"),
    "lemme": ("let", "me"),
    "gimme": ("give", "me"),
    "kinda": ("kind", "of"),
    "sorta": ("sort", "of"),
    "yall": ("you", "all"),
    "ya'll": ("you", "all"),
    "imma": ("i", "am", "going", "to"),
    "cuz": ("because",),
    "cos": ("because",),
    "coz": ("because",),
    "cause": ("because",),
    "em": ("them",),
}
_AUDIO_FEATURES_CACHE: dict[str, Any] = {}


def _benchmark_cache_roots() -> list[Path]:
    roots = [REPO_ROOT / ".cache", Path.home() / ".cache" / "karaoke"]
    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


@dataclass(frozen=True)
class BenchmarkSong:
    manifest_index: int
    artist: str
    title: str
    youtube_id: str
    youtube_url: str
    clip_id: str | None = None
    audio_start_sec: float = 0.0
    lyrics_file: str | None = None

    @property
    def slug(self) -> str:
        safe = self.base_slug
        if self.clip_id:
            clip_safe = re.sub(r"[^a-z0-9]+", "-", self.clip_id.lower()).strip("-")
            if clip_safe:
                safe = f"{safe}-{clip_safe}" if safe else clip_safe
        return safe or self.youtube_id

    @property
    def base_slug(self) -> str:
        safe = f"{self.artist}-{self.title}".lower()
        return re.sub(r"[^a-z0-9]+", "-", safe).strip("-") or self.youtube_id


def _parse_manifest(path: Path) -> list[BenchmarkSong]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Manifest root must be a mapping")

    songs_raw = raw.get("songs", [])
    if not isinstance(songs_raw, list):
        raise ValueError("Manifest 'songs' must be a list")

    songs: list[BenchmarkSong] = []
    for idx, song in enumerate(songs_raw):
        if not isinstance(song, dict):
            raise ValueError(f"songs[{idx}] must be a mapping")
        try:
            audio_start_sec = song.get("audio_start_sec", 0.0)
            if audio_start_sec is None:
                audio_start_sec = 0.0
            songs.append(
                BenchmarkSong(
                    manifest_index=idx + 1,
                    artist=str(song["artist"]),
                    title=str(song["title"]),
                    youtube_id=str(song["youtube_id"]),
                    youtube_url=str(song["youtube_url"]),
                    clip_id=_normalize_optional_manifest_text(song.get("clip_id")),
                    audio_start_sec=float(audio_start_sec),
                    lyrics_file=_resolve_manifest_optional_path(
                        path.parent, song.get("lyrics_file")
                    ),
                )
            )
        except KeyError as exc:
            raise ValueError(f"songs[{idx}] missing required field: {exc}") from exc
    return songs


def _resolve_manifest_optional_path(manifest_dir: Path, value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = (manifest_dir / candidate).resolve()
    return str(candidate)


def _normalize_optional_manifest_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _validate_cli_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.min_dtw_song_coverage_ratio <= 1.0:
        raise ValueError("--min-dtw-song-coverage-ratio must be between 0 and 1")
    if not 0.0 <= args.min_dtw_line_coverage_ratio <= 1.0:
        raise ValueError("--min-dtw-line-coverage-ratio must be between 0 and 1")
    if not 0.0 <= args.min_timing_quality_score_line_weighted <= 1.0:
        raise ValueError(
            "--min-timing-quality-score-line-weighted must be between 0 and 1"
        )
    if args.min_agreement_coverage_gain_for_bad_ratio_warning < 0.0:
        raise ValueError(
            "--min-agreement-coverage-gain-for-bad-ratio-warning must be >= 0"
        )
    if args.max_agreement_bad_ratio_increase_on_coverage_gain < 0.0:
        raise ValueError(
            "--max-agreement-bad-ratio-increase-on-coverage-gain must be >= 0"
        )
    if not 0.0 <= args.max_whisper_phase_share <= 1.0:
        raise ValueError("--max-whisper-phase-share must be between 0 and 1")
    if not 0.0 <= args.max_alignment_phase_share <= 1.0:
        raise ValueError("--max-alignment-phase-share must be between 0 and 1")
    if args.max_scheduler_overhead_sec < 0.0:
        raise ValueError("--max-scheduler-overhead-sec must be >= 0")


def _filter_manifest_songs(
    songs: list[BenchmarkSong], *, match: str | None, max_songs: int
) -> list[BenchmarkSong]:
    selected = songs
    if match:
        regex = re.compile(match, re.IGNORECASE)
        selected = [
            song
            for song in selected
            if regex.search(
                " ".join(
                    part
                    for part in (song.artist, song.title, song.clip_id or "")
                    if part
                )
            )
            is not None
        ]
    if max_songs > 0:
        selected = selected[:max_songs]
    return selected


def _apply_aggregate_only_cached_scope(
    songs: list[BenchmarkSong],
    *,
    aggregate_only: bool,
    match: str | None,
    max_songs: int,
    run_dir: Path,
) -> list[BenchmarkSong]:
    if not aggregate_only or match or max_songs > 0:
        return songs
    cached_slugs = _discover_cached_result_slugs(run_dir)
    selected: list[BenchmarkSong] = []
    for song in songs:
        slug_candidates = [song.slug]
        if song.clip_id:
            slug_candidates.append(song.base_slug)
        if any(slug in cached_slugs for slug in slug_candidates):
            selected.append(song)
    return selected


def _pctile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = pos - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _mean(values: Iterable[float]) -> float | None:
    data = list(values)
    if not data:
        return None
    return float(statistics.fmean(data))


def _env_float(
    name: str, default: float, *, min_value: float, max_value: float
) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        return default
    return max(min_value, min(max_value, parsed))


def _expand_agreement_token(token: str) -> list[str]:
    token_canon = token.lstrip("'")
    if token_canon in _AGREEMENT_COLLOQUIAL_SPECIALS:
        return list(_AGREEMENT_COLLOQUIAL_SPECIALS[token_canon])
    if token_canon in _AGREEMENT_CONTRACTION_SPECIALS:
        return list(_AGREEMENT_CONTRACTION_SPECIALS[token_canon])
    if token.endswith("in'") and len(token) > 4:
        return [f"{token[:-1]}g"]
    if token.endswith("n't") and len(token) > 3:
        return [token[:-3], "not"]
    if token.endswith("'re") and len(token) > 3:
        return [token[:-3], "are"]
    if token.endswith("'ve") and len(token) > 3:
        return [token[:-3], "have"]
    if token.endswith("'ll") and len(token) > 3:
        return [token[:-3], "will"]
    if token.endswith("'m") and len(token) > 2:
        return [token[:-2], "am"]
    if token.endswith("'d") and len(token) > 2:
        return [token[:-2], "would"]
    if token.endswith("'s") and len(token) > 2:
        return [token[:-2]]
    return [token.replace("'", "")]


def _normalize_agreement_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    folded = "".join(
        ch
        for ch in unicodedata.normalize("NFKD", text.lower())
        if unicodedata.category(ch) != "Mn"
    )
    folded = re.sub(r"[’`´]", "'", folded)
    folded = re.sub(r"[^a-z0-9'\s]", " ", folded)
    raw_tokens = [tok for tok in re.sub(r"\s+", " ", folded).strip().split(" ") if tok]
    expanded: list[str] = []
    for token in raw_tokens:
        expanded.extend(_expand_agreement_token(token))

    normalized_tokens: list[str] = []
    prev = ""
    for token in expanded:
        if not token:
            continue
        # Repeated ad-lib fillers should not dominate string-level similarity.
        if token in _AGREEMENT_FILLER_TOKENS and prev == token:
            continue
        normalized_tokens.append(token)
        prev = token
    return " ".join(normalized_tokens).strip()


def _normalize_agreement_text_hook_boundary(text: Any) -> str:
    base = _normalize_agreement_text(text)
    if not base:
        return ""
    tokens = [tok for tok in base.split() if tok]
    if not tokens:
        return ""
    return " ".join(_strip_optional_hook_boundary_tokens(tokens)).strip()


def _agreement_text_similarity(
    left: Any,
    right: Any,
    *,
    normalize_fn: Callable[[Any], str] = _normalize_agreement_text,
) -> float:
    a = normalize_fn(left)
    b = normalize_fn(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _agreement_token_overlap(
    left: Any,
    right: Any,
    *,
    normalize_fn: Callable[[Any], str] = _normalize_agreement_text,
) -> float:
    a = normalize_fn(left).split()
    b = normalize_fn(right).split()
    if not a or not b:
        return 0.0
    a_set = set(a)
    b_set = set(b)
    overlap = len(a_set & b_set)
    return overlap / max(1, min(len(a_set), len(b_set)))


def _normalize_word_text(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    folded = "".join(
        ch
        for ch in unicodedata.normalize("NFKD", raw.strip().lower())
        if unicodedata.category(ch) != "Mn"
    )
    cleaned = re.sub(r"[^a-z0-9'\s]", " ", folded)
    return re.sub(r"\s+", " ", cleaned).strip()


def _lexical_tokens_basic_raw(text: str) -> list[str]:
    tokens: list[str] = []
    folded = "".join(
        ch
        for ch in unicodedata.normalize("NFKD", text)
        if unicodedata.category(ch) != "Mn"
    )
    for raw in folded.split():
        tok = "".join(ch for ch in raw.lower() if ch.isalpha() or ch == "'")
        if tok:
            tokens.append(tok)
    return tokens


def _lexical_tokens_compact_raw(text: str) -> list[str]:
    tokens: list[str] = []
    folded = "".join(
        ch
        for ch in unicodedata.normalize("NFKD", text)
        if unicodedata.category(ch) != "Mn"
    )
    for raw in folded.split():
        tok = "".join(ch for ch in raw.lower() if ch.isalpha())
        if tok:
            tokens.append(tok)
    return tokens


def _strip_optional_hook_boundary_tokens(tokens: list[str]) -> list[str]:
    if len(tokens) < 3:
        return tokens

    def _core_token_count(items: list[str]) -> int:
        return sum(1 for tok in items if tok not in _AGREEMENT_FILLER_TOKENS)

    if not any(tok not in _AGREEMENT_FILLER_TOKENS for tok in tokens):
        return tokens

    out = list(tokens)
    changed = True
    while changed:
        changed = False
        for phrase in _OPTIONAL_HOOK_BOUNDARY_PHRASES:
            phrase_len = len(phrase)
            if (
                len(out) - phrase_len >= 2
                and tuple(out[:phrase_len]) == phrase
                and _core_token_count(out[phrase_len:]) >= 3
            ):
                out = out[phrase_len:]
                changed = True
            if (
                len(out) - phrase_len >= 2
                and tuple(out[-phrase_len:]) == phrase
                and _core_token_count(out[:-phrase_len]) >= 3
            ):
                out = out[:-phrase_len]
                changed = True
        if (
            len(out) >= 4
            and out[0] in _AGREEMENT_FILLER_TOKENS
            and _core_token_count(out[1:]) >= 3
        ):
            out = out[1:]
            changed = True
        if (
            len(out) >= 4
            and out[-1] in _AGREEMENT_FILLER_TOKENS
            and _core_token_count(out[:-1]) >= 3
        ):
            out = out[:-1]
            changed = True
        for idx in range(1, len(out) - 1):
            if (
                len(out) >= 5
                and out[idx] in _AGREEMENT_FILLER_TOKENS
                and out[idx - 1] not in _AGREEMENT_FILLER_TOKENS
                and out[idx + 1] not in _AGREEMENT_FILLER_TOKENS
                and _core_token_count(out[:idx] + out[idx + 1 :]) >= 3
            ):
                out = out[:idx] + out[idx + 1 :]
                changed = True
                break
    return out


def _word_similarity(left: Any, right: Any) -> float:
    a = _normalize_word_text(left)
    b = _normalize_word_text(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _align_words_for_gold_comparison(
    generated_words: list[dict[str, Any]],
    gold_words: list[dict[str, Any]],
    *,
    lookahead: int = 12,
    min_similarity: float = 0.8,
) -> list[tuple[int, int, float]]:
    """Monotonic text-aware matching for generated-vs-gold words."""
    if not generated_words or not gold_words:
        return []

    matches: list[tuple[int, int, float]] = []
    gen_idx = 0
    gold_idx = 0
    gen_len = len(generated_words)
    gold_len = len(gold_words)

    while gen_idx < gen_len and gold_idx < gold_len:
        best: tuple[int, int, float, float] | None = None
        gen_hi = min(gen_len, gen_idx + lookahead + 1)
        gold_hi = min(gold_len, gold_idx + lookahead + 1)

        for i in range(gen_idx, gen_hi):
            for j in range(gold_idx, gold_hi):
                sim = _word_similarity(
                    generated_words[i].get("text", ""),
                    gold_words[j].get("text", ""),
                )
                if sim < min_similarity:
                    continue
                start_delta = abs(
                    float(generated_words[i].get("start", 0.0))
                    - float(gold_words[j].get("start", 0.0))
                )
                # Prefer higher similarity, then minimal skip distance, then timing proximity.
                score = (
                    (sim * 2.0)
                    - (0.05 * ((i - gen_idx) + (j - gold_idx)))
                    - (0.01 * min(start_delta, 10.0))
                )
                if best is None or score > best[3]:
                    best = (i, j, sim, score)

        if best is not None:
            i, j, sim, _score = best
            matches.append((i, j, sim))
            gen_idx = i + 1
            gold_idx = j + 1
            continue

        gen_start = float(generated_words[gen_idx].get("start", 0.0))
        gold_start = float(gold_words[gold_idx].get("start", 0.0))
        if gen_start <= gold_start:
            gen_idx += 1
        else:
            gold_idx += 1

    return matches


def _flatten_words_from_timing_doc(
    doc: dict[str, Any],
    *,
    mark_parenthetical_optional: bool = False,
    suppress_line_indexes: set[int] | None = None,
) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    lines = doc.get("lines", [])
    if not isinstance(lines, list):
        return words
    for line_index, line in enumerate(lines):
        if not isinstance(line, dict):
            continue
        if suppress_line_indexes and line_index in suppress_line_indexes:
            continue
        line_words = line.get("words", [])
        if not isinstance(line_words, list):
            continue
        paren_depth = 0
        line_word_entries: list[dict[str, Any]] = []
        for w in line_words:
            if not isinstance(w, dict):
                continue
            start = w.get("start")
            end = w.get("end")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            text = str(w.get("text", ""))
            optional = False
            if mark_parenthetical_optional:
                open_count = text.count("(")
                close_count = text.count(")")
                # Parenthesized tokens are treated as optional/backing vocals.
                optional = paren_depth > 0 or open_count > 0
                paren_depth = max(0, paren_depth + open_count - close_count)
            line_word_entries.append(
                {
                    "text": text,
                    "start": float(start),
                    "end": float(end),
                    "optional": optional,
                    "line_index": line_index,
                }
            )
        if mark_parenthetical_optional and line_word_entries:
            trailing_optional_seen = False
            for entry in reversed(line_word_entries):
                if bool(entry.get("optional")):
                    trailing_optional_seen = True
                    continue
                entry["followed_by_optional_tail"] = trailing_optional_seen
                trailing_optional_seen = False
        words.extend(line_word_entries)
    return words


def _normalize_interjection_token(text: str) -> str:
    return re.sub(r"[^a-z]+", "", str(text).lower())


def _extract_parenthetical_interjection_lines(
    doc: dict[str, Any],
    *,
    suppress_line_indexes: set[int] | None = None,
) -> list[dict[str, Any]]:
    lines_out: list[dict[str, Any]] = []
    lines = doc.get("lines", [])
    if not isinstance(lines, list):
        return lines_out
    for line_index, line in enumerate(lines):
        if not isinstance(line, dict):
            continue
        if suppress_line_indexes and line_index in suppress_line_indexes:
            continue
        line_words = line.get("words", [])
        if not isinstance(line_words, list) or not line_words:
            continue
        paren_depth = 0
        optional_flags: list[bool] = []
        normalized_tokens: list[str] = []
        for w in line_words:
            if not isinstance(w, dict):
                continue
            text = str(w.get("text", ""))
            open_count = text.count("(")
            close_count = text.count(")")
            optional = paren_depth > 0 or open_count > 0
            paren_depth = max(0, paren_depth + open_count - close_count)
            optional_flags.append(optional)
            token = _normalize_interjection_token(text)
            if token:
                normalized_tokens.append(token)
        if not optional_flags or not all(optional_flags):
            continue
        if not normalized_tokens or not all(
            token in _AGREEMENT_FILLER_TOKENS for token in normalized_tokens
        ):
            continue
        start = line.get("start")
        end = line.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        lines_out.append(
            {
                "line_index": line_index,
                "text": str(line.get("text", "")),
                "normalized_text": " ".join(normalized_tokens),
                "start": float(start),
                "end": float(end),
            }
        )
    return lines_out


def _align_parenthetical_interjection_lines(
    generated_lines: list[dict[str, Any]],
    gold_lines: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    gen_idx = 0
    gold_idx = 0
    while gen_idx < len(generated_lines) and gold_idx < len(gold_lines):
        gen_norm = str(generated_lines[gen_idx].get("normalized_text", "")).strip()
        gold_norm = str(gold_lines[gold_idx].get("normalized_text", "")).strip()
        if gen_norm and gen_norm == gold_norm:
            matches.append((gen_idx, gold_idx))
            gen_idx += 1
            gold_idx += 1
            continue
        gen_idx += 1
    return matches


def _line_is_parenthetical_interjection(line: dict[str, Any]) -> bool:
    doc = {"lines": [line]}
    return bool(_extract_parenthetical_interjection_lines(doc))


def _extract_lines_for_gold_comparison(
    doc: dict[str, Any],
    *,
    suppress_line_indexes: set[int] | None = None,
) -> list[dict[str, Any]]:
    lines_out: list[dict[str, Any]] = []
    lines = doc.get("lines", [])
    if not isinstance(lines, list):
        return lines_out
    for line_index, line in enumerate(lines):
        if not isinstance(line, dict):
            continue
        if suppress_line_indexes and line_index in suppress_line_indexes:
            continue
        start = line.get("start")
        end = line.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            line_words = line.get("words", [])
            if isinstance(line_words, list) and line_words:
                valid_words = [w for w in line_words if isinstance(w, dict)]
                if not isinstance(start, (int, float)) and valid_words:
                    word_start = valid_words[0].get("start")
                    if isinstance(word_start, (int, float)):
                        start = float(word_start)
                if not isinstance(end, (int, float)) and valid_words:
                    word_end = valid_words[-1].get("end")
                    if isinstance(word_end, (int, float)):
                        end = float(word_end)
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        text = str(line.get("text", ""))
        if not text.strip():
            line_words = line.get("words", [])
            if isinstance(line_words, list):
                text = " ".join(
                    str(w.get("text", "")).strip()
                    for w in line_words
                    if isinstance(w, dict) and str(w.get("text", "")).strip()
                )
        normalized_text = _normalize_agreement_text(text)
        if not normalized_text:
            continue
        lines_out.append(
            {
                "line_index": line_index,
                "text": text,
                "normalized_text": normalized_text,
                "start": float(start),
                "end": float(end),
                "duration": max(0.0, float(end) - float(start)),
            }
        )
    return lines_out


def _is_softened_adlib_or_tag_text(text: Any) -> bool:
    tokens = [tok for tok in _normalize_agreement_text(text).split() if tok]
    if not tokens or len(tokens) > 4:
        return False
    allowed = _AGREEMENT_FILLER_TOKENS | _GOLD_SOFTENED_TAG_TOKENS
    return all(token in allowed for token in tokens)


def _whisper_window_text(line: dict[str, Any]) -> str:
    raw_words = line.get("whisper_window_words")
    if not isinstance(raw_words, list):
        return ""
    return " ".join(
        str(word.get("text", "")).strip()
        for word in raw_words
        if isinstance(word, dict) and str(word.get("text", "")).strip()
    ).strip()


def _softened_gold_adlib_line_indexes(report: dict[str, Any]) -> set[int]:
    lines = report.get("lines", [])
    if not isinstance(lines, list):
        return set()
    low_conf_raw = report.get("low_confidence_lines", [])
    low_conf_indexes: set[int] = set()
    if isinstance(low_conf_raw, list):
        for item in low_conf_raw:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            if isinstance(index, int) and index > 0:
                low_conf_indexes.add(index - 1)

    softened: set[int] = set()
    for line_index, line in enumerate(lines):
        if not isinstance(line, dict):
            continue
        if _line_is_parenthetical_interjection(line):
            continue
        text = str(line.get("text", "")).strip()
        if not _is_softened_adlib_or_tag_text(text):
            continue
        window_text = _whisper_window_text(line)
        window_overlap = _agreement_token_overlap(text, window_text)
        avg_prob = line.get("whisper_window_avg_prob")
        window_word_count = line.get("whisper_window_word_count")
        if (
            line_index in low_conf_indexes
            or (isinstance(avg_prob, (int, float)) and float(avg_prob) < 0.35)
            or (isinstance(window_word_count, int) and window_word_count <= 1)
            or window_overlap < 0.35
        ):
            softened.add(line_index)
    return softened


def _align_lines_for_gold_comparison(
    generated_lines: list[dict[str, Any]],
    gold_lines: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    gen_idx = 0
    gold_idx = 0
    while gen_idx < len(generated_lines) and gold_idx < len(gold_lines):
        gen_norm = str(generated_lines[gen_idx].get("normalized_text", "")).strip()
        gold_norm = str(gold_lines[gold_idx].get("normalized_text", "")).strip()
        if gen_norm and gen_norm == gold_norm:
            matches.append((gen_idx, gold_idx))
            gen_idx += 1
            gold_idx += 1
            continue
        gen_idx += 1
    return matches


def _gold_path_for_song(
    index: int, song: BenchmarkSong, gold_root: Path
) -> Path | None:
    roots = [gold_root]
    clip_root = DEFAULT_CLIP_GOLD_ROOT.resolve()
    if song.clip_id and clip_root != gold_root.resolve():
        roots.append(clip_root)
    for root in roots:
        index_candidates = [index]
        if song.manifest_index not in index_candidates:
            index_candidates.append(song.manifest_index)
        explicit_candidates = [
            root / f"{candidate_index:02d}_{song.slug}.gold.json"
            for candidate_index in index_candidates
        ]
        explicit_candidates.append(root / f"{song.slug}.gold.json")
        slug_matches = sorted(root.glob(f"*_{song.slug}.gold.json"))
        candidates: list[Path] = []
        for path in explicit_candidates + slug_matches:
            if path not in candidates:
                candidates.append(path)
        for path in candidates:
            if path.exists():
                return path
    return None


def _default_gold_path_for_song(
    index: int, song: BenchmarkSong, gold_root: Path
) -> Path:
    return gold_root / f"{song.manifest_index:02d}_{song.slug}.gold.json"


def _resolve_gold_rebaseline_path(
    index: int, song: BenchmarkSong, gold_root: Path
) -> Path:
    existing = _gold_path_for_song(index=index, song=song, gold_root=gold_root)
    if existing is not None:
        return existing
    return _default_gold_path_for_song(index=index, song=song, gold_root=gold_root)


def _write_rebaseline_gold(target_path: Path, report_doc: dict[str, Any]) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(report_doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _build_rebaseline_gold_doc(
    *,
    song: BenchmarkSong,
    report_doc: dict[str, Any],
    prior_gold_doc: dict[str, Any] | None,
) -> dict[str, Any]:
    out = dict(report_doc)
    out["candidate_url"] = str(
        out.get("candidate_url")
        or (prior_gold_doc or {}).get("candidate_url")
        or song.youtube_url
    )
    audio_path = (
        out.get("audio_path")
        or (prior_gold_doc or {}).get("audio_path")
        or _resolve_song_audio_path(song, gold_doc=prior_gold_doc)
    )
    if isinstance(audio_path, str) and audio_path.strip():
        out["audio_path"] = audio_path
    return out


def _rebaseline_song_from_report(
    *,
    index: int,
    song: BenchmarkSong,
    report_path: Path,
    gold_root: Path,
) -> Path | None:
    if not report_path.exists():
        return None
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return None
    gold_path = _resolve_gold_rebaseline_path(
        index=index, song=song, gold_root=gold_root
    )
    prior_gold_doc = _load_gold_doc(index=index, song=song, gold_root=gold_root)
    _write_rebaseline_gold(
        gold_path,
        _build_rebaseline_gold_doc(
            song=song,
            report_doc=loaded,
            prior_gold_doc=prior_gold_doc,
        ),
    )
    return gold_path


def _load_gold_doc(
    index: int, song: BenchmarkSong, gold_root: Path
) -> dict[str, Any] | None:
    gold_path = _gold_path_for_song(index=index, song=song, gold_root=gold_root)
    if gold_path is None:
        return None
    try:
        loaded = json.loads(gold_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


def _build_generate_command(
    *,
    python_bin: str,
    song: BenchmarkSong,
    report_path: Path,
    cache_dir: Path | None,
    offline: bool,
    force: bool,
    whisper_map_lrc_dtw: bool,
    strategy: str = "hybrid_dtw",
    drop_lrc_line_timings: bool = False,
    evaluate_lyrics_sources: bool = False,
) -> list[str]:
    cmd = [
        python_bin,
        "-m",
        "y2karaoke.cli",
        "generate",
        song.youtube_url,
        "--title",
        song.title,
        "--artist",
        song.artist,
        "--no-render",
        "--timing-report",
        str(report_path),
    ]
    if cache_dir is not None:
        cmd.extend(["--work-dir", str(cache_dir)])
    if song.lyrics_file:
        cmd.extend(["--lyrics-file", song.lyrics_file])
    if song.audio_start_sec > 0.0:
        cmd.extend(["--audio-start", f"{song.audio_start_sec:g}"])
    if offline:
        cmd.append("--offline")
    if force:
        cmd.append("--force")
    if evaluate_lyrics_sources:
        cmd.append("--evaluate-lyrics")
    if drop_lrc_line_timings:
        cmd.append("--drop-lrc-line-timings")
    if strategy == "hybrid_dtw":
        if whisper_map_lrc_dtw:
            cmd.append("--whisper-map-lrc-dtw")
        else:
            cmd.append("--whisper")
    elif strategy == "hybrid_whisper":
        cmd.append("--whisper")
    elif strategy == "whisper_only":
        cmd.append("--whisper-only")
    elif strategy == "lrc_only":
        pass
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return cmd


def _inc_counter(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _agreement_line_word_count(line: dict[str, Any]) -> int:
    line_words = line.get("words")
    if isinstance(line_words, list) and line_words:
        return len(line_words)
    return len(_normalize_agreement_text(line.get("text")).split())


def _agreement_window_skip_reason(
    line: dict[str, Any], line_word_count: int
) -> str | None:
    window_word_count_raw = line.get("whisper_window_word_count")
    window_word_count = (
        int(window_word_count_raw)
        if isinstance(window_word_count_raw, (int, float))
        else 0
    )
    window_avg_prob = line.get("whisper_window_avg_prob")
    if line_word_count >= 6 and window_word_count < max(2, int(0.35 * line_word_count)):
        return "insufficient_window_words_for_long_line"
    if (
        line_word_count >= 5
        and isinstance(window_avg_prob, (int, float))
        and window_avg_prob < 0.45
        and window_word_count < max(2, int(0.5 * line_word_count))
    ):
        return "low_window_confidence_and_sparse_words"
    if (
        "whisper_window_word_count" in line
        and line_word_count >= 4
        and window_word_count < 2
    ):
        return "explicit_window_too_sparse"
    return None


def _agreement_anchor_outside_window(
    line: dict[str, Any], whisper_anchor_start: float
) -> bool:
    window_start_raw = line.get("whisper_window_start")
    window_end_raw = line.get("whisper_window_end")
    if not isinstance(window_start_raw, (int, float)) or not isinstance(
        window_end_raw, (int, float)
    ):
        return False
    window_start = float(window_start_raw) - 1.0
    window_end = float(window_end_raw) + 1.0
    return whisper_anchor_start < window_start or whisper_anchor_start > window_end


def _iter_agreement_window_tokens(
    line: dict[str, Any],
    *,
    normalize_fn: Callable[[Any], str] = _normalize_agreement_text,
) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    window_words = line.get("whisper_window_words")
    if not isinstance(window_words, list):
        return out
    for entry in window_words:
        if not isinstance(entry, dict):
            continue
        start_raw = entry.get("start")
        if not isinstance(start_raw, (int, float)):
            continue
        normalized = normalize_fn(entry.get("text"))
        if not normalized:
            continue
        start = float(start_raw)
        for token in normalized.split():
            out.append((token, start))
    return out


def _first_agreement_content_token(
    text: Any,
    *,
    normalize_fn: Callable[[Any], str] = _normalize_agreement_text,
) -> str | None:
    for token in normalize_fn(text).split():
        if len(token) >= 3:
            return token
    return None


def _match_agreement_window_token_sequence(
    window_tokens: list[tuple[str, float]],
    start_index: int,
    target_token: str,
) -> tuple[int, float] | None:
    if start_index >= len(window_tokens):
        return None
    token, token_start = window_tokens[start_index]
    if token == target_token:
        return start_index + 1, token_start
    if start_index + 1 >= len(window_tokens):
        return None
    combined = token + window_tokens[start_index + 1][0]
    if combined == target_token:
        return start_index + 2, token_start
    return None


def _agreement_window_tokens_similar(left: str, right: str) -> bool:
    if left == right:
        return True
    if len(left) >= 4 and len(right) >= 4:
        return left.startswith(right) or right.startswith(left)
    return False


def _select_agreement_window_sequence_anchor_start(
    line: dict[str, Any],
    *,
    anchor_start: float,
    normalize_fn: Callable[[Any], str] = _normalize_agreement_text,
) -> float:
    line_tokens = normalize_fn(line.get("text")).split()
    anchor_tokens = normalize_fn(line.get("nearest_segment_start_text")).split()
    window_tokens = _iter_agreement_window_tokens(line, normalize_fn=normalize_fn)
    if len(line_tokens) < 5 or len(window_tokens) < 4:
        return anchor_start

    anchor_text_similarity = _agreement_text_similarity(
        line.get("text"),
        line.get("nearest_segment_start_text"),
        normalize_fn=normalize_fn,
    )
    window_start_raw = line.get("whisper_window_start")
    window_starts_after_anchor = isinstance(window_start_raw, (int, float)) and (
        anchor_start + 0.35 < float(window_start_raw)
    )
    if anchor_text_similarity >= 0.9 and not window_starts_after_anchor:
        return anchor_start

    best_candidate: tuple[tuple[float, float, int, int], float] | None = None
    candidate_cases = [(0, 0)]
    if window_starts_after_anchor:
        candidate_cases.extend([(1, 0), (1, 1)])

    for drop_lead, drop_tail in candidate_cases:
        if drop_tail and (
            not line_tokens or line_tokens[-1] not in _AGREEMENT_TRAILING_FILLER_TOKENS
        ):
            continue
        end_index = len(line_tokens) - drop_tail if drop_tail else len(line_tokens)
        target_tokens = line_tokens[drop_lead:end_index]
        if len(target_tokens) < 4:
            continue
        for start_idx in range(len(window_tokens)):
            cursor = start_idx
            matched = 0
            matched_prob_sum = 0.0
            candidate_start: float | None = None
            for target_token in target_tokens:
                found = False
                while cursor < len(window_tokens):
                    window_token, token_start = window_tokens[cursor]
                    if _agreement_window_tokens_similar(target_token, window_token):
                        if candidate_start is None:
                            candidate_start = token_start
                        matched += 1
                        matched_prob_sum += 1.0
                        cursor += 1
                        found = True
                        break
                    cursor += 1
                if not found:
                    break
            if candidate_start is None:
                continue
            match_ratio = matched / len(target_tokens)
            if match_ratio < 0.85:
                continue
            score = (
                match_ratio,
                matched_prob_sum / max(matched, 1),
                drop_lead,
                drop_tail,
            )
            if best_candidate is None or score > best_candidate[0]:
                best_candidate = (score, candidate_start)

    if best_candidate is None:
        return anchor_start
    candidate_start = best_candidate[1]
    line_start_raw = line.get("start")
    if not isinstance(line_start_raw, (int, float)):
        return anchor_start
    line_start = float(line_start_raw)
    if abs(line_start - candidate_start) + 0.12 >= abs(line_start - anchor_start):
        return anchor_start
    return candidate_start


def _select_agreement_suffix_window_anchor_start(
    line: dict[str, Any],
    *,
    anchor_start: float,
    normalize_fn: Callable[[Any], str] = _normalize_agreement_text,
) -> float:
    line_tokens = normalize_fn(line.get("text")).split()
    window_tokens = _iter_agreement_window_tokens(line, normalize_fn=normalize_fn)
    if len(line_tokens) < 4 or len(window_tokens) < 3:
        return anchor_start
    if (
        _agreement_text_similarity(
            line.get("text"),
            line.get("nearest_segment_start_text"),
            normalize_fn=normalize_fn,
        )
        < 0.94
    ):
        return anchor_start
    first_window_tokens = {token for token, _ in window_tokens[:2]}
    candidate_start: float | None = None
    for drop_lead in (1, 2):
        if len(line_tokens) - drop_lead < 3:
            break
        if any(token in first_window_tokens for token in line_tokens[:drop_lead]):
            continue
        target_tokens = line_tokens[drop_lead:]
        for start_idx in range(len(window_tokens) - len(target_tokens) + 1):
            if all(
                _agreement_window_tokens_similar(
                    target_tokens[offset], window_tokens[start_idx + offset][0]
                )
                for offset in range(len(target_tokens))
            ):
                candidate_start = window_tokens[start_idx][1]
                break
        if candidate_start is not None:
            break
    if candidate_start is None:
        return anchor_start

    line_start_raw = line.get("start")
    if not isinstance(line_start_raw, (int, float)):
        return anchor_start
    line_start = float(line_start_raw)
    if abs(line_start - candidate_start) + 0.12 >= abs(line_start - anchor_start):
        return anchor_start
    return candidate_start


def _select_agreement_prefix_window_anchor_start(
    line: dict[str, Any],
    *,
    anchor_start: float,
    normalize_fn: Callable[[Any], str] = _normalize_agreement_text,
) -> float:
    line_tokens = normalize_fn(line.get("text")).split()
    anchor_tokens = normalize_fn(line.get("nearest_segment_start_text")).split()
    window_tokens = _iter_agreement_window_tokens(line, normalize_fn=normalize_fn)
    if len(line_tokens) < 4 or len(anchor_tokens) <= len(line_tokens):
        return anchor_start
    if anchor_tokens[: len(line_tokens)] != line_tokens:
        return anchor_start
    if line_tokens[0] in {token for token, _ in window_tokens[:2]}:
        return anchor_start

    target_tokens = line_tokens[1:]
    candidate_start: float | None = None
    for start_idx in range(len(window_tokens) - len(target_tokens) + 1):
        if all(
            _agreement_window_tokens_similar(
                target_tokens[offset], window_tokens[start_idx + offset][0]
            )
            for offset in range(len(target_tokens))
        ):
            candidate_start = window_tokens[start_idx][1]
            break
    if candidate_start is None:
        return anchor_start

    line_start_raw = line.get("start")
    if not isinstance(line_start_raw, (int, float)):
        return anchor_start
    line_start = float(line_start_raw)
    if abs(line_start - candidate_start) + 0.12 >= abs(line_start - anchor_start):
        return anchor_start
    return candidate_start


def _select_agreement_token_sequence_anchor_start(
    line: dict[str, Any],
    *,
    anchor_start: float,
    normalize_fn: Callable[[Any], str] = _normalize_agreement_text,
) -> float:
    line_tokens = normalize_fn(line.get("text")).split()
    window_tokens = _iter_agreement_window_tokens(line, normalize_fn=normalize_fn)
    if len(line_tokens) < 3 or len(window_tokens) < 2:
        return anchor_start

    text_similarity = _agreement_text_similarity(
        line.get("text"),
        line.get("nearest_segment_start_text"),
        normalize_fn=normalize_fn,
    )
    token_overlap = _agreement_token_overlap(
        line.get("text"),
        line.get("nearest_segment_start_text"),
        normalize_fn=normalize_fn,
    )
    if text_similarity < 0.6 or token_overlap < 0.6:
        return anchor_start

    first_window_tokens = {token for token, _ in window_tokens[:2]}
    candidate_start: float | None = None
    for drop_lead in (0, 1, 2):
        if len(line_tokens) - drop_lead < 2:
            break
        if drop_lead and any(
            token in first_window_tokens for token in line_tokens[:drop_lead]
        ):
            continue
        target_tokens = line_tokens[drop_lead:]
        for start_idx in range(len(window_tokens)):
            cursor = start_idx
            matched_all = True
            first_start: float | None = None
            for target_token in target_tokens:
                found = False
                while cursor < len(window_tokens):
                    matched = _match_agreement_window_token_sequence(
                        window_tokens,
                        cursor,
                        target_token,
                    )
                    if matched is None:
                        cursor += 1
                        continue
                    cursor, token_start = matched
                    if first_start is None:
                        first_start = token_start
                    found = True
                    break
                if not found:
                    matched_all = False
                    break
            if matched_all and first_start is not None:
                candidate_start = first_start
                break
        if candidate_start is not None:
            break

    if candidate_start is None:
        return anchor_start
    line_start_raw = line.get("start")
    if not isinstance(line_start_raw, (int, float)):
        return anchor_start
    line_start = float(line_start_raw)
    if abs(line_start - candidate_start) + 0.12 >= abs(line_start - anchor_start):
        return anchor_start
    return candidate_start


def _select_agreement_repeated_exact_window_anchor_start(
    line: dict[str, Any],
    *,
    anchor_start: float,
    target_text: Any | None = None,
    normalize_fn: Callable[[Any], str] = _normalize_agreement_text,
) -> float:
    if target_text is None:
        target_text = line.get("nearest_segment_start_text")
    line_tokens = normalize_fn(target_text).split()
    window_tokens = _iter_agreement_window_tokens(line, normalize_fn=normalize_fn)
    if len(line_tokens) < 3 or len(window_tokens) < len(line_tokens):
        return anchor_start

    line_start_raw = line.get("start")
    if not isinstance(line_start_raw, (int, float)):
        return anchor_start
    line_start = float(line_start_raw)

    candidates: list[float] = []
    for start_idx in range(len(window_tokens)):
        cursor = start_idx
        candidate_start: float | None = None
        matched_all = True
        for target_token in line_tokens:
            matched = _match_agreement_window_token_sequence(
                window_tokens,
                cursor,
                target_token,
            )
            if matched is None:
                matched_all = False
                break
            cursor, token_start = matched
            if candidate_start is None:
                candidate_start = token_start
        if matched_all and candidate_start is not None:
            candidates.append(candidate_start)

    if not candidates:
        return anchor_start

    best_candidate = min(
        candidates,
        key=lambda candidate: (
            abs(line_start - candidate),
            abs(anchor_start - candidate),
            candidate,
        ),
    )
    if abs(line_start - best_candidate) + 0.12 >= abs(line_start - anchor_start):
        return anchor_start
    return best_candidate


def _select_agreement_lead_in_anchor_start(
    line: dict[str, Any],
    *,
    anchor_start: float,
    normalize_fn: Callable[[Any], str] = _normalize_agreement_text,
) -> float:
    line_tokens = normalize_fn(line.get("text")).split()
    anchor_tokens = normalize_fn(line.get("nearest_segment_start_text")).split()
    if line_tokens != anchor_tokens or len(line_tokens) < 2:
        return anchor_start
    if line_tokens[0] not in _AGREEMENT_LEAD_IN_TOKENS:
        return anchor_start

    window_tokens = _iter_agreement_window_tokens(line, normalize_fn=normalize_fn)
    if len(window_tokens) < 2:
        return anchor_start

    target_index = 1
    while (
        target_index < len(line_tokens) - 1
        and line_tokens[target_index] in _AGREEMENT_LEAD_IN_TOKENS
    ):
        target_index += 1

    def _find_candidate_start(start_token_index: int) -> float | None:
        candidate: float | None = None
        for idx in range(len(window_tokens)):
            matched = _match_agreement_window_token_sequence(
                window_tokens, idx, line_tokens[start_token_index]
            )
            if matched is None:
                continue
            cursor = matched[0]
            candidate = matched[1]
            matched_all = True
            for target_token in line_tokens[start_token_index + 1 : target_index + 1]:
                found = False
                while cursor < len(window_tokens):
                    seq_match = _match_agreement_window_token_sequence(
                        window_tokens, cursor, target_token
                    )
                    if seq_match is None:
                        cursor += 1
                        continue
                    cursor, candidate = seq_match
                    found = True
                    break
                if not found:
                    matched_all = False
                    break
            if matched_all and candidate is not None:
                return candidate
        return None

    candidate_start = _find_candidate_start(0)

    line_start_raw = line.get("start")
    if not isinstance(line_start_raw, (int, float)):
        return anchor_start
    line_start = float(line_start_raw)
    chosen_start = anchor_start
    if candidate_start is not None:
        if 0.0 <= candidate_start - anchor_start <= 1.2 and abs(
            line_start - candidate_start
        ) + 0.18 < abs(line_start - anchor_start):
            chosen_start = candidate_start

    # If the segment anchor begins materially before the explicit Whisper window,
    # allow the benchmark anchor to start from the first matched meaningful token
    # after a dropped lead-in phrase.
    if anchor_start + 0.35 < window_tokens[0][1] and target_index >= 2:
        skip_index = 1
        while (
            skip_index < target_index
            and line_tokens[skip_index] in _AGREEMENT_LEAD_IN_TOKENS
        ):
            skip_index += 1
        skipped_candidate = _find_candidate_start(skip_index)
        if skipped_candidate is not None and (
            0.0 <= skipped_candidate - anchor_start <= 2.0
            and abs(line_start - skipped_candidate) + 0.12
            < abs(line_start - chosen_start)
        ):
            chosen_start = skipped_candidate

    return chosen_start


def _select_agreement_anchor_start(
    line: dict[str, Any],
    *,
    normalize_fn: Callable[[Any], str] = _normalize_agreement_text,
) -> float | None:
    anchor_start_raw = line.get("nearest_segment_start")
    if not isinstance(anchor_start_raw, (int, float)):
        return None
    anchor_start = float(anchor_start_raw)
    anchor_end_raw = line.get("nearest_segment_end")
    if not isinstance(anchor_end_raw, (int, float)):
        anchor_end = anchor_start
    else:
        anchor_end = float(anchor_end_raw)
    if anchor_end < anchor_start:
        anchor_end = anchor_start

    line_start_raw = line.get("start")
    if not isinstance(line_start_raw, (int, float)):
        return anchor_start
    line_start = float(line_start_raw)

    line_text = normalize_fn(line.get("text"))
    anchor_text = normalize_fn(line.get("nearest_segment_start_text"))
    if not line_text or not anchor_text or line_text not in anchor_text:
        if line_text != anchor_text:
            return _select_agreement_window_sequence_anchor_start(
                line,
                anchor_start=_select_agreement_repeated_exact_window_anchor_start(
                    line,
                    anchor_start=anchor_start,
                    target_text=line.get("nearest_segment_start_text"),
                    normalize_fn=normalize_fn,
                ),
                normalize_fn=normalize_fn,
            )
        word_count = len(line_text.split())
        if word_count < 1 or word_count > 4:
            return _select_agreement_token_sequence_anchor_start(
                line,
                anchor_start=_select_agreement_prefix_window_anchor_start(
                    line,
                    anchor_start=_select_agreement_suffix_window_anchor_start(
                        line,
                        anchor_start=_select_agreement_window_sequence_anchor_start(
                            line,
                            anchor_start=_select_agreement_lead_in_anchor_start(
                                line,
                                anchor_start=anchor_start,
                                normalize_fn=normalize_fn,
                            ),
                            normalize_fn=normalize_fn,
                        ),
                        normalize_fn=normalize_fn,
                    ),
                    normalize_fn=normalize_fn,
                ),
                normalize_fn=normalize_fn,
            )
        if abs(line_start - anchor_end) + 0.25 >= abs(line_start - anchor_start):
            return _select_agreement_token_sequence_anchor_start(
                line,
                anchor_start=_select_agreement_prefix_window_anchor_start(
                    line,
                    anchor_start=_select_agreement_suffix_window_anchor_start(
                        line,
                        anchor_start=_select_agreement_window_sequence_anchor_start(
                            line,
                            anchor_start=_select_agreement_lead_in_anchor_start(
                                line,
                                anchor_start=anchor_start,
                                normalize_fn=normalize_fn,
                            ),
                            normalize_fn=normalize_fn,
                        ),
                        normalize_fn=normalize_fn,
                    ),
                    normalize_fn=normalize_fn,
                ),
                normalize_fn=normalize_fn,
            )
        return _select_agreement_token_sequence_anchor_start(
            line,
            anchor_start=_select_agreement_prefix_window_anchor_start(
                line,
                anchor_start=_select_agreement_suffix_window_anchor_start(
                    line,
                    anchor_start=anchor_end,
                    normalize_fn=normalize_fn,
                ),
                normalize_fn=normalize_fn,
            ),
            normalize_fn=normalize_fn,
        )
    if line_text == anchor_text:
        word_count = len(line_text.split())
        anchor_start = _select_agreement_repeated_exact_window_anchor_start(
            line,
            anchor_start=anchor_start,
            target_text=line.get("nearest_segment_start_text"),
            normalize_fn=normalize_fn,
        )
        if word_count < 1 or word_count > 4:
            return _select_agreement_token_sequence_anchor_start(
                line,
                anchor_start=_select_agreement_prefix_window_anchor_start(
                    line,
                    anchor_start=_select_agreement_suffix_window_anchor_start(
                        line,
                        anchor_start=_select_agreement_window_sequence_anchor_start(
                            line,
                            anchor_start=_select_agreement_lead_in_anchor_start(
                                line,
                                anchor_start=anchor_start,
                                normalize_fn=normalize_fn,
                            ),
                            normalize_fn=normalize_fn,
                        ),
                        normalize_fn=normalize_fn,
                    ),
                    normalize_fn=normalize_fn,
                ),
                normalize_fn=normalize_fn,
            )
        if abs(line_start - anchor_end) + 0.25 >= abs(line_start - anchor_start):
            return _select_agreement_token_sequence_anchor_start(
                line,
                anchor_start=_select_agreement_prefix_window_anchor_start(
                    line,
                    anchor_start=_select_agreement_suffix_window_anchor_start(
                        line,
                        anchor_start=_select_agreement_window_sequence_anchor_start(
                            line,
                            anchor_start=_select_agreement_lead_in_anchor_start(
                                line,
                                anchor_start=anchor_start,
                                normalize_fn=normalize_fn,
                            ),
                            normalize_fn=normalize_fn,
                        ),
                        normalize_fn=normalize_fn,
                    ),
                    normalize_fn=normalize_fn,
                ),
                normalize_fn=normalize_fn,
            )
        return _select_agreement_token_sequence_anchor_start(
            line,
            anchor_start=_select_agreement_prefix_window_anchor_start(
                line,
                anchor_start=_select_agreement_suffix_window_anchor_start(
                    line,
                    anchor_start=anchor_end,
                    normalize_fn=normalize_fn,
                ),
                normalize_fn=normalize_fn,
            ),
            normalize_fn=normalize_fn,
        )
    if abs(line_start - anchor_end) >= abs(line_start - anchor_start):
        return _select_agreement_token_sequence_anchor_start(
            line,
            anchor_start=_select_agreement_prefix_window_anchor_start(
                line,
                anchor_start=_select_agreement_suffix_window_anchor_start(
                    line,
                    anchor_start=_select_agreement_window_sequence_anchor_start(
                        line,
                        anchor_start=anchor_start,
                        normalize_fn=normalize_fn,
                    ),
                    normalize_fn=normalize_fn,
                ),
                normalize_fn=normalize_fn,
            ),
            normalize_fn=normalize_fn,
        )
    return _select_agreement_token_sequence_anchor_start(
        line,
        anchor_start=_select_agreement_prefix_window_anchor_start(
            line,
            anchor_start=_select_agreement_suffix_window_anchor_start(
                line,
                anchor_start=anchor_end,
                normalize_fn=normalize_fn,
            ),
            normalize_fn=normalize_fn,
        ),
        normalize_fn=normalize_fn,
    )


def _evaluate_agreement_line(
    line: dict[str, Any],
    min_text_similarity: float,
    min_token_overlap: float,
    *,
    normalize_fn: Callable[[Any], str] = _normalize_agreement_text,
) -> dict[str, Any]:
    line_word_count = _agreement_line_word_count(line)
    skip_reason = _agreement_window_skip_reason(line, line_word_count)
    if skip_reason is not None:
        return {"skip_reason": skip_reason}

    line_start = line.get("start")
    if not isinstance(line_start, (int, float)):
        return {"skip_reason": "missing_line_start"}
    whisper_anchor_start = _select_agreement_anchor_start(
        line,
        normalize_fn=normalize_fn,
    )
    if whisper_anchor_start is None:
        return {"skip_reason": "missing_anchor_start"}
    if _agreement_anchor_outside_window(line, float(whisper_anchor_start)):
        return {"skip_reason": "anchor_outside_window"}

    line_text = line.get("text")
    whisper_anchor_text = line.get("nearest_segment_start_text")
    window_word_count_raw = line.get("whisper_window_word_count")
    window_word_count = (
        int(window_word_count_raw)
        if isinstance(window_word_count_raw, (int, float))
        else 0
    )
    window_avg_prob = line.get("whisper_window_avg_prob")
    anchor_start_delta = abs(float(line_start) - float(whisper_anchor_start))
    sim = _agreement_text_similarity(
        line_text, whisper_anchor_text, normalize_fn=normalize_fn
    )
    overlap = _agreement_token_overlap(
        line_text, whisper_anchor_text, normalize_fn=normalize_fn
    )
    first_content_token = _first_agreement_content_token(
        line_text, normalize_fn=normalize_fn
    )
    window_tokens = _iter_agreement_window_tokens(line, normalize_fn=normalize_fn)
    leading_window_tokens = [token for token, _ in window_tokens[:4]]
    missing_leading_start_evidence = (
        anchor_start_delta >= 0.8
        and bool(first_content_token)
        and bool(window_tokens)
        and all(
            not _agreement_window_tokens_similar(first_content_token, token)
            for token in leading_window_tokens
        )
        and overlap >= max(0.6, min_token_overlap + 0.1)
    )
    if missing_leading_start_evidence:
        return {"skip_reason": "missing_window_line_start", "eligible": True}
    has_strong_window_evidence = (
        window_word_count >= max(3, int(0.6 * line_word_count))
        and isinstance(window_avg_prob, (int, float))
        and float(window_avg_prob) >= 0.55
    )
    timing_rescue = (
        line_word_count >= 5
        and anchor_start_delta <= 0.22
        and overlap >= max(0.82, min_token_overlap + 0.25)
        and has_strong_window_evidence
    )
    short_line_rescue = (
        3 <= line_word_count <= 4
        and anchor_start_delta <= 0.12
        and overlap >= max(0.9, min_token_overlap + 0.35)
        and window_word_count >= line_word_count
        and isinstance(window_avg_prob, (int, float))
        and float(window_avg_prob) >= 0.65
    )
    high_overlap_tight_delta_rescue = (
        line_word_count >= 3
        and anchor_start_delta <= 0.14
        and overlap >= max(0.72, min_token_overlap + 0.22)
        and window_word_count >= max(2, int(0.5 * line_word_count))
        and isinstance(window_avg_prob, (int, float))
        and float(window_avg_prob) >= 0.55
    )
    weak_lexical_tight_timing_rescue = (
        line_word_count >= 4
        and anchor_start_delta <= 0.2
        and overlap >= max(0.15, min_token_overlap - 0.35)
        and window_word_count >= max(2, int(0.4 * line_word_count))
        and isinstance(window_avg_prob, (int, float))
        and float(window_avg_prob) >= 0.55
    )
    rescue_applies = (
        timing_rescue
        or short_line_rescue
        or high_overlap_tight_delta_rescue
        or weak_lexical_tight_timing_rescue
    )
    if sim < min_text_similarity and not rescue_applies:
        return {"skip_reason": "low_text_similarity", "eligible": True}
    if overlap < min_token_overlap and not rescue_applies:
        return {"skip_reason": "low_token_overlap", "eligible": True}
    return {
        "eligible": True,
        "anchor_start_delta": anchor_start_delta,
        "text_similarity": sim,
        "adaptive_rescue": bool(rescue_applies and sim < min_text_similarity),
    }


def _bucket_counts(
    deltas: list[float],
    agreement_good_start_sec: float,
    agreement_warn_start_sec: float,
) -> tuple[int, int, int, int]:
    good = sum(1 for v in deltas if v <= agreement_good_start_sec)
    warn = sum(
        1 for v in deltas if agreement_good_start_sec < v <= agreement_warn_start_sec
    )
    bad = sum(1 for v in deltas if v > agreement_warn_start_sec)
    severe = sum(1 for v in deltas if v > 1.5)
    return good, warn, bad, severe


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _timing_quality_band(score: float) -> str:
    if score >= 0.8:
        return "excellent"
    if score >= 0.65:
        return "good"
    if score >= 0.5:
        return "fair"
    return "poor"


def _compute_timing_quality_score(values: dict[str, Any]) -> tuple[float, str, str]:
    dtw_line_raw = values.get("dtw_line_coverage")
    dtw_word_raw = values.get("dtw_word_coverage")
    low_conf_raw = values.get("low_confidence_ratio")
    agree_cov_raw = values.get("agreement_coverage_ratio")
    agree_p95_raw = values.get("agreement_start_p95_abs_sec")
    agree_bad_raw = values.get("agreement_bad_ratio")
    anchor_p95_raw = values.get("whisper_anchor_start_p95_abs_sec")
    gold_cov_raw = values.get("gold_word_coverage_ratio")
    gold_start_raw = values.get("gold_start_mean_abs_sec")
    gold_comp_raw = values.get("gold_comparable_word_count")

    dtw_line = float(dtw_line_raw) if isinstance(dtw_line_raw, (int, float)) else 0.0
    dtw_word = float(dtw_word_raw) if isinstance(dtw_word_raw, (int, float)) else 0.0
    low_conf = float(low_conf_raw) if isinstance(low_conf_raw, (int, float)) else 0.0
    agree_cov = float(agree_cov_raw) if isinstance(agree_cov_raw, (int, float)) else 0.0
    agree_p95 = float(agree_p95_raw) if isinstance(agree_p95_raw, (int, float)) else 0.0
    agree_bad = float(agree_bad_raw) if isinstance(agree_bad_raw, (int, float)) else 0.0
    anchor_p95 = (
        float(anchor_p95_raw) if isinstance(anchor_p95_raw, (int, float)) else 0.0
    )
    gold_cov = float(gold_cov_raw) if isinstance(gold_cov_raw, (int, float)) else 0.0
    gold_start_mean = (
        float(gold_start_raw) if isinstance(gold_start_raw, (int, float)) else 0.0
    )
    gold_comparable_words = (
        int(gold_comp_raw) if isinstance(gold_comp_raw, (int, float)) else 0
    )

    agreement_score = (
        (0.45 * _clamp01(agree_cov / 0.5))
        + (0.35 * (1.0 - _clamp01(agree_p95 / 1.5)))
        + (0.2 * (1.0 - _clamp01(agree_bad / 0.25)))
    )
    anchor_score = 1.0 - _clamp01(anchor_p95 / 2.0)
    low_conf_score = 1.0 - _clamp01(low_conf / 0.25)

    if isinstance(dtw_line_raw, (int, float)) and isinstance(
        dtw_word_raw, (int, float)
    ):
        internal_score = (
            (0.42 * _clamp01(dtw_line))
            + (0.26 * _clamp01(dtw_word))
            + (0.22 * low_conf_score)
            + (0.1 * agreement_score)
        )
        score_mode = "dtw_internal"
    else:
        internal_score = (0.7 * anchor_score) + (0.3 * low_conf_score)
        score_mode = "anchor_fallback"

    if gold_comparable_words >= 20:
        gold_score = (0.55 * _clamp01(gold_cov)) + (
            0.45 * (1.0 - _clamp01(gold_start_mean / 1.25))
        )
        final_score = (0.78 * internal_score) + (0.22 * gold_score)
        score_mode = f"{score_mode}+gold"
    else:
        final_score = internal_score

    clamped = _clamp01(final_score)
    rounded = round(clamped, 4)
    return rounded, _timing_quality_band(clamped), score_mode


def _extract_song_metrics(
    report: dict[str, Any],
    gold_doc: dict[str, Any] | None = None,
    *,
    audio_path: str | None = None,
) -> dict[str, Any]:
    lines = report.get("lines", [])
    line_count = len(lines)
    low_conf = report.get("low_confidence_lines", [])
    alignment_method = str(report.get("alignment_method") or "")
    dtw_metrics = report.get("dtw_metrics", {})
    if not isinstance(dtw_metrics, dict):
        dtw_metrics = {}
    dtw_line_coverage = report.get("dtw_line_coverage")
    has_independent_anchor = isinstance(dtw_line_coverage, (int, float))

    # Keep agreement matching conservative, but allow mild lyric-video wording
    # variance so diagnostics have enough comparable anchor lines.
    agreement_min_text_similarity = _env_float(
        "Y2KARAOKE_BENCH_AGREEMENT_MIN_TEXT_SIM",
        0.58,
        min_value=0.0,
        max_value=1.0,
    )
    agreement_min_token_overlap = _env_float(
        "Y2KARAOKE_BENCH_AGREEMENT_MIN_TOKEN_OVERLAP",
        0.50,
        min_value=0.0,
        max_value=1.0,
    )
    agreement_good_start_sec = 0.35
    agreement_warn_start_sec = 0.8
    whisper_anchor_start_abs_deltas: list[float] = []
    pre_whisper_start_shift_abs = [
        abs(float(line.get("start", 0.0)) - float(line.get("pre_whisper_start")))
        for line in lines
        if isinstance(line.get("pre_whisper_start"), (int, float))
    ]
    pre_whisper_late_shift = [
        float(line.get("start", 0.0)) - float(line.get("pre_whisper_start"))
        for line in lines
        if isinstance(line.get("pre_whisper_start"), (int, float))
        and float(line.get("start", 0.0)) > float(line.get("pre_whisper_start"))
    ]
    agreement_text_sims: list[float] = []
    agreement_hook_boundary_text_sims: list[float] = []
    agreement_eligible_line_count = 0
    agreement_hook_boundary_eligible_line_count = 0
    agreement_adaptive_rescue_count = 0
    agreement_hook_boundary_adaptive_rescue_count = 0
    agreement_skip_reason_counts: dict[str, int] = {}
    agreement_hook_boundary_skip_reason_counts: dict[str, int] = {}

    def _inc_agreement_skip(reason: str) -> None:
        agreement_skip_reason_counts[reason] = (
            agreement_skip_reason_counts.get(reason, 0) + 1
        )

    def _inc_hook_boundary_skip(reason: str) -> None:
        agreement_hook_boundary_skip_reason_counts[reason] = (
            agreement_hook_boundary_skip_reason_counts.get(reason, 0) + 1
        )

    for line in lines:
        evaluation = _evaluate_agreement_line(
            line=line,
            min_text_similarity=agreement_min_text_similarity,
            min_token_overlap=agreement_min_token_overlap,
        )
        hook_boundary_evaluation = _evaluate_agreement_line(
            line=line,
            min_text_similarity=agreement_min_text_similarity,
            min_token_overlap=agreement_min_token_overlap,
            normalize_fn=_normalize_agreement_text_hook_boundary,
        )
        if bool(evaluation.get("eligible")):
            agreement_eligible_line_count += 1
        if bool(hook_boundary_evaluation.get("eligible")):
            agreement_hook_boundary_eligible_line_count += 1
        skip_reason = evaluation.get("skip_reason")
        if isinstance(skip_reason, str) and skip_reason:
            _inc_agreement_skip(skip_reason)
        else:
            anchor_start_delta = evaluation.get("anchor_start_delta")
            text_similarity = evaluation.get("text_similarity")
            if isinstance(anchor_start_delta, (int, float)):
                whisper_anchor_start_abs_deltas.append(float(anchor_start_delta))
            if isinstance(text_similarity, (int, float)):
                agreement_text_sims.append(float(text_similarity))
            if bool(evaluation.get("adaptive_rescue")):
                agreement_adaptive_rescue_count += 1

        hook_boundary_skip_reason = hook_boundary_evaluation.get("skip_reason")
        if isinstance(hook_boundary_skip_reason, str) and hook_boundary_skip_reason:
            _inc_hook_boundary_skip(hook_boundary_skip_reason)
        else:
            hook_boundary_text_similarity = hook_boundary_evaluation.get(
                "text_similarity"
            )
            if isinstance(hook_boundary_text_similarity, (int, float)):
                agreement_hook_boundary_text_sims.append(
                    float(hook_boundary_text_similarity)
                )
            if bool(hook_boundary_evaluation.get("adaptive_rescue")):
                agreement_hook_boundary_adaptive_rescue_count += 1

    # Independent agreement metric:
    # only available when we have DTW-based anchors (cross-strategy comparable path).
    agreement_start_abs_deltas = (
        whisper_anchor_start_abs_deltas if has_independent_anchor else []
    )

    (
        agreement_good_count,
        agreement_warn_count,
        agreement_bad_count,
        agreement_severe_count,
    ) = _bucket_counts(
        agreement_start_abs_deltas, agreement_good_start_sec, agreement_warn_start_sec
    )
    anchor_good_count, anchor_warn_count, anchor_bad_count, anchor_severe_count = (
        _bucket_counts(
            whisper_anchor_start_abs_deltas,
            agreement_good_start_sec,
            agreement_warn_start_sec,
        )
    )

    low_conf_ratio = (len(low_conf) / line_count) if line_count else 0.0
    metrics = {
        "alignment_method": alignment_method,
        "agreement_measurement_mode": (
            "independent_dtw_anchor"
            if has_independent_anchor
            else "unavailable_no_dtw_anchor"
        ),
        "line_count": line_count,
        "low_confidence_lines": len(low_conf),
        "low_confidence_ratio": round(low_conf_ratio, 4),
        "dtw_line_coverage": dtw_line_coverage,
        "dtw_word_coverage": report.get("dtw_word_coverage"),
        "dtw_phonetic_similarity_coverage": report.get(
            "dtw_phonetic_similarity_coverage"
        ),
        "fallback_map_attempted": int(
            float(dtw_metrics.get("fallback_map_attempted", 0.0) or 0.0) > 0.0
        ),
        "fallback_map_selected": int(
            float(dtw_metrics.get("fallback_map_selected", 0.0) or 0.0) > 0.0
        ),
        "fallback_map_rejected": int(
            float(dtw_metrics.get("fallback_map_rejected", 0.0) or 0.0) > 0.0
        ),
        "fallback_map_decision_code": (
            int(float(dtw_metrics.get("fallback_map_decision_code", 0.0) or 0.0))
            if isinstance(dtw_metrics.get("fallback_map_decision_code"), (int, float))
            else 0
        ),
        "fallback_map_score_gain": (
            round(float(dtw_metrics.get("fallback_map_score_gain", 0.0) or 0.0), 4)
            if isinstance(dtw_metrics.get("fallback_map_score_gain"), (int, float))
            else 0.0
        ),
        "tail_guardrail_flagged": int(
            float(dtw_metrics.get("tail_guardrail_flagged", 0.0) or 0.0) > 0.0
        ),
        "tail_guardrail_fallback_attempted": int(
            float(dtw_metrics.get("tail_guardrail_fallback_attempted", 0.0) or 0.0)
            > 0.0
        ),
        "tail_guardrail_fallback_applied": int(
            float(dtw_metrics.get("tail_guardrail_fallback_applied", 0.0) or 0.0) > 0.0
        ),
        "tail_guardrail_target_coverage_ratio": (
            round(
                float(
                    dtw_metrics.get("tail_guardrail_target_coverage_ratio", 0.0) or 0.0
                ),
                4,
            )
            if isinstance(
                dtw_metrics.get("tail_guardrail_target_coverage_ratio"),
                (int, float),
            )
            else 0.0
        ),
        "tail_guardrail_target_shortfall_sec": (
            round(
                float(
                    dtw_metrics.get("tail_guardrail_target_shortfall_sec", 0.0) or 0.0
                ),
                4,
            )
            if isinstance(
                dtw_metrics.get("tail_guardrail_target_shortfall_sec"), (int, float)
            )
            else 0.0
        ),
        "local_transcribe_cache_hits": (
            int(float(dtw_metrics.get("local_transcribe_cache_hits", 0.0) or 0.0))
            if isinstance(dtw_metrics.get("local_transcribe_cache_hits"), (int, float))
            else 0
        ),
        "local_transcribe_cache_misses": (
            int(float(dtw_metrics.get("local_transcribe_cache_misses", 0.0) or 0.0))
            if isinstance(
                dtw_metrics.get("local_transcribe_cache_misses"), (int, float)
            )
            else 0
        ),
        "agreement_min_text_similarity": agreement_min_text_similarity,
        "agreement_min_token_overlap": agreement_min_token_overlap,
        "agreement_adaptive_rescue_count": agreement_adaptive_rescue_count,
        "agreement_hook_boundary_adaptive_rescue_count": (
            agreement_hook_boundary_adaptive_rescue_count
        ),
        "agreement_eligible_lines": agreement_eligible_line_count,
        "agreement_hook_boundary_eligible_lines": (
            agreement_hook_boundary_eligible_line_count
        ),
        "agreement_matched_lines": len(whisper_anchor_start_abs_deltas),
        "agreement_eligibility_ratio": round(
            (agreement_eligible_line_count / line_count) if line_count else 0.0, 4
        ),
        "agreement_hook_boundary_eligibility_ratio": round(
            (
                agreement_hook_boundary_eligible_line_count / line_count
                if line_count
                else 0.0
            ),
            4,
        ),
        "agreement_match_ratio_within_eligible": round(
            (
                len(whisper_anchor_start_abs_deltas) / agreement_eligible_line_count
                if agreement_eligible_line_count
                else 0.0
            ),
            4,
        ),
        "agreement_hook_boundary_match_ratio_within_eligible": round(
            (
                len(whisper_anchor_start_abs_deltas)
                / agreement_hook_boundary_eligible_line_count
                if agreement_hook_boundary_eligible_line_count
                else 0.0
            ),
            4,
        ),
        "agreement_skip_reason_counts": dict(
            sorted(agreement_skip_reason_counts.items())
        ),
        "agreement_hook_boundary_skip_reason_counts": dict(
            sorted(agreement_hook_boundary_skip_reason_counts.items())
        ),
        "agreement_count": len(agreement_start_abs_deltas),
        "agreement_coverage_ratio": round(
            (len(agreement_start_abs_deltas) / line_count) if line_count else 0.0, 4
        ),
        "agreement_text_similarity_mean": round(_mean(agreement_text_sims) or 0.0, 4),
        "agreement_hook_boundary_text_similarity_mean": round(
            _mean(agreement_hook_boundary_text_sims) or 0.0, 4
        ),
        "agreement_start_mean_abs_sec": round(
            _mean(agreement_start_abs_deltas) or 0.0, 4
        ),
        "agreement_start_p95_abs_sec": round(
            _pctile(agreement_start_abs_deltas, 0.95), 4
        ),
        "agreement_start_max_abs_sec": round(
            max(agreement_start_abs_deltas, default=0.0), 4
        ),
        "agreement_good_lines": agreement_good_count,
        "agreement_warn_lines": agreement_warn_count,
        "agreement_bad_lines": agreement_bad_count,
        "agreement_severe_lines": agreement_severe_count,
        "agreement_good_ratio": round(
            (agreement_good_count / line_count) if line_count else 0.0, 4
        ),
        "agreement_warn_ratio": round(
            (agreement_warn_count / line_count) if line_count else 0.0, 4
        ),
        "agreement_bad_ratio": round(
            (agreement_bad_count / line_count) if line_count else 0.0, 4
        ),
        "agreement_severe_ratio": round(
            (agreement_severe_count / line_count) if line_count else 0.0, 4
        ),
        # Diagnostic-only metric (self-referential to Whisper anchors) for debugging drift.
        "whisper_anchor_count": len(whisper_anchor_start_abs_deltas),
        "whisper_anchor_coverage_ratio": round(
            (len(whisper_anchor_start_abs_deltas) / line_count) if line_count else 0.0,
            4,
        ),
        "whisper_anchor_start_mean_abs_sec": round(
            _mean(whisper_anchor_start_abs_deltas) or 0.0, 4
        ),
        "whisper_anchor_start_p95_abs_sec": round(
            _pctile(whisper_anchor_start_abs_deltas, 0.95), 4
        ),
        "whisper_anchor_start_max_abs_sec": round(
            max(whisper_anchor_start_abs_deltas, default=0.0), 4
        ),
        "whisper_anchor_good_ratio": round(
            (anchor_good_count / line_count) if line_count else 0.0, 4
        ),
        "whisper_anchor_warn_ratio": round(
            (anchor_warn_count / line_count) if line_count else 0.0, 4
        ),
        "whisper_anchor_bad_ratio": round(
            (anchor_bad_count / line_count) if line_count else 0.0, 4
        ),
        "whisper_anchor_severe_ratio": round(
            (anchor_severe_count / line_count) if line_count else 0.0, 4
        ),
        "pre_whisper_line_count": int(report.get("pre_whisper_line_count", 0) or 0),
        "pre_whisper_start_shift_mean_abs_sec": round(
            _mean(pre_whisper_start_shift_abs) or 0.0, 4
        ),
        "pre_whisper_late_shift_line_count": len(pre_whisper_late_shift),
        "pre_whisper_late_shift_mean_sec": round(
            _mean(pre_whisper_late_shift) or 0.0, 4
        ),
    }

    softened_gold_line_indexes = _softened_gold_adlib_line_indexes(report)
    generated_words = _flatten_words_from_timing_doc(
        report, suppress_line_indexes=softened_gold_line_indexes
    )
    gold_words_all_unsuppressed = _flatten_words_from_timing_doc(
        gold_doc or {}, mark_parenthetical_optional=True
    )
    gold_words_all = _flatten_words_from_timing_doc(
        gold_doc or {},
        mark_parenthetical_optional=True,
        suppress_line_indexes=softened_gold_line_indexes,
    )
    generated_interjection_lines = _extract_parenthetical_interjection_lines(
        report, suppress_line_indexes=softened_gold_line_indexes
    )
    gold_interjection_lines = _extract_parenthetical_interjection_lines(
        gold_doc or {}, suppress_line_indexes=softened_gold_line_indexes
    )
    aligned_interjection_lines = _align_parenthetical_interjection_lines(
        generated_interjection_lines,
        gold_interjection_lines,
    )
    generated_lines_for_gold = _extract_lines_for_gold_comparison(
        report, suppress_line_indexes=softened_gold_line_indexes
    )
    gold_lines_for_gold = _extract_lines_for_gold_comparison(
        gold_doc or {}, suppress_line_indexes=softened_gold_line_indexes
    )
    aligned_gold_lines = _align_lines_for_gold_comparison(
        generated_lines_for_gold,
        gold_lines_for_gold,
    )
    gold_words = [w for w in gold_words_all if not bool(w.get("optional"))]
    aligned_pairs = _align_words_for_gold_comparison(generated_words, gold_words)
    comparable = len(aligned_pairs)
    start_abs_deltas: list[float] = []
    end_abs_deltas: list[float] = []
    end_abs_deltas_strict: list[float] = []
    line_duration_abs_deltas: list[float] = []
    interjection_start_abs_deltas: list[float] = []
    gold_nearest_onset_start_abs_deltas: list[float] = []
    gold_nearest_onset_start_abs_deltas_non_interjection: list[float] = []
    gold_later_onset_choice_improvements: list[float] = []
    pre_whisper_start_abs_deltas_to_gold: list[float] = []
    downstream_gold_regression_improvements: list[float] = []
    text_matches = 0

    for gen_idx, gold_idx, sim in aligned_pairs:
        gen = generated_words[gen_idx]
        gold = gold_words[gold_idx]
        start_abs_deltas.append(abs(gen["start"] - gold["start"]))
        end_abs_delta = abs(gen["end"] - gold["end"])
        end_abs_deltas_strict.append(end_abs_delta)
        if not bool(gold.get("followed_by_optional_tail")):
            end_abs_deltas.append(end_abs_delta)
        if sim >= 0.999:
            text_matches += 1

    for gen_idx, gold_idx in aligned_interjection_lines:
        gen_line = generated_interjection_lines[gen_idx]
        gold_line = gold_interjection_lines[gold_idx]
        interjection_start_abs_deltas.append(
            abs(float(gen_line["start"]) - float(gold_line["start"]))
        )

    for gen_idx, gold_idx in aligned_gold_lines:
        gen_line = generated_lines_for_gold[gen_idx]
        gold_line = gold_lines_for_gold[gold_idx]
        line_duration_abs_deltas.append(
            abs(float(gen_line["duration"]) - float(gold_line["duration"]))
        )
        pre_whisper_start = lines[gen_idx].get("pre_whisper_start")
        if isinstance(pre_whisper_start, (int, float)):
            pre_whisper_abs_delta = abs(
                float(pre_whisper_start) - float(gold_line["start"])
            )
            final_abs_delta = abs(float(gen_line["start"]) - float(gold_line["start"]))
            pre_whisper_start_abs_deltas_to_gold.append(pre_whisper_abs_delta)
            # Count cases where downstream stages materially worsen a line that was
            # already closer to gold before Whisper/post-alignment drift.
            if final_abs_delta - pre_whisper_abs_delta >= 0.25:
                downstream_gold_regression_improvements.append(
                    final_abs_delta - pre_whisper_abs_delta
                )

    gold_later_onset_choice_improvements = _gold_later_onset_choice_rows(
        generated_lines=generated_lines_for_gold,
        gold_lines=gold_lines_for_gold,
        aligned_line_pairs=aligned_gold_lines,
        audio_path=audio_path,
    )

    for delta, is_interjection in _gold_line_nearest_onset_start_deltas(
        gold_doc=gold_doc,
        audio_path=audio_path,
    ):
        gold_nearest_onset_start_abs_deltas.append(delta)
        if not is_interjection:
            gold_nearest_onset_start_abs_deltas_non_interjection.append(delta)

    metrics.update(
        {
            "gold_available": bool(gold_words_all),
            "generated_word_count": len(generated_words),
            "gold_word_count": len(gold_words),
            "gold_optional_word_count": len(gold_words_all) - len(gold_words),
            "gold_comparable_word_count": comparable,
            "gold_alignment_mode": "monotonic_text_window_parenthetical_optional",
            "gold_word_coverage_ratio": round(
                (comparable / len(gold_words)) if gold_words else 0.0, 4
            ),
            "gold_word_text_match_ratio": round(
                (text_matches / comparable) if comparable else 0.0, 4
            ),
            "gold_trailing_parenthetical_softened_word_count": (
                sum(1 for w in gold_words if bool(w.get("followed_by_optional_tail")))
            ),
            "gold_parenthetical_interjection_line_count": len(gold_interjection_lines),
            "gold_parenthetical_interjection_comparable_line_count": len(
                aligned_interjection_lines
            ),
            "gold_softened_adlib_line_count": len(softened_gold_line_indexes),
            "gold_softened_adlib_word_count": sum(
                1
                for word in gold_words_all_unsuppressed
                if int(word.get("line_index", -1)) in softened_gold_line_indexes
            ),
            "gold_comparable_line_count": len(aligned_gold_lines),
            "gold_parenthetical_interjection_start_mean_abs_sec": round(
                _mean(interjection_start_abs_deltas) or 0.0, 4
            ),
            "gold_parenthetical_interjection_start_p95_abs_sec": round(
                _pctile(interjection_start_abs_deltas, 0.95), 4
            ),
            "gold_nearest_onset_start_mean_abs_sec": round(
                _mean(gold_nearest_onset_start_abs_deltas) or 0.0, 4
            ),
            "gold_nearest_onset_start_p95_abs_sec": round(
                _pctile(gold_nearest_onset_start_abs_deltas, 0.95), 4
            ),
            "gold_nearest_onset_start_non_interjection_mean_abs_sec": round(
                _mean(gold_nearest_onset_start_abs_deltas_non_interjection) or 0.0, 4
            ),
            "gold_later_onset_choice_line_count": len(
                gold_later_onset_choice_improvements
            ),
            "gold_later_onset_choice_mean_improvement_sec": round(
                _mean(gold_later_onset_choice_improvements) or 0.0,
                4,
            ),
            "gold_pre_whisper_start_mean_abs_sec": round(
                _mean(pre_whisper_start_abs_deltas_to_gold) or 0.0, 4
            ),
            "gold_downstream_regression_line_count": len(
                downstream_gold_regression_improvements
            ),
            "gold_downstream_regression_mean_improvement_sec": round(
                _mean(downstream_gold_regression_improvements) or 0.0, 4
            ),
            "gold_line_duration_mean_abs_sec": round(
                _mean(line_duration_abs_deltas) or 0.0, 4
            ),
            "gold_line_duration_p95_abs_sec": round(
                _pctile(line_duration_abs_deltas, 0.95), 4
            ),
            "avg_abs_word_start_delta_sec": round(_mean(start_abs_deltas) or 0.0, 4),
            "gold_start_mean_abs_sec": round(_mean(start_abs_deltas) or 0.0, 4),
            "gold_start_p95_abs_sec": round(_pctile(start_abs_deltas, 0.95), 4),
            "gold_start_max_abs_sec": round(max(start_abs_deltas, default=0.0), 4),
            "gold_end_mean_abs_sec": round(_mean(end_abs_deltas) or 0.0, 4),
            "gold_end_p95_abs_sec": round(_pctile(end_abs_deltas, 0.95), 4),
            "gold_end_max_abs_sec": round(max(end_abs_deltas, default=0.0), 4),
            "gold_end_mean_abs_sec_strict": round(
                _mean(end_abs_deltas_strict) or 0.0, 4
            ),
        }
    )

    timing_quality_score, timing_quality_band, timing_quality_score_mode = (
        _compute_timing_quality_score(metrics)
    )
    metrics["timing_quality_score"] = timing_quality_score
    metrics["timing_quality_band"] = timing_quality_band
    metrics["timing_quality_score_mode"] = timing_quality_score_mode

    return metrics


def _resolve_song_audio_path(
    song: BenchmarkSong, gold_doc: dict[str, Any] | None = None
) -> str | None:
    if isinstance(gold_doc, dict):
        gold_audio = gold_doc.get("audio_path")
        if isinstance(gold_audio, str) and gold_audio.strip():
            resolved_gold_audio = Path(gold_audio).expanduser()
            if resolved_gold_audio.exists():
                return str(resolved_gold_audio.resolve())
    for cache_root in _benchmark_cache_roots():
        cache_dir = cache_root / song.youtube_id
        wavs = sorted(cache_dir.glob("*.wav"))
        if wavs:
            return str(_prefer_primary_song_wav(wavs).resolve())
    return _resolve_cached_audio_path_by_slug(song.slug)


def _has_cached_primary_song_audio(song: BenchmarkSong) -> bool:
    for cache_root in _benchmark_cache_roots():
        cache_dir = cache_root / song.youtube_id
        wavs = sorted(cache_dir.glob("*.wav"))
        for wav in wavs:
            name = wav.name.lower()
            if name.startswith("trimmed_from_"):
                continue
            if any(
                key in name
                for key in ["vocals", "instrumental", "bass", "drums", "other"]
            ):
                continue
            return True
    return False


def _has_cached_benchmark_source(song: BenchmarkSong) -> bool:
    for cache_root in _benchmark_cache_roots():
        cache_dir = cache_root / song.youtube_id
        if (cache_dir / "metadata.json").exists() and _has_cached_primary_song_audio(
            song
        ):
            return True
    return False


def _prefer_primary_song_wav(wavs: list[Path]) -> Path:
    ranked = sorted(
        wavs,
        key=lambda path: (
            1 if "_(" in path.name else 0,
            1 if "instrumental" in path.stem.lower() else 0,
            len(path.name),
        ),
    )
    return ranked[0]


def _resolve_cached_audio_path_by_slug(slug: str) -> str | None:
    parts = [part for part in slug.split("-") if len(part) >= 3]
    if not parts:
        return None
    candidates: list[tuple[int, Path]] = []
    for cache_root in _benchmark_cache_roots():
        for wav in cache_root.glob("*/*.wav"):
            name = wav.stem.lower()
            score = sum(1 for part in parts if part in name)
            if score > 0:
                candidates.append((score, wav))
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            -item[0],
            1 if "_(" in item[1].name else 0,
            1 if "instrumental" in item[1].stem.lower() else 0,
            len(item[1].name),
        )
    )
    best_score, best_path = candidates[0]
    if best_score < max(2, min(len(parts), 3)):
        return None
    return str(best_path.resolve())


def _load_audio_features_cached(audio_path: str | None) -> Any | None:
    if not isinstance(audio_path, str) or not audio_path.strip():
        return None
    cached = _AUDIO_FEATURES_CACHE.get(audio_path)
    if cached is not None:
        return cached
    resolved = Path(audio_path).expanduser()
    if not resolved.exists():
        return None
    features = extract_audio_features(str(resolved))
    if features is None:
        return None
    _AUDIO_FEATURES_CACHE[audio_path] = features
    return features


def _nearest_onset_delta(
    onset_times: Any, *, target: float, window: float = 1.25
) -> float | None:
    if onset_times is None or len(onset_times) == 0:
        return None
    next_onset = first_onset_after(onset_times, start=target, window=window)
    prev_onset = None
    prev_idx = int(onset_times.searchsorted(target, side="left")) - 1
    if prev_idx >= 0:
        prev_candidate = float(onset_times[prev_idx])
        if abs(prev_candidate - target) <= window:
            prev_onset = prev_candidate
    candidates = [
        abs(float(onset) - target)
        for onset in (prev_onset, next_onset)
        if onset is not None and abs(float(onset) - target) <= window
    ]
    if not candidates:
        return None
    return min(candidates)


def _gold_line_nearest_onset_start_deltas(
    *, gold_doc: dict[str, Any] | None, audio_path: str | None
) -> list[tuple[float, bool]]:
    if not isinstance(gold_doc, dict):
        return []
    features = _load_audio_features_cached(audio_path)
    if features is None or getattr(features, "onset_times", None) is None:
        return []
    onset_times = features.onset_times
    rows: list[tuple[float, bool]] = []
    for line in gold_doc.get("lines", []):
        if not isinstance(line, dict):
            continue
        start = line.get("start")
        if not isinstance(start, (int, float)):
            continue
        delta = _nearest_onset_delta(onset_times, target=float(start))
        if delta is None:
            continue
        rows.append((delta, _line_is_parenthetical_interjection(line)))
    return rows


def _gold_later_onset_choice_rows(
    *,
    generated_lines: list[dict[str, Any]],
    gold_lines: list[dict[str, Any]],
    aligned_line_pairs: list[tuple[int, int]],
    audio_path: str | None,
) -> list[float]:
    features = _load_audio_features_cached(audio_path)
    if features is None or getattr(features, "onset_times", None) is None:
        return []
    onset_times = features.onset_times
    rows: list[float] = []
    for gen_idx, gold_idx in aligned_line_pairs:
        gen_line = generated_lines[gen_idx]
        gold_line = gold_lines[gold_idx]
        if _line_is_parenthetical_interjection(gold_line):
            continue
        gen_start = gen_line.get("start")
        gold_start = gold_line.get("start")
        if not isinstance(gen_start, (int, float)) or not isinstance(
            gold_start, (int, float)
        ):
            continue
        current_error = abs(float(gen_start) - float(gold_start))
        if current_error < 0.35:
            continue
        candidate_onsets = onset_times[
            (onset_times >= (float(gen_start) + 0.2))
            & (onset_times <= (float(gen_start) + 1.6))
        ]
        if len(candidate_onsets) == 0:
            continue
        candidate_errors = [
            abs(float(onset) - float(gold_start)) for onset in candidate_onsets
        ]
        best_error = min(candidate_errors, default=None)
        if best_error is None:
            continue
        improvement = current_error - best_error
        if improvement >= 0.25:
            rows.append(improvement)
    return rows


def _issue_tag(message: str) -> str:
    msg = message.lower()
    if "lyrics source disagreement triggered routing" in msg:
        return "lyrics_source_disagreement"
    if "tail completeness guardrail" in msg:
        return "tail_completeness_guardrail"
    if "low whisper confidence" in msg:
        return "low_whisper_confidence"
    if "timing delta" in msg and "clamp" in msg:
        return "timing_delta_clamped"
    if "duration mismatch" in msg:
        return "duration_mismatch"
    if "no lyrics" in msg:
        return "missing_lyrics"
    return "other"


def _fallback_map_reason_from_code(raw: Any) -> str:
    if not isinstance(raw, (int, float)):
        return "unknown"
    code = int(raw)
    if code == 0:
        return "skipped_quality_gate"
    if code == 1:
        return "selected_score_gain"
    if code == 2:
        return "rejected_insufficient_score_gain"
    if code == 3:
        return "forced_map_mode"
    if code == 4:
        return "selected_coverage_promotion"
    return "unknown"


def _extract_alignment_diagnostics(report: dict[str, Any]) -> dict[str, Any]:
    alignment_method = str(report.get("alignment_method") or "")
    lyrics_source = str(report.get("lyrics_source") or "")
    lyrics_source_provider = (
        lyrics_source.split("(", 1)[0].strip() if lyrics_source else ""
    )
    issues = report.get("issues", [])
    issue_tags: list[str] = []
    if isinstance(issues, list):
        for item in issues:
            if isinstance(item, str) and item.strip():
                issue_tags.append(_issue_tag(item))
    issue_tag_counts: dict[str, int] = {}
    for tag in issue_tags:
        issue_tag_counts[tag] = issue_tag_counts.get(tag, 0) + 1
    dtw_metrics = report.get("dtw_metrics", {})
    dtw_enabled = isinstance(dtw_metrics, dict) and bool(dtw_metrics)
    fallback_map_attempted = (
        bool(float(dtw_metrics.get("fallback_map_attempted", 0.0) or 0.0))
        if isinstance(dtw_metrics, dict)
        else False
    )
    fallback_map_selected = (
        bool(float(dtw_metrics.get("fallback_map_selected", 0.0) or 0.0))
        if isinstance(dtw_metrics, dict)
        else False
    )
    fallback_map_rejected = (
        bool(float(dtw_metrics.get("fallback_map_rejected", 0.0) or 0.0))
        if isinstance(dtw_metrics, dict)
        else False
    )
    fallback_map_decision_reason = (
        _fallback_map_reason_from_code(dtw_metrics.get("fallback_map_decision_code"))
        if isinstance(dtw_metrics, dict)
        else "unknown"
    )
    fallback_map_score_gain = (
        float(dtw_metrics.get("fallback_map_score_gain", 0.0) or 0.0)
        if isinstance(dtw_metrics, dict)
        and isinstance(dtw_metrics.get("fallback_map_score_gain"), (int, float))
        else 0.0
    )
    return {
        "alignment_method": alignment_method,
        "lyrics_source": lyrics_source,
        "lyrics_source_provider": lyrics_source_provider,
        "lyrics_source_audio_scoring_used": bool(
            report.get("lyrics_source_audio_scoring_used", False)
        ),
        "lyrics_source_disagreement_flagged": bool(
            report.get("lyrics_source_disagreement_flagged", False)
        ),
        "lyrics_source_disagreement_reasons": list(
            report.get("lyrics_source_disagreement_reasons", []) or []
        ),
        "lyrics_source_candidate_count": int(
            report.get("lyrics_source_candidate_count", 0) or 0
        ),
        "lyrics_source_comparable_candidate_count": int(
            report.get("lyrics_source_comparable_candidate_count", 0) or 0
        ),
        "lyrics_source_selection_mode": str(
            report.get("lyrics_source_selection_mode") or "default"
        ),
        "lyrics_source_routing_skip_reason": str(
            report.get("lyrics_source_routing_skip_reason") or "none"
        ),
        "whisper_requested": bool(report.get("whisper_requested", False)),
        "whisper_used": bool(report.get("whisper_used", False)),
        "whisper_force_dtw": bool(report.get("whisper_force_dtw", False)),
        "whisper_corrections": int(report.get("whisper_corrections", 0) or 0),
        "dtw_metrics_present": bool(dtw_enabled),
        "fallback_map_attempted": fallback_map_attempted,
        "fallback_map_selected": fallback_map_selected,
        "fallback_map_rejected": fallback_map_rejected,
        "fallback_map_decision_reason": fallback_map_decision_reason,
        "fallback_map_score_gain": round(fallback_map_score_gain, 4),
        "tail_guardrail_flagged": (
            bool(float(dtw_metrics.get("tail_guardrail_flagged", 0.0) or 0.0))
            if isinstance(dtw_metrics, dict)
            else False
        ),
        "tail_guardrail_fallback_attempted": (
            bool(
                float(dtw_metrics.get("tail_guardrail_fallback_attempted", 0.0) or 0.0)
            )
            if isinstance(dtw_metrics, dict)
            else False
        ),
        "tail_guardrail_fallback_applied": (
            bool(float(dtw_metrics.get("tail_guardrail_fallback_applied", 0.0) or 0.0))
            if isinstance(dtw_metrics, dict)
            else False
        ),
        "tail_guardrail_target_coverage_ratio": (
            round(
                float(
                    dtw_metrics.get("tail_guardrail_target_coverage_ratio", 0.0) or 0.0
                ),
                4,
            )
            if isinstance(dtw_metrics, dict)
            and isinstance(
                dtw_metrics.get("tail_guardrail_target_coverage_ratio"), (int, float)
            )
            else 0.0
        ),
        "tail_guardrail_target_shortfall_sec": (
            round(
                float(
                    dtw_metrics.get("tail_guardrail_target_shortfall_sec", 0.0) or 0.0
                ),
                4,
            )
            if isinstance(dtw_metrics, dict)
            and isinstance(
                dtw_metrics.get("tail_guardrail_target_shortfall_sec"), (int, float)
            )
            else 0.0
        ),
        "issue_count": len(issue_tags),
        "issue_tags": sorted(set(issue_tags)),
        "issue_tag_counts": issue_tag_counts,
    }


def _lexical_tokens_basic(text: str) -> list[str]:
    return _strip_optional_hook_boundary_tokens(_lexical_tokens_basic_raw(text))


def _lexical_tokens_compact(text: str) -> list[str]:
    return _strip_optional_hook_boundary_tokens(_lexical_tokens_compact_raw(text))


def _best_lexical_match_index(
    token: str, pool: list[str], used: list[bool], min_ratio: float = 0.86
) -> int | None:
    best_idx: int | None = None
    best_score = 0.0
    for idx, cand in enumerate(pool):
        if used[idx]:
            continue
        if token == cand:
            return idx
        score = SequenceMatcher(None, token, cand).ratio()
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx is None or best_score < min_ratio:
        return None
    return best_idx


def _max_contiguous_exact_run(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    best = 0
    for i in range(len(left)):
        for j in range(len(right)):
            run = 0
            while i + run < len(left) and j + run < len(right):
                if left[i + run] != right[j + run]:
                    break
                run += 1
            if run > best:
                best = run
    return best


def _compute_lexical_line_diagnostics(
    *,
    line: dict[str, Any],
    line_text: str,
    whisper_text: str,
) -> dict[str, Any] | None:
    line_basic_raw = _lexical_tokens_basic_raw(line_text)
    line_compact_raw = _lexical_tokens_compact_raw(line_text)
    wh_basic_raw = _lexical_tokens_basic_raw(whisper_text)
    wh_compact_raw = _lexical_tokens_compact_raw(whisper_text)
    line_basic = _strip_optional_hook_boundary_tokens(line_basic_raw)
    line_compact = _strip_optional_hook_boundary_tokens(line_compact_raw)
    wh_basic = _strip_optional_hook_boundary_tokens(wh_basic_raw)
    wh_compact = _strip_optional_hook_boundary_tokens(wh_compact_raw)
    if not line_basic or not wh_basic:
        return None

    repetitive_phrase = len(line_basic) >= 4 and len(set(line_compact)) <= max(
        2, len(line_compact) // 2
    )
    compact_line_set = set(line_compact)
    compact_wh_set = set(wh_compact)
    max_run = _max_contiguous_exact_run(line_compact, wh_compact)
    truncation_pattern = bool(
        max_run >= 3
        and compact_line_set
        and len(compact_line_set & compact_wh_set) < len(compact_line_set)
    )
    hook_boundary_variant = bool(
        (
            line_basic_raw != line_basic
            or wh_basic_raw != wh_basic
            or line_compact_raw != line_compact
            or wh_compact_raw != wh_compact
        )
        and max_run >= 3
    )

    used_basic = [False] * len(wh_basic)
    used_compact = [False] * len(wh_compact)
    rescued_tokens: list[str] = []
    apostrophe_tokens: list[str] = []
    unmatched_tokens: list[str] = []
    compact_rescue = 0
    apostrophe_rescue = 0

    for i, tok_basic in enumerate(line_basic):
        tok_compact = (
            line_compact[i] if i < len(line_compact) else tok_basic.replace("'", "")
        )
        idx_basic = _best_lexical_match_index(tok_basic, wh_basic, used_basic)
        if idx_basic is not None:
            used_basic[idx_basic] = True
            continue
        unmatched_tokens.append(tok_basic)
        idx_compact = _best_lexical_match_index(tok_compact, wh_compact, used_compact)
        if idx_compact is None:
            continue
        used_compact[idx_compact] = True
        rescued_tokens.append(tok_basic)
        compact_rescue += 1
        if "'" in tok_basic:
            apostrophe_tokens.append(tok_basic)
            apostrophe_rescue += 1

    sample = None
    if rescued_tokens:
        sample = {
            "line_index": int(line.get("index") or 0),
            "line_text": line_text,
            "whisper_window_excerpt": whisper_text[:180],
            "rescued_tokens": rescued_tokens[:8],
            "apostrophe_rescued_tokens": apostrophe_tokens[:8],
            "unmatched_tokens": unmatched_tokens[:8],
        }
    return {
        "line_token_count": len(line_basic),
        "compact_rescue": compact_rescue,
        "apostrophe_rescue": apostrophe_rescue,
        "hook_boundary_variant": hook_boundary_variant,
        "repetitive_phrase": repetitive_phrase,
        "truncation_pattern": truncation_pattern,
        "sample": sample,
    }


def _extract_lexical_mismatch_diagnostics(
    report: dict[str, Any],
    metrics: dict[str, Any],
    alignment_policy_hint: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    hint = ""
    if isinstance(alignment_policy_hint, dict):
        hint = str(alignment_policy_hint.get("hint") or "")
    if hint != "review_dtw_lexical_matching":
        return None

    lines = report.get("lines")
    if not isinstance(lines, list) or not lines:
        return None

    line_count_analyzed = 0
    total_line_tokens = 0
    compact_rescue = 0
    apostrophe_rescue = 0
    hook_boundary_variant_count = 0
    truncation_pattern_count = 0
    repetitive_phrase_line_count = 0
    samples: list[dict[str, Any]] = []

    for line in lines:
        if not isinstance(line, dict):
            continue
        line_text = str(line.get("text") or "")
        win_words = line.get("whisper_window_words")
        if not line_text or not isinstance(win_words, list):
            continue
        whisper_text = " ".join(
            str(w.get("text") or "")
            for w in win_words
            if isinstance(w, dict) and str(w.get("text") or "").strip()
        )
        if not whisper_text:
            continue
        line_count_analyzed += 1
        line_diag = _compute_lexical_line_diagnostics(
            line=line,
            line_text=line_text,
            whisper_text=whisper_text,
        )
        if line_diag is None:
            continue

        total_line_tokens += int(line_diag.get("line_token_count", 0) or 0)
        compact_rescue += int(line_diag.get("compact_rescue", 0) or 0)
        apostrophe_rescue += int(line_diag.get("apostrophe_rescue", 0) or 0)
        if bool(line_diag.get("hook_boundary_variant")):
            hook_boundary_variant_count += 1
        if bool(line_diag.get("repetitive_phrase")):
            repetitive_phrase_line_count += 1
        if bool(line_diag.get("truncation_pattern")):
            truncation_pattern_count += 1
        sample = line_diag.get("sample")
        if isinstance(sample, dict) and len(samples) < 8:
            samples.append(sample)

    if total_line_tokens <= 0:
        return None

    return {
        "active": True,
        "line_count_analyzed": line_count_analyzed,
        "dtw_line_coverage": float(metrics.get("dtw_line_coverage") or 0.0),
        "dtw_word_coverage": float(metrics.get("dtw_word_coverage") or 0.0),
        "compact_rescue_token_count": int(compact_rescue),
        "apostrophe_rescue_token_count": int(apostrophe_rescue),
        "compact_rescue_ratio": round(compact_rescue / total_line_tokens, 4),
        "apostrophe_rescue_ratio": round(apostrophe_rescue / total_line_tokens, 4),
        "hook_boundary_variant_count": int(hook_boundary_variant_count),
        "hook_boundary_variant_ratio": (
            round(hook_boundary_variant_count / line_count_analyzed, 4)
            if line_count_analyzed
            else 0.0
        ),
        "truncation_pattern_count": int(truncation_pattern_count),
        "truncation_pattern_ratio": (
            round(truncation_pattern_count / line_count_analyzed, 4)
            if line_count_analyzed
            else 0.0
        ),
        "repetitive_phrase_line_count": int(repetitive_phrase_line_count),
        "repetitive_phrase_line_ratio": (
            round(repetitive_phrase_line_count / line_count_analyzed, 4)
            if line_count_analyzed
            else 0.0
        ),
        "samples": samples,
    }


def _infer_reference_divergence_suspicion(
    metrics: dict[str, Any],
    alignment_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {"suspected": False, "score": 0.0, "confidence": "low", "evidence": []}

    def _f(key: str) -> float:
        v = metrics.get(key)
        return float(v) if isinstance(v, (int, float)) else 0.0

    line_count = int(metrics.get("line_count", 0) or 0)
    dtw_line_cov = _f("dtw_line_coverage")
    dtw_word_cov = _f("dtw_word_coverage")
    agree_cov = _f("agreement_coverage_ratio")
    agree_sim = _f("agreement_text_similarity_mean")
    agree_p95 = _f("agreement_start_p95_abs_sec")
    low_conf = _f("low_confidence_ratio")
    source_disagreement_flagged = bool(
        (alignment_diagnostics or {}).get("lyrics_source_disagreement_flagged", False)
    )

    if not bool(metrics.get("gold_available")):
        return _infer_reference_divergence_no_gold(
            line_count=line_count,
            dtw_line_cov=dtw_line_cov,
            dtw_word_cov=dtw_word_cov,
            agree_cov=agree_cov,
            agree_sim=agree_sim,
            agree_p95=agree_p95,
            low_conf=low_conf,
        )
    score = 0.0
    evidence: list[str] = []

    gold_cov = _f("gold_word_coverage_ratio")
    gold_start_mean = _f("gold_start_mean_abs_sec")
    gold_start_p95 = _f("gold_start_p95_abs_sec")
    agree_bad = _f("agreement_bad_ratio")
    comparable_words = int(metrics.get("gold_comparable_word_count", 0) or 0)

    if comparable_words < 20 or line_count < 10:
        return _infer_reference_divergence_insufficient(
            gold_cov=gold_cov,
            gold_start_mean=gold_start_mean,
            gold_start_p95=gold_start_p95,
            dtw_line_cov=dtw_line_cov,
            agree_cov=agree_cov,
            agree_sim=agree_sim,
            low_conf=low_conf,
            comparable_words=comparable_words,
        )

    strong_dtw_internal = (
        dtw_line_cov >= 0.9 and dtw_word_cov >= 0.4 and low_conf <= 0.12
    )
    strong_agreement_subset = agree_cov >= 0.5 and agree_sim >= 0.9 and low_conf <= 0.08
    if strong_dtw_internal:
        score += 1.0
        evidence.append("strong_dtw_internal_consistency")
    if strong_agreement_subset:
        score += 1.0
        evidence.append("strong_lrc_whisper_agreement_subset")
    if low_conf <= 0.1:
        score += 0.25
        evidence.append("low_internal_uncertainty")

    severe_gold_timing_mismatch = gold_start_mean >= 20.0 or gold_start_p95 >= 35.0
    gold_coverage_timing_combo = gold_cov <= 0.68 and gold_start_mean >= 10.0
    high_gold_mismatch_with_strong_dtw = (
        dtw_line_cov >= 0.88
        and dtw_word_cov >= 0.55
        and low_conf <= 0.12
        and agree_cov <= 0.12
        and gold_start_mean >= 10.0
        and gold_start_p95 >= 18.0
    )
    if severe_gold_timing_mismatch:
        score += 1.5
        evidence.append("severe_gold_timing_mismatch")
    elif gold_start_mean >= 10.0 and gold_start_p95 >= 20.0:
        score += 0.5
        evidence.append("elevated_gold_timing_mismatch")
    if high_gold_mismatch_with_strong_dtw:
        score += 1.25
        evidence.append("high_gold_mismatch_with_strong_dtw")
    if source_disagreement_flagged and high_gold_mismatch_with_strong_dtw:
        score += 0.5
        evidence.append("multi_source_disagreement_supports_reference_mismatch")
    if gold_coverage_timing_combo:
        score += 1.0
        evidence.append("gold_coverage_timing_combo_mismatch")

    # If internal signals are weak, treat this as likely pipeline quality rather than reference mismatch.
    if not (strong_dtw_internal or strong_agreement_subset):
        score -= 1.5
        evidence.append("insufficient_internal_consistency")
    if dtw_line_cov < 0.75 or low_conf > 0.2:
        score -= 0.5
        evidence.append("weak_internal_alignment_signals")
    if agree_cov > 0 and agree_bad > 0.2:
        score -= 0.5
        evidence.append("high_agreement_bad_ratio")

    suspected = bool(score >= 2.5)
    confidence = "high" if score >= 4.0 else "medium" if score >= 2.5 else "low"
    return {
        "suspected": suspected,
        "score": round(score, 3),
        "confidence": confidence,
        "evidence": sorted(set(evidence)),
        "signals": {
            "gold_word_coverage_ratio": round(gold_cov, 4),
            "gold_start_mean_abs_sec": round(gold_start_mean, 4),
            "gold_start_p95_abs_sec": round(gold_start_p95, 4),
            "dtw_line_coverage": round(dtw_line_cov, 4),
            "dtw_word_coverage": round(dtw_word_cov, 4),
            "agreement_coverage_ratio": round(agree_cov, 4),
            "agreement_text_similarity_mean": round(agree_sim, 4),
            "agreement_bad_ratio": round(agree_bad, 4),
            "low_confidence_ratio": round(low_conf, 4),
            "gold_comparable_word_count": comparable_words,
            "lyrics_source_disagreement_flagged": source_disagreement_flagged,
        },
    }


def _infer_reference_divergence_no_gold(
    *,
    line_count: int,
    dtw_line_cov: float,
    dtw_word_cov: float,
    agree_cov: float,
    agree_sim: float,
    agree_p95: float,
    low_conf: float,
) -> dict[str, Any]:
    no_gold_suspected = (
        line_count >= 40
        and dtw_line_cov <= 0.6
        and dtw_word_cov <= 0.45
        and agree_cov >= 0.07
        and agree_sim >= 0.9
        and agree_p95 <= 1.2
        and low_conf <= 0.08
    )
    if not no_gold_suspected:
        return {
            "suspected": False,
            "score": 0.0,
            "confidence": "low",
            "evidence": ["no_gold_reference"],
        }
    return {
        "suspected": True,
        "score": 1.75,
        "confidence": "medium",
        "evidence": [
            "no_gold_reference",
            "low_dtw_with_strong_anchor_agreement",
        ],
        "signals": {
            "dtw_line_coverage": round(dtw_line_cov, 4),
            "dtw_word_coverage": round(dtw_word_cov, 4),
            "agreement_coverage_ratio": round(agree_cov, 4),
            "agreement_text_similarity_mean": round(agree_sim, 4),
            "agreement_start_p95_abs_sec": round(agree_p95, 4),
            "low_confidence_ratio": round(low_conf, 4),
            "line_count": line_count,
        },
    }


def _infer_reference_divergence_insufficient(
    *,
    gold_cov: float,
    gold_start_mean: float,
    gold_start_p95: float,
    dtw_line_cov: float,
    agree_cov: float,
    agree_sim: float,
    low_conf: float,
    comparable_words: int,
) -> dict[str, Any]:
    return {
        "suspected": False,
        "score": 0.0,
        "confidence": "low",
        "evidence": ["insufficient_comparable_coverage"],
        "signals": {
            "gold_word_coverage_ratio": round(gold_cov, 4),
            "gold_start_mean_abs_sec": round(gold_start_mean, 4),
            "gold_start_p95_abs_sec": round(gold_start_p95, 4),
            "dtw_line_coverage": round(dtw_line_cov, 4),
            "agreement_coverage_ratio": round(agree_cov, 4),
            "agreement_text_similarity_mean": round(agree_sim, 4),
            "low_confidence_ratio": round(low_conf, 4),
            "gold_comparable_word_count": comparable_words,
        },
    }


def _infer_alignment_policy_hint(
    metrics: dict[str, Any],
    alignment_diagnostics: dict[str, Any] | None = None,
    reference_divergence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {"hint": "none", "confidence": "low", "reasons": []}
    if isinstance(reference_divergence, dict) and bool(
        reference_divergence.get("suspected")
    ):
        return {
            "hint": "reference_mismatch_review",
            "confidence": str(reference_divergence.get("confidence", "medium")),
            "reasons": ["reference_divergence_suspected"],
        }

    def _f(key: str) -> float:
        v = metrics.get(key)
        return float(v) if isinstance(v, (int, float)) else 0.0

    dtw_line = _f("dtw_line_coverage")
    dtw_word = _f("dtw_word_coverage")
    agree_cov = _f("agreement_coverage_ratio")
    agree_p95 = _f("agreement_start_p95_abs_sec")
    agree_bad = _f("agreement_bad_ratio")
    low_conf = _f("low_confidence_ratio")
    gold_cov = _f("gold_word_coverage_ratio")
    gold_start_mean = _f("gold_start_mean_abs_sec")

    reasons: list[str] = []
    hint = "none"
    confidence = "low"

    issue_tags: set[str] = set()
    if isinstance(alignment_diagnostics, dict):
        issue_tags = set(
            str(v)
            for v in (alignment_diagnostics.get("issue_tags") or [])
            if isinstance(v, str)
        )

    # Candidate for ignoring provider LRC line timings and deriving timing from audio+lyrics.
    # Keep this conservative: it should only trigger when we have stronger evidence that
    # timing disagreement is real (not just sparse/noisy agreement matches).
    gold_timing_mismatch_evidence = (
        gold_cov >= 0.6
        and gold_start_mean >= 1.0
        and agree_cov >= 0.3
        and agree_p95 >= 1.0
        and (agree_bad >= 0.1 or "timing_delta_clamped" in issue_tags)
        and dtw_line >= 0.65
        and low_conf <= 0.15
    )
    internal_timing_mismatch_evidence = (
        "timing_delta_clamped" in issue_tags
        and agree_cov >= 0.45
        and agree_p95 >= 2.0
        and agree_bad >= 0.2
        and dtw_line >= 0.75
        and dtw_word >= 0.6
        and low_conf <= 0.15
    )
    already_strong_hybrid_alignment = (
        dtw_line >= 0.95
        and dtw_word >= 0.9
        and agree_cov >= 0.5
        and agree_p95 <= 2.0
        and low_conf <= 0.05
    )
    if gold_timing_mismatch_evidence or internal_timing_mismatch_evidence:
        if already_strong_hybrid_alignment:
            hint = "timing_delta_clamped_review"
            confidence = "medium"
            reasons.extend(
                [
                    "timing_delta_clamped",
                    "hybrid_alignment_already_strong",
                ]
            )
        else:
            hint = "consider_lyrics_no_timing"
            confidence = "high" if gold_timing_mismatch_evidence else "medium"
            if gold_timing_mismatch_evidence:
                reasons.extend(
                    [
                        "gold_timing_mismatch_with_good_coverage",
                        "dtw_line_coverage_present",
                    ]
                )
            if internal_timing_mismatch_evidence:
                reasons.extend(
                    [
                        "agreement_start_p95_high",
                        "agreement_bad_ratio_high",
                        "agreement_coverage_present",
                        "dtw_line_coverage_present",
                    ]
                )

    # Candidate for Whisper-heavy / audio-first review: weak DTW lexical coverage but low internal uncertainty.
    if (
        hint == "none"
        and dtw_line >= 0.7
        and dtw_word < 0.65
        and low_conf <= 0.1
        and gold_cov >= 0.75
        and agree_cov >= 0.4
        and agree_p95 <= 0.9
    ):
        hint = "review_dtw_lexical_matching"
        confidence = "medium"
        reasons.extend(
            [
                "dtw_alignment_weaker_than_gold_agreement",
                "low_internal_uncertainty",
            ]
        )

    if isinstance(alignment_diagnostics, dict):
        if "timing_delta_clamped" in issue_tags and hint == "none":
            hint = "timing_delta_clamped_review"
            confidence = "medium"
            reasons.append("timing_delta_clamped")
        elif "timing_delta_clamped" in issue_tags and hint != "none":
            reasons.append("timing_delta_clamped")

    return {
        "hint": hint,
        "confidence": confidence,
        "reasons": sorted(set(reasons)),
    }


def _classify_quality_diagnosis(
    metrics: dict[str, Any],
    *,
    alignment_policy_hint: dict[str, Any] | None = None,
    reference_divergence: dict[str, Any] | None = None,
    lexical_mismatch_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {
            "verdict": "insufficient_evidence",
            "confidence": "low",
            "reasons": ["metrics_unavailable"],
            "signals": {},
        }

    def _f(key: str) -> float:
        value = metrics.get(key)
        return float(value) if isinstance(value, (int, float)) else 0.0

    line_count = int(metrics.get("line_count", 0) or 0)
    dtw_line = _f("dtw_line_coverage")
    dtw_word = _f("dtw_word_coverage")
    low_conf = _f("low_confidence_ratio")
    agree_cov = _f("agreement_coverage_ratio")
    agree_p95 = _f("agreement_start_p95_abs_sec")
    gold_cov = _f("gold_word_coverage_ratio")
    gold_start_mean = _f("gold_start_mean_abs_sec")
    gold_start_p95 = _f("gold_start_p95_abs_sec")
    pre_whisper_start_mean = _f("gold_pre_whisper_start_mean_abs_sec")
    downstream_regression_lines = int(
        metrics.get("gold_downstream_regression_line_count", 0) or 0
    )
    downstream_regression_mean = _f("gold_downstream_regression_mean_improvement_sec")
    gold_comparable_words = int(metrics.get("gold_comparable_word_count", 0) or 0)
    has_gold = bool(metrics.get("gold_available"))
    has_dtw = isinstance(metrics.get("dtw_line_coverage"), (int, float))
    reasons: list[str] = []
    verdict = "needs_pipeline_work"
    confidence = "low"

    if isinstance(reference_divergence, dict) and bool(
        reference_divergence.get("suspected")
    ):
        evidence = reference_divergence.get("evidence", [])
        evidence_reasons = (
            [str(v) for v in evidence if isinstance(v, str)]
            if isinstance(evidence, list)
            else []
        )
        reasons.extend(evidence_reasons or ["reference_divergence_suspected"])
        return {
            "verdict": "likely_reference_divergence",
            "confidence": str(reference_divergence.get("confidence", "medium")),
            "reasons": sorted(set(reasons)),
            "signals": {
                "line_count": line_count,
                "dtw_line_coverage": round(dtw_line, 4),
                "dtw_word_coverage": round(dtw_word, 4),
                "gold_word_coverage_ratio": round(gold_cov, 4),
                "gold_start_mean_abs_sec": round(gold_start_mean, 4),
                "low_confidence_ratio": round(low_conf, 4),
            },
        }

    if line_count < 8 or (not has_dtw and not has_gold):
        reasons.append("insufficient_coverage")
        return {
            "verdict": "insufficient_evidence",
            "confidence": "low",
            "reasons": reasons,
            "signals": {
                "line_count": line_count,
                "dtw_available": has_dtw,
                "gold_available": has_gold,
            },
        }

    strong_internal = dtw_line >= 0.85 and dtw_word >= 0.55 and low_conf <= 0.12
    stable_agreement = agree_cov <= 0.0 or agree_p95 <= 0.9
    hook_agree_eligibility = _f("agreement_hook_boundary_eligibility_ratio")
    hook_agree_text_similarity = _f("agreement_hook_boundary_text_similarity_mean")
    agree_text_similarity = _f("agreement_text_similarity_mean")
    strong_gold = (
        has_gold
        and gold_comparable_words >= 40
        and gold_cov >= 0.8
        and gold_start_mean <= 0.65
        and gold_start_p95 <= 1.9
    )
    strong_gold_agreement = (
        strong_gold and agree_cov >= 0.4 and agree_p95 <= 0.9 and low_conf <= 0.1
    )
    downstream_regression_dominant = (
        has_gold
        and downstream_regression_lines >= max(6, int(line_count * 0.1))
        and downstream_regression_mean >= 0.6
        and pre_whisper_start_mean + 0.15 < gold_start_mean
    )
    hook_boundary_ratio = 0.0
    if isinstance(lexical_mismatch_diagnostics, dict):
        hook_boundary_ratio = float(
            lexical_mismatch_diagnostics.get("hook_boundary_variant_ratio", 0.0) or 0.0
        )
    hint = ""
    if isinstance(alignment_policy_hint, dict):
        hint = str(alignment_policy_hint.get("hint") or "")

    if strong_internal and stable_agreement and (not has_gold or strong_gold):
        verdict = "likely_pipeline_ok"
        confidence = "high" if strong_gold else "medium"
        reasons.extend(["strong_internal_alignment"])
        if has_gold:
            reasons.append("gold_timing_consistent")
    elif hint == "review_dtw_lexical_matching" and strong_gold_agreement:
        verdict = "needs_manual_review"
        confidence = "medium"
        reasons.extend(
            [
                "review_dtw_lexical_matching",
                "gold_timing_consistent",
                "agreement_consistent",
            ]
        )
        if hook_boundary_ratio >= 0.25:
            reasons.append("hook_boundary_variants_dominant")
        if hook_agree_eligibility >= max(
            0.85, agree_cov + 0.2
        ) and hook_agree_text_similarity >= max(0.9, agree_text_similarity):
            reasons.append("hook_normalized_agreement_consistent")
    elif downstream_regression_dominant:
        verdict = "needs_pipeline_work"
        confidence = "high"
        reasons.extend(
            [
                "downstream_regression_dominant",
                "pre_whisper_timing_stronger_than_final",
            ]
        )
    else:
        severe_pipeline_signals = (
            dtw_line < 0.75
            or dtw_word < 0.45
            or low_conf > 0.2
            or (agree_cov >= 0.35 and agree_p95 > 1.2 and not strong_gold)
        )
        if severe_pipeline_signals:
            verdict = "needs_pipeline_work"
            confidence = "high"
            if dtw_line < 0.75:
                reasons.append("low_dtw_line_coverage")
            if dtw_word < 0.45:
                reasons.append("low_dtw_word_coverage")
            if low_conf > 0.2:
                reasons.append("high_low_confidence_ratio")
            if agree_cov >= 0.35 and agree_p95 > 1.2 and not strong_gold:
                reasons.append("high_agreement_p95")
        else:
            verdict = "needs_manual_review"
            confidence = "medium"
            reasons.append("mixed_signals")

    if isinstance(alignment_policy_hint, dict):
        hint = str(alignment_policy_hint.get("hint") or "")
        if hint and hint != "none":
            reasons.append(f"policy_hint:{hint}")

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasons": sorted(set(reasons)),
        "signals": {
            "line_count": line_count,
            "dtw_line_coverage": round(dtw_line, 4),
            "dtw_word_coverage": round(dtw_word, 4),
            "low_confidence_ratio": round(low_conf, 4),
            "agreement_coverage_ratio": round(agree_cov, 4),
            "agreement_start_p95_abs_sec": round(agree_p95, 4),
            "gold_word_coverage_ratio": round(gold_cov, 4),
            "gold_start_mean_abs_sec": round(gold_start_mean, 4),
            "gold_pre_whisper_start_mean_abs_sec": round(pre_whisper_start_mean, 4),
            "gold_downstream_regression_line_count": downstream_regression_lines,
            "gold_downstream_regression_mean_improvement_sec": round(
                downstream_regression_mean, 4
            ),
            "gold_comparable_word_count": gold_comparable_words,
        },
    }


def _build_triage_rankings(
    succeeded: list[dict[str, Any]], *, top_n: int = 5
) -> dict[str, list[dict[str, Any]]]:
    """Rank songs by likely reference divergence vs likely pipeline failure."""

    reference_rows: list[dict[str, Any]] = []
    pipeline_rows: list[dict[str, Any]] = []

    for record in succeeded:
        metrics = record.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        song_name = f"{record['artist']} - {record['title']}"
        reference_row = _build_reference_triage_row(
            record=record, metrics=metrics, song_name=song_name
        )
        if reference_row is not None:
            reference_rows.append(reference_row)
        pipeline_row = _build_pipeline_triage_row(
            record=record, metrics=metrics, song_name=song_name
        )
        if pipeline_row is not None:
            pipeline_rows.append(pipeline_row)

    reference_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    pipeline_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return {
        "likely_reference_divergence": reference_rows[:top_n],
        "likely_pipeline_failure": pipeline_rows[:top_n],
    }


def _triage_metric_float(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _build_reference_triage_row(
    *, record: dict[str, Any], metrics: dict[str, Any], song_name: str
) -> dict[str, Any] | None:
    dtw_line = _triage_metric_float(metrics, "dtw_line_coverage")
    dtw_word = _triage_metric_float(metrics, "dtw_word_coverage")
    low_conf = _triage_metric_float(metrics, "low_confidence_ratio")
    agree_cov = _triage_metric_float(metrics, "agreement_coverage_ratio")
    agree_sim = _triage_metric_float(metrics, "agreement_text_similarity_mean")
    agree_p95 = _triage_metric_float(metrics, "agreement_start_p95_abs_sec")

    reference_score = 0.0
    reference_reasons: list[str] = []
    ref_div = record.get("reference_divergence", {})
    if isinstance(ref_div, dict) and bool(ref_div.get("suspected")):
        reference_score += 2.0 + min(float(ref_div.get("score", 0.0) or 0.0), 4.0) / 2
        reference_reasons.append("reference_divergence_suspected")
    if (
        not bool(metrics.get("gold_available"))
        and dtw_line <= 0.6
        and dtw_word <= 0.45
        and agree_cov >= 0.07
        and agree_sim >= 0.9
        and agree_p95 <= 1.2
        and low_conf <= 0.08
    ):
        reference_score += 1.0
        reference_reasons.append("low_dtw_with_strong_anchor_agreement")
    if reference_score <= 0.0:
        return None
    return {
        "song": song_name,
        "score": round(reference_score, 3),
        "reasons": sorted(set(reference_reasons)),
    }


def _build_pipeline_triage_row(
    *, record: dict[str, Any], metrics: dict[str, Any], song_name: str
) -> dict[str, Any] | None:
    dtw_line = _triage_metric_float(metrics, "dtw_line_coverage")
    dtw_word = _triage_metric_float(metrics, "dtw_word_coverage")
    low_conf = _triage_metric_float(metrics, "low_confidence_ratio")
    agree_cov = _triage_metric_float(metrics, "agreement_coverage_ratio")
    agree_p95 = _triage_metric_float(metrics, "agreement_start_p95_abs_sec")
    agree_bad = _triage_metric_float(metrics, "agreement_bad_ratio")
    hook_agree_eligibility = _triage_metric_float(
        metrics, "agreement_hook_boundary_eligibility_ratio"
    )
    hook_agree_text_similarity = _triage_metric_float(
        metrics, "agreement_hook_boundary_text_similarity_mean"
    )
    timing_quality_score = _triage_metric_float(metrics, "timing_quality_score")
    agreement_reliability = max(0.3, min(1.0, agree_cov / 0.5))
    ref_div = record.get("reference_divergence", {})
    lexical_diag = record.get("lexical_mismatch_diagnostics", {})
    hook_boundary_ratio = 0.0
    if isinstance(lexical_diag, dict):
        hook_boundary_ratio = float(
            lexical_diag.get("hook_boundary_variant_ratio", 0.0) or 0.0
        )

    pipeline_score = 0.0
    pipeline_reasons: list[str] = []
    if dtw_line < 0.75:
        pipeline_score += (0.75 - dtw_line) * 2.0
        pipeline_reasons.append("low_dtw_line_coverage")
    if dtw_word < 0.6:
        pipeline_score += (0.6 - dtw_word) * 1.5
        pipeline_reasons.append("low_dtw_word_coverage")
    if low_conf > 0.1:
        pipeline_score += min(low_conf, 0.5)
        pipeline_reasons.append("high_low_confidence_ratio")
    if agree_p95 > 1.0:
        pipeline_score += min((agree_p95 - 1.0) / 2.0, 1.5) * agreement_reliability
        pipeline_reasons.append("high_agreement_p95")
    if agree_bad > 0.1:
        pipeline_score += (agree_bad - 0.1) * 2.0 * agreement_reliability
        pipeline_reasons.append("high_agreement_bad_ratio")
    if isinstance(metrics.get("timing_quality_score"), (int, float)):
        if timing_quality_score < 0.55:
            pipeline_score += min((0.55 - timing_quality_score) * 2.0, 0.8)
            pipeline_reasons.append("low_timing_quality_score")
        elif timing_quality_score >= 0.78:
            pipeline_score = max(0.0, pipeline_score - 0.2)
            pipeline_reasons.append("timing_quality_score_good")
    if (
        hook_boundary_ratio >= 0.25
        and hook_agree_eligibility >= max(0.85, agree_cov + 0.2)
        and hook_agree_text_similarity >= 0.9
        and agree_p95 <= 0.9
        and low_conf <= 0.1
    ):
        pipeline_score = max(0.0, pipeline_score - 0.45)
        pipeline_reasons.append("hook_boundary_variants_dominant")
        pipeline_reasons.append("hook_normalized_agreement_consistent")
    if isinstance(ref_div, dict) and bool(ref_div.get("suspected")):
        pipeline_score = max(0.0, pipeline_score - 1.0)
        pipeline_reasons.append("downgraded_due_to_reference_divergence")
    if pipeline_score <= 0.25:
        return None
    return {
        "song": song_name,
        "score": round(pipeline_score, 3),
        "reasons": sorted(set(pipeline_reasons)),
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:  # noqa: C901
    succeeded = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] != "ok"]
    metrics = [r.get("metrics", {}) for r in succeeded]

    def metric_values(key: str) -> list[float]:
        vals: list[float] = []
        for m in metrics:
            value = m.get(key)
            if isinstance(value, (int, float)):
                vals.append(float(value))
        return vals

    def weighted_metric_mean(
        key: str, *, weight_key: str = "line_count"
    ) -> float | None:
        weighted_sum = 0.0
        total_weight = 0.0
        for m in metrics:
            value = m.get(key)
            weight = m.get(weight_key)
            if not isinstance(value, (int, float)):
                continue
            if not isinstance(weight, (int, float)) or float(weight) <= 0:
                continue
            weighted_sum += float(value) * float(weight)
            total_weight += float(weight)
        if total_weight <= 0:
            return None
        return weighted_sum / total_weight

    def weighted_metric_mean_for_rows(
        rows: list[dict[str, Any]], key: str, *, weight_key: str = "line_count"
    ) -> float | None:
        weighted_sum = 0.0
        total_weight = 0.0
        for row in rows:
            metrics_obj = row.get("metrics", {})
            if not isinstance(metrics_obj, dict):
                continue
            value = metrics_obj.get(key)
            weight = metrics_obj.get(weight_key)
            if not isinstance(value, (int, float)):
                continue
            if not isinstance(weight, (int, float)) or float(weight) <= 0:
                continue
            weighted_sum += float(value) * float(weight)
            total_weight += float(weight)
        if total_weight <= 0:
            return None
        return weighted_sum / total_weight

    def metric_mean(key: str) -> float | None:
        result = _mean(metric_values(key))
        return float(result) if result is not None else None

    def metric_mean_for_rows(rows: list[dict[str, Any]], key: str) -> float | None:
        values: list[float] = []
        for row in rows:
            metrics_obj = row.get("metrics", {})
            if not isinstance(metrics_obj, dict):
                continue
            value = metrics_obj.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        result = _mean(values)
        return float(result) if result is not None else None

    total_lines = int(sum(metric_values("line_count")))
    low_conf_total = int(sum(metric_values("low_confidence_lines")))
    agreement_count_total = int(sum(metric_values("agreement_count")))
    agreement_eligible_lines_total = int(sum(metric_values("agreement_eligible_lines")))
    agreement_matched_anchor_lines_total = int(
        sum(metric_values("agreement_matched_lines"))
    )
    whisper_anchor_count_total = int(sum(metric_values("whisper_anchor_count")))
    agreement_good_total = int(sum(metric_values("agreement_good_lines")))
    agreement_warn_total = int(sum(metric_values("agreement_warn_lines")))
    agreement_bad_total = int(sum(metric_values("agreement_bad_lines")))
    agreement_severe_total = int(sum(metric_values("agreement_severe_lines")))
    local_transcribe_cache_hits_total = int(
        sum(metric_values("local_transcribe_cache_hits"))
    )
    local_transcribe_cache_misses_total = int(
        sum(metric_values("local_transcribe_cache_misses"))
    )
    local_transcribe_cache_events_total = (
        local_transcribe_cache_hits_total + local_transcribe_cache_misses_total
    )
    low_conf_ratio = (low_conf_total / total_lines) if total_lines else 0.0
    agreement_coverage_ratio = (
        agreement_count_total / total_lines if total_lines else 0.0
    )
    whisper_anchor_coverage_ratio = (
        whisper_anchor_count_total / total_lines if total_lines else 0.0
    )
    agreement_good_ratio = (agreement_good_total / total_lines) if total_lines else 0.0
    agreement_warn_ratio = (agreement_warn_total / total_lines) if total_lines else 0.0
    agreement_bad_ratio = (agreement_bad_total / total_lines) if total_lines else 0.0
    agreement_severe_ratio = (
        agreement_severe_total / total_lines if total_lines else 0.0
    )
    measured_dtw_songs = [
        r
        for r in succeeded
        if isinstance(r.get("metrics", {}).get("dtw_line_coverage"), (int, float))
    ]
    measured_dtw_song_count = len(measured_dtw_songs)
    measured_dtw_line_count = int(
        sum(
            float(r.get("metrics", {}).get("line_count", 0))
            for r in measured_dtw_songs
            if isinstance(r.get("metrics", {}).get("line_count"), (int, float))
        )
    )
    gold_measured_songs = [
        r
        for r in succeeded
        if isinstance(
            r.get("metrics", {}).get("avg_abs_word_start_delta_sec"), (int, float)
        )
        and int(r.get("metrics", {}).get("gold_comparable_word_count", 0) or 0) > 0
    ]
    gold_metric_song_count = len(gold_measured_songs)
    gold_word_count_total = int(sum(metric_values("gold_word_count")))
    gold_comparable_word_count_total = int(
        sum(metric_values("gold_comparable_word_count"))
    )
    gold_word_coverage_ratio_total = (
        (gold_comparable_word_count_total / gold_word_count_total)
        if gold_word_count_total
        else 0.0
    )
    sum_song_elapsed_total = round(
        sum(
            float(r.get("elapsed_sec", 0.0))
            for r in results
            if isinstance(r.get("elapsed_sec"), (int, float))
        ),
        2,
    )
    sum_song_elapsed_executed = round(
        sum(
            float(r.get("elapsed_sec", 0.0))
            for r in results
            if isinstance(r.get("elapsed_sec"), (int, float))
            and not bool(r.get("result_reused", False))
        ),
        2,
    )
    phase_totals: dict[str, float] = {}
    for r in results:
        phase_map = r.get("phase_durations_sec")
        if not isinstance(phase_map, dict):
            continue
        for phase_name, raw_val in phase_map.items():
            if not isinstance(raw_val, (int, float)):
                continue
            phase_totals[phase_name] = phase_totals.get(phase_name, 0.0) + float(
                raw_val
            )
    phase_totals = {k: round(v, 2) for k, v in sorted(phase_totals.items())}
    phase_shares = (
        {k: round(v / sum_song_elapsed_executed, 4) for k, v in phase_totals.items()}
        if sum_song_elapsed_executed > 0
        else {}
    )

    def _hotspot_records(
        *,
        key: str,
        top_n: int = 3,
        reverse: bool = False,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for r in succeeded:
            m = r.get("metrics", {})
            if not isinstance(m, dict):
                continue
            value = m.get(key)
            if not isinstance(value, (int, float)):
                continue
            rows.append(
                {
                    "song": f"{r['artist']} - {r['title']}",
                    "value": round(float(value), 4),
                    "line_count": (
                        int(m.get("line_count", 0))
                        if isinstance(m.get("line_count"), (int, float))
                        else 0
                    ),
                }
            )
        rows.sort(key=lambda row: float(row["value"]), reverse=reverse)
        return rows[:top_n]

    def _lexical_hook_boundary_hotspots(*, top_n: int = 3) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for r in succeeded:
            lexical_diag = r.get("lexical_mismatch_diagnostics")
            if not isinstance(lexical_diag, dict):
                continue
            ratio = lexical_diag.get("hook_boundary_variant_ratio")
            if not isinstance(ratio, (int, float)):
                continue
            count = int(lexical_diag.get("hook_boundary_variant_count", 0) or 0)
            if count <= 0 or float(ratio) <= 0.0:
                continue
            rows.append(
                {
                    "song": f"{r['artist']} - {r['title']}",
                    "value": round(float(ratio), 4),
                    "count": count,
                }
            )
        rows.sort(
            key=lambda row: (float(row["value"]), int(row["count"])), reverse=True
        )
        return rows[:top_n]

    def _cache_bucket(decision: Any) -> str:
        if not isinstance(decision, str):
            return "unknown"
        normalized = decision.strip().lower()
        if normalized.startswith("hit"):
            return "hit"
        if normalized.startswith("likely_hit"):
            return "likely_hit"
        if normalized.startswith("miss"):
            return "miss"
        if normalized.startswith("computed"):
            return "computed"
        return "unknown"

    cache_summary: dict[str, dict[str, Any]] = {}
    for phase_name in ("audio", "separation", "whisper", "alignment"):
        decisions: list[str] = []
        for r in succeeded:
            if bool(r.get("result_reused", False)):
                continue
            cache = r.get("cache_decisions", {})
            if not isinstance(cache, dict):
                continue
            value = cache.get(phase_name)
            if isinstance(value, str):
                decisions.append(value)
        counts = {
            "hit": 0,
            "likely_hit": 0,
            "miss": 0,
            "computed": 0,
            "unknown": 0,
        }
        for item in decisions:
            counts[_cache_bucket(item)] += 1
        total = len(decisions)
        cache_summary[phase_name] = {
            "total": total,
            "hit_count": counts["hit"],
            "likely_hit_count": counts["likely_hit"],
            "miss_count": counts["miss"],
            "computed_count": counts["computed"],
            "unknown_count": counts["unknown"],
            "cached_count": counts["hit"] + counts["likely_hit"],
            "cached_ratio": round(
                ((counts["hit"] + counts["likely_hit"]) / total) if total else 0.0,
                4,
            ),
            "miss_ratio": round((counts["miss"] / total) if total else 0.0, 4),
        }

    alignment_method_counts: dict[str, int] = {}
    lyrics_source_provider_counts: dict[str, int] = {}
    issue_tag_totals: dict[str, int] = {}
    policy_hint_counts: dict[str, int] = {}
    lyrics_source_selection_mode_counts: dict[str, int] = {}
    lyrics_source_routing_skip_reason_counts: dict[str, int] = {}
    fallback_map_decision_counts: dict[str, int] = {}
    fallback_map_attempted_count = 0
    fallback_map_selected_count = 0
    fallback_map_rejected_count = 0
    lyrics_source_disagreement_song_count = 0
    lyrics_source_audio_scoring_song_count = 0
    fallback_map_song_report: list[dict[str, Any]] = []
    for r in succeeded:
        diag = r.get("alignment_diagnostics", {})
        if not isinstance(diag, dict):
            continue
        method = str(diag.get("alignment_method", "") or "unknown")
        alignment_method_counts[method] = alignment_method_counts.get(method, 0) + 1
        provider = str(diag.get("lyrics_source_provider", "") or "unknown")
        lyrics_source_provider_counts[provider] = (
            lyrics_source_provider_counts.get(provider, 0) + 1
        )
        tag_counts = diag.get("issue_tag_counts", {})
        if isinstance(tag_counts, dict):
            for tag, raw_count in tag_counts.items():
                if not isinstance(tag, str) or not isinstance(raw_count, (int, float)):
                    continue
                issue_tag_totals[tag] = issue_tag_totals.get(tag, 0) + int(raw_count)
        policy_hint = r.get("alignment_policy_hint", {})
        if isinstance(policy_hint, dict):
            hint = str(policy_hint.get("hint", "") or "none")
            policy_hint_counts[hint] = policy_hint_counts.get(hint, 0) + 1
        selection_mode = str(diag.get("lyrics_source_selection_mode", "") or "default")
        lyrics_source_selection_mode_counts[selection_mode] = (
            lyrics_source_selection_mode_counts.get(selection_mode, 0) + 1
        )
        skip_reason = str(diag.get("lyrics_source_routing_skip_reason", "") or "none")
        lyrics_source_routing_skip_reason_counts[skip_reason] = (
            lyrics_source_routing_skip_reason_counts.get(skip_reason, 0) + 1
        )
        if bool(diag.get("lyrics_source_disagreement_flagged", False)):
            lyrics_source_disagreement_song_count += 1
        if bool(diag.get("lyrics_source_audio_scoring_used", False)):
            lyrics_source_audio_scoring_song_count += 1
        attempted = bool(diag.get("fallback_map_attempted", False))
        selected = bool(diag.get("fallback_map_selected", False))
        rejected = bool(diag.get("fallback_map_rejected", False))
        decision_reason = str(diag.get("fallback_map_decision_reason", "") or "unknown")
        fallback_map_decision_counts[decision_reason] = (
            fallback_map_decision_counts.get(decision_reason, 0) + 1
        )
        if attempted:
            fallback_map_attempted_count += 1
        if selected:
            fallback_map_selected_count += 1
        if rejected:
            fallback_map_rejected_count += 1
        fallback_map_song_report.append(
            {
                "song": f"{r['artist']} - {r['title']}",
                "attempted": attempted,
                "selected": selected,
                "rejected": rejected,
                "decision_reason": decision_reason,
                "score_gain": float(diag.get("fallback_map_score_gain", 0.0) or 0.0),
            }
        )

    reference_divergence_suspects = [
        {
            "song": f"{r['artist']} - {r['title']}",
            "score": float(r.get("reference_divergence", {}).get("score", 0.0) or 0.0),
            "confidence": str(
                r.get("reference_divergence", {}).get("confidence", "low") or "low"
            ),
            "evidence": list(
                r.get("reference_divergence", {}).get("evidence", []) or []
            ),
        }
        for r in succeeded
        if isinstance(r.get("reference_divergence"), dict)
        and bool(r["reference_divergence"].get("suspected"))
    ]
    reference_divergence_suspects.sort(
        key=lambda row: cast(float, row["score"]), reverse=True
    )
    reference_divergence_song_keys = {
        row["song"] for row in reference_divergence_suspects if isinstance(row, dict)
    }
    gold_measured_non_reference_songs = [
        row
        for row in gold_measured_songs
        if f"{row['artist']} - {row['title']}" not in reference_divergence_song_keys
    ]
    gold_metric_song_count_excluding_reference = len(gold_measured_non_reference_songs)
    curated_canary_song_names = [
        f"{row['artist']} - {row['title']}" for row in gold_measured_non_reference_songs
    ]
    curated_canary_gold_word_count_total = int(
        sum(
            int(r.get("metrics", {}).get("gold_word_count", 0) or 0)
            for r in gold_measured_non_reference_songs
        )
    )
    curated_canary_gold_comparable_word_count_total = int(
        sum(
            int(r.get("metrics", {}).get("gold_comparable_word_count", 0) or 0)
            for r in gold_measured_non_reference_songs
        )
    )
    curated_canary_gold_word_coverage_ratio_total = (
        (
            curated_canary_gold_comparable_word_count_total
            / curated_canary_gold_word_count_total
        )
        if curated_canary_gold_word_count_total
        else 0.0
    )
    avg_abs_word_start_delta_word_weighted_mean_excluding_reference = (
        weighted_metric_mean_for_rows(
            gold_measured_non_reference_songs,
            "avg_abs_word_start_delta_sec",
            weight_key="gold_comparable_word_count",
        )
    )
    curated_canary_gold_end_mean_abs_sec_mean = metric_mean_for_rows(
        gold_measured_non_reference_songs, "gold_end_mean_abs_sec"
    )
    curated_canary_gold_line_duration_mean_abs_sec_mean = metric_mean_for_rows(
        gold_measured_non_reference_songs, "gold_line_duration_mean_abs_sec"
    )
    curated_canary_gold_pre_whisper_start_mean_abs_sec_mean = metric_mean_for_rows(
        gold_measured_non_reference_songs, "gold_pre_whisper_start_mean_abs_sec"
    )
    curated_canary_gold_start_p95_abs_sec_mean = metric_mean_for_rows(
        gold_measured_non_reference_songs, "gold_start_p95_abs_sec"
    )
    curated_canary_gold_parenthetical_interjection_start_mean_abs_sec_mean = (
        metric_mean_for_rows(
            gold_measured_non_reference_songs,
            "gold_parenthetical_interjection_start_mean_abs_sec",
        )
    )
    curated_canary_gold_parenthetical_interjection_start_p95_abs_sec_mean = (
        metric_mean_for_rows(
            gold_measured_non_reference_songs,
            "gold_parenthetical_interjection_start_p95_abs_sec",
        )
    )
    curated_canary_gold_nearest_onset_start_mean_abs_sec_mean = metric_mean_for_rows(
        gold_measured_non_reference_songs,
        "gold_nearest_onset_start_mean_abs_sec",
    )
    curated_canary_gold_nearest_onset_start_non_interjection_mean_abs_sec_mean = (
        metric_mean_for_rows(
            gold_measured_non_reference_songs,
            "gold_nearest_onset_start_non_interjection_mean_abs_sec",
        )
    )
    curated_canary_gold_later_onset_choice_line_count_total = int(
        sum(
            int(r.get("metrics", {}).get("gold_later_onset_choice_line_count", 0) or 0)
            for r in gold_measured_non_reference_songs
        )
    )
    curated_canary_gold_later_onset_choice_mean_improvement_sec_mean = (
        metric_mean_for_rows(
            gold_measured_non_reference_songs,
            "gold_later_onset_choice_mean_improvement_sec",
        )
    )
    curated_canary_gold_downstream_regression_line_count_total = int(
        sum(
            int(
                r.get("metrics", {}).get("gold_downstream_regression_line_count", 0)
                or 0
            )
            for r in gold_measured_non_reference_songs
        )
    )
    curated_canary_gold_downstream_regression_mean_improvement_sec_mean = (
        metric_mean_for_rows(
            gold_measured_non_reference_songs,
            "gold_downstream_regression_mean_improvement_sec",
        )
    )
    triage_rankings = _build_triage_rankings(succeeded, top_n=5)
    diagnosis_counts: dict[str, int] = {}
    for row in succeeded:
        diagnosis = row.get("quality_diagnosis", {})
        if not isinstance(diagnosis, dict):
            continue
        verdict = str(diagnosis.get("verdict", "") or "unknown")
        diagnosis_counts[verdict] = diagnosis_counts.get(verdict, 0) + 1
    diagnosis_counts = dict(sorted(diagnosis_counts.items()))
    diagnosis_ratios = {
        key: round((value / len(succeeded)) if succeeded else 0.0, 4)
        for key, value in diagnosis_counts.items()
    }
    lexical_review_song_count = 0
    lexical_hook_boundary_variant_song_count = 0
    lexical_hook_boundary_variant_line_count_total = 0
    lexical_truncation_pattern_line_count_total = 0
    lexical_repetitive_phrase_line_count_total = 0
    lexical_hook_boundary_variant_ratio_values: list[float] = []
    lexical_truncation_pattern_ratio_values: list[float] = []
    lexical_repetitive_phrase_ratio_values: list[float] = []
    for row in succeeded:
        lexical_diag = row.get("lexical_mismatch_diagnostics", {})
        if not isinstance(lexical_diag, dict) or not bool(lexical_diag.get("active")):
            continue
        hook_variant_ratio = lexical_diag.get("hook_boundary_variant_ratio")
        trunc_ratio = lexical_diag.get("truncation_pattern_ratio")
        rep_ratio = lexical_diag.get("repetitive_phrase_line_ratio")
        lexical_review_song_count += 1
        if (
            isinstance(hook_variant_ratio, (int, float))
            and float(hook_variant_ratio) >= 0.12
        ):
            lexical_hook_boundary_variant_song_count += 1
        lexical_hook_boundary_variant_line_count_total += int(
            lexical_diag.get("hook_boundary_variant_count", 0) or 0
        )
        lexical_truncation_pattern_line_count_total += int(
            lexical_diag.get("truncation_pattern_count", 0) or 0
        )
        lexical_repetitive_phrase_line_count_total += int(
            lexical_diag.get("repetitive_phrase_line_count", 0) or 0
        )
        if isinstance(hook_variant_ratio, (int, float)):
            lexical_hook_boundary_variant_ratio_values.append(float(hook_variant_ratio))
        if isinstance(trunc_ratio, (int, float)):
            lexical_truncation_pattern_ratio_values.append(float(trunc_ratio))
        if isinstance(rep_ratio, (int, float)):
            lexical_repetitive_phrase_ratio_values.append(float(rep_ratio))
    timing_quality_band_counts: dict[str, int] = {}
    for row in succeeded:
        metrics_row = row.get("metrics", {})
        if not isinstance(metrics_row, dict):
            continue
        band = str(metrics_row.get("timing_quality_band", "") or "")
        if not band:
            continue
        timing_quality_band_counts[band] = timing_quality_band_counts.get(band, 0) + 1
    timing_quality_band_counts = dict(sorted(timing_quality_band_counts.items()))
    timing_quality_band_ratios = {
        key: round((value / len(succeeded)) if succeeded else 0.0, 4)
        for key, value in timing_quality_band_counts.items()
    }
    agreement_skip_reason_totals: dict[str, int] = {}
    agreement_comparability_report: list[dict[str, Any]] = []
    for row in succeeded:
        metrics_row = row.get("metrics", {})
        if not isinstance(metrics_row, dict):
            continue
        skip_map = metrics_row.get("agreement_skip_reason_counts", {})
        if isinstance(skip_map, dict):
            for reason, raw_count in skip_map.items():
                if not isinstance(reason, str) or not isinstance(
                    raw_count, (int, float)
                ):
                    continue
                agreement_skip_reason_totals[reason] = agreement_skip_reason_totals.get(
                    reason, 0
                ) + int(raw_count)
        agreement_comparability_report.append(
            {
                "song": f"{row['artist']} - {row['title']}",
                "line_count": int(metrics_row.get("line_count", 0) or 0),
                "eligible_lines": int(
                    metrics_row.get("agreement_eligible_lines", 0) or 0
                ),
                "matched_lines_anchor": int(
                    metrics_row.get("agreement_matched_lines", 0) or 0
                ),
                "matched_lines_independent": int(
                    metrics_row.get("agreement_count", 0) or 0
                ),
                "eligibility_ratio": metrics_row.get("agreement_eligibility_ratio"),
                "match_ratio_within_eligible": metrics_row.get(
                    "agreement_match_ratio_within_eligible"
                ),
                "measurement_mode": metrics_row.get("agreement_measurement_mode"),
                "skip_reasons": (
                    dict(sorted(skip_map.items())) if isinstance(skip_map, dict) else {}
                ),
            }
        )
    agreement_comparability_report.sort(
        key=lambda row: (
            float(row.get("match_ratio_within_eligible") or 0.0),
            float(row.get("eligibility_ratio") or 0.0),
        )
    )

    return {
        "songs_total": len(results),
        "songs_succeeded": len(succeeded),
        "songs_failed": len(failed),
        "success_rate": round((len(succeeded) / len(results)) if results else 0.0, 4),
        "line_count_total": total_lines,
        "low_confidence_lines_total": low_conf_total,
        "low_confidence_ratio_mean": (
            round(float(metric_mean("low_confidence_ratio") or 0.0), 4)
            if metric_mean("low_confidence_ratio") is not None
            else None
        ),
        "low_confidence_ratio_line_weighted_mean": (
            round(float(weighted_metric_mean("low_confidence_ratio") or 0.0), 4)
            if weighted_metric_mean("low_confidence_ratio") is not None
            else None
        ),
        "low_confidence_ratio_total": round(low_conf_ratio, 4),
        "agreement_count_total": agreement_count_total,
        "agreement_eligible_lines_total": agreement_eligible_lines_total,
        "agreement_matched_anchor_lines_total": agreement_matched_anchor_lines_total,
        "agreement_coverage_ratio_total": round(agreement_coverage_ratio, 4),
        "whisper_anchor_count_total": whisper_anchor_count_total,
        "whisper_anchor_coverage_ratio_total": round(whisper_anchor_coverage_ratio, 4),
        "agreement_good_lines_total": agreement_good_total,
        "agreement_warn_lines_total": agreement_warn_total,
        "agreement_bad_lines_total": agreement_bad_total,
        "agreement_severe_lines_total": agreement_severe_total,
        "agreement_good_ratio_total": round(agreement_good_ratio, 4),
        "agreement_warn_ratio_total": round(agreement_warn_ratio, 4),
        "agreement_bad_ratio_total": round(agreement_bad_ratio, 4),
        "agreement_severe_ratio_total": round(agreement_severe_ratio, 4),
        "local_transcribe_cache_hits_total": local_transcribe_cache_hits_total,
        "local_transcribe_cache_misses_total": local_transcribe_cache_misses_total,
        "local_transcribe_cache_events_total": local_transcribe_cache_events_total,
        "local_transcribe_cache_hit_ratio": round(
            (
                local_transcribe_cache_hits_total / local_transcribe_cache_events_total
                if local_transcribe_cache_events_total
                else 0.0
            ),
            4,
        ),
        "dtw_line_coverage_mean": (
            round(float(metric_mean("dtw_line_coverage") or 0.0), 4)
            if metric_mean("dtw_line_coverage") is not None
            else None
        ),
        "dtw_word_coverage_mean": (
            round(float(metric_mean("dtw_word_coverage") or 0.0), 4)
            if metric_mean("dtw_word_coverage") is not None
            else None
        ),
        "dtw_phonetic_similarity_coverage_mean": (
            round(float(metric_mean("dtw_phonetic_similarity_coverage") or 0.0), 4)
            if metric_mean("dtw_phonetic_similarity_coverage") is not None
            else None
        ),
        "dtw_line_coverage_line_weighted_mean": (
            round(float(weighted_metric_mean("dtw_line_coverage") or 0.0), 4)
            if weighted_metric_mean("dtw_line_coverage") is not None
            else None
        ),
        "dtw_word_coverage_line_weighted_mean": (
            round(float(weighted_metric_mean("dtw_word_coverage") or 0.0), 4)
            if weighted_metric_mean("dtw_word_coverage") is not None
            else None
        ),
        "dtw_phonetic_similarity_coverage_line_weighted_mean": (
            round(
                float(weighted_metric_mean("dtw_phonetic_similarity_coverage") or 0.0),
                4,
            )
            if weighted_metric_mean("dtw_phonetic_similarity_coverage") is not None
            else None
        ),
        "agreement_coverage_ratio_mean": (
            round(float(metric_mean("agreement_coverage_ratio") or 0.0), 4)
            if metric_mean("agreement_coverage_ratio") is not None
            else None
        ),
        "agreement_hook_boundary_eligibility_ratio_mean": (
            round(
                float(metric_mean("agreement_hook_boundary_eligibility_ratio") or 0.0),
                4,
            )
            if metric_mean("agreement_hook_boundary_eligibility_ratio") is not None
            else None
        ),
        "agreement_text_similarity_mean": (
            round(float(metric_mean("agreement_text_similarity_mean") or 0.0), 4)
            if metric_mean("agreement_text_similarity_mean") is not None
            else None
        ),
        "agreement_hook_boundary_text_similarity_mean": (
            round(
                float(
                    metric_mean("agreement_hook_boundary_text_similarity_mean") or 0.0
                ),
                4,
            )
            if metric_mean("agreement_hook_boundary_text_similarity_mean") is not None
            else None
        ),
        "agreement_start_mean_abs_sec_mean": (
            round(float(metric_mean("agreement_start_mean_abs_sec") or 0.0), 4)
            if metric_mean("agreement_start_mean_abs_sec") is not None
            else None
        ),
        "agreement_start_mean_abs_sec_line_weighted_mean": (
            round(float(weighted_metric_mean("agreement_start_mean_abs_sec") or 0.0), 4)
            if weighted_metric_mean("agreement_start_mean_abs_sec") is not None
            else None
        ),
        "agreement_start_max_abs_sec_mean": (
            round(float(metric_mean("agreement_start_max_abs_sec") or 0.0), 4)
            if metric_mean("agreement_start_max_abs_sec") is not None
            else None
        ),
        "agreement_start_p95_abs_sec_mean": (
            round(float(metric_mean("agreement_start_p95_abs_sec") or 0.0), 4)
            if metric_mean("agreement_start_p95_abs_sec") is not None
            else None
        ),
        "agreement_start_p95_abs_sec_line_weighted_mean": (
            round(float(weighted_metric_mean("agreement_start_p95_abs_sec") or 0.0), 4)
            if weighted_metric_mean("agreement_start_p95_abs_sec") is not None
            else None
        ),
        "agreement_bad_ratio_mean": (
            round(float(metric_mean("agreement_bad_ratio") or 0.0), 4)
            if metric_mean("agreement_bad_ratio") is not None
            else None
        ),
        "agreement_severe_ratio_mean": (
            round(float(metric_mean("agreement_severe_ratio") or 0.0), 4)
            if metric_mean("agreement_severe_ratio") is not None
            else None
        ),
        "timing_quality_score_mean": (
            round(float(metric_mean("timing_quality_score") or 0.0), 4)
            if metric_mean("timing_quality_score") is not None
            else None
        ),
        "timing_quality_score_line_weighted_mean": (
            round(float(weighted_metric_mean("timing_quality_score") or 0.0), 4)
            if weighted_metric_mean("timing_quality_score") is not None
            else None
        ),
        "whisper_anchor_start_mean_abs_sec_mean": (
            round(float(metric_mean("whisper_anchor_start_mean_abs_sec") or 0.0), 4)
            if metric_mean("whisper_anchor_start_mean_abs_sec") is not None
            else None
        ),
        "whisper_anchor_start_mean_abs_sec_line_weighted_mean": (
            round(
                float(weighted_metric_mean("whisper_anchor_start_mean_abs_sec") or 0.0),
                4,
            )
            if weighted_metric_mean("whisper_anchor_start_mean_abs_sec") is not None
            else None
        ),
        "whisper_anchor_start_p95_abs_sec_mean": (
            round(float(metric_mean("whisper_anchor_start_p95_abs_sec") or 0.0), 4)
            if metric_mean("whisper_anchor_start_p95_abs_sec") is not None
            else None
        ),
        "whisper_anchor_start_p95_abs_sec_line_weighted_mean": (
            round(
                float(weighted_metric_mean("whisper_anchor_start_p95_abs_sec") or 0.0),
                4,
            )
            if weighted_metric_mean("whisper_anchor_start_p95_abs_sec") is not None
            else None
        ),
        "whisper_anchor_bad_ratio_mean": (
            round(float(metric_mean("whisper_anchor_bad_ratio") or 0.0), 4)
            if metric_mean("whisper_anchor_bad_ratio") is not None
            else None
        ),
        "whisper_anchor_severe_ratio_mean": (
            round(float(metric_mean("whisper_anchor_severe_ratio") or 0.0), 4)
            if metric_mean("whisper_anchor_severe_ratio") is not None
            else None
        ),
        "dtw_metric_song_count": measured_dtw_song_count,
        "dtw_metric_song_coverage_ratio": round(
            (measured_dtw_song_count / len(succeeded)) if succeeded else 0.0, 4
        ),
        "dtw_metric_line_count": measured_dtw_line_count,
        "dtw_metric_line_coverage_ratio": round(
            (measured_dtw_line_count / total_lines) if total_lines else 0.0, 4
        ),
        "gold_metric_song_count": gold_metric_song_count,
        "gold_metric_song_coverage_ratio": round(
            (gold_metric_song_count / len(succeeded)) if succeeded else 0.0, 4
        ),
        "gold_metric_song_count_excluding_reference_divergence": (
            gold_metric_song_count_excluding_reference
        ),
        "gold_metric_song_coverage_ratio_excluding_reference_divergence": round(
            (
                (gold_metric_song_count_excluding_reference / len(succeeded))
                if succeeded
                else 0.0
            ),
            4,
        ),
        "curated_canary_song_count": gold_metric_song_count_excluding_reference,
        "curated_canary_song_coverage_ratio": round(
            (
                (gold_metric_song_count_excluding_reference / len(succeeded))
                if succeeded
                else 0.0
            ),
            4,
        ),
        "curated_canary_song_names": curated_canary_song_names,
        "curated_canary_gold_word_count_total": curated_canary_gold_word_count_total,
        "curated_canary_gold_comparable_word_count_total": (
            curated_canary_gold_comparable_word_count_total
        ),
        "curated_canary_gold_word_coverage_ratio_total": round(
            curated_canary_gold_word_coverage_ratio_total, 4
        ),
        "curated_canary_avg_abs_word_start_delta_sec_word_weighted_mean": (
            round(
                float(
                    avg_abs_word_start_delta_word_weighted_mean_excluding_reference
                    or 0.0
                ),
                4,
            )
            if avg_abs_word_start_delta_word_weighted_mean_excluding_reference
            is not None
            else None
        ),
        "curated_canary_gold_start_p95_abs_sec_mean": (
            round(float(curated_canary_gold_start_p95_abs_sec_mean or 0.0), 4)
            if curated_canary_gold_start_p95_abs_sec_mean is not None
            else None
        ),
        "curated_canary_gold_end_mean_abs_sec_mean": (
            round(float(curated_canary_gold_end_mean_abs_sec_mean or 0.0), 4)
            if curated_canary_gold_end_mean_abs_sec_mean is not None
            else None
        ),
        "curated_canary_gold_line_duration_mean_abs_sec_mean": (
            round(float(curated_canary_gold_line_duration_mean_abs_sec_mean or 0.0), 4)
            if curated_canary_gold_line_duration_mean_abs_sec_mean is not None
            else None
        ),
        "curated_canary_gold_pre_whisper_start_mean_abs_sec_mean": (
            round(
                float(curated_canary_gold_pre_whisper_start_mean_abs_sec_mean or 0.0),
                4,
            )
            if curated_canary_gold_pre_whisper_start_mean_abs_sec_mean is not None
            else None
        ),
        "curated_canary_gold_parenthetical_interjection_start_mean_abs_sec_mean": (
            round(
                float(
                    curated_canary_gold_parenthetical_interjection_start_mean_abs_sec_mean
                    or 0.0
                ),
                4,
            )
            if curated_canary_gold_parenthetical_interjection_start_mean_abs_sec_mean
            is not None
            else None
        ),
        "curated_canary_gold_parenthetical_interjection_start_p95_abs_sec_mean": (
            round(
                float(
                    curated_canary_gold_parenthetical_interjection_start_p95_abs_sec_mean
                    or 0.0
                ),
                4,
            )
            if curated_canary_gold_parenthetical_interjection_start_p95_abs_sec_mean
            is not None
            else None
        ),
        "curated_canary_gold_nearest_onset_start_mean_abs_sec_mean": (
            round(
                float(curated_canary_gold_nearest_onset_start_mean_abs_sec_mean or 0.0),
                4,
            )
            if curated_canary_gold_nearest_onset_start_mean_abs_sec_mean is not None
            else None
        ),
        "curated_canary_gold_nearest_onset_start_non_interjection_mean_abs_sec_mean": (
            round(
                float(
                    curated_canary_gold_nearest_onset_start_non_interjection_mean_abs_sec_mean
                    or 0.0
                ),
                4,
            )
            if curated_canary_gold_nearest_onset_start_non_interjection_mean_abs_sec_mean
            is not None
            else None
        ),
        "curated_canary_gold_later_onset_choice_line_count_total": (
            curated_canary_gold_later_onset_choice_line_count_total
        ),
        "curated_canary_gold_later_onset_choice_mean_improvement_sec_mean": (
            round(
                float(
                    curated_canary_gold_later_onset_choice_mean_improvement_sec_mean
                    or 0.0
                ),
                4,
            )
            if curated_canary_gold_later_onset_choice_mean_improvement_sec_mean
            is not None
            else None
        ),
        "curated_canary_gold_downstream_regression_line_count_total": (
            curated_canary_gold_downstream_regression_line_count_total
        ),
        "curated_canary_gold_downstream_regression_mean_improvement_sec_mean": (
            round(
                float(
                    curated_canary_gold_downstream_regression_mean_improvement_sec_mean
                    or 0.0
                ),
                4,
            )
            if curated_canary_gold_downstream_regression_mean_improvement_sec_mean
            is not None
            else None
        ),
        "curated_canary_reference_watchlist_count": len(reference_divergence_suspects),
        "curated_canary_reference_watchlist": [
            row["song"]
            for row in reference_divergence_suspects
            if isinstance(row, dict)
        ],
        "reference_divergence_suspected_count": len(reference_divergence_suspects),
        "reference_divergence_suspected_ratio": round(
            (len(reference_divergence_suspects) / len(succeeded)) if succeeded else 0.0,
            4,
        ),
        "likely_reference_divergence_count": len(
            triage_rankings.get("likely_reference_divergence", [])
        ),
        "likely_pipeline_failure_count": len(
            triage_rankings.get("likely_pipeline_failure", [])
        ),
        "quality_diagnosis_counts": diagnosis_counts,
        "quality_diagnosis_ratios": diagnosis_ratios,
        "lexical_review_song_count": lexical_review_song_count,
        "lexical_hook_boundary_variant_song_count": (
            lexical_hook_boundary_variant_song_count
        ),
        "lexical_hook_boundary_variant_line_count_total": (
            lexical_hook_boundary_variant_line_count_total
        ),
        "lexical_truncation_pattern_line_count_total": (
            lexical_truncation_pattern_line_count_total
        ),
        "lexical_repetitive_phrase_line_count_total": (
            lexical_repetitive_phrase_line_count_total
        ),
        "lexical_hook_boundary_variant_ratio_mean": (
            round(float(_mean(lexical_hook_boundary_variant_ratio_values) or 0.0), 4)
            if lexical_hook_boundary_variant_ratio_values
            else None
        ),
        "lexical_truncation_pattern_ratio_mean": (
            round(float(_mean(lexical_truncation_pattern_ratio_values) or 0.0), 4)
            if lexical_truncation_pattern_ratio_values
            else None
        ),
        "lexical_repetitive_phrase_line_ratio_mean": (
            round(float(_mean(lexical_repetitive_phrase_ratio_values) or 0.0), 4)
            if lexical_repetitive_phrase_ratio_values
            else None
        ),
        "agreement_skip_reason_totals": dict(
            sorted(agreement_skip_reason_totals.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "agreement_comparability_report": agreement_comparability_report,
        "timing_quality_band_counts": timing_quality_band_counts,
        "timing_quality_band_ratios": timing_quality_band_ratios,
        "gold_word_count_total": gold_word_count_total,
        "gold_comparable_word_count_total": gold_comparable_word_count_total,
        "gold_word_coverage_ratio_total": round(gold_word_coverage_ratio_total, 4),
        "avg_abs_word_start_delta_sec_mean": (
            round(float(metric_mean("avg_abs_word_start_delta_sec") or 0.0), 4)
            if metric_mean("avg_abs_word_start_delta_sec") is not None
            else None
        ),
        "avg_abs_word_start_delta_sec_word_weighted_mean": (
            round(
                float(
                    weighted_metric_mean(
                        "avg_abs_word_start_delta_sec",
                        weight_key="gold_comparable_word_count",
                    )
                    or 0.0
                ),
                4,
            )
            if weighted_metric_mean(
                "avg_abs_word_start_delta_sec",
                weight_key="gold_comparable_word_count",
            )
            is not None
            else None
        ),
        "avg_abs_word_start_delta_sec_word_weighted_mean_excluding_reference_divergence": (
            round(
                float(
                    avg_abs_word_start_delta_word_weighted_mean_excluding_reference
                    or 0.0
                ),
                4,
            )
            if avg_abs_word_start_delta_word_weighted_mean_excluding_reference
            is not None
            else None
        ),
        "gold_start_p95_abs_sec_mean": (
            round(float(metric_mean("gold_start_p95_abs_sec") or 0.0), 4)
            if metric_mean("gold_start_p95_abs_sec") is not None
            else None
        ),
        "gold_end_mean_abs_sec_mean": (
            round(float(metric_mean("gold_end_mean_abs_sec") or 0.0), 4)
            if metric_mean("gold_end_mean_abs_sec") is not None
            else None
        ),
        "gold_line_duration_mean_abs_sec_mean": (
            round(float(metric_mean("gold_line_duration_mean_abs_sec") or 0.0), 4)
            if metric_mean("gold_line_duration_mean_abs_sec") is not None
            else None
        ),
        "gold_nearest_onset_start_mean_abs_sec_mean": (
            round(float(metric_mean("gold_nearest_onset_start_mean_abs_sec") or 0.0), 4)
            if metric_mean("gold_nearest_onset_start_mean_abs_sec") is not None
            else None
        ),
        "gold_end_p95_abs_sec_mean": (
            round(float(metric_mean("gold_end_p95_abs_sec") or 0.0), 4)
            if metric_mean("gold_end_p95_abs_sec") is not None
            else None
        ),
        "songs_without_dtw_metrics": [
            f"{r['artist']} - {r['title']}"
            for r in succeeded
            if not isinstance(
                r.get("metrics", {}).get("dtw_line_coverage"), (int, float)
            )
        ],
        "songs_without_gold_metrics": [
            f"{r['artist']} - {r['title']}"
            for r in succeeded
            if not isinstance(
                r.get("metrics", {}).get("avg_abs_word_start_delta_sec"), (int, float)
            )
            or int(r.get("metrics", {}).get("gold_comparable_word_count", 0) or 0) <= 0
        ],
        "sum_song_elapsed_sec": sum_song_elapsed_executed,
        "sum_song_elapsed_total_sec": sum_song_elapsed_total,
        "phase_totals_sec": phase_totals,
        "phase_shares_of_song_elapsed": phase_shares,
        "quality_hotspots": {
            "highest_timing_quality_score": _hotspot_records(
                key="timing_quality_score", top_n=3, reverse=True
            ),
            "lowest_dtw_line_coverage": _hotspot_records(
                key="dtw_line_coverage", top_n=3, reverse=False
            ),
            "highest_low_confidence_ratio": _hotspot_records(
                key="low_confidence_ratio", top_n=3, reverse=True
            ),
            "lowest_agreement_coverage_ratio": _hotspot_records(
                key="agreement_coverage_ratio", top_n=3, reverse=False
            ),
            "highest_agreement_start_p95_abs_sec": _hotspot_records(
                key="agreement_start_p95_abs_sec", top_n=3, reverse=True
            ),
            "highest_agreement_bad_ratio": _hotspot_records(
                key="agreement_bad_ratio", top_n=3, reverse=True
            ),
            "highest_agreement_severe_ratio": _hotspot_records(
                key="agreement_severe_ratio", top_n=3, reverse=True
            ),
            "lowest_timing_quality_score": _hotspot_records(
                key="timing_quality_score", top_n=3, reverse=False
            ),
            "highest_whisper_anchor_start_p95_abs_sec": _hotspot_records(
                key="whisper_anchor_start_p95_abs_sec", top_n=3, reverse=True
            ),
            "highest_avg_abs_word_start_delta_sec": _hotspot_records(
                key="avg_abs_word_start_delta_sec", top_n=3, reverse=True
            ),
            "lowest_gold_word_coverage_ratio": _hotspot_records(
                key="gold_word_coverage_ratio", top_n=3, reverse=False
            ),
            "lexical_hook_boundary_variants": _lexical_hook_boundary_hotspots(),
            "reference_divergence_suspects": reference_divergence_suspects[:5],
            "likely_reference_divergence": triage_rankings.get(
                "likely_reference_divergence", []
            ),
            "likely_pipeline_failure": triage_rankings.get(
                "likely_pipeline_failure", []
            ),
            "quality_diagnosis": [
                {
                    "song": f"{r['artist']} - {r['title']}",
                    "verdict": str(
                        r.get("quality_diagnosis", {}).get("verdict", "unknown")
                    ),
                    "confidence": str(
                        r.get("quality_diagnosis", {}).get("confidence", "low")
                    ),
                    "reasons": list(
                        r.get("quality_diagnosis", {}).get("reasons", []) or []
                    ),
                }
                for r in succeeded
                if isinstance(r.get("quality_diagnosis"), dict)
            ][:8],
        },
        "cache_summary": cache_summary,
        "alignment_diagnostics_summary": {
            "alignment_method_counts": dict(sorted(alignment_method_counts.items())),
            "lyrics_source_provider_counts": dict(
                sorted(lyrics_source_provider_counts.items())
            ),
            "alignment_policy_hint_counts": dict(sorted(policy_hint_counts.items())),
            "lyrics_source_selection_mode_counts": dict(
                sorted(lyrics_source_selection_mode_counts.items())
            ),
            "lyrics_source_routing_skip_reason_counts": dict(
                sorted(lyrics_source_routing_skip_reason_counts.items())
            ),
            "lyrics_source_disagreement_song_count": (
                lyrics_source_disagreement_song_count
            ),
            "lyrics_source_audio_scoring_song_count": (
                lyrics_source_audio_scoring_song_count
            ),
            "fallback_map_attempted_count": fallback_map_attempted_count,
            "fallback_map_selected_count": fallback_map_selected_count,
            "fallback_map_rejected_count": fallback_map_rejected_count,
            "fallback_map_decision_counts": dict(
                sorted(fallback_map_decision_counts.items())
            ),
            "fallback_map_song_report": fallback_map_song_report,
            "issue_tag_totals": dict(
                sorted(issue_tag_totals.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
        },
        "failed_songs": [f"{r['artist']} - {r['title']}" for r in failed],
    }


def _quality_coverage_warnings(
    *,
    aggregate: dict[str, Any],
    dtw_enabled: bool,
    min_song_coverage_ratio: float,
    min_line_coverage_ratio: float,
    min_timing_quality_score_line_weighted: float,
    suite_wall_elapsed_sec: float,
) -> list[str]:
    warnings: list[str] = []
    song_cov = float(aggregate.get("dtw_metric_song_coverage_ratio", 0.0) or 0.0)
    line_cov = float(aggregate.get("dtw_metric_line_coverage_ratio", 0.0) or 0.0)
    if not dtw_enabled and (
        song_cov < min_song_coverage_ratio or line_cov < min_line_coverage_ratio
    ):
        warnings.append(
            "DTW mapping is disabled (--no-whisper-map-lrc-dtw) and DTW coverage is low; "
            "quality metrics may be partially unmeasured."
        )
    if song_cov < min_song_coverage_ratio:
        warnings.append(
            "DTW song coverage below threshold: "
            f"{song_cov:.3f} < {min_song_coverage_ratio:.3f}"
        )
    if line_cov < min_line_coverage_ratio:
        warnings.append(
            "DTW line coverage below threshold: "
            f"{line_cov:.3f} < {min_line_coverage_ratio:.3f}"
        )
    agreement_count = int(aggregate.get("agreement_count_total", 0) or 0)
    if agreement_count == 0:
        warnings.append(
            "Independent line-start agreement is unavailable for this strategy "
            "(no DTW anchor coverage); use Whisper-anchor diagnostics only for debugging."
        )
    else:
        warnings.extend(_agreement_coverage_warnings(aggregate))
    warnings.extend(_diagnosis_ratio_warnings(aggregate))
    timing_quality = aggregate.get("timing_quality_score_line_weighted_mean")
    if isinstance(timing_quality, (int, float)) and float(timing_quality) < float(
        min_timing_quality_score_line_weighted
    ):
        warnings.append(
            "Line-weighted timing quality score is below target: "
            f"{float(timing_quality):.3f} < {float(min_timing_quality_score_line_weighted):.3f}"
        )
    timing_band_ratios = aggregate.get("timing_quality_band_ratios", {})
    if isinstance(timing_band_ratios, dict):
        poor_ratio = timing_band_ratios.get("poor")
        if isinstance(poor_ratio, (int, float)) and float(poor_ratio) > 0.25:
            warnings.append(
                "Too many songs are in poor timing-quality band: "
                f"{float(poor_ratio):.3f} > 0.250"
            )
    warnings.extend(_gold_metric_warnings(aggregate))

    sum_song_elapsed = float(aggregate.get("sum_song_elapsed_sec", 0.0) or 0.0)
    if suite_wall_elapsed_sec > 0 and sum_song_elapsed > suite_wall_elapsed_sec:
        warnings.append(
            "Per-song elapsed sum exceeds suite wall elapsed; compare runs using "
            "suite_wall_elapsed_sec and sum_song_elapsed_sec explicitly."
        )
    return warnings


def _agreement_coverage_warnings(aggregate: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    agreement_cov = float(aggregate.get("agreement_coverage_ratio_mean", 0.0) or 0.0)
    if agreement_cov < 0.4:
        warnings.append(
            "LRC-Whisper agreement coverage is low: "
            f"{agreement_cov:.3f} < 0.400 (not enough lexically comparable lines)"
        )
    agreement_p95 = float(aggregate.get("agreement_start_p95_abs_sec_mean", 0.0) or 0.0)
    if agreement_p95 > 0.8:
        warnings.append(
            "Line-start agreement p95 is high on comparable lines: "
            f"{agreement_p95:.3f}s > 0.800s"
        )
    agreement_bad_ratio = float(aggregate.get("agreement_bad_ratio_total", 0.0) or 0.0)
    if agreement_bad_ratio > 0.1:
        warnings.append(
            "Too many comparable lines have poor start agreement (>0.8s): "
            f"{agreement_bad_ratio:.3f} > 0.100"
        )
    agreement_severe_ratio = float(
        aggregate.get("agreement_severe_ratio_total", 0.0) or 0.0
    )
    if agreement_severe_ratio > 0.03:
        warnings.append(
            "Comparable lines with severe agreement error (>1.5s) are high: "
            f"{agreement_severe_ratio:.3f} > 0.030"
        )
    return warnings


def _diagnosis_ratio_warnings(aggregate: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    diagnosis_ratios = aggregate.get("quality_diagnosis_ratios", {})
    if not isinstance(diagnosis_ratios, dict):
        return warnings
    pipeline_ratio = diagnosis_ratios.get("needs_pipeline_work")
    if isinstance(pipeline_ratio, (int, float)) and float(pipeline_ratio) > 0.35:
        warnings.append(
            "Many songs are diagnosed as pipeline work needed: "
            f"{float(pipeline_ratio):.3f} > 0.350"
        )
    reference_ratio = diagnosis_ratios.get("likely_reference_divergence")
    if isinstance(reference_ratio, (int, float)) and float(reference_ratio) > 0.35:
        warnings.append(
            "Many songs are diagnosed as likely reference divergence: "
            f"{float(reference_ratio):.3f} > 0.350"
        )
    return warnings


def _gold_metric_warnings(aggregate: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    curated_canary_song_count = int(aggregate.get("curated_canary_song_count", 0) or 0)
    gold_metric_song_count = int(aggregate.get("gold_metric_song_count", 0) or 0)
    use_curated_canaries = curated_canary_song_count > 0
    active_song_count = (
        curated_canary_song_count if use_curated_canaries else gold_metric_song_count
    )
    if active_song_count <= 0:
        return warnings
    coverage_key = (
        "curated_canary_song_coverage_ratio"
        if use_curated_canaries
        else "gold_metric_song_coverage_ratio"
    )
    word_cov_key = (
        "curated_canary_gold_word_coverage_ratio_total"
        if use_curated_canaries
        else "gold_word_coverage_ratio_total"
    )
    start_mean_key = (
        "curated_canary_avg_abs_word_start_delta_sec_word_weighted_mean"
        if use_curated_canaries
        else "avg_abs_word_start_delta_sec_word_weighted_mean"
    )
    label = "Curated-canary gold-set" if use_curated_canaries else "Gold-set"
    gold_song_cov = float(aggregate.get(coverage_key, 0.0) or 0.0)
    if gold_song_cov < 0.5:
        warnings.append(
            f"{label} metric song coverage is low: " f"{gold_song_cov:.3f} < 0.500"
        )
    gold_word_cov = float(aggregate.get(word_cov_key, 0.0) or 0.0)
    if gold_word_cov < 0.8:
        warnings.append(
            f"{label} comparable word coverage is low: " f"{gold_word_cov:.3f} < 0.800"
        )
    gold_start_mean = float(aggregate.get(start_mean_key, 0.0) or 0.0)
    if gold_start_mean > 0.35:
        warnings.append(
            f"{label} avg abs word-start delta is high: "
            f"{gold_start_mean:.3f}s > 0.350s"
        )
    return warnings


def _agreement_tradeoff_warnings(
    *,
    aggregate: dict[str, Any],
    baseline_aggregate: dict[str, Any] | None,
    min_coverage_gain: float,
    max_bad_ratio_increase: float,
) -> list[str]:
    if (
        baseline_aggregate is None
        or min_coverage_gain <= 0.0
        or max_bad_ratio_increase < 0.0
    ):
        return []

    current_cov = aggregate.get("agreement_coverage_ratio_mean")
    baseline_cov = baseline_aggregate.get("agreement_coverage_ratio_mean")
    current_bad = aggregate.get("agreement_bad_ratio_mean")
    baseline_bad = baseline_aggregate.get("agreement_bad_ratio_mean")
    if not isinstance(current_cov, (int, float)) or not isinstance(
        baseline_cov, (int, float)
    ):
        return []
    if not isinstance(current_bad, (int, float)) or not isinstance(
        baseline_bad, (int, float)
    ):
        return []

    coverage_delta = float(current_cov) - float(baseline_cov)
    bad_ratio_delta = float(current_bad) - float(baseline_bad)
    if coverage_delta >= float(min_coverage_gain) and bad_ratio_delta > float(
        max_bad_ratio_increase
    ):
        return [
            "Agreement tradeoff warning: agreement_coverage_ratio_mean increased "
            f"by {coverage_delta:+.4f} (>= {float(min_coverage_gain):.4f}) while "
            "agreement_bad_ratio_mean increased by "
            f"{bad_ratio_delta:+.4f} (> {float(max_bad_ratio_increase):.4f})"
        ]
    return []


def _cache_expectation_warnings(
    *,
    aggregate: dict[str, Any],
    expect_cached_separation: bool,
    expect_cached_whisper: bool,
) -> list[str]:
    warnings: list[str] = []
    cache_summary = aggregate.get("cache_summary", {})
    if not isinstance(cache_summary, dict):
        return warnings

    if expect_cached_separation:
        sep = cache_summary.get("separation", {})
        if isinstance(sep, dict):
            miss = int(sep.get("miss_count", 0) or 0)
            total = int(sep.get("total", 0) or 0)
            unknown = int(sep.get("unknown_count", 0) or 0)
            cached_ratio = float(sep.get("cached_ratio", 0.0) or 0.0)
            if total == 0:
                warnings.append(
                    "Expected cached separation but no executed-song cache data was available "
                    "(results likely reused). Re-run with --rerun-completed."
                )
            elif miss > 0:
                warnings.append(
                    "Expected cached separation but misses were observed: "
                    f"{miss}/{total} miss, cached_ratio={cached_ratio:.3f}"
                )
            elif unknown > 0:
                warnings.append(
                    "Expected cached separation but cache state was unknown for "
                    f"{unknown}/{total} song(s)."
                )
            elif cached_ratio < 1.0:
                warnings.append(
                    "Expected cached separation but cached_ratio was below 1.0: "
                    f"{cached_ratio:.3f}"
                )

    if expect_cached_whisper:
        whisper = cache_summary.get("whisper", {})
        if isinstance(whisper, dict):
            miss = int(whisper.get("miss_count", 0) or 0)
            total = int(whisper.get("total", 0) or 0)
            unknown = int(whisper.get("unknown_count", 0) or 0)
            cached_ratio = float(whisper.get("cached_ratio", 0.0) or 0.0)
            if total == 0:
                warnings.append(
                    "Expected cached whisper but no executed-song cache data was available "
                    "(results likely reused). Re-run with --rerun-completed."
                )
            elif miss > 0:
                warnings.append(
                    "Expected cached whisper but misses were observed: "
                    f"{miss}/{total} miss, cached_ratio={cached_ratio:.3f}"
                )
            elif unknown > 0:
                warnings.append(
                    "Expected cached whisper but cache state was unknown for "
                    f"{unknown}/{total} song(s)."
                )
            elif cached_ratio < 1.0:
                warnings.append(
                    "Expected cached whisper but cached_ratio was below 1.0: "
                    f"{cached_ratio:.3f}"
                )
    return warnings


def _runtime_budget_warnings(
    *,
    aggregate: dict[str, Any],
    suite_wall_elapsed_sec: float,
    max_whisper_phase_share: float,
    max_alignment_phase_share: float,
    max_scheduler_overhead_sec: float,
) -> list[str]:
    warnings: list[str] = []
    phase_shares = aggregate.get("phase_shares_of_song_elapsed", {})
    if not isinstance(phase_shares, dict):
        phase_shares = {}

    if max_whisper_phase_share > 0:
        whisper_share_raw = phase_shares.get("whisper")
        whisper_share = (
            float(whisper_share_raw)
            if isinstance(whisper_share_raw, (int, float))
            else 0.0
        )
        if whisper_share > max_whisper_phase_share:
            warnings.append(
                "Whisper phase share exceeds budget: "
                f"{whisper_share:.3f} > {max_whisper_phase_share:.3f}"
            )

    if max_alignment_phase_share > 0:
        alignment_share_raw = phase_shares.get("alignment")
        alignment_share = (
            float(alignment_share_raw)
            if isinstance(alignment_share_raw, (int, float))
            else 0.0
        )
        if alignment_share > max_alignment_phase_share:
            warnings.append(
                "Alignment phase share exceeds budget: "
                f"{alignment_share:.3f} > {max_alignment_phase_share:.3f}"
            )

    if max_scheduler_overhead_sec > 0:
        sum_song_elapsed = float(aggregate.get("sum_song_elapsed_sec", 0.0) or 0.0)
        scheduler_overhead = max(0.0, suite_wall_elapsed_sec - sum_song_elapsed)
        if scheduler_overhead > max_scheduler_overhead_sec:
            warnings.append(
                "Scheduler overhead exceeds budget: "
                f"{scheduler_overhead:.2f}s > {max_scheduler_overhead_sec:.2f}s"
            )

    return warnings


def _write_markdown_summary(  # noqa: C901
    path: Path,
    *,
    run_id: str,
    manifest: Path,
    aggregate: dict[str, Any],
    songs: list[dict[str, Any]],
) -> None:
    def _fmt_num(value: Any, *, unit: str = "", digits: int = 3) -> str:
        if isinstance(value, (int, float)):
            return f"{float(value):.{digits}f}{unit}"
        return "-"

    lines: list[str] = []
    lines.append("# Benchmark Timing Quality Report")
    lines.append("")
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Manifest: `{manifest}`")
    lines.append(
        f"- Songs: `{aggregate['songs_succeeded']}/{aggregate['songs_total']}` succeeded"
    )
    lines.append(
        f"- Mean DTW line coverage: `{_fmt_num(aggregate.get('dtw_line_coverage_mean'))}`"
    )
    lines.append(
        "- Gold metric coverage: "
        f"`{aggregate.get('gold_metric_song_count', 0)}/{aggregate.get('songs_succeeded', 0)}` songs, "
        f"`{aggregate.get('gold_comparable_word_count_total', 0)}/{aggregate.get('gold_word_count_total', 0)}` words"
    )
    curated_canary_song_count = int(aggregate.get("curated_canary_song_count", 0) or 0)
    if curated_canary_song_count > 0:
        lines.append(
            "- Curated canary gold coverage: "
            f"`{curated_canary_song_count}/{aggregate.get('songs_succeeded', 0)}` songs, "
            f"`{aggregate.get('curated_canary_gold_comparable_word_count_total', 0)}`/"
            f"`{aggregate.get('curated_canary_gold_word_count_total', 0)}` words"
        )
        lines.append(
            "- Curated canary primary metric (avg abs word-start delta): "
            f"`{_fmt_num(aggregate.get('curated_canary_avg_abs_word_start_delta_sec_word_weighted_mean'), unit='s')}` "
            "(word-weighted)"
        )
        lines.append(
            "- Curated canary p95 abs word-start delta: "
            f"`{_fmt_num(aggregate.get('curated_canary_gold_start_p95_abs_sec_mean'), unit='s')}`"
        )
        lines.append(
            "- Curated canary mean abs line-duration delta: "
            f"`{_fmt_num(aggregate.get('curated_canary_gold_line_duration_mean_abs_sec_mean'), unit='s')}`"
        )
        pre_whisper_gold_mean = aggregate.get(
            "curated_canary_gold_pre_whisper_start_mean_abs_sec_mean"
        )
        if pre_whisper_gold_mean is not None:
            lines.append(
                "- Curated canary pre-Whisper gold start delta: "
                f"`{_fmt_num(pre_whisper_gold_mean, unit='s')}`"
            )
        nearest_onset_mean = aggregate.get(
            "curated_canary_gold_nearest_onset_start_mean_abs_sec_mean"
        )
        nearest_onset_non_interjection_mean = aggregate.get(
            "curated_canary_gold_nearest_onset_start_non_interjection_mean_abs_sec_mean"
        )
        if (
            nearest_onset_mean is not None
            or nearest_onset_non_interjection_mean is not None
        ):
            lines.append(
                "- Curated canary nearest-onset start delta: "
                f"`{_fmt_num(nearest_onset_mean, unit='s')}` overall, "
                f"`{_fmt_num(nearest_onset_non_interjection_mean, unit='s')}` non-interjection"
            )
        later_onset_choice_count = aggregate.get(
            "curated_canary_gold_later_onset_choice_line_count_total"
        )
        later_onset_choice_mean = aggregate.get(
            "curated_canary_gold_later_onset_choice_mean_improvement_sec_mean"
        )
        if later_onset_choice_count:
            lines.append(
                "- Curated canary later-onset choice opportunities: "
                f"`{later_onset_choice_count}` lines, "
                f"`{_fmt_num(later_onset_choice_mean, unit='s')}` mean potential improvement"
            )
        downstream_regression_count = aggregate.get(
            "curated_canary_gold_downstream_regression_line_count_total"
        )
        downstream_regression_mean = aggregate.get(
            "curated_canary_gold_downstream_regression_mean_improvement_sec_mean"
        )
        if downstream_regression_count:
            lines.append(
                "- Curated canary downstream gold regressions: "
                f"`{downstream_regression_count}` lines, "
                f"`{_fmt_num(downstream_regression_mean, unit='s')}` mean avoidable drift"
            )
        interjection_mean = aggregate.get(
            "curated_canary_gold_parenthetical_interjection_start_mean_abs_sec_mean"
        )
        interjection_p95 = aggregate.get(
            "curated_canary_gold_parenthetical_interjection_start_p95_abs_sec_mean"
        )
        if interjection_mean is not None or interjection_p95 is not None:
            lines.append(
                "- Curated canary parenthetical-interjection start delta: "
                f"`{_fmt_num(interjection_mean, unit='s')}` mean, "
                f"`{_fmt_num(interjection_p95, unit='s')}` p95"
            )
        watchlist = aggregate.get("curated_canary_reference_watchlist", [])
        if isinstance(watchlist, list) and watchlist:
            lines.append(
                "- Curated reference-divergence watchlist: "
                + ", ".join(f"`{item}`" for item in watchlist)
            )
    lines.append(
        "- Gold comparable word coverage ratio: "
        f"`{aggregate.get('gold_word_coverage_ratio_total', 0.0):.3f}`"
    )
    lines.append(
        "- Reference divergence suspects (first-pass heuristic): "
        f"`{aggregate.get('reference_divergence_suspected_count', 0)}` "
        f"({aggregate.get('reference_divergence_suspected_ratio', 0.0):.3f})"
    )
    diagnosis_counts = aggregate.get("quality_diagnosis_counts", {})
    if isinstance(diagnosis_counts, dict) and diagnosis_counts:
        diagnosis_summary = ", ".join(
            f"{k}={v}" for k, v in sorted(diagnosis_counts.items())
        )
        lines.append(f"- Per-song quality diagnosis: `{diagnosis_summary}`")
    lexical_review_song_count = int(aggregate.get("lexical_review_song_count", 0) or 0)
    if lexical_review_song_count > 0:
        lines.append(
            "- Lexical-review hotspots: "
            f"`{lexical_review_song_count}` song(s), "
            f"hook-boundary songs `{int(aggregate.get('lexical_hook_boundary_variant_song_count', 0) or 0)}`, "
            f"hook-boundary ratio `{_fmt_num(aggregate.get('lexical_hook_boundary_variant_ratio_mean'))}`, "
            f"truncation-pattern ratio "
            f"`{_fmt_num(aggregate.get('lexical_truncation_pattern_ratio_mean'))}`, "
            f"repetitive-phrase ratio "
            f"`{_fmt_num(aggregate.get('lexical_repetitive_phrase_line_ratio_mean'))}`"
        )
        hook_boundary_eligibility = aggregate.get(
            "agreement_hook_boundary_eligibility_ratio_mean"
        )
        hook_boundary_text_similarity = aggregate.get(
            "agreement_hook_boundary_text_similarity_mean"
        )
        if (
            hook_boundary_eligibility is not None
            or hook_boundary_text_similarity is not None
        ):
            lines.append(
                "- Hook-normalized agreement signal: "
                f"eligibility `{_fmt_num(hook_boundary_eligibility)}`, "
                f"text similarity `{_fmt_num(hook_boundary_text_similarity)}`"
            )
    timing_band_counts = aggregate.get("timing_quality_band_counts", {})
    if isinstance(timing_band_counts, dict) and timing_band_counts:
        timing_band_summary = ", ".join(
            f"{k}={v}" for k, v in sorted(timing_band_counts.items())
        )
        lines.append(f"- Timing quality bands: `{timing_band_summary}`")
    lines.append(
        "- Primary metric (avg abs word-start delta): "
        f"`{_fmt_num(aggregate.get('avg_abs_word_start_delta_sec_word_weighted_mean'), unit='s')}` (word-weighted)"
    )
    primary_metric_ex_ref = _fmt_num(
        aggregate.get(
            "avg_abs_word_start_delta_sec_word_weighted_mean_excluding_reference_divergence"
        ),
        unit="s",
    )
    lines.append(
        "- Primary metric excluding reference-divergence suspects: "
        f"`{primary_metric_ex_ref}` "
        "(word-weighted)"
    )
    lines.append(
        "- Secondary metric (avg abs word-end delta): "
        f"`{_fmt_num(aggregate.get('gold_end_mean_abs_sec_mean'), unit='s')}`"
    )
    lines.append(
        "- Secondary metric (avg abs line-duration delta): "
        f"`{_fmt_num(aggregate.get('gold_line_duration_mean_abs_sec_mean'), unit='s')}`"
    )
    lines.append(
        "- Secondary metric (mean abs nearest-onset start delta): "
        f"`{_fmt_num(aggregate.get('gold_nearest_onset_start_mean_abs_sec_mean'), unit='s')}`"
    )
    lines.append(
        f"- Mean DTW word coverage: `{_fmt_num(aggregate.get('dtw_word_coverage_mean'))}`"
    )
    lines.append(
        "- Mean DTW phonetic similarity coverage: "
        f"`{_fmt_num(aggregate.get('dtw_phonetic_similarity_coverage_mean'))}`"
    )
    lines.append(
        f"- Low-confidence line ratio: `{aggregate['low_confidence_ratio_total']:.3f}`"
    )
    lines.append(
        "- Agreement coverage ratio: "
        f"`{_fmt_num(aggregate.get('agreement_coverage_ratio_total'))}`"
    )
    lines.append(
        "- Agreement comparability (eligible/matched anchor lines): "
        f"`{aggregate.get('agreement_eligible_lines_total', 0)}`/"
        f"`{aggregate.get('agreement_matched_anchor_lines_total', 0)}`"
    )
    lines.append(
        "- Mean agreement text similarity: "
        f"`{_fmt_num(aggregate.get('agreement_text_similarity_mean'))}`"
    )
    lines.append(
        "- Mean abs line-start agreement error (comparable lines): "
        f"`{_fmt_num(aggregate.get('agreement_start_mean_abs_sec_mean'), unit='s')}`"
    )
    lines.append(
        "- p95 abs line-start agreement error (comparable lines): "
        f"`{_fmt_num(aggregate.get('agreement_start_p95_abs_sec_mean'), unit='s')}`"
    )
    lines.append(
        "- Poor agreement ratio (>0.8s): "
        f"`{aggregate.get('agreement_bad_ratio_total', 0.0):.3f}`"
    )
    lines.append(
        "- Severe agreement ratio (>1.5s): "
        f"`{aggregate.get('agreement_severe_ratio_total', 0.0):.3f}`"
    )
    agreement_skip_totals = aggregate.get("agreement_skip_reason_totals", {})
    if isinstance(agreement_skip_totals, dict) and agreement_skip_totals:
        top_skip = list(agreement_skip_totals.items())[:5]
        lines.append(
            "- Common agreement skip reasons: "
            + ", ".join(f"`{k}` x{v}" for k, v in top_skip)
        )
    lines.append(
        "- Whisper-anchor diagnostic p95 abs line-start delta: "
        f"`{_fmt_num(aggregate.get('whisper_anchor_start_p95_abs_sec_mean'), unit='s')}`"
    )
    lines.append(
        "- DTW metric coverage: "
        f"`{aggregate.get('dtw_metric_song_count', 0)}/{aggregate.get('songs_succeeded', 0)}` "
        f"songs, `{aggregate.get('dtw_metric_line_count', 0)}/{aggregate.get('line_count_total', 0)}` lines"
    )
    lines.append(
        "- Mean DTW line coverage (line-weighted): "
        f"`{_fmt_num(aggregate.get('dtw_line_coverage_line_weighted_mean'))}`"
    )
    lines.append(
        "- Timing quality score (line-weighted): "
        f"`{_fmt_num(aggregate.get('timing_quality_score_line_weighted_mean'))}`"
    )
    cache_summary = aggregate.get("cache_summary", {})
    if isinstance(cache_summary, dict):
        sep = cache_summary.get("separation")
        if isinstance(sep, dict):
            lines.append(
                "- Separation cache ratio: "
                f"`{float(sep.get('cached_ratio', 0.0)):.3f}` "
                f"({int(sep.get('cached_count', 0) or 0)}/{int(sep.get('total', 0) or 0)})"
            )
    local_cache_events = int(
        aggregate.get("local_transcribe_cache_events_total", 0) or 0
    )
    if local_cache_events > 0:
        lines.append(
            "- Local transcription cache reuse (within song run): "
            f"`{float(aggregate.get('local_transcribe_cache_hit_ratio', 0.0) or 0.0):.3f}` "
            f"({int(aggregate.get('local_transcribe_cache_hits_total', 0) or 0)}/{local_cache_events} hits)"
        )
    phase_totals = aggregate.get("phase_totals_sec", {})
    if isinstance(phase_totals, dict) and phase_totals:
        top_phase = max(phase_totals, key=lambda key: float(phase_totals.get(key, 0.0)))
        lines.append(
            "- Slowest phase by summed song time: "
            f"`{top_phase}` (`{float(phase_totals[top_phase]):.2f}s`)"
        )
    diag_summary = aggregate.get("alignment_diagnostics_summary", {})
    if isinstance(diag_summary, dict):
        method_counts = diag_summary.get("alignment_method_counts", {})
        provider_counts = diag_summary.get("lyrics_source_provider_counts", {})
        policy_hint_counts = diag_summary.get("alignment_policy_hint_counts", {})
        source_selection_mode_counts = diag_summary.get(
            "lyrics_source_selection_mode_counts", {}
        )
        source_routing_skip_reason_counts = diag_summary.get(
            "lyrics_source_routing_skip_reason_counts", {}
        )
        source_disagreement_song_count = int(
            diag_summary.get("lyrics_source_disagreement_song_count", 0) or 0
        )
        source_audio_scoring_song_count = int(
            diag_summary.get("lyrics_source_audio_scoring_song_count", 0) or 0
        )
        fallback_decision_counts = diag_summary.get("fallback_map_decision_counts", {})
        fallback_attempted_count = int(
            diag_summary.get("fallback_map_attempted_count", 0) or 0
        )
        fallback_selected_count = int(
            diag_summary.get("fallback_map_selected_count", 0) or 0
        )
        fallback_rejected_count = int(
            diag_summary.get("fallback_map_rejected_count", 0) or 0
        )
        issue_totals = diag_summary.get("issue_tag_totals", {})
        if isinstance(method_counts, dict) and method_counts:
            lines.append(
                "- Alignment methods used: "
                + ", ".join(f"`{k}` x{v}" for k, v in sorted(method_counts.items()))
            )
        if isinstance(provider_counts, dict) and provider_counts:
            lines.append(
                "- Lyrics providers seen: "
                + ", ".join(f"`{k}` x{v}" for k, v in sorted(provider_counts.items()))
            )
        if (
            isinstance(source_selection_mode_counts, dict)
            and source_selection_mode_counts
        ):
            lines.append(
                "- Lyrics source selection modes: "
                + ", ".join(
                    f"`{k}` x{v}"
                    for k, v in sorted(source_selection_mode_counts.items())
                )
            )
        if (
            isinstance(source_routing_skip_reason_counts, dict)
            and source_routing_skip_reason_counts
        ):
            lines.append(
                "- Lyrics source routing skip reasons: "
                + ", ".join(
                    f"`{k}` x{v}"
                    for k, v in sorted(source_routing_skip_reason_counts.items())
                )
            )
        if source_disagreement_song_count or source_audio_scoring_song_count:
            lines.append(
                "- Lyrics source routing: "
                f"`{source_disagreement_song_count}` disagreement-triggered, "
                f"`{source_audio_scoring_song_count}` audio-scored"
            )
        if isinstance(policy_hint_counts, dict) and policy_hint_counts:
            lines.append(
                "- Alignment policy hints: "
                + ", ".join(
                    f"`{k}` x{v}" for k, v in sorted(policy_hint_counts.items())
                )
            )
        lines.append(
            "- Fallback map decisions: "
            f"attempted={fallback_attempted_count}, "
            f"selected={fallback_selected_count}, "
            f"rejected={fallback_rejected_count}"
        )
        if isinstance(fallback_decision_counts, dict) and fallback_decision_counts:
            lines.append(
                "- Fallback map reason counts: "
                + ", ".join(
                    f"`{k}` x{v}" for k, v in sorted(fallback_decision_counts.items())
                )
            )
        if isinstance(issue_totals, dict) and issue_totals:
            top_issue_tags = sorted(
                issue_totals.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))
            )[:5]
            lines.append(
                "- Common timing report issue tags: "
                + ", ".join(f"`{k}` x{v}" for k, v in top_issue_tags)
            )
    lines.append("")
    lines.append("## Per-song")
    lines.append("")
    lines.append(
        "| Song | Status | Gold words | Gold cov | Word start abs mean | Word start p95 | Word end abs mean | Alignment | DTW line | DTW word | Fallback map | Policy hint | Ref mismatch suspect | Diagnosis | Elapsed |"  # noqa: E501
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for song in songs:
        metrics = song.get("metrics", {})
        alignment_diag = song.get("alignment_diagnostics", {})
        fallback_cell = "-"
        if isinstance(alignment_diag, dict):
            attempted = bool(alignment_diag.get("fallback_map_attempted", False))
            selected = bool(alignment_diag.get("fallback_map_selected", False))
            rejected = bool(alignment_diag.get("fallback_map_rejected", False))
            reason = str(
                alignment_diag.get("fallback_map_decision_reason", "") or "unknown"
            )
            gain = alignment_diag.get("fallback_map_score_gain")
            if attempted:
                if selected:
                    fallback_cell = f"selected ({reason}, gain={_fmt_num(gain)})"
                elif rejected:
                    fallback_cell = f"rejected ({reason}, gain={_fmt_num(gain)})"
                else:
                    fallback_cell = f"attempted ({reason})"
            else:
                fallback_cell = f"no ({reason})"
        policy_hint = song.get("alignment_policy_hint", {})
        policy_hint_cell = "-"
        if isinstance(policy_hint, dict) and policy_hint:
            hint = str(policy_hint.get("hint", "none") or "none")
            conf = str(policy_hint.get("confidence", "low") or "low")
            if hint != "none":
                policy_hint_cell = f"{hint} ({conf})"
            else:
                policy_hint_cell = "none"
        ref_div = song.get("reference_divergence", {})
        ref_div_cell = "-"
        if isinstance(ref_div, dict) and ref_div:
            if bool(ref_div.get("suspected")):
                ref_div_cell = (
                    f"yes ({ref_div.get('confidence', 'low')}, "
                    f"{float(ref_div.get('score', 0.0) or 0.0):.2f})"
                )
            else:
                ref_div_cell = f"no ({float(ref_div.get('score', 0.0) or 0.0):.2f})"
        diagnosis = song.get("quality_diagnosis", {})
        diagnosis_cell = "-"
        if isinstance(diagnosis, dict) and diagnosis:
            verdict = str(diagnosis.get("verdict", "unknown") or "unknown")
            conf = str(diagnosis.get("confidence", "low") or "low")
            diagnosis_cell = f"{verdict} ({conf})"
        lines.append(
            "| "
            + f"{song['artist']} - {song['title']} | "
            + f"{song['status']} | "
            + f"{metrics.get('gold_comparable_word_count', '-')}/{metrics.get('gold_word_count', '-')}"
            + " | "
            + f"{metrics.get('gold_word_coverage_ratio', '-')}"
            + " | "
            + f"{metrics.get('avg_abs_word_start_delta_sec', '-')}"
            + " | "
            + f"{metrics.get('gold_start_p95_abs_sec', '-')}"
            + " | "
            + f"{metrics.get('gold_end_mean_abs_sec', '-')}"
            + " | "
            + f"{metrics.get('alignment_method', '-')}"
            + " | "
            + f"{metrics.get('dtw_line_coverage', '-')}"
            + " | "
            + f"{metrics.get('dtw_word_coverage', '-')}"
            + " | "
            + fallback_cell
            + " | "
            + policy_hint_cell
            + " | "
            + ref_div_cell
            + " | "
            + diagnosis_cell
            + " | "
            + f"{song.get('elapsed_sec', '-')}"
            + "s |"
        )
    comparability_rows = aggregate.get("agreement_comparability_report", [])
    if isinstance(comparability_rows, list) and comparability_rows:
        lines.append("")
        lines.append("## Agreement Comparability")
        lines.append("")
        lines.append(
            "| Song | Eligible | Matched(anchor) | Matched(independent) | Eligible ratio | Match/Eligible | Top skip reasons |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for row in comparability_rows[:12]:
            skip_reasons = row.get("skip_reasons", {})
            skip_text = "-"
            if isinstance(skip_reasons, dict) and skip_reasons:
                pairs = list(skip_reasons.items())[:3]
                skip_text = ", ".join(f"{k}:{v}" for k, v in pairs)
            lines.append(
                "| "
                + f"{row.get('song', '-')} | "
                + f"{row.get('eligible_lines', 0)} | "
                + f"{row.get('matched_lines_anchor', 0)} | "
                + f"{row.get('matched_lines_independent', 0)} | "
                + f"{_fmt_num(row.get('eligibility_ratio'))} | "
                + f"{_fmt_num(row.get('match_ratio_within_eligible'))} | "
                + f"{skip_text} |"
            )
    lines.append("")
    hotspots = aggregate.get("quality_hotspots", {})
    if isinstance(hotspots, dict):
        low_dtw = hotspots.get("lowest_dtw_line_coverage", [])
        high_low_conf = hotspots.get("highest_low_confidence_ratio", [])
        low_agree_cov = hotspots.get("lowest_agreement_coverage_ratio", [])
        high_agree_p95 = hotspots.get("highest_agreement_start_p95_abs_sec", [])
        high_agree_bad = hotspots.get("highest_agreement_bad_ratio", [])
        high_agree_severe = hotspots.get("highest_agreement_severe_ratio", [])
        high_timing_quality = hotspots.get("highest_timing_quality_score", [])
        low_timing_quality = hotspots.get("lowest_timing_quality_score", [])
        high_gold_start = hotspots.get("highest_avg_abs_word_start_delta_sec", [])
        low_gold_cov = hotspots.get("lowest_gold_word_coverage_ratio", [])
        hook_boundary_rows = hotspots.get("lexical_hook_boundary_variants", [])
        ref_div_suspects = hotspots.get("reference_divergence_suspects", [])
        likely_ref_div = hotspots.get("likely_reference_divergence", [])
        likely_pipeline = hotspots.get("likely_pipeline_failure", [])
        diagnosis_rows = hotspots.get("quality_diagnosis", [])
        if (
            low_dtw
            or high_low_conf
            or low_agree_cov
            or high_agree_p95
            or high_agree_bad
            or high_agree_severe
            or high_timing_quality
            or low_timing_quality
            or high_gold_start
            or low_gold_cov
            or hook_boundary_rows
            or ref_div_suspects
            or likely_ref_div
            or likely_pipeline
            or diagnosis_rows
        ):
            lines.append("## Hotspots")
            lines.append("")
            if low_dtw:
                lines.append("- Lowest DTW line coverage:")
                for item in low_dtw:
                    if isinstance(item, dict):
                        lines.append(f"  - {item.get('song')}: {item.get('value')}")
            if high_low_conf:
                lines.append("- Highest low-confidence ratio:")
                for item in high_low_conf:
                    if isinstance(item, dict):
                        lines.append(f"  - {item.get('song')}: {item.get('value')}")
            if low_agree_cov:
                lines.append("- Lowest agreement coverage ratio:")
                for item in low_agree_cov:
                    if isinstance(item, dict):
                        lines.append(f"  - {item.get('song')}: {item.get('value')}")
            if high_agree_p95:
                lines.append("- Highest agreement start p95 error:")
                for item in high_agree_p95:
                    if isinstance(item, dict):
                        lines.append(f"  - {item.get('song')}: {item.get('value')}s")
            if high_agree_bad:
                lines.append("- Highest poor agreement ratio (>0.8s):")
                for item in high_agree_bad:
                    if isinstance(item, dict):
                        lines.append(f"  - {item.get('song')}: {item.get('value')}")
            if high_agree_severe:
                lines.append("- Highest severe agreement ratio (>1.5s):")
                for item in high_agree_severe:
                    if isinstance(item, dict):
                        lines.append(f"  - {item.get('song')}: {item.get('value')}")
            if high_timing_quality:
                lines.append("- Highest timing quality score:")
                for item in high_timing_quality:
                    if isinstance(item, dict):
                        lines.append(f"  - {item.get('song')}: {item.get('value')}")
            if low_timing_quality:
                lines.append("- Lowest timing quality score:")
                for item in low_timing_quality:
                    if isinstance(item, dict):
                        lines.append(f"  - {item.get('song')}: {item.get('value')}")
            if high_gold_start:
                lines.append("- Highest avg abs word-start delta (gold):")
                for item in high_gold_start:
                    if isinstance(item, dict):
                        lines.append(f"  - {item.get('song')}: {item.get('value')}s")
            if low_gold_cov:
                lines.append("- Lowest gold comparable word coverage ratio:")
                for item in low_gold_cov:
                    if isinstance(item, dict):
                        lines.append(f"  - {item.get('song')}: {item.get('value')}")
            if hook_boundary_rows:
                lines.append("- Highest hook-boundary lexical variant ratio:")
                for item in hook_boundary_rows:
                    if isinstance(item, dict):
                        lines.append(
                            "  - "
                            f"{item.get('song')}: {item.get('value')} "
                            f"(count={int(item.get('count', 0) or 0)})"
                        )
            if ref_div_suspects:
                lines.append("- Reference divergence suspects (heuristic):")
                for item in ref_div_suspects:
                    if isinstance(item, dict):
                        evidence = item.get("evidence") or []
                        if isinstance(evidence, list):
                            evidence_text = ", ".join(str(v) for v in evidence[:3])
                        else:
                            evidence_text = str(evidence)
                        lines.append(
                            "  - "
                            f"{item.get('song')}: score={item.get('score')} "
                            f"({item.get('confidence')}); {evidence_text}"
                        )
            if likely_ref_div:
                lines.append("- Triage ranking: likely reference divergence:")
                for item in likely_ref_div:
                    if isinstance(item, dict):
                        reasons = item.get("reasons") or []
                        reason_text = (
                            ", ".join(str(v) for v in reasons[:3])
                            if isinstance(reasons, list)
                            else str(reasons)
                        )
                        lines.append(
                            f"  - {item.get('song')}: score={item.get('score')} ({reason_text})"
                        )
            if likely_pipeline:
                lines.append("- Triage ranking: likely pipeline failure:")
                for item in likely_pipeline:
                    if isinstance(item, dict):
                        reasons = item.get("reasons") or []
                        reason_text = (
                            ", ".join(str(v) for v in reasons[:3])
                            if isinstance(reasons, list)
                            else str(reasons)
                        )
                        lines.append(
                            f"  - {item.get('song')}: score={item.get('score')} ({reason_text})"
                        )
            if diagnosis_rows:
                lines.append("- Quality diagnosis snapshot:")
                for item in diagnosis_rows:
                    if isinstance(item, dict):
                        reasons = item.get("reasons") or []
                        reason_text = (
                            ", ".join(str(v) for v in reasons[:2])
                            if isinstance(reasons, list)
                            else str(reasons)
                        )
                        lines.append(
                            "  - "
                            + f"{item.get('song')}: {item.get('verdict')} ({item.get('confidence')})"
                            + (f"; {reason_text}" if reason_text else "")
                        )
            lines.append("")
    if aggregate["failed_songs"]:
        lines.append("## Failures")
        lines.append("")
        for item in aggregate["failed_songs"]:
            lines.append(f"- {item}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_run_dir(
    *,
    output_root: Path,
    run_id: str | None,
    resume_run_dir: Path | None,
    resume_latest: bool,
) -> tuple[Path, str]:
    if resume_run_dir is not None:
        return resume_run_dir.resolve(), resume_run_dir.resolve().name

    if resume_latest:
        candidates = sorted(
            [p for p in output_root.resolve().glob("20*") if p.is_dir()],
            key=lambda p: (p.stat().st_mtime, p.name),
            reverse=True,
        )
        if candidates:
            return candidates[0], candidates[0].name

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_root.resolve() / run_id, run_id


def _song_result_path(run_dir: Path, index: int, slug: str) -> Path:
    return run_dir / f"{index:02d}_{slug}_result.json"


def _discover_cached_result_slugs(run_dir: Path) -> set[str]:
    slugs: set[str] = set()
    for path in run_dir.glob("*_result.json"):
        name = path.name
        if not name.endswith("_result.json"):
            continue
        parts = name.split("_", 1)
        if len(parts) != 2:
            continue
        slug = parts[1][: -len("_result.json")]
        if slug:
            slugs.add(slug)
    return slugs


def _load_song_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _load_run_metadata(run_dir: Path) -> dict[str, Any] | None:
    for name in ("benchmark_progress.json", "benchmark_report.json"):
        path = run_dir / name
        if not path.exists():
            continue
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(loaded, dict):
            return loaded
    return None


def _infer_gold_root_from_cached_results(run_dir: Path) -> Path | None:
    candidates: list[str] = []
    for path in sorted(run_dir.glob("*_result.json")):
        loaded = _load_song_result(path)
        if not loaded:
            continue
        run_signature = loaded.get("run_signature")
        if isinstance(run_signature, dict):
            value = run_signature.get("gold_root")
            if isinstance(value, str) and value.strip():
                candidates.append(value)
                continue
        gold_path = loaded.get("gold_path")
        if isinstance(gold_path, str) and gold_path.strip():
            candidates.append(str(Path(gold_path).resolve().parent))
    if not candidates:
        return None
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate] = counts.get(candidate, 0) + 1
    best = max(counts.items(), key=lambda item: item[1])[0]
    return Path(best).resolve()


def _inherit_resume_gold_root(args: argparse.Namespace, run_dir: Path) -> Path | None:
    if not args.aggregate_only or args.resume_run_dir is None:
        return None
    try:
        requested_gold_root = args.gold_root.resolve()
    except Exception:
        return None
    if requested_gold_root != DEFAULT_GOLD_ROOT.resolve():
        return None
    inferred_from_results = _infer_gold_root_from_cached_results(run_dir)
    if (
        inferred_from_results is not None
        and inferred_from_results != requested_gold_root
    ):
        return inferred_from_results
    metadata = _load_run_metadata(run_dir)
    if not metadata:
        return None
    options = metadata.get("options")
    if not isinstance(options, dict):
        return None
    prior_gold_root = options.get("gold_root")
    if not isinstance(prior_gold_root, str) or not prior_gold_root.strip():
        return None
    inherited = Path(prior_gold_root).resolve()
    return inherited if inherited != requested_gold_root else None


def _refresh_cached_metrics(
    record: dict[str, Any],
    *,
    index: int | None = None,
    song: BenchmarkSong | None = None,
    gold_root: Path = DEFAULT_GOLD_ROOT,
) -> dict[str, Any]:
    """Backfill metrics from timing report for legacy cached result files."""
    if str(record.get("status", "")) != "ok":
        return record
    report_path_raw = record.get("report_path")
    if not isinstance(report_path_raw, str) or not report_path_raw:
        return record
    report_path = Path(report_path_raw)
    if not report_path.exists():
        return record
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return record
    return _refresh_metrics_from_loaded_report(
        record,
        report=report,
        index=index,
        song=song,
        gold_root=gold_root,
    )


def _refresh_metrics_from_loaded_report(
    record: dict[str, Any],
    *,
    report: Any,
    index: int | None = None,
    song: BenchmarkSong | None = None,
    gold_root: Path = DEFAULT_GOLD_ROOT,
) -> dict[str, Any]:
    if not isinstance(report, dict):
        return record
    gold_doc = None
    if index is not None and song is not None:
        gold_doc = _load_gold_doc(index=index, song=song, gold_root=gold_root)
    song_audio_path = (
        _resolve_song_audio_path(song, gold_doc=gold_doc) if song is not None else None
    )
    record["metrics"] = _extract_song_metrics(
        report,
        gold_doc=gold_doc,
        audio_path=song_audio_path,
    )
    record["alignment_diagnostics"] = _extract_alignment_diagnostics(report)
    record["reference_divergence"] = _infer_reference_divergence_suspicion(
        record["metrics"],
        alignment_diagnostics=record.get("alignment_diagnostics"),
    )
    record["alignment_policy_hint"] = _infer_alignment_policy_hint(
        record["metrics"],
        alignment_diagnostics=record.get("alignment_diagnostics"),
        reference_divergence=record.get("reference_divergence"),
    )
    record["lexical_mismatch_diagnostics"] = _extract_lexical_mismatch_diagnostics(
        report,
        record["metrics"],
        alignment_policy_hint=record.get("alignment_policy_hint"),
    )
    record["quality_diagnosis"] = _classify_quality_diagnosis(
        record["metrics"],
        alignment_policy_hint=record.get("alignment_policy_hint"),
        reference_divergence=record.get("reference_divergence"),
        lexical_mismatch_diagnostics=record.get("lexical_mismatch_diagnostics"),
    )
    return record


def _shift_report_to_clip_window(
    report: dict[str, Any],
    *,
    song: BenchmarkSong,
    gold_doc: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not song.clip_id or song.audio_start_sec <= 0.0:
        return report
    if not isinstance(gold_doc, dict):
        return None
    gold_lines = gold_doc.get("lines")
    if not isinstance(gold_lines, list) or not gold_lines:
        return None
    clip_end = max(
        (float(line.get("end", 0.0)) for line in gold_lines if isinstance(line, dict)),
        default=0.0,
    )
    if clip_end <= 0.0:
        return None
    absolute_start = song.audio_start_sec
    # Give clip scoring a little slack at the tail so a line that legitimately
    # overlaps the clipped region is not dropped for ending a few frames late.
    absolute_end = absolute_start + clip_end + 1.0
    shifted_lines: list[dict[str, Any]] = []
    shift_keys = (
        "start",
        "end",
        "pre_whisper_start",
        "pre_whisper_end",
        "whisper_window_start",
        "whisper_window_end",
        "nearest_segment_start",
        "nearest_segment_end",
        "nearest_segment_start_end",
        "nearest_segment_end_start",
    )
    for line in report.get("lines", []):
        if not isinstance(line, dict):
            continue
        start = line.get("start")
        end = line.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        if float(end) <= absolute_start or float(start) >= absolute_end:
            continue
        shifted = dict(line)
        for key in shift_keys:
            value = shifted.get(key)
            if isinstance(value, (int, float)):
                clamped_value = min(max(float(value), absolute_start), absolute_end)
                shifted[key] = round(clamped_value - absolute_start, 3)
        words_out: list[dict[str, Any]] = []
        for word in line.get("words", []):
            if not isinstance(word, dict):
                continue
            ws = word.get("start")
            we = word.get("end")
            if not isinstance(ws, (int, float)) or not isinstance(we, (int, float)):
                continue
            if float(we) <= absolute_start or float(ws) >= absolute_end:
                continue
            shifted_word = dict(word)
            shifted_word["start"] = round(
                min(max(float(ws), absolute_start), absolute_end) - absolute_start, 3
            )
            shifted_word["end"] = round(
                min(max(float(we), absolute_start), absolute_end) - absolute_start, 3
            )
            words_out.append(shifted_word)
        if words_out:
            shifted["words"] = words_out
        window_words_out: list[dict[str, Any]] = []
        for word in line.get("whisper_window_words", []):
            if not isinstance(word, dict):
                continue
            shifted_word = dict(word)
            ws = word.get("start")
            we = word.get("end")
            if isinstance(ws, (int, float)):
                clamped_ws = min(max(float(ws), absolute_start), absolute_end)
                shifted_word["start"] = round(clamped_ws - absolute_start, 3)
            if isinstance(we, (int, float)):
                clamped_we = min(max(float(we), absolute_start), absolute_end)
                shifted_word["end"] = round(clamped_we - absolute_start, 3)
            window_words_out.append(shifted_word)
        if window_words_out:
            shifted["whisper_window_words"] = window_words_out
        shifted_lines.append(shifted)
    shifted_report = dict(report)
    shifted_report["lines"] = shifted_lines
    return shifted_report


def _cached_result_slug_candidates(song: BenchmarkSong) -> list[str]:
    candidates = [song.slug]
    if song.clip_id:
        candidates.append(song.base_slug)
    return candidates


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_aggregate_only_results(
    *,
    songs: list[BenchmarkSong],
    run_dir: Path,
    gold_root: Path,
    rebaseline: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    song_results: list[dict[str, Any]] = []
    skipped_missing: list[str] = []
    for index, song in enumerate(songs, start=1):
        report_path = run_dir / f"{index:02d}_{song.slug}_timing_report.json"
        result_path = _song_result_path(run_dir, index, song.slug)
        prior = _load_song_result(result_path)
        loaded_via_base_slug = False
        if prior is None:
            slug_matches: list[Path] = []
            for slug in _cached_result_slug_candidates(song):
                slug_matches.extend(sorted(run_dir.glob(f"*_{slug}_result.json")))
            deduped: list[Path] = []
            for match in slug_matches:
                if match not in deduped:
                    deduped.append(match)
            if len(deduped) == 1:
                result_path = deduped[0]
                prior = _load_song_result(result_path)
                loaded_via_base_slug = song.clip_id is not None and (
                    result_path.name.endswith(f"{song.base_slug}_result.json")
                )
        print(f"[{index}/{len(songs)}] {song.artist} - {song.title}")
        if prior is None:
            skipped_missing.append(f"{song.artist} - {song.title}")
            print("  -> skipped (missing cached result)")
            continue

        if loaded_via_base_slug and song.clip_id:
            report_raw = prior.get("report_path")
            report_doc: dict[str, Any] | None = None
            if isinstance(report_raw, str) and report_raw:
                report_path = Path(report_raw)
                if report_path.exists():
                    try:
                        loaded_report = json.loads(
                            report_path.read_text(encoding="utf-8")
                        )
                    except Exception:
                        loaded_report = None
                    gold_doc = _load_gold_doc(
                        index=index, song=song, gold_root=gold_root
                    )
                    if isinstance(loaded_report, dict):
                        report_doc = _shift_report_to_clip_window(
                            loaded_report, song=song, gold_doc=gold_doc
                        )
            if report_doc is None:
                skipped_missing.append(f"{song.artist} - {song.title}")
                print("  -> skipped (missing cached clip-compatible report)")
                continue
            prior = dict(prior)
            prior["clip_scored_from_full_song"] = True
            prior = _refresh_metrics_from_loaded_report(
                prior,
                report=report_doc,
                index=index,
                song=song,
                gold_root=gold_root,
            )
        else:
            prior = _refresh_cached_metrics(
                prior, index=index, song=song, gold_root=gold_root
            )
        if rebaseline and prior.get("status") == "ok":
            rebased_path = _rebaseline_song_from_report(
                index=index,
                song=song,
                report_path=Path(str(prior.get("report_path", report_path))),
                gold_root=gold_root,
            )
            prior["gold_rebaselined"] = rebased_path is not None
            if rebased_path is not None:
                prior["gold_path"] = str(rebased_path)
                print(f"  -> gold rebaselined: {rebased_path}")
        prior["result_reused"] = True
        prior["aggregate_only_recomputed"] = True
        _write_json(result_path, prior)
        song_results.append(prior)
        print(f"  -> {prior.get('status', 'unknown')} (aggregate-only)")
    return song_results, skipped_missing


def _write_checkpoint(
    *,
    run_id: str,
    run_dir: Path,
    manifest_path: Path,
    args: argparse.Namespace,
    song_results: list[dict[str, Any]],
    suite_elapsed: float,
) -> None:
    aggregate = _aggregate(song_results)
    suite_wall_elapsed = round(suite_elapsed, 2)
    sum_song_elapsed = float(aggregate.get("sum_song_elapsed_sec", 0.0) or 0.0)
    sum_song_elapsed_total = float(
        aggregate.get("sum_song_elapsed_total_sec", sum_song_elapsed) or 0.0
    )
    overhead_base = sum_song_elapsed_total if args.aggregate_only else sum_song_elapsed
    report_json = {
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "repo_root": str(REPO_ROOT),
        "started_at_utc": run_id,
        "elapsed_sec": suite_wall_elapsed,
        "suite_wall_elapsed_sec": suite_wall_elapsed,
        "sum_song_elapsed_sec": round(sum_song_elapsed, 2),
        "sum_song_elapsed_total_sec": round(sum_song_elapsed_total, 2),
        "scheduler_overhead_sec": round(suite_wall_elapsed - overhead_base, 2),
        "status": "running",
        "options": _build_common_report_options(args),
        "aggregate": aggregate,
        "songs": song_results,
    }
    _write_json(run_dir / "benchmark_progress.json", report_json)


def _build_common_report_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "offline": args.offline,
        "force": args.force,
        "strategy": args.strategy,
        "scenario": args.scenario,
        "whisper_map_lrc_dtw": not args.no_whisper_map_lrc_dtw,
        "timeout_sec": args.timeout_sec,
        "heartbeat_sec": args.heartbeat_sec,
        "match": args.match,
        "max_songs": args.max_songs,
        "min_dtw_song_coverage_ratio": args.min_dtw_song_coverage_ratio,
        "min_dtw_line_coverage_ratio": args.min_dtw_line_coverage_ratio,
        "min_timing_quality_score_line_weighted": (
            args.min_timing_quality_score_line_weighted
        ),
        "strict_quality_coverage": args.strict_quality_coverage,
        "agreement_baseline_report": (
            str(args.agreement_baseline_report.resolve())
            if args.agreement_baseline_report
            else None
        ),
        "min_agreement_coverage_gain_for_bad_ratio_warning": (
            args.min_agreement_coverage_gain_for_bad_ratio_warning
        ),
        "max_agreement_bad_ratio_increase_on_coverage_gain": (
            args.max_agreement_bad_ratio_increase_on_coverage_gain
        ),
        "strict_agreement_tradeoff": args.strict_agreement_tradeoff,
        "expect_cached_separation": args.expect_cached_separation,
        "expect_cached_whisper": args.expect_cached_whisper,
        "strict_cache_expectations": args.strict_cache_expectations,
        "rebaseline": args.rebaseline,
        "gold_root": str(args.gold_root.resolve()),
        "aggregate_only": args.aggregate_only,
    }


def _build_final_report_options(
    args: argparse.Namespace, *, aggregate_only_skipped_missing_count: int
) -> dict[str, Any]:
    options = _build_common_report_options(args)
    options.update(
        {
            "max_whisper_phase_share": args.max_whisper_phase_share,
            "max_alignment_phase_share": args.max_alignment_phase_share,
            "max_scheduler_overhead_sec": args.max_scheduler_overhead_sec,
            "strict_runtime_budgets": args.strict_runtime_budgets,
            "aggregate_only_skipped_missing_count": (
                aggregate_only_skipped_missing_count
            ),
        }
    )
    return options


def _compute_suite_elapsed_accounting(
    *,
    aggregate: dict[str, Any],
    measured_suite_elapsed: float,
    aggregate_only: bool,
) -> tuple[float, float, float, float]:
    measured_suite_elapsed = round(float(measured_suite_elapsed), 2)
    sum_song_elapsed = float(aggregate.get("sum_song_elapsed_sec", 0.0) or 0.0)
    sum_song_elapsed_total = float(
        aggregate.get("sum_song_elapsed_total_sec", sum_song_elapsed) or 0.0
    )
    suite_elapsed = (
        round(max(measured_suite_elapsed, sum_song_elapsed_total), 2)
        if aggregate_only
        else measured_suite_elapsed
    )
    overhead_base = sum_song_elapsed_total if aggregate_only else sum_song_elapsed
    return suite_elapsed, sum_song_elapsed, sum_song_elapsed_total, overhead_base


def _build_final_report_json(
    *,
    run_id: str,
    manifest_path: Path,
    args: argparse.Namespace,
    aggregate: dict[str, Any],
    song_results: list[dict[str, Any]],
    suite_elapsed: float,
    sum_song_elapsed: float,
    sum_song_elapsed_total: float,
    overhead_base: float,
    quality_warnings: list[str],
    cache_warnings: list[str],
    runtime_warnings: list[str],
    run_warnings: list[str],
    aggregate_only_skipped_missing: list[str],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "repo_root": str(REPO_ROOT),
        "started_at_utc": run_id,
        "elapsed_sec": suite_elapsed,
        "suite_wall_elapsed_sec": suite_elapsed,
        "sum_song_elapsed_sec": round(sum_song_elapsed, 2),
        "sum_song_elapsed_total_sec": round(sum_song_elapsed_total, 2),
        "scheduler_overhead_sec": round(suite_elapsed - overhead_base, 2),
        "options": _build_final_report_options(
            args,
            aggregate_only_skipped_missing_count=len(aggregate_only_skipped_missing),
        ),
        "status": "finished_with_warnings" if run_warnings else "finished",
        "quality_warnings": quality_warnings,
        "cache_warnings": cache_warnings,
        "runtime_warnings": runtime_warnings,
        "warnings": run_warnings,
        "aggregate_only_skipped_missing_songs": aggregate_only_skipped_missing,
        "aggregate": aggregate,
        "songs": song_results,
    }


def _compute_run_warnings(
    *,
    args: argparse.Namespace,
    aggregate: dict[str, Any],
    suite_elapsed: float,
    baseline_aggregate: dict[str, Any] | None,
    aggregate_only_skipped_missing: list[str],
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    quality_warnings = _quality_coverage_warnings(
        aggregate=aggregate,
        dtw_enabled=(args.strategy == "hybrid_dtw" and not args.no_whisper_map_lrc_dtw),
        min_song_coverage_ratio=args.min_dtw_song_coverage_ratio,
        min_line_coverage_ratio=args.min_dtw_line_coverage_ratio,
        min_timing_quality_score_line_weighted=(
            args.min_timing_quality_score_line_weighted
        ),
        suite_wall_elapsed_sec=suite_elapsed,
    )
    agreement_tradeoff_warnings = _agreement_tradeoff_warnings(
        aggregate=aggregate,
        baseline_aggregate=baseline_aggregate,
        min_coverage_gain=args.min_agreement_coverage_gain_for_bad_ratio_warning,
        max_bad_ratio_increase=(args.max_agreement_bad_ratio_increase_on_coverage_gain),
    )
    quality_warnings.extend(agreement_tradeoff_warnings)
    cache_warnings = _cache_expectation_warnings(
        aggregate=aggregate,
        expect_cached_separation=args.expect_cached_separation,
        expect_cached_whisper=args.expect_cached_whisper,
    )
    runtime_warnings = _runtime_budget_warnings(
        aggregate=aggregate,
        suite_wall_elapsed_sec=suite_elapsed,
        max_whisper_phase_share=args.max_whisper_phase_share,
        max_alignment_phase_share=args.max_alignment_phase_share,
        max_scheduler_overhead_sec=args.max_scheduler_overhead_sec,
    )
    run_warnings = quality_warnings + cache_warnings + runtime_warnings
    if aggregate_only_skipped_missing:
        run_warnings.append(
            "Aggregate-only skipped songs without cached results: "
            f"{len(aggregate_only_skipped_missing)}"
        )
    return (
        quality_warnings,
        agreement_tradeoff_warnings,
        cache_warnings,
        runtime_warnings,
        run_warnings,
    )


def _derive_run_status(aggregate: dict[str, Any], run_warnings: list[str]) -> str:
    status = "OK" if aggregate.get("songs_failed", 0) == 0 else "FAIL"
    if status == "OK" and run_warnings:
        return "WARN"
    return status


def _determine_exit_code(
    *,
    aggregate: dict[str, Any],
    args: argparse.Namespace,
    quality_warnings: list[str],
    agreement_tradeoff_warnings: list[str],
    cache_warnings: list[str],
    runtime_warnings: list[str],
) -> int:
    if aggregate.get("songs_failed", 0) > 0:
        return 2
    if quality_warnings and args.strict_quality_coverage:
        return 3
    if agreement_tradeoff_warnings and args.strict_agreement_tradeoff:
        return 6
    if cache_warnings and args.strict_cache_expectations:
        return 4
    if runtime_warnings and args.strict_runtime_budgets:
        return 5
    return 0


def _load_baseline_aggregate(
    report_arg: Path | None,
) -> dict[str, Any] | None:
    if report_arg is None:
        return None
    baseline_path = report_arg.expanduser().resolve()
    if baseline_path.is_dir():
        baseline_path = baseline_path / "benchmark_report.json"
    baseline_doc = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_raw = baseline_doc.get("aggregate")
    return baseline_raw if isinstance(baseline_raw, dict) else None


def _print_run_summary(
    *,
    status: str,
    run_dir: Path,
    json_path: Path,
    md_path: Path,
    aggregate: dict[str, Any],
    suite_elapsed: float,
    sum_song_elapsed: float,
    overhead_base: float,
    run_warnings: list[str],
) -> None:
    print(f"benchmark_suite: {status}")
    print(f"- run_dir: {run_dir}")
    print(f"- json: {json_path}")
    print(f"- markdown: {md_path}")
    print(
        "- success: "
        f"{aggregate['songs_succeeded']}/{aggregate['songs_total']} "
        f"({aggregate['success_rate'] * 100:.1f}%)"
    )
    print(
        "- mean metrics: "
        f"gold_start_abs_mean_weighted={(aggregate.get('avg_abs_word_start_delta_sec_word_weighted_mean') or 0.0):.3f}s, "
        "gold_start_abs_mean_weighted_ex_ref_div="
        f"{(aggregate.get('avg_abs_word_start_delta_sec_word_weighted_mean_excluding_reference_divergence') or 0.0):.3f}s, "
        f"gold_start_p95_mean={(aggregate.get('gold_start_p95_abs_sec_mean') or 0.0):.3f}s, "
        f"gold_end_abs_mean={(aggregate.get('gold_end_mean_abs_sec_mean') or 0.0):.3f}s, "
        f"gold_cov={(aggregate.get('gold_word_coverage_ratio_total') or 0.0):.3f}, "
        f"dtw_line={(aggregate.get('dtw_line_coverage_mean') or 0.0):.3f}, "
        f"dtw_line_weighted={(aggregate.get('dtw_line_coverage_line_weighted_mean') or 0.0):.3f}, "
        f"dtw_word={(aggregate.get('dtw_word_coverage_mean') or 0.0):.3f}, "
        f"phonetic={(aggregate.get('dtw_phonetic_similarity_coverage_mean') or 0.0):.3f}, "
        f"low_conf_ratio={(aggregate.get('low_confidence_ratio_total') or 0.0):.3f}, "
        f"agreement_cov={(aggregate.get('agreement_coverage_ratio_total') or 0.0):.3f}, "
        f"agreement_text_sim={(aggregate.get('agreement_text_similarity_mean') or 0.0):.3f}, "
        f"agreement_start_abs_mean={(aggregate.get('agreement_start_mean_abs_sec_mean') or 0.0):.3f}s, "
        f"agreement_start_p95={(aggregate.get('agreement_start_p95_abs_sec_mean') or 0.0):.3f}s, "
        f"agreement_bad_ratio={(aggregate.get('agreement_bad_ratio_total') or 0.0):.3f}, "
        f"agreement_severe_ratio={(aggregate.get('agreement_severe_ratio_total') or 0.0):.3f}"
    )
    print(
        "- gold coverage: "
        f"songs={aggregate.get('gold_metric_song_coverage_ratio', 0.0):.3f}, "
        f"words={aggregate.get('gold_comparable_word_count_total', 0)}/{aggregate.get('gold_word_count_total', 0)}"
    )
    print(
        "- dtw coverage: "
        f"songs={aggregate['dtw_metric_song_coverage_ratio']:.3f}, "
        f"lines={aggregate['dtw_metric_line_coverage_ratio']:.3f}"
    )
    print(
        "- elapsed: "
        f"suite_wall={suite_elapsed:.2f}s, "
        f"sum_song={sum_song_elapsed:.2f}s, "
        f"overhead={suite_elapsed - overhead_base:.2f}s"
    )
    cache_summary = aggregate.get("cache_summary", {})
    if isinstance(cache_summary, dict):
        sep = cache_summary.get("separation")
        if isinstance(sep, dict):
            print(
                "- separation cache: "
                f"cached_ratio={float(sep.get('cached_ratio', 0.0) or 0.0):.3f}, "
                f"miss_count={int(sep.get('miss_count', 0) or 0)}"
            )
    if run_warnings:
        print("- warnings:")
        for warning in run_warnings:
            print(f"  - {warning}")


def _persist_final_report_outputs(
    *,
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    manifest_path: Path,
    aggregate: dict[str, Any],
    song_results: list[dict[str, Any]],
    report_json: dict[str, Any],
) -> tuple[Path, Path]:
    json_path = run_dir / "benchmark_report.json"
    md_path = run_dir / "benchmark_report.md"
    json_path.write_text(
        json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_json(run_dir / "benchmark_progress.json", report_json)
    _write_markdown_summary(
        md_path,
        run_id=run_id,
        manifest=manifest_path,
        aggregate=aggregate,
        songs=song_results,
    )
    latest = args.output_root.resolve() / "latest.json"
    latest.write_text(str(json_path) + "\n", encoding="utf-8")
    return json_path, md_path


def _build_run_signature(
    args: argparse.Namespace, manifest_path: Path
) -> dict[str, Any]:
    gold_root = getattr(args, "gold_root", None)
    return {
        "manifest_path": str(manifest_path),
        "offline": bool(args.offline),
        "force": bool(args.force),
        "strategy": str(args.strategy),
        "scenario": str(args.scenario),
        "whisper_map_lrc_dtw": not bool(args.no_whisper_map_lrc_dtw),
        "cache_dir": str(args.cache_dir.resolve()) if args.cache_dir else None,
        "gold_root": str(gold_root.resolve()) if gold_root else None,
        "min_timing_quality_score_line_weighted": float(
            args.min_timing_quality_score_line_weighted
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Benchmark manifest YAML path",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory where benchmark run folders are created",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache dir passed to y2karaoke via --work-dir",
    )
    parser.add_argument(
        "--gold-root",
        type=Path,
        default=DEFAULT_GOLD_ROOT,
        help="Directory containing per-song *.gold.json files",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Use a stable run id (folder name under --output-root)",
    )
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        default=None,
        help="Resume a specific existing run directory",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume the most recently modified run directory in --output-root",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter to invoke for each song run",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=1800,
        help="Per-song timeout (seconds)",
    )
    parser.add_argument(
        "--max-songs",
        type=int,
        default=0,
        help="Run only the first N songs (0 means all)",
    )
    parser.add_argument(
        "--match",
        type=str,
        default="",
        help="Only run songs whose artist/title matches this case-insensitive regex",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Pass --offline to y2karaoke generate (requires cached data)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to y2karaoke generate",
    )
    parser.add_argument(
        "--no-whisper-map-lrc-dtw",
        action="store_true",
        help="Disable --whisper-map-lrc-dtw while running benchmark songs",
    )
    parser.add_argument(
        "--strategy",
        choices=["hybrid_dtw", "hybrid_whisper", "whisper_only", "lrc_only"],
        default="hybrid_dtw",
        help=(
            "Benchmark strategy: hybrid_dtw (default), hybrid_whisper, "
            "whisper_only, or lrc_only."
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=["default", "lyrics_no_timing"],
        default="default",
        help=(
            "Optional benchmark scenario variant. "
            "'lyrics_no_timing' ignores provider LRC timestamps to isolate "
            "lyrics-text + audio alignment behavior."
        ),
    )
    parser.add_argument(
        "--evaluate-lyrics-sources",
        action="store_true",
        help=(
            "Pass --evaluate-lyrics to y2karaoke generate so all available timed "
            "lyrics sources are compared and the best-scoring source is selected."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at first failing song",
    )
    parser.add_argument(
        "--heartbeat-sec",
        type=int,
        default=30,
        help="Progress heartbeat interval while a song is running",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="When resuming, rerun songs previously marked failed",
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="When resuming, rerun songs previously marked ok",
    )
    parser.add_argument(
        "--reuse-mismatched-results",
        action="store_true",
        help="Reuse cached per-song results even when run options differ",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help=(
            "Do not run generation; recompute aggregate/report from cached "
            "<index>_*_result.json files in run directory."
        ),
    )
    parser.add_argument(
        "--min-dtw-song-coverage-ratio",
        type=float,
        default=0.9,
        help="Warn if successful-song DTW coverage ratio is below this threshold",
    )
    parser.add_argument(
        "--min-dtw-line-coverage-ratio",
        type=float,
        default=0.9,
        help="Warn if benchmark-line DTW coverage ratio is below this threshold",
    )
    parser.add_argument(
        "--strict-quality-coverage",
        action="store_true",
        help="Return non-zero if quality coverage warnings are present",
    )
    parser.add_argument(
        "--agreement-baseline-report",
        type=Path,
        default=None,
        help=(
            "Optional baseline benchmark report path (or run dir) used to detect "
            "agreement-coverage vs agreement-bad-ratio tradeoffs."
        ),
    )
    parser.add_argument(
        "--min-agreement-coverage-gain-for-bad-ratio-warning",
        type=float,
        default=0.0,
        help=(
            "When baseline report is provided, warn if agreement coverage gain is "
            "at least this amount while bad-ratio increase exceeds configured max "
            "(0 disables, default: 0)."
        ),
    )
    parser.add_argument(
        "--max-agreement-bad-ratio-increase-on-coverage-gain",
        type=float,
        default=0.0,
        help=(
            "Maximum allowed agreement_bad_ratio_mean increase under coverage-gain "
            "warning condition (default: 0.0)."
        ),
    )
    parser.add_argument(
        "--strict-agreement-tradeoff",
        action="store_true",
        help=(
            "Return non-zero if agreement tradeoff warning is triggered "
            "(requires --agreement-baseline-report + gain threshold)."
        ),
    )
    parser.add_argument(
        "--min-timing-quality-score-line-weighted",
        type=float,
        default=0.58,
        help=(
            "Warn if line-weighted timing quality score is below this threshold "
            "(0 disables)."
        ),
    )
    parser.add_argument(
        "--expect-cached-separation",
        action="store_true",
        help="Warn if separation phase is not fully cache-hit across successful songs",
    )
    parser.add_argument(
        "--expect-cached-whisper",
        action="store_true",
        help="Warn if whisper phase is not fully cache-hit across successful songs",
    )
    parser.add_argument(
        "--strict-cache-expectations",
        action="store_true",
        help="Return non-zero if cache expectation warnings are present",
    )
    parser.add_argument(
        "--max-whisper-phase-share",
        type=float,
        default=0.0,
        help=(
            "Warn if whisper phase share of cumulative song elapsed exceeds this "
            "ratio (0 disables)."
        ),
    )
    parser.add_argument(
        "--max-alignment-phase-share",
        type=float,
        default=0.0,
        help=(
            "Warn if alignment phase share of cumulative song elapsed exceeds this "
            "ratio (0 disables)."
        ),
    )
    parser.add_argument(
        "--max-scheduler-overhead-sec",
        type=float,
        default=0.0,
        help="Warn if suite wall time minus summed song time exceeds this budget.",
    )
    parser.add_argument(
        "--strict-runtime-budgets",
        action="store_true",
        help="Return non-zero if runtime budget warnings are present",
    )
    parser.add_argument(
        "--rebaseline",
        action="store_true",
        help=(
            "Update per-song gold files from successful timing reports. "
            "Use --match/--max-songs to constrain scope."
        ),
    )
    return parser.parse_args()


def _tail_text(data: str | bytes | None, line_count: int = 30) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        text = data.decode("utf-8", errors="replace")
    else:
        text = data
    return "\n".join(text.splitlines()[-line_count:])


def _coerce_text(data: str | bytes | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


def _extract_stage_hint(stdout_text: str, stderr_text: str) -> str | None:
    """Infer a readable stage hint from accumulated subprocess output."""
    merged = f"{stdout_text}\n{stderr_text}"
    lines = [line.strip() for line in merged.splitlines()]
    candidates = [line for line in lines if line]
    if not candidates:
        return None

    # Prefer explicit y2karaoke log lines if present.
    log_lines = [
        line
        for line in candidates
        if "INFO:y2karaoke" in line
        or "WARNING:y2karaoke" in line
        or "ERROR:y2karaoke" in line
    ]
    target_lines = log_lines if log_lines else candidates

    stage_markers = [
        ("karaoke generation complete", "complete"),
        ("rendering karaoke video", "render"),
        ("skipping video rendering", "render_skip"),
        ("creating background segments", "backgrounds"),
        ("wrote timing report", "timing_report"),
        ("scaling lyrics timing", "timing_scale"),
        ("fetching lyrics", "lyrics_fetch"),
        ("whisper", "whisper_alignment"),
        ("transcrib", "whisper_alignment"),
        ("align", "whisper_alignment"),
        ("separat", "separation"),
        ("demucs", "separation"),
        ("stem", "separation"),
        ("using cached audio", "media_cached_audio"),
        ("downloading audio", "media_download_audio"),
        ("using cached video", "media_cached_video"),
        ("downloading video", "media_download_video"),
        ("identifying track", "identify_track"),
        ("video id:", "identify_track"),
    ]

    # Ignore tqdm/progress-noise lines and return the newest meaningful line.
    newest_line: str | None = None
    for raw in reversed(target_lines):
        line = raw.replace("\r", " ").strip()
        if not line:
            continue
        if line.startswith("%|") or "/it]" in line:
            continue
        if line.startswith("[") and line.endswith("]"):
            continue
        if len(line) > 180:
            line = line[-180:]
        newest_line = line
        break

    # Prefer stage inferred from the newest meaningful line so label and text match.
    if newest_line is not None:
        newest_lower = newest_line.lower()
        for marker, label in stage_markers:
            if marker in newest_lower:
                return f"[{label}] {newest_line}"
        return newest_line

    # Fallback: infer a coarse stage from all buffered output.
    merged_lower = merged.lower()
    for marker, label in stage_markers:
        if marker in merged_lower:
            return f"[{label}]"
    return None


def _stage_label_from_hint(stage_hint: str | None) -> str | None:
    if not stage_hint or not stage_hint.startswith("["):
        return None
    end = stage_hint.find("]")
    if end <= 1:
        return None
    return stage_hint[1:end]


def _read_process_cpu_percent(pid: int) -> float | None:
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "%cpu="],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            return None
        value = proc.stdout.strip()
        if not value:
            return None
        return float(value)
    except Exception:
        return None


def _find_flag_value(cmd: list[str], flag: str) -> str | None:
    for idx, token in enumerate(cmd):
        if token == flag and idx + 1 < len(cmd):
            return cmd[idx + 1]
    return None


def _extract_video_id_from_command(cmd: list[str]) -> str | None:
    for token in cmd:
        if token.startswith("http://") or token.startswith("https://"):
            parsed = urlparse(token)
            if parsed.netloc in {"www.youtube.com", "youtube.com", "m.youtube.com"}:
                values = parse_qs(parsed.query).get("v")
                if values and values[0]:
                    return values[0]
    return None


def _collect_process_tree_commands(root_pid: int) -> list[str]:
    try:
        proc = subprocess.run(
            ["ps", "-ax", "-o", "pid=,ppid=,command="],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            return []
        children_by_parent: dict[int, list[int]] = {}
        command_by_pid: dict[int, str] = {}
        for raw in proc.stdout.splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = line.split(maxsplit=2)
            if len(parts) < 3:
                continue
            pid_s, ppid_s, command = parts
            try:
                pid = int(pid_s)
                ppid = int(ppid_s)
            except ValueError:
                continue
            command_by_pid[pid] = command
            children_by_parent.setdefault(ppid, []).append(pid)

        commands: list[str] = []
        stack = [root_pid]
        seen: set[int] = set()
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            seen.add(pid)
            command_value = command_by_pid.get(pid)
            if command_value:
                commands.append(command_value)
            stack.extend(children_by_parent.get(pid, []))
        return commands
    except Exception:
        return []


def _candidate_video_cache_dirs_for_command(cmd: list[str]) -> list[Path]:
    video_id = _extract_video_id_from_command(cmd)
    if not video_id:
        return []
    dirs: list[Path] = []
    cache_dir_value = _find_flag_value(cmd, "--work-dir")
    if cache_dir_value:
        dirs.append(Path(cache_dir_value) / video_id)
    for cache_root in _benchmark_cache_roots():
        candidate = cache_root / video_id
        if candidate not in dirs:
            dirs.append(candidate)
    return dirs


def _infer_compute_substage(
    *,
    cmd: list[str],
    proc_pid: int,
    stage_hint: str | None,
    report_path: Path,
) -> str | None:
    commands = [c.lower() for c in _collect_process_tree_commands(proc_pid)]
    joined = "\n".join(commands)
    if any(key in joined for key in ["audio-separator", "demucs", "vocals", "stems"]):
        return "separation"
    if any(
        key in joined
        for key in ["whisperx", "whisper", "faster-whisper", "ctranslate2"]
    ):
        return "whisper"

    for video_cache in _candidate_video_cache_dirs_for_command(cmd):
        if not video_cache.exists():
            continue
        whisper_files = list(video_cache.glob("*_whisper_*.json"))
        has_whisper_output = bool(whisper_files)
        stem_files = [
            p
            for p in video_cache.glob("*.wav")
            if any(
                key in p.name.lower()
                for key in ["vocals", "instrumental", "bass", "drums", "other"]
            )
        ]
        has_stems = bool(stem_files)
        if not has_stems and not report_path.exists():
            return "separation"
        if has_whisper_output and not report_path.exists():
            return "alignment"
        if has_stems and not has_whisper_output and not report_path.exists():
            return "whisper"

    hint_lower = (stage_hint or "").lower()
    if "lyrics_fetch" in hint_lower or "whisper_alignment" in hint_lower:
        return "alignment"
    return None


def _phase_from_stage_label(stage_label: str | None) -> str | None:
    if stage_label is None:
        return None
    mapping = {
        "identify_track": "identify",
        "media_cached_audio": "media_prepare",
        "media_download_audio": "media_prepare",
        "media_cached_video": "media_prepare",
        "media_download_video": "media_prepare",
        "separation": "separation",
        "lyrics_fetch": "lyrics_fetch",
        "whisper": "whisper",
        "whisper_alignment": "alignment",
        "alignment": "alignment",
        "timing_scale": "timing_finalize",
        "timing_report": "timing_finalize",
        "backgrounds": "render",
        "render": "render",
        "render_skip": "render_skip",
        "complete": "complete",
    }
    return mapping.get(stage_label, stage_label)


def _collect_cache_state(cmd: list[str], report_path: Path) -> dict[str, Any]:
    state: dict[str, Any] = {
        "audio_files": 0,
        "stem_files": 0,
        "whisper_files": 0,
        "report_exists": report_path.exists(),
    }
    cache_dir_value = _find_flag_value(cmd, "--work-dir")
    video_id = _extract_video_id_from_command(cmd)
    if not cache_dir_value or not video_id:
        return state

    cache_dir = Path(cache_dir_value)
    video_cache = cache_dir / video_id
    if not video_cache.exists():
        return state

    wav_files = list(video_cache.glob("*.wav"))
    stem_keys = ["vocals", "instrumental", "bass", "drums", "other"]
    stem_files = [p for p in wav_files if any(k in p.name.lower() for k in stem_keys)]
    audio_files = [p for p in wav_files if p not in stem_files]
    whisper_files = list(video_cache.glob("*_whisper_*.json"))

    state["audio_files"] = len(audio_files)
    state["stem_files"] = len(stem_files)
    state["whisper_files"] = len(whisper_files)
    return state


def _infer_cache_decisions(
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    combined_output: str,
    report_exists: bool,
) -> dict[str, str]:
    out_lower = combined_output.lower()
    decisions: dict[str, str] = {}

    if "using cached audio" in out_lower:
        decisions["audio"] = "hit (logged cached audio)"
    elif "downloading audio" in out_lower:
        decisions["audio"] = "miss (downloaded)"
    elif before.get("audio_files", 0) > 0:
        decisions["audio"] = "likely_hit (audio files already present)"
    else:
        decisions["audio"] = "unknown"

    if after.get("stem_files", 0) > before.get("stem_files", 0):
        decisions["separation"] = "miss (generated stems)"
    elif "using cached vocal separation" in out_lower:
        decisions["separation"] = "hit (logged cached vocal separation)"
    elif before.get("stem_files", 0) > 0:
        decisions["separation"] = "likely_hit (stems already present)"
    else:
        decisions["separation"] = "unknown"

    if after.get("whisper_files", 0) > before.get("whisper_files", 0):
        decisions["whisper"] = "miss (generated whisper output)"
    elif "loaded cached whisper transcription" in out_lower:
        decisions["whisper"] = "hit (logged cached whisper transcription)"
    elif before.get("whisper_files", 0) > 0:
        decisions["whisper"] = "likely_hit (whisper output already present)"
    else:
        decisions["whisper"] = "unknown"

    if report_exists:
        decisions["alignment"] = "computed (timing report written)"
    else:
        decisions["alignment"] = "unknown"
    return decisions


def _compose_heartbeat_stage_text(
    *,
    stage_hint: str | None,
    last_stage_hint: str | None,
    cpu_percent: float | None,
    compute_substage: str | None = None,
) -> str | None:
    base_hint = stage_hint or last_stage_hint
    if cpu_percent is None:
        return base_hint

    if base_hint:
        label = _stage_label_from_hint(base_hint)
        if (
            label in {"media_cached_audio", "media_download_audio", "identify_track"}
            and cpu_percent >= 120.0
        ):
            if compute_substage:
                return f"[{compute_substage}] cpu={cpu_percent:.1f}%"
            return f"[compute_active] {base_hint} (cpu={cpu_percent:.1f}%)"
        return f"{base_hint} (cpu={cpu_percent:.1f}%)"

    if cpu_percent >= 120.0:
        if compute_substage:
            return f"[{compute_substage}] cpu={cpu_percent:.1f}%"
        return f"[compute_active] cpu={cpu_percent:.1f}% (likely separation/whisper/alignment)"
    if cpu_percent >= 20.0:
        return f"[active] cpu={cpu_percent:.1f}%"
    return None


def _collect_heartbeat_state(
    *,
    cmd: list[str],
    proc_pid: int,
    report_path: Path,
    out_accum: str,
    err_accum: str,
    last_stage_hint: str | None,
) -> dict[str, Any]:
    stage_hint = _extract_stage_hint(out_accum, err_accum)
    updated_last_stage_hint = stage_hint or last_stage_hint
    cpu_percent = _read_process_cpu_percent(proc_pid)
    compute_substage = _infer_compute_substage(
        cmd=cmd,
        proc_pid=proc_pid,
        stage_hint=stage_hint,
        report_path=report_path,
    )
    heartbeat_stage_text = _compose_heartbeat_stage_text(
        stage_hint=stage_hint,
        last_stage_hint=updated_last_stage_hint,
        cpu_percent=cpu_percent,
        compute_substage=compute_substage,
    )
    prefer_substage = (
        compute_substage is not None
        and cpu_percent is not None
        and cpu_percent >= 120.0
    )
    stage_label = (
        compute_substage
        if prefer_substage
        else (
            _stage_label_from_hint(stage_hint)
            or _stage_label_from_hint(updated_last_stage_hint)
        )
    )
    return {
        "last_stage_hint": updated_last_stage_hint,
        "stage_label": stage_label,
        "heartbeat_stage_text": heartbeat_stage_text,
    }


def _close_current_phase(
    *,
    current_phase: str | None,
    phase_started_at: dict[str, float],
    phase_durations: dict[str, float],
    elapsed: float,
) -> None:
    if current_phase is None:
        return
    start_time = phase_started_at.get(current_phase, elapsed)
    phase_durations[current_phase] = phase_durations.get(current_phase, 0.0) + max(
        elapsed - start_time, 0.0
    )


def _print_phase_summary(phase_durations: dict[str, float]) -> None:
    if not phase_durations:
        return
    summary = ", ".join(f"{k}={v:.1f}s" for k, v in sorted(phase_durations.items()))
    print(f"    >>> phase_summary {summary}")


def _execute_song_process(
    *,
    cmd: list[str],
    env: dict[str, str],
    start: float,
    report_path: Path,
    timeout_sec: int,
    heartbeat_sec: int,
) -> dict[str, Any]:  # noqa: C901
    phase_started_at: dict[str, float] = {}
    phase_durations: dict[str, float] = {}
    current_phase: str | None = None

    def _begin_or_advance_phase(next_phase: str | None, elapsed_running: float) -> None:
        nonlocal current_phase
        if next_phase is None:
            return
        if current_phase is None:
            current_phase = next_phase
            # Attribute pre-heartbeat runtime to the first inferred phase.
            phase_started_at[next_phase] = 0.0
            print(f"    >>> phase_start {next_phase} at {elapsed_running:.1f}s")
            return
        if next_phase == current_phase:
            return
        start_time = phase_started_at.get(current_phase, elapsed_running)
        phase_durations[current_phase] = phase_durations.get(current_phase, 0.0) + max(
            elapsed_running - start_time, 0.0
        )
        current_phase = next_phase
        phase_started_at[next_phase] = elapsed_running
        print(f"    >>> phase_start {next_phase} at {elapsed_running:.1f}s")

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out_accum = ""
    err_accum = ""
    last_stage_hint: str | None = None
    try:
        while True:
            try:
                stdout_text, stderr_text = proc.communicate(timeout=heartbeat_sec)
                out_accum += _coerce_text(stdout_text)
                err_accum += _coerce_text(stderr_text)
                break
            except subprocess.TimeoutExpired as exc:
                out_accum += _coerce_text(exc.stdout)
                err_accum += _coerce_text(exc.stderr)
                elapsed_running = round(time.monotonic() - start, 1)
                heartbeat_state = _collect_heartbeat_state(
                    cmd=cmd,
                    proc_pid=proc.pid,
                    report_path=report_path,
                    out_accum=out_accum,
                    err_accum=err_accum,
                    last_stage_hint=last_stage_hint,
                )
                last_stage_hint_raw = heartbeat_state.get("last_stage_hint")
                last_stage_hint = (
                    str(last_stage_hint_raw)
                    if last_stage_hint_raw is not None
                    else last_stage_hint
                )
                stage_label = (
                    str(heartbeat_state.get("stage_label"))
                    if heartbeat_state.get("stage_label") is not None
                    else None
                )
                heartbeat_stage_text = (
                    str(heartbeat_state.get("heartbeat_stage_text"))
                    if heartbeat_state.get("heartbeat_stage_text") is not None
                    else None
                )
                _begin_or_advance_phase(
                    _phase_from_stage_label(stage_label), elapsed_running
                )
                stage_suffix = (
                    f" stage: {heartbeat_stage_text}" if heartbeat_stage_text else ""
                )
                print(
                    f"    ... running {elapsed_running}s "
                    f"(timeout {timeout_sec}s){stage_suffix}"
                )
                if elapsed_running >= timeout_sec:
                    proc.kill()
                    stdout_text, stderr_text = proc.communicate()
                    out_accum += _coerce_text(stdout_text)
                    err_accum += _coerce_text(stderr_text)
                    raise subprocess.TimeoutExpired(
                        cmd=cmd,
                        timeout=timeout_sec,
                        output=out_accum,
                        stderr=err_accum,
                    )
    except subprocess.TimeoutExpired:
        elapsed = round(time.monotonic() - start, 2)
        _close_current_phase(
            current_phase=current_phase,
            phase_started_at=phase_started_at,
            phase_durations=phase_durations,
            elapsed=elapsed,
        )
        _print_phase_summary(phase_durations)
        raise

    elapsed = round(time.monotonic() - start, 2)
    if current_phase is None:
        final_stage_hint = _extract_stage_hint(out_accum, err_accum) or last_stage_hint
        final_stage_label = _stage_label_from_hint(final_stage_hint)
        inferred_phase = _phase_from_stage_label(final_stage_label)
        if inferred_phase is not None:
            current_phase = inferred_phase
            phase_started_at[inferred_phase] = 0.0
    _close_current_phase(
        current_phase=current_phase,
        phase_started_at=phase_started_at,
        phase_durations=phase_durations,
        elapsed=elapsed,
    )
    _print_phase_summary(phase_durations)
    return {
        "out_accum": out_accum,
        "err_accum": err_accum,
        "elapsed": elapsed,
        "return_code": int(proc.returncode or 0),
        "last_stage_hint": last_stage_hint,
        "phase_durations_sec": (
            {key: round(value, 2) for key, value in phase_durations.items()}
            if phase_durations
            else {}
        ),
    }


def _run_song_command(
    *,
    cmd: list[str],
    env: dict[str, str],
    start: float,
    song: BenchmarkSong,
    report_path: Path,
    song_log_path: Path,
    timeout_sec: int,
    heartbeat_sec: int,
    run_signature: dict[str, Any],
    gold_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "artist": song.artist,
        "title": song.title,
        "youtube_id": song.youtube_id,
        "report_path": str(report_path),
        "command": cmd,
        "song_log_path": str(song_log_path),
        "run_signature": run_signature,
    }
    before_cache_state = _collect_cache_state(cmd, report_path)
    combined_output_for_cache = ""
    try:
        execution = _execute_song_process(
            cmd=cmd,
            env=env,
            start=start,
            report_path=report_path,
            timeout_sec=timeout_sec,
            heartbeat_sec=heartbeat_sec,
        )
        out_accum = str(execution["out_accum"])
        err_accum = str(execution["err_accum"])
        combined_output_for_cache = out_accum + "\n" + err_accum
        record["elapsed_sec"] = float(execution["elapsed"])
        phase_durations = execution.get("phase_durations_sec", {})
        if phase_durations:
            record["phase_durations_sec"] = phase_durations

        song_log_path.write_text(
            (
                f"$ {' '.join(cmd)}\n\n"
                + "=== STDOUT ===\n"
                + out_accum
                + "\n=== STDERR ===\n"
                + err_accum
            ),
            encoding="utf-8",
        )

        final_stage_hint = _extract_stage_hint(out_accum, err_accum) or execution.get(
            "last_stage_hint"
        )
        if final_stage_hint:
            record["last_stage_hint"] = str(final_stage_hint)

        record["return_code"] = int(execution["return_code"])
        if record["return_code"] != 0:
            record["status"] = "failed"
            record["error"] = f"command exited {record['return_code']}"
            record["stdout_tail"] = "\n".join(out_accum.splitlines()[-30:])
            record["stderr_tail"] = "\n".join(err_accum.splitlines()[-30:])
        elif not report_path.exists():
            record["status"] = "failed"
            record["error"] = "timing report was not produced"
        else:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            record["metrics"] = _extract_song_metrics(
                report,
                gold_doc=gold_doc,
                audio_path=_resolve_song_audio_path(song, gold_doc=gold_doc),
            )
            record["alignment_diagnostics"] = _extract_alignment_diagnostics(report)
            record["reference_divergence"] = _infer_reference_divergence_suspicion(
                record["metrics"],
                alignment_diagnostics=record.get("alignment_diagnostics"),
            )
            record["alignment_policy_hint"] = _infer_alignment_policy_hint(
                record["metrics"],
                alignment_diagnostics=record.get("alignment_diagnostics"),
                reference_divergence=record.get("reference_divergence"),
            )
            record["lexical_mismatch_diagnostics"] = (
                _extract_lexical_mismatch_diagnostics(
                    report,
                    record["metrics"],
                    alignment_policy_hint=record.get("alignment_policy_hint"),
                )
            )
            record["quality_diagnosis"] = _classify_quality_diagnosis(
                record["metrics"],
                alignment_policy_hint=record.get("alignment_policy_hint"),
                reference_divergence=record.get("reference_divergence"),
                lexical_mismatch_diagnostics=record.get("lexical_mismatch_diagnostics"),
            )
            record["status"] = "ok"
    except subprocess.TimeoutExpired as exc:
        elapsed = round(time.monotonic() - start, 2)
        record["elapsed_sec"] = elapsed
        record["status"] = "failed"
        record["error"] = f"timeout after {timeout_sec}s"
        out = _coerce_text(exc.stdout)
        err = _coerce_text(exc.stderr)
        combined_output_for_cache = out + "\n" + err
        stage_hint = _extract_stage_hint(out, err)
        if stage_hint:
            record["last_stage_hint"] = stage_hint
        song_log_path.write_text(
            (
                f"$ {' '.join(cmd)}\n\n"
                + "=== STDOUT (partial) ===\n"
                + out
                + "\n=== STDERR (partial) ===\n"
                + err
            ),
            encoding="utf-8",
        )
        record["stdout_tail"] = _tail_text(out)
        record["stderr_tail"] = _tail_text(err)

    after_cache_state = _collect_cache_state(cmd, report_path)
    cache_decisions = _infer_cache_decisions(
        before=before_cache_state,
        after=after_cache_state,
        combined_output=combined_output_for_cache,
        report_exists=bool(report_path.exists()),
    )
    record["cache_decisions"] = cache_decisions
    print(
        "    >>> cache_decisions "
        + ", ".join(f"{k}={v}" for k, v in cache_decisions.items())
    )
    return record


def _try_reuse_cached_song_result(
    *,
    args: argparse.Namespace,
    run_signature: dict[str, Any],
    index: int,
    total_songs: int,
    song: BenchmarkSong,
    result_path: Path,
    report_path: Path,
    gold_root: Path,
) -> dict[str, Any] | None:
    prior = _load_song_result(result_path)
    if prior is None:
        return None
    prior_signature = prior.get("run_signature")
    signature_matches = prior_signature == run_signature
    if not signature_matches and not args.reuse_mismatched_results:
        print(f"[{index}/{total_songs}] {song.artist} - {song.title}")
        print("  -> cached result ignored (run options changed)")
        return None

    prior_status = str(prior.get("status", ""))
    if prior_status == "ok" and not args.rerun_completed:
        prior = _refresh_cached_metrics(
            prior, index=index, song=song, gold_root=gold_root
        )
        if args.rebaseline:
            rebased_path = _rebaseline_song_from_report(
                index=index,
                song=song,
                report_path=Path(str(prior.get("report_path", report_path))),
                gold_root=gold_root,
            )
            prior["gold_rebaselined"] = rebased_path is not None
            if rebased_path is not None:
                prior["gold_path"] = str(rebased_path)
                print(f"  -> gold rebaselined: {rebased_path}")
        prior["result_reused"] = True
        print(f"[{index}/{total_songs}] {song.artist} - {song.title}")
        print("  -> ok (cached result)")
        return prior
    if prior_status == "failed" and not args.rerun_failed:
        prior["result_reused"] = True
        print(f"[{index}/{total_songs}] {song.artist} - {song.title}")
        print("  -> failed (cached result)")
        return prior
    return None


def _run_single_song_generation(
    *,
    args: argparse.Namespace,
    index: int,
    total_songs: int,
    song: BenchmarkSong,
    run_dir: Path,
    run_signature: dict[str, Any],
    gold_root: Path,
    env: dict[str, str],
) -> tuple[dict[str, Any], Path]:
    report_path = run_dir / f"{index:02d}_{song.slug}_timing_report.json"
    offline = bool(args.offline or _has_cached_benchmark_source(song))
    cmd = _build_generate_command(
        python_bin=args.python_bin,
        song=song,
        report_path=report_path,
        cache_dir=args.cache_dir,
        offline=offline,
        force=args.force,
        whisper_map_lrc_dtw=not args.no_whisper_map_lrc_dtw,
        strategy=args.strategy,
        drop_lrc_line_timings=(args.scenario == "lyrics_no_timing"),
        evaluate_lyrics_sources=bool(getattr(args, "evaluate_lyrics_sources", False)),
    )
    print(f"[{index}/{total_songs}] {song.artist} - {song.title}")
    start = time.monotonic()
    song_log_path = run_dir / f"{index:02d}_{song.slug}_generate.log"
    gold_doc = _load_gold_doc(index=index, song=song, gold_root=gold_root)
    record = _run_song_command(
        cmd=cmd,
        env=env,
        start=start,
        song=song,
        report_path=report_path,
        song_log_path=song_log_path,
        timeout_sec=args.timeout_sec,
        heartbeat_sec=args.heartbeat_sec,
        run_signature=run_signature,
        gold_doc=gold_doc,
    )
    if gold_doc is not None:
        gold_path = _gold_path_for_song(index=index, song=song, gold_root=gold_root)
        if gold_path is not None:
            record["gold_path"] = str(gold_path)
    if args.rebaseline and record.get("status") == "ok":
        rebased_path = _rebaseline_song_from_report(
            index=index,
            song=song,
            report_path=report_path,
            gold_root=gold_root,
        )
        record["gold_rebaselined"] = rebased_path is not None
        if rebased_path is not None:
            record["gold_path"] = str(rebased_path)
            print(f"  -> gold rebaselined: {rebased_path}")
    return record, _song_result_path(run_dir, index, song.slug)


def _build_runner_env() -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'src'}:{existing_pythonpath}"
        if existing_pythonpath
        else str(REPO_ROOT / "src")
    )
    return env


def _append_result_and_checkpoint(
    *,
    record: dict[str, Any],
    song_results: list[dict[str, Any]],
    run_id: str,
    run_dir: Path,
    manifest_path: Path,
    args: argparse.Namespace,
    suite_start: float,
    result_path: Path | None = None,
) -> None:
    song_results.append(record)
    if result_path is not None:
        _write_json(result_path, record)
    _write_checkpoint(
        run_id=run_id,
        run_dir=run_dir,
        manifest_path=manifest_path,
        args=args,
        song_results=song_results,
        suite_elapsed=time.monotonic() - suite_start,
    )


def _collect_single_song_result(
    *,
    args: argparse.Namespace,
    song_results: list[dict[str, Any]],
    index: int,
    total_songs: int,
    song: BenchmarkSong,
    run_signature: dict[str, Any],
    run_id: str,
    run_dir: Path,
    manifest_path: Path,
    gold_root: Path,
    env: dict[str, str],
    suite_start: float,
) -> dict[str, Any]:
    report_path = run_dir / f"{index:02d}_{song.slug}_timing_report.json"
    result_path = _song_result_path(run_dir, index, song.slug)
    reused = _try_reuse_cached_song_result(
        args=args,
        run_signature=run_signature,
        index=index,
        total_songs=total_songs,
        song=song,
        result_path=result_path,
        report_path=report_path,
        gold_root=gold_root,
    )
    if reused is not None:
        _append_result_and_checkpoint(
            record=reused,
            song_results=song_results,
            run_id=run_id,
            run_dir=run_dir,
            manifest_path=manifest_path,
            args=args,
            suite_start=suite_start,
        )
        return reused

    record, result_path = _run_single_song_generation(
        args=args,
        index=index,
        total_songs=total_songs,
        song=song,
        run_dir=run_dir,
        run_signature=run_signature,
        gold_root=gold_root,
        env=env,
    )
    record = _refresh_cached_metrics(
        record,
        index=index,
        song=song,
        gold_root=gold_root,
    )
    _append_result_and_checkpoint(
        record=record,
        song_results=song_results,
        run_id=run_id,
        run_dir=run_dir,
        manifest_path=manifest_path,
        args=args,
        suite_start=suite_start,
        result_path=result_path,
    )
    return record


def _collect_song_results(
    *,
    args: argparse.Namespace,
    songs: list[BenchmarkSong],
    run_id: str,
    run_dir: Path,
    manifest_path: Path,
    run_signature: dict[str, Any],
    gold_root: Path,
    env: dict[str, str],
    suite_start: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    song_results: list[dict[str, Any]] = []
    aggregate_only_skipped_missing: list[str] = []
    if args.aggregate_only:
        return _load_aggregate_only_results(
            songs=songs,
            run_dir=run_dir,
            gold_root=gold_root,
            rebaseline=args.rebaseline,
        )

    total_songs = len(songs)
    for index, song in enumerate(songs, start=1):
        record = _collect_single_song_result(
            args=args,
            song_results=song_results,
            index=index,
            total_songs=total_songs,
            song=song,
            run_signature=run_signature,
            run_id=run_id,
            run_dir=run_dir,
            manifest_path=manifest_path,
            gold_root=gold_root,
            env=env,
            suite_start=suite_start,
        )
        if not record.get("result_reused"):
            print(f"  -> {record['status']} ({record.get('elapsed_sec', 0.0)}s)")
        if record["status"] != "ok" and args.fail_fast:
            break

    return song_results, aggregate_only_skipped_missing


def _prepare_run_context(
    args: argparse.Namespace,
) -> tuple[
    Path,
    list[BenchmarkSong],
    Path,
    str,
    dict[str, str],
    dict[str, Any],
    Path,
    dict[str, Any] | None,
]:
    manifest_path = args.manifest.resolve()
    songs = _filter_manifest_songs(
        _parse_manifest(manifest_path), match=args.match, max_songs=args.max_songs
    )
    run_dir, run_id = _resolve_run_dir(
        output_root=args.output_root,
        run_id=args.run_id,
        resume_run_dir=args.resume_run_dir,
        resume_latest=args.resume_latest,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    inherited_gold_root = _inherit_resume_gold_root(args, run_dir)
    if inherited_gold_root is not None:
        args.gold_root = inherited_gold_root
    songs = _apply_aggregate_only_cached_scope(
        songs,
        aggregate_only=args.aggregate_only,
        match=args.match,
        max_songs=args.max_songs,
        run_dir=run_dir,
    )
    env = _build_runner_env()
    run_signature = _build_run_signature(args, manifest_path)
    gold_root = args.gold_root.resolve()
    baseline_aggregate = _load_baseline_aggregate(args.agreement_baseline_report)
    return (
        manifest_path,
        songs,
        run_dir,
        run_id,
        env,
        run_signature,
        gold_root,
        baseline_aggregate,
    )


def main() -> int:  # noqa: C901
    args = _parse_args()
    _validate_cli_args(args)
    (
        manifest_path,
        songs,
        run_dir,
        run_id,
        env,
        run_signature,
        gold_root,
        baseline_aggregate,
    ) = _prepare_run_context(args)

    if not songs:
        print("benchmark_suite: no songs selected")
        if args.aggregate_only:
            print("  (aggregate-only mode found no cached result songs)")
        return 1

    suite_start = time.monotonic()
    song_results, aggregate_only_skipped_missing = _collect_song_results(
        args=args,
        songs=songs,
        run_id=run_id,
        run_dir=run_dir,
        manifest_path=manifest_path,
        run_signature=run_signature,
        gold_root=gold_root,
        env=env,
        suite_start=suite_start,
    )

    aggregate = _aggregate(song_results)
    suite_elapsed, sum_song_elapsed, sum_song_elapsed_total, overhead_base = (
        _compute_suite_elapsed_accounting(
            aggregate=aggregate,
            measured_suite_elapsed=(time.monotonic() - suite_start),
            aggregate_only=args.aggregate_only,
        )
    )
    quality_warnings = _quality_coverage_warnings(
        aggregate=aggregate,
        dtw_enabled=(args.strategy == "hybrid_dtw" and not args.no_whisper_map_lrc_dtw),
        min_song_coverage_ratio=args.min_dtw_song_coverage_ratio,
        min_line_coverage_ratio=args.min_dtw_line_coverage_ratio,
        min_timing_quality_score_line_weighted=(
            args.min_timing_quality_score_line_weighted
        ),
        suite_wall_elapsed_sec=suite_elapsed,
    )
    agreement_tradeoff_warnings = _agreement_tradeoff_warnings(
        aggregate=aggregate,
        baseline_aggregate=baseline_aggregate,
        min_coverage_gain=args.min_agreement_coverage_gain_for_bad_ratio_warning,
        max_bad_ratio_increase=(args.max_agreement_bad_ratio_increase_on_coverage_gain),
    )
    quality_warnings.extend(agreement_tradeoff_warnings)
    cache_warnings = _cache_expectation_warnings(
        aggregate=aggregate,
        expect_cached_separation=args.expect_cached_separation,
        expect_cached_whisper=args.expect_cached_whisper,
    )
    runtime_warnings = _runtime_budget_warnings(
        aggregate=aggregate,
        suite_wall_elapsed_sec=suite_elapsed,
        max_whisper_phase_share=args.max_whisper_phase_share,
        max_alignment_phase_share=args.max_alignment_phase_share,
        max_scheduler_overhead_sec=args.max_scheduler_overhead_sec,
    )
    run_warnings = quality_warnings + cache_warnings + runtime_warnings
    if aggregate_only_skipped_missing:
        run_warnings.append(
            "Aggregate-only skipped songs without cached results: "
            f"{len(aggregate_only_skipped_missing)}"
        )
    report_json = _build_final_report_json(
        run_id=run_id,
        manifest_path=manifest_path,
        args=args,
        aggregate=aggregate,
        song_results=song_results,
        suite_elapsed=suite_elapsed,
        sum_song_elapsed=sum_song_elapsed,
        sum_song_elapsed_total=sum_song_elapsed_total,
        overhead_base=overhead_base,
        quality_warnings=quality_warnings,
        cache_warnings=cache_warnings,
        runtime_warnings=runtime_warnings,
        run_warnings=run_warnings,
        aggregate_only_skipped_missing=aggregate_only_skipped_missing,
    )

    json_path, md_path = _persist_final_report_outputs(
        args=args,
        run_id=run_id,
        run_dir=run_dir,
        manifest_path=manifest_path,
        aggregate=aggregate,
        song_results=song_results,
        report_json=report_json,
    )

    status = _derive_run_status(aggregate, run_warnings)
    _print_run_summary(
        status=status,
        run_dir=run_dir,
        json_path=json_path,
        md_path=md_path,
        aggregate=aggregate,
        suite_elapsed=suite_elapsed,
        sum_song_elapsed=sum_song_elapsed,
        overhead_base=overhead_base,
        run_warnings=run_warnings,
    )
    return _determine_exit_code(
        aggregate=aggregate,
        args=args,
        quality_warnings=quality_warnings,
        agreement_tradeoff_warnings=agreement_tradeoff_warnings,
        cache_warnings=cache_warnings,
        runtime_warnings=runtime_warnings,
    )


if __name__ == "__main__":
    sys.exit(main())
