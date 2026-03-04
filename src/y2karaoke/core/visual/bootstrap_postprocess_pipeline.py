"""Pipeline helpers for constructing initial refined bootstrap line outputs."""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..models import TargetLine
from ..text_utils import normalize_text_basic
from .reconstruction import snap


def nearest_known_word_indices(
    known_indices: List[int], n_words: int
) -> tuple[List[int], List[int]]:
    prev_known = [-1] * n_words
    next_known = [-1] * n_words

    cursor = -1
    for i in range(n_words):
        if cursor + 1 < len(known_indices) and known_indices[cursor + 1] == i:
            cursor += 1
        if cursor >= 0:
            prev_known[i] = known_indices[cursor]

    cursor = len(known_indices)
    for i in range(n_words - 1, -1, -1):
        if cursor - 1 >= 0 and known_indices[cursor - 1] == i:
            cursor -= 1
        if cursor < len(known_indices):
            next_known[i] = known_indices[cursor]

    return prev_known, next_known


def _order_target_lines_for_output(t_lines: list[TargetLine]) -> list[TargetLine]:
    ordered_lines = list(t_lines)
    if not any(
        getattr(ln, "block_order_hint", None) is not None for ln in ordered_lines
    ):
        return ordered_lines

    def _block_hint_sort_value(ln: TargetLine) -> int:
        hint = getattr(ln, "block_order_hint", None)
        return int(hint) if hint is not None else 10_000_000

    ordered_lines.sort(
        key=lambda ln: (
            0 if getattr(ln, "block_order_hint", None) is not None else 1,
            _block_hint_sort_value(ln),
            (
                float(ln.visibility_start)
                if ln.visibility_start is not None
                else float(ln.start)
            ),
            float(ln.y),
            int(ln.line_index),
        )
    )
    return ordered_lines


def _build_word_output_with_no_word_starts(
    line: TargetLine,
    *,
    line_start: float,
) -> list[dict[str, Any]]:
    n_words = len(line.words)
    line_duration = max((line.end or (line_start + 1.0)) - line_start, 1.0)
    step = line_duration / max(n_words, 1)
    out: list[dict[str, Any]] = []
    for j, txt in enumerate(line.words):
        ws = line_start + j * step
        we = ws + step
        out.append(
            {
                "word_index": j + 1,
                "text": txt,
                "start": snap(ws),
                "end": snap(we),
                "confidence": 0.0,
            }
        )
    return out


def _build_word_output_from_word_starts(
    line: TargetLine,
    *,
    line_start: float,
    prev_line_end: float,
    block_first_mode: bool,
    clamp_confidence_fn: Callable[[Optional[float], float], float],
    nearest_known_word_indices_fn: Callable[
        [List[int], int], tuple[List[int], List[int]]
    ],
) -> list[dict[str, Any]]:
    n_words = len(line.words)
    word_starts = line.word_starts or [None] * n_words
    word_ends = line.word_ends or [None] * n_words
    word_confidences = line.word_confidences or [None] * n_words

    vs = [j for j, s in enumerate(word_starts) if s is not None]
    prev_known, next_known = nearest_known_word_indices_fn(vs, n_words)
    out_s: list[float] = []
    out_e: list[float] = []
    out_c: list[float] = []

    for j in range(n_words):
        ws_val = word_starts[j]
        if ws_val is not None:
            out_s.append(ws_val)
            out_e.append(word_ends[j] or ws_val + 0.1)
            out_c.append(clamp_confidence_fn(word_confidences[j], 0.5))
        else:
            prev_v = prev_known[j]
            next_v = next_known[j]

            if prev_v == -1:
                base = line_start
                first_vs_val = word_starts[vs[0]] if vs else base + 1.0
                assert first_vs_val is not None
                next_t = first_vs_val
                step = max(0.1, (next_t - base) / (len(vs) + 1 if vs else 2))
                out_s.append(
                    max(
                        base,
                        (next_t - (vs[0] - j + 1) * step if vs else base + j * 0.5),
                    )
                )
            elif next_v == -1:
                base = out_e[prev_v]
                out_s.append(base + (j - prev_v) * 0.3)
            else:
                frac = (j - prev_v) / (next_v - prev_v)
                next_vs_val = word_starts[next_v]
                assert next_vs_val is not None
                out_s.append(out_e[prev_v] + frac * (next_vs_val - out_e[prev_v]))
            out_e.append(out_s[-1] + 0.1)
            out_c.append(0.25)

    out: list[dict[str, Any]] = []
    for j in range(n_words):
        if j == 0:
            if not block_first_mode:
                out_s[j] = max(out_s[j], prev_line_end)
        else:
            out_s[j] = max(out_s[j], out_e[j - 1] + 0.05)

        out_e[j] = min(max(out_e[j], out_s[j] + 0.1), out_s[j] + 0.8)

        out.append(
            {
                "word_index": j + 1,
                "text": line.words[j],
                "start": snap(out_s[j]),
                "end": snap(out_e[j]),
                "confidence": round(out_c[j], 3),
            }
        )
    return out


def _build_line_output_dict(
    *,
    line: TargetLine,
    line_index: int,
    words: list[dict[str, Any]],
) -> dict[str, Any]:
    line_start = words[0]["start"]
    line_end = words[-1]["end"]
    return {
        "line_index": line_index,
        "text": " ".join(w["text"] for w in words),
        "start": line_start,
        "end": line_end,
        "confidence": round(
            sum(w["confidence"] for w in words) / max(len(words), 1), 3
        ),
        "words": words,
        "y": line.y,
        "word_rois": line.word_rois,
        "char_rois": [],
        "_reconstruction_meta": line.reconstruction_meta or {},
        "_visibility_start": (
            float(line.visibility_start) if line.visibility_start is not None else None
        ),
        "_visibility_end": (
            float(line.visibility_end) if line.visibility_end is not None else None
        ),
    }


def build_initial_lines_output(
    t_lines: list[TargetLine],
    *,
    artist: Optional[str],
    title: Optional[str],
    split_fused_output_words_fn: Callable[..., list[dict[str, Any]]],
    clamp_confidence_fn: Callable[[Optional[float], float], float],
    nearest_known_word_indices_fn: Callable[
        [List[int], int], tuple[List[int], List[int]]
    ],
) -> list[dict[str, Any]]:
    lines_out: list[dict[str, Any]] = []
    prev_line_end = 5.0
    normalized_title = normalize_text_basic(title or "")
    normalized_artist = normalize_text_basic(artist or "")
    ordered_lines = _order_target_lines_for_output(t_lines)

    for idx, ln in enumerate(ordered_lines):
        if ln.start < 7.0 and (
            not ln.word_starts or all(s is None for s in ln.word_starts)
        ):
            continue

        norm_txt = normalize_text_basic(ln.text)
        if norm_txt in [normalized_title, normalized_artist]:
            continue

        block_meta = (
            ln.reconstruction_meta.get("block_first", {})
            if isinstance(ln.reconstruction_meta, dict)
            and isinstance(ln.reconstruction_meta.get("block_first"), dict)
            else {}
        )
        block_first_mode = bool(block_meta)
        w_out: list[dict[str, Any]] = []
        l_start = (
            float(ln.visibility_start)
            if ln.visibility_start is not None
            else max(ln.start, prev_line_end)
        )

        if not ln.word_starts or all(s is None for s in ln.word_starts):
            w_out = _build_word_output_with_no_word_starts(ln, line_start=l_start)
        else:
            w_out = _build_word_output_from_word_starts(
                ln,
                line_start=l_start,
                prev_line_end=prev_line_end,
                block_first_mode=block_first_mode,
                clamp_confidence_fn=clamp_confidence_fn,
                nearest_known_word_indices_fn=nearest_known_word_indices_fn,
            )

        w_out = split_fused_output_words_fn(
            w_out, reconstruction_meta=ln.reconstruction_meta
        )
        if not w_out:
            continue

        line_end = w_out[-1]["end"]
        if not block_first_mode:
            prev_line_end = line_end

        lines_out.append(
            _build_line_output_dict(line=ln, line_index=idx + 1, words=w_out)
        )

    for i, line_dict in enumerate(lines_out):
        line_dict["line_index"] = i + 1
    return lines_out
