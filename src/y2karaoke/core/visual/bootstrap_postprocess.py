"""Post-processing helpers for karaoke visual bootstrap outputs."""

from __future__ import annotations

from typing import Any, List, Optional

from ..models import TargetLine
from ..text_utils import (
    LYRIC_FUNCTION_WORDS,
    normalize_ocr_line,
    normalize_text_basic,
)
from .reconstruction import snap


def _clamp_confidence(value: Optional[float], default: float = 0.0) -> float:
    if value is None:
        value = default
    return max(0.0, min(1.0, float(value)))


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


def build_refined_lines_output(
    t_lines: list[TargetLine], artist: Optional[str], title: Optional[str]
) -> list[dict[str, Any]]:
    lines_out: list[dict[str, Any]] = []
    prev_line_end = 5.0
    normalized_title = normalize_text_basic(title or "")
    normalized_artist = normalize_text_basic(artist or "")

    for idx, ln in enumerate(t_lines):
        if ln.start < 7.0 and (
            not ln.word_starts or all(s is None for s in ln.word_starts)
        ):
            continue

        norm_txt = normalize_text_basic(ln.text)
        if norm_txt in [normalized_title, normalized_artist]:
            continue

        w_out: list[dict[str, Any]] = []
        n_words = len(ln.words)
        # Use visibility_start as the absolute floor if available, otherwise fallback to sequential logic
        l_start = (
            float(ln.visibility_start)
            if ln.visibility_start is not None
            else max(ln.start, prev_line_end)
        )

        if not ln.word_starts or all(s is None for s in ln.word_starts):
            line_duration = max((ln.end or (l_start + 1.0)) - l_start, 1.0)
            step = line_duration / max(n_words, 1)
            for j, txt in enumerate(ln.words):
                ws = l_start + j * step
                we = ws + step
                w_out.append(
                    {
                        "word_index": j + 1,
                        "text": txt,
                        "start": snap(ws),
                        "end": snap(we),
                        "confidence": 0.0,
                    }
                )
        else:
            word_starts = ln.word_starts
            word_ends = ln.word_ends or [None] * n_words
            word_confidences = ln.word_confidences or [None] * n_words

            vs = [j for j, s in enumerate(word_starts) if s is not None]
            prev_known, next_known = nearest_known_word_indices(vs, n_words)
            out_s: list[float] = []
            out_e: list[float] = []
            out_c: list[float] = []

            for j in range(n_words):
                ws_val = word_starts[j]
                if ws_val is not None:
                    out_s.append(ws_val)
                    out_e.append(word_ends[j] or ws_val + 0.1)
                    out_c.append(_clamp_confidence(word_confidences[j], default=0.5))
                else:
                    prev_v = prev_known[j]
                    next_v = next_known[j]

                    if prev_v == -1:
                        base = ln.start
                        first_vs_val = word_starts[vs[0]] if vs else base + 1.0
                        assert first_vs_val is not None
                        next_t = first_vs_val
                        step = max(0.1, (next_t - base) / (len(vs) + 1 if vs else 2))
                        out_s.append(
                            max(
                                base,
                                (
                                    next_t - (vs[0] - j + 1) * step
                                    if vs
                                    else base + j * 0.5
                                ),
                            )
                        )
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
                    out_c.append(0.25)

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
                        "start": snap(out_s[j]),
                        "end": snap(out_e[j]),
                        "confidence": round(out_c[j], 3),
                    }
                )

        w_out = _split_fused_output_words(w_out)
        if not w_out:
            continue

        line_start = w_out[0]["start"]
        line_end = w_out[-1]["end"]
        prev_line_end = line_end

        lines_out.append(
            {
                "line_index": idx + 1,
                "text": ln.text,
                "start": line_start,
                "end": line_end,
                "confidence": round(
                    sum(w["confidence"] for w in w_out) / max(len(w_out), 1), 3
                ),
                "words": w_out,
                "y": ln.y,
                "word_rois": ln.word_rois,
                "char_rois": [],
            }
        )

    for i, line_dict in enumerate(lines_out):
        line_dict["line_index"] = i + 1
    _retime_short_interstitial_output_lines(lines_out)
    _rebalance_compressed_middle_four_line_sequences(lines_out)
    _filter_singer_label_prefixes(lines_out, artist=artist)
    _normalize_output_casing(lines_out)
    return lines_out


def _normalize_output_casing(lines_out: list[dict[str, Any]]) -> None:
    """If the output is almost entirely ALL CAPS, convert to Title Case for readability."""
    if not lines_out:
        return
    total_chars = 0
    upper_chars = 0
    for ln in lines_out:
        txt = ln.get("text", "")
        alpha = [ch for ch in txt if ch.isalpha()]
        total_chars += len(alpha)
        upper_chars += sum(1 for ch in alpha if ch.isupper())

    if total_chars > 100 and upper_chars > 0.9 * total_chars:
        import string

        for ln in lines_out:
            for w in ln.get("words", []):
                w["text"] = string.capwords(w["text"].lower())
            ln["text"] = " ".join(w["text"] for w in ln.get("words", []))


def _filter_singer_label_prefixes(
    lines_out: list[dict[str, Any]], artist: Optional[str]
) -> None:
    """Remove words that appear as prefixes with high frequency or match artist name."""
    if not lines_out:
        return

    banned_prefixes = _identify_banned_prefixes(lines_out, artist)
    if not banned_prefixes:
        return

    for ln in lines_out:
        _remove_prefix_from_line(ln, banned_prefixes)

    # Remove now-empty lines
    lines_out[:] = [ln for ln in lines_out if ln.get("words")]
    for i, ln in enumerate(lines_out):
        ln["line_index"] = i + 1


def _identify_banned_prefixes(
    lines_out: list[dict[str, Any]], artist: Optional[str]
) -> set[str]:
    counts: dict[str, int] = {}
    for ln in lines_out:
        words = ln.get("words", [])
        if words:
            prefix = normalize_text_basic(words[0]["text"])
            if prefix:
                counts[prefix] = counts.get(prefix, 0) + 1

    artist_norm = normalize_text_basic(artist or "").split()
    banned_prefixes: set[str] = set()
    total = len(lines_out)

    for prefix, count in counts.items():
        if prefix in LYRIC_FUNCTION_WORDS:
            continue
        # If it appears in > 10% of lines as a prefix and is not a function word
        if count > 0.1 * total and count >= 3:
            banned_prefixes.add(prefix)
        # Or if it matches a part of the artist name
        elif artist_norm and prefix in artist_norm:
            banned_prefixes.add(prefix)
    return banned_prefixes


def _remove_prefix_from_line(line: dict[str, Any], banned_prefixes: set[str]) -> None:
    words = line.get("words", [])
    if not words:
        return
    prefix = normalize_text_basic(words[0]["text"])
    if prefix in banned_prefixes:
        words.pop(0)
        if not words:
            line["words"] = []
            line["text"] = ""
            return
        line["words"] = words
        line["text"] = " ".join(w["text"] for w in words)
        line["start"] = words[0]["start"]
        for i, w in enumerate(words):
            w["word_index"] = i + 1


def _retime_short_interstitial_output_lines(lines_out: list[dict[str, Any]]) -> None:
    """Delay short bridge lines that are tightly attached to a previous long line."""
    for i in range(1, len(lines_out) - 1):
        prev = lines_out[i - 1]
        cur = lines_out[i]
        nxt = lines_out[i + 1]

        prev_end = float(prev.get("end", 0.0))
        cur_start = float(cur.get("start", 0.0))
        cur_end = float(cur.get("end", cur_start))
        next_start = float(nxt.get("start", cur_end))
        cur_words = cur.get("words", [])
        prev_words = prev.get("words", [])
        if len(cur_words) > 2 or len(prev_words) < 4:
            continue

        cur_dur = cur_end - cur_start
        if cur_dur > 1.2:
            continue
        lead_gap = cur_start - prev_end
        tail_gap = next_start - cur_end
        if lead_gap >= 0.45 or tail_gap <= 0.6:
            continue

        shift = min(0.85, max(0.45, 0.8 - lead_gap), tail_gap - 0.15)
        if shift < 0.25:
            continue
        new_start = snap(cur_start + shift)
        new_end = snap(cur_end + shift)
        if new_end >= next_start - 0.1:
            continue

        cur["start"] = new_start
        cur["end"] = new_end
        for w in cur_words:
            w["start"] = snap(float(w["start"]) + shift)
            w["end"] = snap(float(w["end"]) + shift)


def _rebalance_compressed_middle_four_line_sequences(
    lines_out: list[dict[str, Any]],
) -> None:
    """Spread middle starts when a 4-line run has compressed middle gaps."""
    for i in range(len(lines_out) - 3):
        a = lines_out[i]
        b = lines_out[i + 1]
        c = lines_out[i + 2]
        d = lines_out[i + 3]
        sa = float(a.get("start", 0.0))
        sb = float(b.get("start", sa))
        sc = float(c.get("start", sb))
        sd = float(d.get("start", sc))
        if not (sa < sb < sc < sd):
            continue
        gap_ab = sb - sa
        gap_bc = sc - sb
        gap_cd = sd - sc
        if gap_ab > 1.4 or gap_bc > 1.1 or gap_cd < 2.0:
            continue
        span = sd - sa
        if span < 3.2:
            continue

        tb = sa + span / 3.0
        tc = sa + 2.0 * span / 3.0
        if tb <= sb + 0.2 and tc <= sc + 0.2:
            continue

        for rec, old_s, target_s in ((b, sb, tb), (c, sc, tc)):
            words = rec.get("words", [])
            if not words:
                continue
            old_e = float(rec.get("end", old_s))
            dur = max(0.7, old_e - old_s)
            new_s = max(old_s, target_s)
            if rec is b:
                new_s = min(new_s, float(c.get("start", sc)) - 0.15)
            else:
                new_s = min(new_s, sd - 0.15)
            if new_s <= old_s + 0.2:
                continue
            shift = new_s - old_s
            rec["start"] = snap(new_s)
            rec["end"] = snap(new_s + dur)
            for w in words:
                w["start"] = snap(float(w["start"]) + shift)
                w["end"] = snap(float(w["end"]) + shift)


def _split_fused_output_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in words:
        txt = str(rec.get("text", "")).strip()
        normalized = normalize_ocr_line(txt).strip()
        parts = [p for p in normalized.split() if p]
        if len(parts) <= 1:
            out.append(rec)
            continue

        start = float(rec.get("start", 0.0))
        end = float(rec.get("end", start + 0.1))
        span = max(0.1, end - start)
        conf = float(rec.get("confidence", 0.0))

        weights = [max(len(p), 1) for p in parts]
        total = float(sum(weights))
        cursor = start
        for idx, (part, w) in enumerate(zip(parts, weights)):
            dur = span * (float(w) / total)
            seg_end = end if idx == len(parts) - 1 else cursor + dur
            out.append(
                {
                    "word_index": 0,
                    "text": part,
                    "start": snap(cursor),
                    "end": snap(seg_end),
                    "confidence": conf,
                }
            )
            cursor = seg_end

    for i, rec in enumerate(out):
        rec["word_index"] = i + 1
    return out
