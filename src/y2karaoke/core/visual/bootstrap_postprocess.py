"""Post-processing helpers for karaoke visual bootstrap outputs."""

from __future__ import annotations

from functools import lru_cache
from difflib import SequenceMatcher
import os
import re
from typing import Any, List, Optional

from ..models import TargetLine
from ..text_utils import (
    LYRIC_FUNCTION_WORDS,
    normalize_ocr_line,
    normalize_text_basic,
)
from .reconstruction import snap
from .reconstruction_intro_filters import filter_intro_non_lyrics

_FUSED_FALLBACK_PREFIX_ANCHORS = (
    "i",
    "my",
)
_FUSED_FALLBACK_SHORT_FUNCTIONS = ("i", "a", "in", "on", "to", "of", "my")
_FUSED_FALLBACK_SUFFIX_ANCHORS = ("in",)
_VOCALIZATION_NOISE_TOKENS = {
    "oh",
    "ooh",
    "oooh",
    "woah",
    "whoa",
    "woo",
    "ah",
    "aah",
    "la",
    "na",
    "mm",
    "mmm",
    "hmm",
}
_HUM_NOISE_TOKENS = {"mm", "mmm", "hmm"}
_COMMON_LYRIC_CHANT_TOKENS = {
    "hey",
    "yeah",
    "yea",
    "yo",
    "no",
    "na",
    "la",
    "oh",
    "ah",
    "woo",
    "ooh",
}
_COLLOQUIAL_EXPANSIONS = {
    "wanna": ("want", "to"),
    "gonna": ("going", "to"),
    "gotta": ("got", "to"),
    "lemme": ("let", "me"),
    "gimme": ("give", "me"),
}
_CONTRACTION_RESTORES = {
    "wont": "won't",
    "cant": "can't",
    "dont": "don't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "isnt": "isn't",
    "arent": "aren't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "couldnt": "couldn't",
    "wouldnt": "wouldn't",
    "shouldnt": "shouldn't",
    "im": "I'm",
    "ive": "I've",
    "ill": "I'll",
}
_OCR_SUB_CHAR_MAP = {
    "i": ("l",),
    "l": ("i",),
    "1": ("l", "i"),
    "|": ("l", "i"),
    "y": ("w",),
    "w": ("y",),
    "0": ("o",),
    "o": ("0",),
}
_OVERLAY_PLATFORM_TOKENS = {
    "youtube",
    "youtu",
    "tube",
    "facebook",
    "twitter",
    "instagram",
    "tiktok",
}
_OVERLAY_CTA_TOKENS = {
    "subscribe",
    "subscribers",
    "subscriber",
    "follow",
    "like",
    "click",
    "watch",
    "channel",
}
_OVERLAY_LEGAL_TOKENS = {
    "rights",
    "reserved",
    "association",
    "ltd",
    "limited",
    "copyright",
    "produced",
}
_OVERLAY_BRAND_TOKENS = {
    "karaoke",
    "instrumental",
    "collection",
}


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

        w_out = _split_fused_output_words(
            w_out, reconstruction_meta=ln.reconstruction_meta
        )
        if not w_out:
            continue

        line_start = w_out[0]["start"]
        line_end = w_out[-1]["end"]
        prev_line_end = line_end

        lines_out.append(
            {
                "line_index": idx + 1,
                "text": " ".join(w["text"] for w in w_out),
                "start": line_start,
                "end": line_end,
                "confidence": round(
                    sum(w["confidence"] for w in w_out) / max(len(w_out), 1), 3
                ),
                "words": w_out,
                "y": ln.y,
                "word_rois": ln.word_rois,
                "char_rois": [],
                "_reconstruction_meta": ln.reconstruction_meta or {},
            }
        )

    for i, line_dict in enumerate(lines_out):
        line_dict["line_index"] = i + 1
    _retime_short_interstitial_output_lines(lines_out)
    _rebalance_compressed_middle_four_line_sequences(lines_out)
    _filter_singer_label_prefixes(lines_out, artist=artist)
    lines_out = filter_intro_non_lyrics(lines_out, artist=artist)
    _remove_overlay_credit_lines(lines_out)
    _remove_weaker_near_duplicate_lines(lines_out)
    _canonicalize_repeated_line_text_variants(lines_out)
    _remove_repeated_singleton_noise_lines(lines_out, artist=artist, title=title)
    if os.environ.get("Y2K_VISUAL_ENABLE_CHANT_NOISE_FILTER", "0") == "1":
        _remove_repeated_chant_noise_lines(lines_out)
    _remove_repeated_fragment_noise_lines(lines_out, artist=artist, title=title)
    _repair_strong_local_chronology_inversions(lines_out)
    _repair_large_adjacent_time_inversions(lines_out)
    _remove_vocalization_noise_runs(lines_out)
    _normalize_output_casing(lines_out)
    _strip_internal_line_metadata(lines_out)
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


def _strip_internal_line_metadata(lines_out: list[dict[str, Any]]) -> None:
    for ln in lines_out:
        ln.pop("_reconstruction_meta", None)


def _overlay_line_signal_score(line: dict[str, Any]) -> int:  # noqa: C901
    text = str(line.get("text", "") or "")
    if not text:
        return 0
    words = line.get("words", [])
    n_words = len(words)
    norm = normalize_text_basic(text)
    toks = [t for t in norm.split() if t]
    if not toks:
        return 0

    text_lower = text.lower()
    token_set = set(toks)
    score = 0

    if n_words >= 6:
        score += 1
    if n_words >= 10:
        score += 1

    platform_hits = len(token_set & _OVERLAY_PLATFORM_TOKENS)
    cta_hits = len(token_set & _OVERLAY_CTA_TOKENS)
    legal_hits = len(token_set & _OVERLAY_LEGAL_TOKENS)
    brand_hits = len(token_set & _OVERLAY_BRAND_TOKENS)

    if platform_hits:
        score += 3
    if platform_hits >= 2:
        score += 1
    if cta_hits and platform_hits:
        score += 3
    elif cta_hits >= 2 and n_words >= 8:
        score += 2
    if legal_hits >= 2:
        score += 3
    elif legal_hits and ("reserved" in token_set or "copyright" in token_set):
        score += 2
    if brand_hits and (platform_hits or cta_hits or legal_hits):
        score += 2

    urlish = (
        "www" in text_lower
        or ".com" in text_lower
        or ".co.uk" in text_lower
        or ".co" in text_lower
    )
    if urlish:
        score += 4

    if "all rights reserved" in text_lower:
        score += 4
    if "follow us" in text_lower or "like us" in text_lower:
        score += 3
    if "produced by" in text_lower:
        score += 2
    if "in association with" in text_lower:
        score += 3

    alnum_long_tokens = 0
    for raw in [str(w.get("text", "")) for w in words]:
        raw_low = raw.lower()
        if len(raw_low) >= 10 and any(ch.isdigit() for ch in raw_low):
            alnum_long_tokens += 1
        if any(ch in "./" for ch in raw_low) and len(raw_low) >= 6:
            alnum_long_tokens += 1
    if alnum_long_tokens:
        score += 2

    return score


def _remove_overlay_credit_lines(lines_out: list[dict[str, Any]]) -> None:
    if not lines_out:
        return

    kept: list[dict[str, Any]] = []
    for ln in lines_out:
        score = _overlay_line_signal_score(ln)
        if score >= 6:
            continue
        kept.append(ln)

    if len(kept) == len(lines_out):
        return
    lines_out[:] = kept
    for i, ln in enumerate(lines_out):
        ln["line_index"] = i + 1


def _line_duplicate_quality_score(line: dict[str, Any]) -> float:
    conf = float(line.get("confidence", 0.0) or 0.0)
    meta = line.get("_reconstruction_meta", {})
    uncertainty = 0.0
    support = 1.0
    if isinstance(meta, dict):
        uncertainty = float(meta.get("uncertainty_score", 0.0) or 0.0)
        support = float(meta.get("selected_text_support_ratio", 1.0) or 0.0)
    return (0.8 * conf) + (0.4 * support) - (0.8 * uncertainty)


def _line_uncertainty(line: dict[str, Any]) -> float:
    meta = line.get("_reconstruction_meta", {})
    if not isinstance(meta, dict):
        return 0.0
    return float(meta.get("uncertainty_score", 0.0) or 0.0)


def _remove_weaker_near_duplicate_lines(  # noqa: C901
    lines_out: list[dict[str, Any]],
) -> None:
    if len(lines_out) < 2:
        return

    norms = [normalize_text_basic(str(ln.get("text", ""))) for ln in lines_out]
    token_counts = [len([t for t in n.split() if t]) for n in norms]
    drops: set[int] = set()

    for i, base in enumerate(lines_out):
        if i in drops:
            continue
        base_norm = norms[i]
        if not base_norm:
            continue
        base_count = token_counts[i]
        if base_count < 3:
            continue
        base_start = float(base.get("start", 0.0) or 0.0)

        for j in range(i + 1, min(i + 5, len(lines_out))):
            if j in drops:
                continue
            cand_norm = norms[j]
            if not cand_norm:
                continue
            cand_count = token_counts[j]
            if cand_count < 3:
                continue
            if abs(cand_count - base_count) > 4:
                continue
            cand_start = float(lines_out[j].get("start", 0.0) or 0.0)
            if cand_start - base_start > 12.0:
                break
            ratio = SequenceMatcher(None, base_norm, cand_norm).ratio()
            if ratio < 0.84:
                continue

            qi = _line_duplicate_quality_score(base)
            qj = _line_duplicate_quality_score(lines_out[j])
            diff = abs(qi - qj)
            if diff < 0.18:
                continue
            # Only suppress when one side is clearly weak by evidence.
            weak_idx = i if qi < qj else j
            weak_line = lines_out[weak_idx]
            weak_conf = float(weak_line.get("confidence", 0.0) or 0.0)
            weak_uncertainty = _line_uncertainty(weak_line)
            if weak_conf > 0.55 and weak_uncertainty < 0.18:
                continue
            drops.add(weak_idx)

    if not drops:
        return

    lines_out[:] = [ln for idx, ln in enumerate(lines_out) if idx not in drops]
    for i, ln in enumerate(lines_out):
        ln["line_index"] = i + 1


def _canonicalize_repeated_line_text_variants(  # noqa: C901
    lines_out: list[dict[str, Any]],
) -> None:
    if len(lines_out) < 3:
        return

    n = len(lines_out)
    norms = [normalize_text_basic(str(ln.get("text", ""))) for ln in lines_out]
    token_lists = [[t for t in nrm.split() if t] for nrm in norms]
    parents = list(range(n))

    def find(x: int) -> int:
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parents[rb] = ra

    for i in range(n):
        toks_i = token_lists[i]
        if len(toks_i) < 3:
            continue
        for j in range(i + 1, n):
            toks_j = token_lists[j]
            if len(toks_j) < 3:
                continue
            if abs(len(toks_i) - len(toks_j)) > 1:
                continue
            # Prefer chorus/refrain-scale repeats, not adjacent line cleanup.
            dt = abs(
                float(lines_out[j].get("start", 0.0))
                - float(lines_out[i].get("start", 0.0))
            )
            if dt < 8.0:
                continue
            ratio = SequenceMatcher(None, norms[i], norms[j]).ratio()
            if ratio < 0.82:
                continue
            shared = len(set(toks_i) & set(toks_j))
            if shared < min(2, len(set(toks_i))):
                continue
            union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    for group in groups.values():
        if len(group) < 2:
            continue
        # Pick canonical line by best combined quality.
        group_sorted = sorted(
            group,
            key=lambda idx: (
                _line_duplicate_quality_score(lines_out[idx]),
                len(token_lists[idx]),
            ),
            reverse=True,
        )
        canon_idx = group_sorted[0]
        canon_words = lines_out[canon_idx].get("words", [])
        canon_tokens = [str(w.get("text", "")) for w in canon_words]
        canon_norm = norms[canon_idx]
        if len(canon_tokens) < 3:
            continue

        for idx in group_sorted[1:]:
            line = lines_out[idx]
            words = line.get("words", [])
            if len(words) != len(canon_tokens):
                continue
            if len(words) != len(token_lists[idx]):
                continue
            if (
                _line_duplicate_quality_score(lines_out[canon_idx])
                - _line_duplicate_quality_score(line)
                < 0.18
            ):
                continue
            line_conf = float(line.get("confidence", 0.0) or 0.0)
            if line_conf > 0.6 and _line_uncertainty(line) < 0.2:
                continue
            ratio = SequenceMatcher(None, norms[idx], canon_norm).ratio()
            if ratio < 0.86:
                continue

            # Require at least one lexical improvement opportunity.
            current_tokens = [str(w.get("text", "")) for w in words]
            if all(
                normalize_text_basic(a) == normalize_text_basic(b)
                for a, b in zip(current_tokens, canon_tokens)
            ):
                continue

            for w, replacement in zip(words, canon_tokens):
                w["text"] = replacement
            line["text"] = " ".join(str(w.get("text", "")) for w in words)


def _remove_repeated_singleton_noise_lines(  # noqa: C901
    lines_out: list[dict[str, Any]], artist: Optional[str], title: Optional[str]
) -> None:
    if len(lines_out) < 3:
        return

    artist_parts = set(normalize_text_basic(artist or "").split())
    title_parts = set(normalize_text_basic(title or "").split())
    protected = {t for t in (artist_parts | title_parts) if t}

    singleton_counts: dict[str, int] = {}
    singleton_indices: dict[str, list[int]] = {}
    singleton_conf_sums: dict[str, float] = {}
    for idx, ln in enumerate(lines_out):
        words = ln.get("words", [])
        if len(words) != 1:
            continue
        token = normalize_text_basic(str(words[0].get("text", "")))
        if not token:
            continue
        singleton_counts[token] = singleton_counts.get(token, 0) + 1
        singleton_indices.setdefault(token, []).append(idx)
        singleton_conf_sums[token] = singleton_conf_sums.get(token, 0.0) + float(
            ln.get("confidence", 0.0) or 0.0
        )

    drops: set[int] = set()
    for token, count in singleton_counts.items():
        if count < 4:
            continue
        if (
            token in protected
            or token in LYRIC_FUNCTION_WORDS
            or token in _VOCALIZATION_NOISE_TOKENS
        ):
            continue
        if len(token) < 3:
            continue
        avg_conf = singleton_conf_sums[token] / max(count, 1)
        if avg_conf > 0.35:
            continue

        for idx in singleton_indices[token]:
            ln = lines_out[idx]
            words = ln.get("words", [])
            if len(words) != 1:
                continue
            start = float(ln.get("start", 0.0) or 0.0)
            end = float(ln.get("end", start) or start)
            if end - start > 1.4:
                continue

            neighbor_mentions = False
            for j in range(max(0, idx - 2), min(len(lines_out), idx + 3)):
                if j == idx:
                    continue
                n_words = lines_out[j].get("words", [])
                if len(n_words) <= 1:
                    continue
                n_text = normalize_text_basic(str(lines_out[j].get("text", "")))
                n_toks = [t for t in n_text.split() if t]
                if token in n_toks:
                    neighbor_mentions = True
                    break
            if neighbor_mentions:
                continue
            drops.add(idx)

    if not drops:
        return

    lines_out[:] = [ln for idx, ln in enumerate(lines_out) if idx not in drops]
    for i, ln in enumerate(lines_out):
        ln["line_index"] = i + 1


def _is_chant_noise_signature(tokens: list[str]) -> bool:
    if len(tokens) < 2:
        return False
    if len(tokens) > 8:
        return False
    if any(len(t) < 2 for t in tokens):
        return False
    # Avoid suppressing normal repeated lyric phrases ("no no no", "yeah yeah yeah")
    if any(t in _VOCALIZATION_NOISE_TOKENS for t in tokens):
        return False
    base = tokens[0]
    if base in _COMMON_LYRIC_CHANT_TOKENS:
        return False
    if len(base) > 5:
        return False
    for tok in tokens[1:]:
        if SequenceMatcher(None, base, tok).ratio() < 0.75:
            return False
        if tok in _COMMON_LYRIC_CHANT_TOKENS:
            return False
    return True


def _remove_repeated_chant_noise_lines(  # noqa: C901
    lines_out: list[dict[str, Any]],
) -> None:
    if len(lines_out) < 4:
        return

    norms = [normalize_text_basic(str(ln.get("text", ""))) for ln in lines_out]
    token_lists = [[t for t in n.split() if t] for n in norms]
    sig_counts: dict[tuple[str, ...], int] = {}
    sig_indices: dict[tuple[str, ...], list[int]] = {}
    sig_conf_sums: dict[tuple[str, ...], float] = {}

    for idx, (ln, toks) in enumerate(zip(lines_out, token_lists)):
        if not _is_chant_noise_signature(toks):
            continue
        # Normalize chant token variants to a common signature key.
        base = min(toks, key=len)
        sig: tuple[str, ...] = (base[:4],)
        sig_counts[sig] = sig_counts.get(sig, 0) + 1
        sig_indices.setdefault(sig, []).append(idx)
        sig_conf_sums[sig] = sig_conf_sums.get(sig, 0.0) + float(
            ln.get("confidence", 0.0) or 0.0
        )

    drops: set[int] = set()
    for sig, count in sig_counts.items():
        if count < 3:
            continue
        avg_conf = sig_conf_sums[sig] / max(count, 1)
        if avg_conf > 0.4 and not (count >= 6 and avg_conf <= 0.55):
            continue
        for idx in sig_indices[sig]:
            ln = lines_out[idx]
            start = float(ln.get("start", 0.0) or 0.0)
            end = float(ln.get("end", start) or start)
            if end - start > 2.4:
                continue
            # Keep if a nearby longer line explicitly contains the chant token.
            chant_token = sig[0]
            neighbor_mentions = False
            for j in range(max(0, idx - 2), min(len(lines_out), idx + 3)):
                if j == idx:
                    continue
                n_words = token_lists[j]
                if len(n_words) <= len(token_lists[idx]):
                    continue
                if _is_chant_noise_signature(n_words):
                    continue
                if chant_token in n_words:
                    neighbor_mentions = True
                    break
            if neighbor_mentions:
                continue
            drops.add(idx)

    if not drops:
        return
    lines_out[:] = [ln for idx, ln in enumerate(lines_out) if idx not in drops]
    for i, ln in enumerate(lines_out):
        ln["line_index"] = i + 1


def _tokens_contiguous_subphrase(needle: list[str], haystack: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    n = len(needle)
    for i in range(0, len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return True
    return False


def _remove_repeated_fragment_noise_lines(  # noqa: C901
    lines_out: list[dict[str, Any]], artist: Optional[str], title: Optional[str]
) -> None:
    if len(lines_out) < 4:
        return

    artist_parts = set(normalize_text_basic(artist or "").split())
    title_parts = set(normalize_text_basic(title or "").split())
    protected = {t for t in (artist_parts | title_parts) if t}

    line_norms = [normalize_text_basic(str(ln.get("text", ""))) for ln in lines_out]
    line_tokens = [[t for t in n.split() if t] for n in line_norms]

    frag_counts: dict[tuple[str, ...], int] = {}
    frag_conf_sums: dict[tuple[str, ...], float] = {}
    frag_indices: dict[tuple[str, ...], list[int]] = {}
    for idx, (ln, toks) in enumerate(zip(lines_out, line_tokens)):
        if not (1 <= len(toks) <= 3):
            continue
        if any(t in protected for t in toks):
            continue
        if all(
            t in LYRIC_FUNCTION_WORDS or t in _VOCALIZATION_NOISE_TOKENS for t in toks
        ):
            continue
        key = tuple(toks)
        frag_counts[key] = frag_counts.get(key, 0) + 1
        frag_conf_sums[key] = frag_conf_sums.get(key, 0.0) + float(
            ln.get("confidence", 0.0) or 0.0
        )
        frag_indices.setdefault(key, []).append(idx)

    drops: set[int] = set()
    for key, count in frag_counts.items():
        if count < 3:
            continue
        avg_conf = frag_conf_sums[key] / max(count, 1)
        if avg_conf > 0.4:
            continue

        for idx in frag_indices[key]:
            if idx in drops:
                continue
            ln = lines_out[idx]
            start = float(ln.get("start", 0.0) or 0.0)
            end = float(ln.get("end", start) or start)
            if end - start > 2.2:
                continue

            supported_by_neighbor = False
            for j in range(max(0, idx - 3), min(len(lines_out), idx + 4)):
                if j == idx:
                    continue
                neigh_toks = line_tokens[j]
                if len(neigh_toks) <= len(key):
                    continue
                n_conf = float(lines_out[j].get("confidence", 0.0) or 0.0)
                if n_conf + 0.1 < float(ln.get("confidence", 0.0) or 0.0):
                    continue
                if _tokens_contiguous_subphrase(list(key), neigh_toks):
                    supported_by_neighbor = True
                    break
            if not supported_by_neighbor:
                continue
            drops.add(idx)

    if not drops:
        return

    lines_out[:] = [ln for idx, ln in enumerate(lines_out) if idx not in drops]
    for i, ln in enumerate(lines_out):
        ln["line_index"] = i + 1


def _repair_strong_local_chronology_inversions(lines_out: list[dict[str, Any]]) -> None:
    if len(lines_out) < 2:
        return

    def _toks(line: dict[str, Any]) -> list[str]:
        return [t for t in normalize_text_basic(str(line.get("text", ""))).split() if t]

    max_passes = min(4, len(lines_out))
    for _ in range(max_passes):
        changed = False
        for i in range(len(lines_out) - 1):
            a = lines_out[i]
            b = lines_out[i + 1]
            sa = float(a.get("start", 0.0) or 0.0)
            sb = float(b.get("start", 0.0) or 0.0)
            if sa <= sb + 0.75:
                continue

            ta = _toks(a)
            tb = _toks(b)
            if not ta or not tb:
                continue
            ratio = SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()
            comparable = ratio >= 0.82 or ta == tb
            short_fragment_case = (
                len(ta) <= 3
                and len(tb) <= 4
                and (
                    float(a.get("confidence", 0.0) or 0.0) < 0.45
                    or float(b.get("confidence", 0.0) or 0.0) < 0.45
                )
            )
            if not (comparable or short_fragment_case):
                continue

            qa = _line_duplicate_quality_score(a)
            qb = _line_duplicate_quality_score(b)
            if abs(qa - qb) < 0.05 and ratio < 0.9:
                continue

            lines_out[i], lines_out[i + 1] = lines_out[i + 1], lines_out[i]
            changed = True
        if not changed:
            break

    for i, ln in enumerate(lines_out):
        ln["line_index"] = i + 1


def _repair_large_adjacent_time_inversions(lines_out: list[dict[str, Any]]) -> None:
    if len(lines_out) < 2:
        return
    max_passes = min(6, len(lines_out))
    for _ in range(max_passes):
        changed = False
        for i in range(len(lines_out) - 1):
            a = lines_out[i]
            b = lines_out[i + 1]
            sa = float(a.get("start", 0.0) or 0.0)
            sb = float(b.get("start", 0.0) or 0.0)
            if sa - sb < 2.0:
                continue
            ea = float(a.get("end", sa) or sa)
            eb = float(b.get("end", sb) or sb)
            # Avoid swapping heavily overlapping near-simultaneous lanes.
            overlap = max(0.0, min(ea, eb) - max(sa, sb))
            if overlap > 0.8:
                continue
            lines_out[i], lines_out[i + 1] = lines_out[i + 1], lines_out[i]
            changed = True
        if not changed:
            break
    for i, ln in enumerate(lines_out):
        ln["line_index"] = i + 1


def _vocalization_noise_tokens(line: dict[str, Any]) -> list[str] | None:
    words = line.get("words", [])
    if len(words) < 2:
        return None
    toks = [normalize_text_basic(str(w.get("text", ""))) for w in words]
    toks = [t for t in toks if t]
    if len(toks) < 2:
        return None
    uniq = set(toks)
    if len(uniq) > 2:
        return None
    if not uniq.issubset(_VOCALIZATION_NOISE_TOKENS):
        return None
    return toks


def _remove_vocalization_noise_runs(lines_out: list[dict[str, Any]]) -> None:
    if not lines_out:
        return
    keep: list[dict[str, Any]] = []
    i = 0
    while i < len(lines_out):
        toks = _vocalization_noise_tokens(lines_out[i])
        if toks is None:
            keep.append(lines_out[i])
            i += 1
            continue

        j = i
        run: list[dict[str, Any]] = []
        run_token_count = 0
        run_vocab: set[str] = set()
        while j < len(lines_out):
            jtoks = _vocalization_noise_tokens(lines_out[j])
            if jtoks is None:
                break
            run.append(lines_out[j])
            run_token_count += len(jtoks)
            run_vocab.update(jtoks)
            j += 1

        min_tokens = 2 if run_vocab and run_vocab.issubset(_HUM_NOISE_TOKENS) else 10
        if run_token_count >= min_tokens and len(run_vocab) <= 2:
            i = j
            continue

        keep.extend(run)
        i = j

    lines_out[:] = keep
    for i, ln in enumerate(lines_out):
        ln["line_index"] = i + 1


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


def _is_high_uncertainty_reconstruction(meta: Optional[dict[str, Any]]) -> bool:
    if not isinstance(meta, dict):
        return False
    uncertainty = float(meta.get("uncertainty_score", 0.0) or 0.0)
    support_ratio = float(meta.get("selected_text_support_ratio", 1.0) or 0.0)
    text_variants = int(meta.get("text_variant_count", 1) or 1)
    weak_votes = int(meta.get("weak_vote_positions", 0) or 0)
    return (
        uncertainty >= 0.2
        or support_ratio < 0.7
        or text_variants >= 4
        or weak_votes >= 2
        or bool(meta.get("used_observed_fallback"))
    )


def _allow_token_level_ocr_fallback(token: str) -> bool:
    low = token.strip().lower()
    if not low.isalpha() or len(low) < 6:
        return False
    if not (low.endswith("in") or low.endswith("i")):
        return False
    return not _is_spelled_word(low)


def _looks_fused_prefix_candidate(token: str) -> bool:
    low = token.strip().lower()
    if not low.isalpha() or len(low) < 5:
        return False
    for anchor in _FUSED_FALLBACK_PREFIX_ANCHORS:
        if low.startswith(anchor) and len(low) > len(anchor) + 2:
            return True
    return False


def _fallback_split_fused_token(token: str) -> list[str] | None:  # noqa: C901
    if not token or " " in token:
        return None
    if not token.isalpha():
        return None
    lower = token.lower()
    if len(lower) < 5:
        for i in range(1, len(lower)):
            left = lower[:i]
            right = lower[i:]
            if left in _FUSED_FALLBACK_SHORT_FUNCTIONS and right in (
                _FUSED_FALLBACK_SHORT_FUNCTIONS
            ):
                return ["I" if left == "i" else left, "I" if right == "i" else right]
        return None

    for anchor in _FUSED_FALLBACK_PREFIX_ANCHORS:
        if not lower.startswith(anchor):
            continue
        if len(lower) <= len(anchor):
            continue
        right = lower[len(anchor) :]
        min_right_len = 4 if len(anchor) == 1 else 3
        if len(right) < min_right_len:
            continue
        if sum(ch in "aeiouy" for ch in right) < 1:
            continue
        left_out = "I" if anchor == "i" else anchor
        return [left_out, right]

    # Split internal single-letter function-word anchors when both sides look lexical.
    if len(lower) >= 8:
        for anchor in ("a",):
            for idx in range(2, len(lower) - 2):
                if lower[idx] != anchor:
                    continue
                left = lower[:idx]
                right = lower[idx + 1 :]
                if len(left) < 4 or len(right) < 4:
                    continue
                if not right.endswith("in"):
                    continue
                if sum(ch in "aeiouy" for ch in left) < 1:
                    continue
                if sum(ch in "aeiouy" for ch in right) < 1:
                    continue
                right_parts = _fallback_split_fused_token(right)
                if right_parts:
                    return [left, anchor, *right_parts]
                return [left, anchor, _repair_fallback_token(right)]

    for anchor in _FUSED_FALLBACK_SUFFIX_ANCHORS:
        if not lower.endswith(anchor):
            continue
        left = lower[: -len(anchor)]
        if len(left) < 4:
            continue
        if sum(ch in "aeiouy" for ch in left) < 1:
            continue
        if anchor == "in" and not left[-1].isalpha():
            continue
        if anchor == "in" and left.endswith("ng"):
            return [_repair_fallback_token(lower)]
        if anchor == "in" and not left.endswith("r"):
            continue
        return [left, anchor]

    # Common OCR fusion pattern where a trailing pronoun "I" gets attached.
    if lower.endswith("i") and len(lower) >= 6:
        left = lower[:-1]
        if sum(ch in "aeiouy" for ch in left) >= 1 and left.isalpha():
            return [left, "I"]

    spell_split = _fallback_spell_validated_split(lower)
    if spell_split:
        return spell_split
    return None


def _repair_fallback_token(token: str) -> str:
    lower = token.lower()
    # Common lyric OCR artifact: dropped terminal "g" in gerunds (singin -> singing).
    if len(lower) >= 6 and lower.endswith("in") and lower[:-2].endswith("ng"):
        return lower[:-2] + "ing"
    return token


def _case_like(source: str, token: str) -> str:
    if token.lower() == "i":
        return "I"
    if source.isupper():
        return token.upper()
    if source[:1].isupper() and source[1:].islower():
        return token.capitalize()
    if source.islower():
        return token.lower()
    return token


@lru_cache(maxsize=1)
def _fallback_spell_checker() -> Any:
    try:
        from AppKit import NSSpellChecker

        return NSSpellChecker.sharedSpellChecker()
    except Exception:
        return None


@lru_cache(maxsize=4096)
def _is_spelled_word(token: str) -> bool:
    low = token.lower()
    if low in {"i", "a"}:
        return True
    if not low.isalpha() or len(low) < 2:
        return False
    checker = _fallback_spell_checker()
    if checker is None:
        return False
    try:
        return checker.checkSpellingOfString_startingAt_(low, 0).length == 0
    except Exception:
        return False


def _fallback_spell_validated_split(token: str) -> list[str] | None:
    if not token.isalpha() or len(token) < 7:
        return None
    if _is_spelled_word(token):
        return None

    best: tuple[int, tuple[str, str]] | None = None
    for i in range(2, len(token) - 1):
        left = token[:i]
        right = token[i:]
        if len(left) < 2 or len(right) < 2:
            continue
        if not (_is_spelled_word(left) and _is_spelled_word(right)):
            continue
        # Require a strong signal to avoid over-splitting uncertain-but-valid words.
        score = 0
        if left in _FUSED_FALLBACK_SHORT_FUNCTIONS:
            score += 6
        if right in _FUSED_FALLBACK_SHORT_FUNCTIONS:
            score += 6
        if len(left) <= 4:
            score += 2
        if len(right) <= 4:
            score += 2
        if abs(len(left) - len(right)) <= 2:
            score += 1
        if score < 4:
            continue
        if best is None or score > best[0]:
            best = (score, (left, right))

    if best is None:
        return None
    left, right = best[1]
    return ["I" if left == "i" else left, "I" if right == "i" else right]


def _contextual_compound_split(
    token: str, next_token: str, confidence: float
) -> list[str] | None:
    low = token.lower()
    next_norm = normalize_text_basic(next_token or "")
    if confidence > 0.55:
        return None
    if not low.isalpha() or not (8 <= len(low) <= 12):
        return None
    if not next_norm or next_norm not in LYRIC_FUNCTION_WORDS:
        return None
    if not _is_spelled_word(low):
        return None

    best: tuple[int, tuple[str, str]] | None = None
    for i in range(4, len(low) - 3):
        left = low[:i]
        right = low[i:]
        if len(left) < 4 or len(right) < 4:
            continue
        if abs(len(left) - len(right)) > 1:
            continue
        if not (_is_spelled_word(left) and _is_spelled_word(right)):
            continue
        score = 0
        if len(left) == len(right):
            score += 3
        if right.endswith(("ing", "ed", "s")):
            score -= 3
        if right.endswith(("ive", "all", "ight", "ove")):
            score += 1
        if score < 2:
            continue
        if best is None or score > best[0]:
            best = (score, (left, right))

    if best is None:
        return None
    left, right = best[1]
    return [_case_like(token, left), right]


def _fallback_spell_guess(token: str) -> str | None:
    checker = _fallback_spell_checker()
    if checker is None:
        return None
    try:
        missed = checker.checkSpellingOfString_startingAt_(token, 0)
        if missed.length <= 0:
            return None
        guesses = checker.guessesForWordRange_inString_language_inSpellDocumentWithTag_(
            missed, token, "en", 0
        )
        if not guesses:
            return None
        return str(guesses[0])
    except Exception:
        return None


def _is_safe_spell_guess_correction(source: str, guess: str) -> bool:
    s = source.lower()
    g = guess.lower()
    if not s.isalpha() or not g.isalpha():
        return False
    if s == g:
        return False
    if len(s) < 4 or len(g) < 4:
        return False
    if abs(len(s) - len(g)) > 1:
        return False
    if SequenceMatcher(None, s, g).ratio() < 0.75:
        return False
    # Require shared suffix or prefix to avoid wild substitutions.
    if not (s[:2] == g[:2] or s[-3:] == g[-3:] or s[1:] == g[1:]):
        return False
    return True


def _ocr_substitution_candidates(token: str) -> list[str]:
    low = token.lower()
    if not low.isalpha() or len(low) < 3 or len(low) > 8:
        return []

    candidates: set[str] = set()
    chars = list(low)
    # 1-edit substitutions
    for i, ch in enumerate(chars):
        for repl in _OCR_SUB_CHAR_MAP.get(ch, ()):
            if repl == ch:
                continue
            cand_chars = chars.copy()
            cand_chars[i] = repl
            candidates.add("".join(cand_chars))
    # 2-edit substitutions (bounded to short invalid tokens)
    if len(low) <= 6 and not _is_spelled_word(low):
        one_edit = list(candidates)
        for base in one_edit:
            bchars = list(base)
            for i, ch in enumerate(bchars):
                for repl in _OCR_SUB_CHAR_MAP.get(ch, ()):
                    if repl == ch:
                        continue
                    cand_chars = bchars.copy()
                    cand_chars[i] = repl
                    candidates.add("".join(cand_chars))

    out = []
    for cand in sorted(candidates):
        if cand == low:
            continue
        if _is_spelled_word(cand):
            out.append(cand)
    return out


def _ocr_insertion_candidates(token: str) -> list[str]:
    low = token.lower()
    if not low.isalpha() or len(low) < 2 or len(low) > 5:
        return []
    if _is_spelled_word(low):
        return []
    alphabet = ("w", "l", "i", "e", "o", "a", "u", "r", "n")
    out = []
    seen: set[str] = set()
    for i in range(len(low) + 1):
        for ch in alphabet:
            cand = low[:i] + ch + low[i:]
            if cand in seen:
                continue
            seen.add(cand)
            if _is_spelled_word(cand):
                out.append(cand)
    return out


def _best_ocr_substitution(token: str) -> str | None:
    low = token.lower()
    if _is_spelled_word(low):
        return None
    best: tuple[float, str] | None = None
    for cand in _ocr_substitution_candidates(low):
        score = SequenceMatcher(None, low, cand).ratio()
        # Lower threshold than spell-guess path because OCR confusions can require 2 edits.
        if score < 0.58:
            continue
        if best is None or score > best[0]:
            best = (score, cand)
    if best is None:
        for cand in _ocr_insertion_candidates(low):
            score = SequenceMatcher(None, low, cand).ratio()
            if score < 0.72:
                continue
            if best is None or score > best[0]:
                best = (score, cand)
    if best is None:
        return None
    return _case_like(token, best[1])


def _maybe_repair_output_token(text: str, confidence: float) -> str:
    token = text.strip()
    if not token or " " in token:
        return token
    low = token.lower()
    if not low.isalpha():
        return token
    if _is_spelled_word(low):
        return token
    if _looks_fused_prefix_candidate(token):
        return token
    if confidence > 0.55 and not _allow_token_level_ocr_fallback(token):
        return token

    ocr_candidate = _best_ocr_substitution(token)
    if ocr_candidate and ocr_candidate != token:
        return ocr_candidate

    guess = _fallback_spell_guess(token)
    if not guess:
        return token
    if not _is_safe_spell_guess_correction(token, guess):
        return token
    return _case_like(token, guess)


def _maybe_expand_colloquial_token(text: str, confidence: float) -> list[str] | None:
    token = text.strip()
    if not token:
        return None
    if confidence > 0.55:
        return None

    low = token.lower()
    exp = _COLLOQUIAL_EXPANSIONS.get(low)
    if exp:
        first, second = exp
        return [_case_like(token, first), second]

    # Normalize dropped-g gerunds (singin' -> singing) conservatively.
    compact = low.replace("’", "'")
    if re.fullmatch(r"[a-z]{3,}in'", compact):
        return [_case_like(token, compact[:-1] + "g")]
    return None


def _maybe_restore_contraction_token(text: str, confidence: float) -> str | None:
    token = text.strip()
    if confidence > 0.55:
        return None
    restored = _CONTRACTION_RESTORES.get(token.lower())
    if not restored:
        return None
    return _case_like(token, restored)


def _maybe_contextual_inflection_token(
    text: str, confidence: float, prev_token: str, next_token: str
) -> str | None:
    token = text.strip()
    low = token.lower()
    if confidence > 0.55 or not low.isalpha() or len(low) < 2:
        return None
    prev_norm = normalize_text_basic(prev_token or "")
    next_norm = normalize_text_basic(next_token or "")

    # Singular/plural agreement hints around be-verbs.
    if next_norm in {"are", "were"} and _is_spelled_word(low + "s"):
        return _case_like(token, low + "s")
    if next_norm in {"is", "was"} and low.endswith("s") and _is_spelled_word(low[:-1]):
        return _case_like(token, low[:-1])

    # Optional past-tense normalization in a narrow, common pattern ("___ as").
    if next_norm == "as" and _is_spelled_word(low + "ed"):
        return _case_like(token, low + "ed")

    # Be-verb + dropped-g gerund ("are singin" -> "are singing").
    if prev_norm in {"am", "is", "are", "was", "were", "be", "been"}:
        if low.endswith("in") and _is_spelled_word(low + "g"):
            return _case_like(token, low + "g")

    # Directional/action context often loses a trailing "w" in OCR ("Ble down" -> "Blew down").
    if next_norm in {"down", "up", "away"} and _is_spelled_word(low + "w"):
        return _case_like(token, low + "w")

    # Line-start interjection confusion ("Aw who" -> "Oh who").
    if prev_norm == "" and low == "aw" and next_norm in {"who", "what", "why", "yeah"}:
        return _case_like(token, "oh")

    # Title/name OCR prefix drop ("Saint eter" -> "Saint Peter").
    if prev_norm in {"saint", "st"} and len(low) >= 3 and not _is_spelled_word(low):
        candidates: list[str] = []
        for ch in "abcdefghijklmnopqrstuvwxyz":
            cand = ch + low
            if _is_spelled_word(cand):
                candidates.append(cand)
        if len(candidates) == 1:
            return _case_like(token, candidates[0])

    return None


def _split_fused_output_words(  # noqa: C901
    words: list[dict[str, Any]],
    *,
    reconstruction_meta: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    allow_fallback = _is_high_uncertainty_reconstruction(reconstruction_meta)
    for rec_idx, rec in enumerate(words):
        txt = str(rec.get("text", "")).strip()
        conf = float(rec.get("confidence", 0.0))
        prev_txt = (
            str(words[rec_idx - 1].get("text", "")).strip() if rec_idx - 1 >= 0 else ""
        )
        next_txt = (
            str(words[rec_idx + 1].get("text", "")).strip()
            if rec_idx + 1 < len(words)
            else ""
        )
        normalized = normalize_ocr_line(txt).strip()
        parts = [p for p in normalized.split() if p]
        if len(parts) <= 1 and (allow_fallback or _allow_token_level_ocr_fallback(txt)):
            fallback_parts = _fallback_split_fused_token(txt)
            if fallback_parts:
                parts = fallback_parts
        if len(parts) <= 1 and not allow_fallback:
            fallback_parts = _fallback_spell_validated_split(txt.lower())
            if fallback_parts:
                parts = fallback_parts
        if len(parts) <= 1:
            fallback_parts = _contextual_compound_split(txt, next_txt, conf)
            if fallback_parts:
                parts = fallback_parts
        if len(parts) <= 1:
            single = parts[0] if parts else txt
            expanded_single = _maybe_expand_colloquial_token(single, conf)
            if expanded_single:
                parts = expanded_single
            else:
                contraction_restored = _maybe_restore_contraction_token(single, conf)
                if contraction_restored:
                    patched = dict(rec)
                    patched["text"] = contraction_restored
                    out.append(patched)
                    continue
                inflected = _maybe_contextual_inflection_token(
                    single, conf, prev_txt, next_txt
                )
                if inflected and inflected != single:
                    patched = dict(rec)
                    patched["text"] = inflected
                    out.append(patched)
                    continue
                repaired_single = _maybe_repair_output_token(single, conf)
                if repaired_single != txt and repaired_single:
                    patched = dict(rec)
                    patched["text"] = repaired_single
                    out.append(patched)
                else:
                    out.append(rec)
                continue

        start = float(rec.get("start", 0.0))
        end = float(rec.get("end", start + 0.1))
        span = max(0.1, end - start)
        expanded_parts: list[str] = []
        for p in parts:
            expanded = _maybe_expand_colloquial_token(p, conf)
            if expanded:
                expanded_parts.extend(expanded)
            else:
                expanded_parts.append(_maybe_repair_output_token(p, conf))
        parts = expanded_parts

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
