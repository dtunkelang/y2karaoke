from __future__ import annotations

import statistics
from typing import Any

_CONFUSABLE_CHARS = set("ohil1|!")


def _compact(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _as_token(words: list[dict[str, Any]]) -> str:
    if not words:
        return ""
    parts = [str(w.get("text", "")).strip() for w in words]
    parts = [p for p in parts if p]
    if not parts:
        return ""
    # Tight visual group => likely same lexical token, so concatenate.
    return "".join(parts)


def _is_confusable_short_token(text: str) -> bool:
    compact = _compact(text)
    if not compact or len(compact) > 8:
        return False
    return all(ch in _CONFUSABLE_CHARS for ch in compact)


def _robust_scale(ln_w: list[dict[str, Any]]) -> float:
    widths = [max(1.0, float(w.get("w", 1.0))) for w in ln_w]
    heights = [max(1.0, float(w.get("h", 1.0))) for w in ln_w]
    if not widths:
        return 1.0
    # Character-like horizontal scale proxy; robust to occasional long words.
    return max(1.0, min(statistics.median(widths), statistics.median(heights) * 1.4))


def _estimate_tiny_merge_threshold(norm_gaps: list[float]) -> float:
    if not norm_gaps:
        return 0.08
    s = sorted(norm_gaps)
    low_half = s[: max(1, len(s) // 2)]
    base = statistics.median(low_half)
    # Conservative merge: only near-touching components should be glued.
    return max(0.03, min(0.08, base * 0.45 + 0.015))


def _pair_merge_threshold(
    merge_threshold: float,
    *,
    prev_text: str,
    cur_text: str,
) -> float:
    prev_compact = _compact(prev_text)
    cur_compact = _compact(cur_text)
    pair_threshold = merge_threshold
    if len(prev_compact) > 1 or len(cur_compact) > 1:
        pair_threshold = min(pair_threshold, 0.04)
    if "oh" in {prev_compact, cur_compact}:
        pair_threshold = min(pair_threshold, 0.03)
    return pair_threshold


def segment_line_tokens_by_visual_gaps(ln_w: list[dict[str, Any]]) -> list[str]:
    if not ln_w:
        return []
    words = sorted(ln_w, key=lambda w: float(w.get("x", 0.0)))
    if len(words) == 1:
        text = str(words[0].get("text", "")).strip()
        return [text] if text else []

    out: list[str] = []
    i = 0
    while i < len(words):
        text = str(words[i].get("text", "")).strip()
        if not _is_confusable_short_token(text):
            if _compact(text) or text == "'":
                out.append(text)
            i += 1
            continue

        j = i + 1
        while j < len(words):
            nxt_text = str(words[j].get("text", "")).strip()
            if not _is_confusable_short_token(nxt_text):
                break
            j += 1

        run = words[i:j]
        if len(run) == 1:
            if _compact(text) or text == "'":
                out.append(text)
            i = j
            continue

        scale = _robust_scale(run)
        gaps = []
        for k in range(len(run) - 1):
            right = float(run[k].get("x", 0.0)) + float(run[k].get("w", 0.0))
            nxt = float(run[k + 1].get("x", 0.0))
            gaps.append(max(0.0, nxt - right))
        norm_gaps = [g / scale for g in gaps]
        merge_threshold = _estimate_tiny_merge_threshold(norm_gaps)

        groups: list[list[dict[str, Any]]] = [[run[0]]]
        for k, w in enumerate(run[1:], start=1):
            prev = run[k - 1]
            pair_threshold = _pair_merge_threshold(
                merge_threshold,
                prev_text=str(prev.get("text", "")).strip(),
                cur_text=str(w.get("text", "")).strip(),
            )

            if norm_gaps[k - 1] > pair_threshold:
                groups.append([w])
            else:
                groups[-1].append(w)

        for grp in groups:
            tok = _as_token(grp)
            if _compact(tok):
                out.append(tok)
        i = j

    return out
