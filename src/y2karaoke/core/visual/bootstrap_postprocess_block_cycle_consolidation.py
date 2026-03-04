"""Consolidation helpers for block-first postprocess rows."""

from __future__ import annotations

from typing import Any, Callable, Optional


def consolidate_block_first_fragment_rows(  # noqa: C901
    lines_out: list[dict[str, Any]],
    *,
    normalize_text_basic_fn: Callable[[str], str],
) -> None:
    """Merge obvious fragment rows into fuller rows within a block-first block."""
    if len(lines_out) < 2:
        return

    def _block_meta(line: dict[str, Any]) -> Optional[dict[str, Any]]:
        meta = line.get("_reconstruction_meta", {})
        if not isinstance(meta, dict):
            return None
        bf = meta.get("block_first")
        return bf if isinstance(bf, dict) else None

    def _cycle_count(line: dict[str, Any]) -> int:
        bf = _block_meta(line)
        if not bf:
            return 1
        try:
            return int(bf.get("cycle_count", 1) or 1)
        except Exception:
            return 1

    def _block_id(line: dict[str, Any]) -> object:
        bf = _block_meta(line)
        return bf.get("block_id") if bf else None

    def _row_order(line: dict[str, Any]) -> int:
        bf = _block_meta(line)
        if not bf:
            return 10_000
        try:
            return int(bf.get("row_order", 10_000))
        except Exception:
            return 10_000

    def _tokens(line: dict[str, Any]) -> list[str]:
        return [
            t for t in normalize_text_basic_fn(str(line.get("text", ""))).split() if t
        ]

    def _append_words(dst: dict[str, Any], src: dict[str, Any]) -> None:
        d_words = list(dst.get("words", []) or [])
        s_words = list(src.get("words", []) or [])
        if not d_words or not s_words:
            return
        start_idx = len(d_words)
        for k, w in enumerate(s_words):
            w2 = dict(w)
            w2["word_index"] = start_idx + k + 1
            d_words.append(w2)
        dst["words"] = d_words
        dst["text"] = " ".join(str(w.get("text", "")) for w in d_words if w.get("text"))
        dst["end"] = float(
            s_words[-1].get("end", dst.get("end", dst.get("start", 0.0)))
        )
        dst["confidence"] = max(
            float(dst.get("confidence", 0.0) or 0.0),
            float(src.get("confidence", 0.0) or 0.0),
        )

    drops: set[int] = set()
    for i in range(len(lines_out) - 1):
        if i in drops:
            continue
        a = lines_out[i]
        b = lines_out[i + 1]
        bid_a = _block_id(a)
        if bid_a is None or bid_a != _block_id(b):
            continue
        if _cycle_count(a) > 1 or _cycle_count(b) > 1:
            continue
        if abs(_row_order(a) - _row_order(b)) > 1:
            continue
        ta = _tokens(a)
        tb = _tokens(b)
        if not ta or not tb:
            continue
        if len(tb) > 3 or len(ta) < 2:
            continue
        if len(tb) >= len(ta):
            continue
        if ta[-len(tb) :] != tb:
            continue
        sa = float(a.get("start", 0.0) or 0.0)
        sb = float(b.get("start", 0.0) or 0.0)
        if abs(sa - sb) > 0.45:
            continue
        ya = float(a.get("y", 0.0) or 0.0)
        yb = float(b.get("y", 0.0) or 0.0)
        if yb <= ya + 10.0:
            continue
        _append_words(a, b)
        drops.add(i + 1)

    if not drops:
        return
    lines_out[:] = [ln for idx, ln in enumerate(lines_out) if idx not in drops]
    for idx, ln in enumerate(lines_out):
        ln["line_index"] = idx + 1
