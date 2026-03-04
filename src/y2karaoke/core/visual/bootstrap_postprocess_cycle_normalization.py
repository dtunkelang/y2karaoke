"""Cycle normalization passes for block-first bootstrap postprocessing."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Callable, Optional


def normalize_block_first_repeated_cycles(  # noqa: C901
    lines_out: list[dict[str, Any]],
    *,
    normalize_text_basic_fn: Callable[[str], str],
) -> None:
    """Deduplicate tiny repeated cycle artifacts in block-first outputs."""
    if len(lines_out) < 4:
        return

    def _bf_meta(line: dict[str, Any]) -> Optional[dict[str, Any]]:
        meta = line.get("_reconstruction_meta", {})
        if not isinstance(meta, dict):
            return None
        bf = meta.get("block_first")
        return bf if isinstance(bf, dict) else None

    def _block_id(line: dict[str, Any]) -> Optional[int]:
        bf = _bf_meta(line)
        if bf:
            try:
                if int(bf.get("cycle_count", 1) or 1) > 1:
                    return None
            except Exception:
                return None
        bid = bf.get("block_id") if bf else None
        return bid if isinstance(bid, int) else None

    def _row_order(line: dict[str, Any]) -> int:
        bf = _bf_meta(line)
        if not bf:
            return 999
        try:
            return int(bf.get("row_order", 999))
        except Exception:
            return 999

    def _norm_tokens(line: dict[str, Any]) -> tuple[str, ...]:
        return tuple(
            t for t in normalize_text_basic_fn(str(line.get("text", ""))).split() if t
        )

    by_block: dict[int, list[int]] = {}
    for idx, ln in enumerate(lines_out):
        bid = _block_id(ln)
        if bid is None:
            continue
        by_block.setdefault(bid, []).append(idx)

    block_signatures: dict[int, tuple[tuple[str, ...], ...]] = {}
    for bid, idxs in by_block.items():
        rows = [lines_out[i] for i in idxs]
        if not (2 <= len(rows) <= 5):
            continue
        rows = sorted(
            rows, key=lambda ln: (_row_order(ln), float(ln.get("y", 0.0) or 0.0))
        )
        sig = tuple(_norm_tokens(ln) for ln in rows)
        if all(sig):
            block_signatures[bid] = sig

    repeated_bids: set[int] = set()
    for bid, sig in block_signatures.items():
        if block_signatures.get(bid + 1) == sig:
            repeated_bids.add(bid)
            repeated_bids.add(bid + 1)

    if not repeated_bids:
        return

    remove_idxs: set[int] = set()
    seen: dict[tuple[int, tuple[str, ...]], list[int]] = {}
    for idx, ln in enumerate(lines_out):
        if idx in remove_idxs:
            continue
        bid = _block_id(ln)
        if bid not in repeated_bids:
            continue
        ro = _row_order(ln)
        toks = _norm_tokens(ln)
        if not toks:
            continue
        key = (ro, toks)
        start = float(ln.get("start", 0.0) or 0.0)
        end = float(ln.get("end", start) or start)
        span = max(0.0, end - start)
        conf = float(ln.get("confidence", 0.0) or 0.0)
        prior = seen.setdefault(key, [])
        kept = False
        for pidx in list(prior):
            pln = lines_out[pidx]
            pstart = float(pln.get("start", 0.0) or 0.0)
            if abs(start - pstart) > 0.85:
                continue
            pend = float(pln.get("end", pstart) or pstart)
            pspan = max(0.0, pend - pstart)
            pconf = float(pln.get("confidence", 0.0) or 0.0)
            if (span, conf) > (pspan, pconf):
                remove_idxs.add(pidx)
                prior.remove(pidx)
                prior.append(idx)
            else:
                remove_idxs.add(idx)
            kept = True
            break
        if not kept:
            prior.append(idx)

    if remove_idxs:
        lines_out[:] = [ln for i, ln in enumerate(lines_out) if i not in remove_idxs]
        for idx, ln in enumerate(lines_out):
            ln["line_index"] = idx + 1


def rebalance_compressed_middle_four_line_sequences(  # noqa: C901
    lines_out: list[dict[str, Any]],
    *,
    normalize_text_basic_fn: Callable[[str], str],
    snap_fn: Callable[[float], float],
) -> None:
    """Spread middle starts when a 4-line run has compressed middle gaps."""

    def _has_later_repeat(seed_idx: int, lookahead: int = 32) -> bool:
        seed = lines_out[seed_idx]
        seed_text = normalize_text_basic_fn(str(seed.get("text", "")))
        seed_tokens = [t for t in seed_text.split() if t]
        if not seed_tokens or len(seed_tokens) > 6:
            return False
        seed_start = float(seed.get("start", 0.0) or 0.0)
        for j in range(seed_idx + 1, min(len(lines_out), seed_idx + 1 + lookahead)):
            other = lines_out[j]
            other_start = float(other.get("start", 0.0) or 0.0)
            if other_start - seed_start < 6.0:
                continue
            other_text = normalize_text_basic_fn(str(other.get("text", "")))
            if not other_text:
                continue
            if SequenceMatcher(None, seed_text, other_text).ratio() >= 0.9:
                return True
        return False

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
        if any(_has_later_repeat(k) for k in (i, i + 1, i + 2, i + 3)):
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
            rec["start"] = snap_fn(new_s)
            rec["end"] = snap_fn(new_s + dur)
            for w in words:
                w["start"] = snap_fn(float(w["start"]) + shift)
                w["end"] = snap_fn(float(w["end"]) + shift)
