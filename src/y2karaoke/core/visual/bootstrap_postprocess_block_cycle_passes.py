"""Block/cycle-oriented postprocessing passes for visual bootstrap outputs."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Optional

from ..text_utils import normalize_text_basic
from .bootstrap_postprocess_block_cycle_filters import (
    filter_singer_label_prefixes as _filter_singer_label_prefixes_impl,
    identify_banned_prefixes as _identify_banned_prefixes_impl,
    remove_prefix_from_line as _remove_prefix_from_line_impl,
    remove_vocalization_noise_runs as _remove_vocalization_noise_runs_impl,
    vocalization_noise_tokens as _vocalization_noise_tokens_impl,
)
from .bootstrap_postprocess_chronology import (
    repair_large_adjacent_time_inversions as _repair_large_adjacent_time_inversions_impl,
    repair_strong_local_chronology_inversions as _repair_strong_local_chronology_inversions_impl,
)
from .reconstruction import snap
from .bootstrap_postprocess_line_passes import (
    _HUM_NOISE_TOKENS,
    _VOCALIZATION_NOISE_TOKENS,
    _line_duplicate_quality_score,
)


def _trim_leading_vocalization_prefixes(lines_out: list[dict[str, Any]]) -> None:
    """Drop a leading vocalization token when the same phrase appears nearby without it.

    This targets rows like "Ooh take that money" where the ad-lib is fused onto a
    lyric row in one repetition but the same row appears without the prefix in nearby
    repeated cycles. Keep it conservative: low confidence only and requires local
    support for the suffix phrase.
    """
    if len(lines_out) < 2:
        return

    norm_lines = [normalize_text_basic(str(ln.get("text", ""))) for ln in lines_out]
    token_lines = [[t for t in n.split() if t] for n in norm_lines]

    for i, ln in enumerate(lines_out):
        words = ln.get("words", [])
        toks = token_lines[i]
        if len(words) != len(toks) or len(toks) < 3:
            continue
        lead = toks[0]
        if lead not in _VOCALIZATION_NOISE_TOKENS:
            continue
        suffix = toks[1:]
        if len(suffix) < 2:
            continue
        conf = float(ln.get("confidence", 0.0) or 0.0)
        if conf > 0.4:
            continue
        start = float(ln.get("start", 0.0) or 0.0)
        end = float(ln.get("end", start) or start)

        supported = False
        for j in range(max(0, i - 10), min(len(lines_out), i + 11)):
            if j == i:
                continue
            other = lines_out[j]
            other_toks = token_lines[j]
            if other_toks != suffix:
                continue
            other_start = float(other.get("start", 0.0) or 0.0)
            other_end = float(other.get("end", other_start) or other_start)
            # Nearby in time or part of the same repeated cluster region.
            if max(0.0, min(end, other_end) - max(start, other_start)) > 0.0 or (
                abs(other_start - start) <= 20.0
            ):
                supported = True
                break
        if not supported:
            continue

        kept_words = words[1:]
        if len(kept_words) < 2:
            continue
        ln["words"] = kept_words
        ln["text"] = " ".join(
            str(w.get("text", "")) for w in kept_words if w.get("text")
        )
        ln["start"] = float(kept_words[0].get("start", start) or start)
        for wi, w in enumerate(kept_words):
            w["word_index"] = wi + 1


def _repair_strong_local_chronology_inversions(lines_out: list[dict[str, Any]]) -> None:
    _repair_strong_local_chronology_inversions_impl(
        lines_out,
        line_duplicate_quality_score_fn=_line_duplicate_quality_score,
    )


def _repair_large_adjacent_time_inversions(lines_out: list[dict[str, Any]]) -> None:
    _repair_large_adjacent_time_inversions_impl(lines_out)


def _dedupe_block_first_cycle_rows(lines_out: list[dict[str, Any]]) -> None:
    """Keep one line per (block_id, cycle_index, row_order) for block-first cycle rows."""
    if not lines_out:
        return

    def _cycle_key(line: dict[str, Any]) -> tuple[object, int, int] | None:
        meta = line.get("_reconstruction_meta", {})
        if not isinstance(meta, dict):
            return None
        bf = meta.get("block_first")
        if not isinstance(bf, dict):
            return None
        cycle_count = int(bf.get("cycle_count", 1) or 1)
        if cycle_count <= 1:
            return None
        block_id = bf.get("block_id")
        cycle_idx = bf.get("cycle_index")
        row_order = bf.get("row_order")
        if cycle_idx is None or row_order is None:
            return None
        return (block_id, int(cycle_idx), int(row_order))

    def _quality(line: dict[str, Any]) -> tuple[float, float, int, int]:
        conf = float(line.get("confidence", 0.0) or 0.0)
        start = float(line.get("start", 0.0) or 0.0)
        end = float(line.get("end", start) or start)
        dur = max(0.0, end - start)
        toks = [t for t in normalize_text_basic(str(line.get("text", ""))).split() if t]
        words = line.get("words", []) or []
        return (conf, dur, len(toks), len(words))

    keep_by_key: dict[tuple[object, int, int], int] = {}
    drops: set[int] = set()
    for idx, ln in enumerate(lines_out):
        key = _cycle_key(ln)
        if key is None:
            continue
        prev_idx = keep_by_key.get(key)
        if prev_idx is None:
            keep_by_key[key] = idx
            continue
        if _quality(ln) > _quality(lines_out[prev_idx]):
            drops.add(prev_idx)
            keep_by_key[key] = idx
        else:
            drops.add(idx)

    if not drops:
        return

    lines_out[:] = [ln for idx, ln in enumerate(lines_out) if idx not in drops]
    for i, ln in enumerate(lines_out):
        ln["line_index"] = i + 1


def _trim_leading_vocalization_in_block_first_cycle_rows(  # noqa: C901
    lines_out: list[dict[str, Any]],
) -> None:
    """Trim leading vocalization tokens on cycle rows when a sibling cycle has the suffix."""
    if len(lines_out) < 2:
        return

    def _bf(line: dict[str, Any]) -> dict[str, Any] | None:
        meta = line.get("_reconstruction_meta", {})
        if not isinstance(meta, dict):
            return None
        bf = meta.get("block_first")
        return bf if isinstance(bf, dict) else None

    entries: list[tuple[int, dict[str, Any], tuple[str, ...]]] = []
    by_block: dict[object, list[int]] = {}
    for idx, ln in enumerate(lines_out):
        bf = _bf(ln)
        if not bf:
            continue
        try:
            if int(bf.get("cycle_count", 1) or 1) <= 1:
                continue
        except Exception:
            continue
        toks = tuple(
            t for t in normalize_text_basic(str(ln.get("text", ""))).split() if t
        )
        entries.append((idx, bf, toks))
        by_block.setdefault(bf.get("block_id"), []).append(idx)

    if not by_block:
        return

    token_map = {
        idx: tuple(
            t
            for t in normalize_text_basic(str(lines_out[idx].get("text", ""))).split()
            if t
        )
        for idxs in by_block.values()
        for idx in idxs
    }

    for idxs in by_block.values():
        tok_set = {token_map[i] for i in idxs if token_map.get(i)}
        for idx in idxs:
            toks_list = list(token_map.get(idx, ()))
            if len(toks_list) < 3:
                continue
            if toks_list[0] not in _VOCALIZATION_NOISE_TOKENS:
                continue
            suffix = tuple(toks_list[1:])
            if suffix not in tok_set:
                continue
            ln = lines_out[idx]
            conf = float(ln.get("confidence", 0.0) or 0.0)
            if conf > 0.45:
                continue
            words = list(ln.get("words", []) or [])
            if len(words) != len(toks_list) or len(words) < 3:
                continue
            kept = words[1:]
            ln["words"] = kept
            ln["text"] = " ".join(str(w.get("text", "")) for w in kept if w.get("text"))
            ln["start"] = float(
                kept[0].get("start", ln.get("start", 0.0)) or ln.get("start", 0.0)
            )
            for wi, w in enumerate(kept):
                w["word_index"] = wi + 1


def _vocalization_noise_tokens(line: dict[str, Any]) -> list[str] | None:
    return _vocalization_noise_tokens_impl(
        line, vocalization_noise_tokens_set=_VOCALIZATION_NOISE_TOKENS
    )


def _remove_vocalization_noise_runs(lines_out: list[dict[str, Any]]) -> None:
    _remove_vocalization_noise_runs_impl(
        lines_out,
        vocalization_noise_tokens_set=_VOCALIZATION_NOISE_TOKENS,
        hum_noise_tokens_set=_HUM_NOISE_TOKENS,
    )


def _filter_singer_label_prefixes(
    lines_out: list[dict[str, Any]], artist: Optional[str]
) -> None:
    _filter_singer_label_prefixes_impl(lines_out, artist)


def _identify_banned_prefixes(
    lines_out: list[dict[str, Any]], artist: Optional[str]
) -> set[str]:
    return _identify_banned_prefixes_impl(lines_out, artist)


def _remove_prefix_from_line(line: dict[str, Any], banned_prefixes: set[str]) -> None:
    _remove_prefix_from_line_impl(line, banned_prefixes)


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


def _consolidate_block_first_fragment_rows(  # noqa: C901
    lines_out: list[dict[str, Any]],
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
        return [t for t in normalize_text_basic(str(line.get("text", ""))).split() if t]

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
        # Only merge adjacent rows in the same block and nearby row order.
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
        # Merge suffix fragment rows that start nearly together.
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


def _normalize_block_first_row_timings(  # noqa: C901
    lines_out: list[dict[str, Any]],
) -> None:
    """Normalize starts within block-first rows using row order.

    When block-first ordering is enabled, rows may still carry inconsistent starts
    from the line-first refinement path. This pass preserves list order and assigns
    a monotonic start progression within each block using observed starts.
    """
    if len(lines_out) < 2:
        return

    def _block_id(line: dict[str, Any]) -> object:
        meta = line.get("_reconstruction_meta", {})
        if not isinstance(meta, dict):
            return None
        bf = meta.get("block_first")
        if not isinstance(bf, dict):
            return None
        try:
            if int(bf.get("cycle_count", 1) or 1) > 1:
                return None
        except Exception:
            return None
        return bf.get("block_id")

    def _row_order(line: dict[str, Any]) -> int:
        meta = line.get("_reconstruction_meta", {})
        if not isinstance(meta, dict):
            return 10_000
        bf = meta.get("block_first")
        if not isinstance(bf, dict):
            return 10_000
        try:
            return int(bf.get("row_order", 10_000))
        except Exception:
            return 10_000

    def _shift_line(line: dict[str, Any], new_start: float) -> None:
        old_start = float(line.get("start", 0.0) or 0.0)
        old_end = float(line.get("end", old_start) or old_start)
        shift = new_start - old_start
        line["start"] = snap(new_start)
        line["end"] = snap(max(new_start + 0.1, old_end + shift))
        for w in line.get("words", []) or []:
            if "start" in w:
                w["start"] = snap(float(w["start"]) + shift)
            if "end" in w:
                w["end"] = snap(float(w["end"]) + shift)

    i = 0
    while i < len(lines_out):
        bid = _block_id(lines_out[i])
        if bid is None:
            i += 1
            continue
        j = i + 1
        while j < len(lines_out) and _block_id(lines_out[j]) == bid:
            j += 1

        block = lines_out[i:j]
        if 2 <= len(block) <= 6:
            starts = [float(ln.get("start", 0.0) or 0.0) for ln in block]
            by_row = sorted(
                block, key=lambda ln: (_row_order(ln), float(ln.get("y", 0.0) or 0.0))
            )
            row_starts = [float(ln.get("start", 0.0) or 0.0) for ln in by_row]
            has_inversion = any(
                row_starts[k] > row_starts[k + 1] + 0.35
                for k in range(len(row_starts) - 1)
            )
            if has_inversion:
                target_starts = sorted(starts)
                prev_end: Optional[float] = None
                for k, ln in enumerate(by_row):
                    new_start = target_starts[k]
                    if prev_end is not None:
                        new_start = max(new_start, prev_end + 0.05)
                    vis_start = ln.get("_visibility_start")
                    vis_end = ln.get("_visibility_end")
                    if vis_start is not None:
                        new_start = max(new_start, float(vis_start) - 0.15)
                    if vis_end is not None:
                        new_start = min(new_start, float(vis_end) - 0.12)
                    old_start = float(ln.get("start", 0.0) or 0.0)
                    if abs(new_start - old_start) > 0.15:
                        _shift_line(ln, new_start)
                    prev_end = float(ln.get("end", new_start) or new_start)
        i = j

    for idx, ln in enumerate(lines_out):
        ln["line_index"] = idx + 1


def _normalize_block_first_repeated_cycles(  # noqa: C901
    lines_out: list[dict[str, Any]],
) -> None:
    """Deduplicate tiny repeated cycle artifacts in block-first outputs.

    Generalized for common karaoke screen row counts (2..5). This targets cases where
    block-first cycle splitting leaves micro-duplicate rows at repeated-screen
    boundaries, which hurts repeat-capped precision.
    """
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
            t for t in normalize_text_basic(str(line.get("text", ""))).split() if t
        )

    # Collect block-first rows by block id.
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

    # Remove near-duplicate rows for same row slot/text within repeated-cycle regions.
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
            # Keep the stronger row; drop micro-duplicates.
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


def _rebalance_compressed_middle_four_line_sequences(  # noqa: C901
    lines_out: list[dict[str, Any]],
) -> None:
    """Spread middle starts when a 4-line run has compressed middle gaps."""

    def _has_later_repeat(seed_idx: int, lookahead: int = 32) -> bool:
        seed = lines_out[seed_idx]
        seed_text = normalize_text_basic(str(seed.get("text", "")))
        seed_tokens = [t for t in seed_text.split() if t]
        if not seed_tokens or len(seed_tokens) > 6:
            return False
        seed_start = float(seed.get("start", 0.0) or 0.0)
        for j in range(seed_idx + 1, min(len(lines_out), seed_idx + 1 + lookahead)):
            other = lines_out[j]
            other_start = float(other.get("start", 0.0) or 0.0)
            if other_start - seed_start < 6.0:
                continue
            other_text = normalize_text_basic(str(other.get("text", "")))
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
        # Avoid chorus/repeated-block windows; this local rebalancing can pull an
        # early repeated line toward a later occurrence.
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
            rec["start"] = snap(new_s)
            rec["end"] = snap(new_s + dur)
            for w in words:
                w["start"] = snap(float(w["start"]) + shift)
                w["end"] = snap(float(w["end"]) + shift)
