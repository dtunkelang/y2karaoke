"""Block/cycle-oriented postprocessing passes for visual bootstrap outputs."""

from __future__ import annotations

from typing import Any, Optional

from ..text_utils import normalize_text_basic
from .bootstrap_postprocess_cycle_normalization import (
    normalize_block_first_repeated_cycles as _normalize_block_first_repeated_cycles_impl,
    rebalance_compressed_middle_four_line_sequences as _rebalance_compressed_middle_four_line_sequences_impl,
)
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
from .bootstrap_postprocess_vocalization_trimming import (
    trim_leading_vocalization_in_block_first_cycle_rows as _trim_leading_vocalization_in_block_first_cycle_rows_impl,
    trim_leading_vocalization_prefixes as _trim_leading_vocalization_prefixes_impl,
)
from .bootstrap_postprocess_block_cycle_dedupe import (
    dedupe_block_first_cycle_rows as _dedupe_block_first_cycle_rows_impl,
)
from .bootstrap_postprocess_interstitial_retime import (
    retime_short_interstitial_output_lines as _retime_short_interstitial_output_lines_impl,
)
from .reconstruction import snap
from .bootstrap_postprocess_line_passes import (
    _HUM_NOISE_TOKENS,
    _VOCALIZATION_NOISE_TOKENS,
    _line_duplicate_quality_score,
)


def _trim_leading_vocalization_prefixes(lines_out: list[dict[str, Any]]) -> None:
    _trim_leading_vocalization_prefixes_impl(
        lines_out,
        vocalization_noise_tokens_set=_VOCALIZATION_NOISE_TOKENS,
    )


def _repair_strong_local_chronology_inversions(lines_out: list[dict[str, Any]]) -> None:
    _repair_strong_local_chronology_inversions_impl(
        lines_out,
        line_duplicate_quality_score_fn=_line_duplicate_quality_score,
    )


_repair_large_adjacent_time_inversions = _repair_large_adjacent_time_inversions_impl


def _dedupe_block_first_cycle_rows(lines_out: list[dict[str, Any]]) -> None:
    _dedupe_block_first_cycle_rows_impl(lines_out)


def _trim_leading_vocalization_in_block_first_cycle_rows(
    lines_out: list[dict[str, Any]],
) -> None:
    _trim_leading_vocalization_in_block_first_cycle_rows_impl(
        lines_out,
        vocalization_noise_tokens_set=_VOCALIZATION_NOISE_TOKENS,
    )


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


_filter_singer_label_prefixes = _filter_singer_label_prefixes_impl


_identify_banned_prefixes = _identify_banned_prefixes_impl


_remove_prefix_from_line = _remove_prefix_from_line_impl


def _retime_short_interstitial_output_lines(lines_out: list[dict[str, Any]]) -> None:
    _retime_short_interstitial_output_lines_impl(lines_out, snap_fn=snap)


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
    _normalize_block_first_repeated_cycles_impl(
        lines_out, normalize_text_basic_fn=normalize_text_basic
    )


def _rebalance_compressed_middle_four_line_sequences(  # noqa: C901
    lines_out: list[dict[str, Any]],
) -> None:
    _rebalance_compressed_middle_four_line_sequences_impl(
        lines_out,
        normalize_text_basic_fn=normalize_text_basic,
        snap_fn=snap,
    )
