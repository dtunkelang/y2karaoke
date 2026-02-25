from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..text_utils import normalize_text_basic


@dataclass
class _EntryRef:
    index: int
    entry: dict[str, Any]
    first_visible: float
    last: float
    y: float


@dataclass
class _Block:
    block_id: int
    first_visible: float
    last: float
    rows: list[_EntryRef]


def _entry_first_visible(ent: dict[str, Any]) -> float:
    return float(ent.get("first_visible", ent.get("first", 0.0)) or 0.0)


def _entry_last(ent: dict[str, Any]) -> float:
    return float(ent.get("last", ent.get("first", 0.0)) or 0.0)


def _entry_y(ent: dict[str, Any]) -> float:
    return float(ent.get("y", 0.0) or 0.0)


def _tokens(ent: dict[str, Any]) -> list[str]:
    return [t for t in normalize_text_basic(str(ent.get("text", ""))).split() if t]


def build_persistent_visibility_blocks(
    entries: list[dict[str, Any]],
    *,
    max_visibility_start_span: float = 1.4,
    min_overlap_sec: float = 0.75,
    max_block_size: int = 6,
) -> list[_Block]:
    refs: list[_EntryRef] = []
    for idx, ent in enumerate(entries):
        fv = _entry_first_visible(ent)
        last = _entry_last(ent)
        if last <= fv:
            continue
        refs.append(_EntryRef(idx, ent, fv, last, _entry_y(ent)))

    if len(refs) < 2:
        return []

    refs.sort(key=lambda r: (r.first_visible, r.last, r.y, r.index))

    blocks: list[_Block] = []
    curr: list[_EntryRef] = []
    curr_vs_min = curr_vs_max = 0.0
    curr_last_max = 0.0

    def flush() -> None:
        nonlocal curr
        if len(curr) >= 2:
            rows = sorted(curr, key=lambda r: (r.y, r.first_visible, r.index))
            blocks.append(
                _Block(
                    block_id=len(blocks),
                    first_visible=min(r.first_visible for r in curr),
                    last=max(r.last for r in curr),
                    rows=rows[:max_block_size],
                )
            )
        curr = []

    for ref in refs:
        if not curr:
            curr = [ref]
            curr_vs_min = curr_vs_max = ref.first_visible
            curr_last_max = ref.last
            continue

        prospective_vs_min = min(curr_vs_min, ref.first_visible)
        prospective_vs_max = max(curr_vs_max, ref.first_visible)
        overlap = min(curr_last_max, ref.last) - max(curr_vs_min, ref.first_visible)
        rec_dur = max(0.0, ref.last - ref.first_visible)
        span_limit = max_visibility_start_span
        min_overlap = min_overlap_sec
        if rec_dur <= 0.8:
            span_limit = max(span_limit, 1.8)
            min_overlap = min(min_overlap, 0.15)
        if (
            (prospective_vs_max - prospective_vs_min) > span_limit
            or overlap < min_overlap
            or len(curr) >= max_block_size
        ):
            flush()
            curr = [ref]
            curr_vs_min = curr_vs_max = ref.first_visible
            curr_last_max = ref.last
            continue

        curr.append(ref)
        curr_vs_min = prospective_vs_min
        curr_vs_max = prospective_vs_max
        curr_last_max = max(curr_last_max, ref.last)

    flush()
    return blocks


def _append_entry_words(dst: dict[str, Any], src: dict[str, Any]) -> None:
    dst_words = list(dst.get("words", []) or [])
    src_words = list(src.get("words", []) or [])
    if not dst_words or not src_words:
        return
    for i, w in enumerate(src_words):
        dst_words.append(dict(w))
        # word_index is reconstructed later downstream if needed; keep shape here.
    dst["words"] = dst_words
    dst["text"] = " ".join(str(w.get("text", "")) for w in dst_words if w.get("text"))
    dst["last"] = max(_entry_last(dst), _entry_last(src))
    if (
        "w_rois" in dst
        and "w_rois" in src
        and isinstance(dst["w_rois"], list)
        and isinstance(src["w_rois"], list)
    ):
        dst["w_rois"] = list(dst["w_rois"]) + list(src["w_rois"])


def _consolidate_fragment_rows_in_blocks(
    entries: list[dict[str, Any]], blocks: list[_Block]
) -> list[dict[str, Any]]:
    drops: set[int] = set()
    for block in blocks:
        rows = block.rows
        for i in range(len(rows) - 1):
            a = rows[i]
            b = rows[i + 1]
            if a.index in drops or b.index in drops:
                continue
            ta = _tokens(a.entry)
            tb = _tokens(b.entry)
            if not ta or not tb:
                continue
            if len(tb) > 3 or len(tb) >= len(ta):
                continue
            if ta[-len(tb) :] != tb:
                continue
            if (
                abs(
                    float(a.entry.get("first", a.first_visible))
                    - float(b.entry.get("first", b.first_visible))
                )
                > 0.45
            ):
                continue
            if b.y <= a.y + 10.0:
                continue
            _append_entry_words(a.entry, b.entry)
            drops.add(b.index)

    if not drops:
        return entries
    return [ent for idx, ent in enumerate(entries) if idx not in drops]


def apply_block_first_ordering_to_persistent_entries(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return persistent entries reordered by explicit visibility blocks + chronology."""
    if len(entries) < 2:
        return entries

    blocks = build_persistent_visibility_blocks(entries)
    if not blocks:
        return entries

    entries2 = _consolidate_fragment_rows_in_blocks(entries, blocks)
    # Rebuild blocks after any drops/merges.
    blocks = build_persistent_visibility_blocks(entries2)

    block_members: dict[int, tuple[int, float, int, float]] = {}
    for block in blocks:
        for row_order, row in enumerate(block.rows):
            block_members[row.index] = (
                block.block_id,
                block.first_visible,
                row_order,
                row.y,
            )
            meta = row.entry.get("reconstruction_meta")
            if not isinstance(meta, dict):
                meta = {}
                row.entry["reconstruction_meta"] = meta
            bf = meta.get("block_first")
            if not isinstance(bf, dict):
                bf = {}
                meta["block_first"] = bf
            bf["block_id"] = block.block_id
            bf["row_order"] = row_order
            bf["row_y"] = row.y

    ordered_records: list[tuple[tuple[float, int, int, float, int], dict[str, Any]]] = (
        []
    )
    for idx, ent in enumerate(entries2):
        mem = block_members.get(idx)
        if mem is not None:
            block_id, bstart, row_order, row_y = mem
            key = (float(bstart), 0, int(block_id), float(row_order), idx)
            ordered_records.append((key, ent))
            continue
        fv = _entry_first_visible(ent)
        key = (fv, 1, 0, _entry_y(ent), idx)
        ordered_records.append((key, ent))

    ordered_records.sort(key=lambda rec: rec[0])
    return [ent for _k, ent in ordered_records]
