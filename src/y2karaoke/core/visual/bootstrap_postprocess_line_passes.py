"""Line-level postprocessing passes for visual bootstrap outputs."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Optional

from ..text_utils import LYRIC_FUNCTION_WORDS, normalize_text_basic
from .reconstruction import snap

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
_ADLIB_TAIL_TOKENS = {
    "uh",
    "ah",
    "aww",
    "oh",
    "hey",
    "come",
    "on",
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
        ln.pop("_visibility_start", None)
        ln.pop("_visibility_end", None)
        ln.pop("_orig_order", None)


def _reorder_clean_visibility_blocks(  # noqa: C901
    lines_out: list[dict[str, Any]],
) -> None:
    """Sort simple lyric blocks by visibility window, then top-to-bottom.

    This handles the common karaoke layout where one screenful of lyrics is shown
    at a time and lines in a screen are sung in vertical order.
    """
    if len(lines_out) < 2:
        return

    for idx, ln in enumerate(lines_out):
        ln.setdefault("_orig_order", idx)

    enriched: list[tuple[int, float, float, float, float]] = []
    for idx, ln in enumerate(lines_out):
        vs = ln.get("_visibility_start")
        ve = ln.get("_visibility_end")
        if vs is None or ve is None:
            continue
        vsf = float(vs)
        vef = float(ve)
        if vef <= vsf:
            continue
        enriched.append(
            (
                idx,
                vsf,
                vef,
                float(ln.get("y", 0.0) or 0.0),
                float(ln.get("start", 0.0) or 0.0),
            )
        )
    if len(enriched) < 2:
        return

    enriched.sort(key=lambda t: (t[1], t[2], t[3], t[4], t[0]))

    blocks: list[list[tuple[int, float, float, float, float]]] = []
    current: list[tuple[int, float, float, float, float]] = []
    current_start = -1.0
    current_end = -1.0
    for rec in enriched:
        if not current:
            current = [rec]
            current_start = rec[1]
            current_end = rec[2]
            continue

        _, rec_vs, rec_ve, _, _ = rec
        overlap_with_block = min(current_end, rec_ve) - max(current_start, rec_vs)
        # For simple karaoke layouts, all lines in a screen usually become visible
        # together. Cluster primarily by visibility onset, with an overlap check to
        # avoid joining distant repeated blocks.
        if (rec_vs - current_start) <= 1.4 and overlap_with_block > 0.2:
            current.append(rec)
            current_start = min(current_start, rec_vs)
            current_end = max(current_end, rec[2])
        else:
            blocks.append(current)
            current = [rec]
            current_start = rec[1]
            current_end = rec[2]
    if current:
        blocks.append(current)

    # Build desired order only for blocks we explicitly choose to reorder.
    desired_positions: dict[int, int] = {}

    def _shift_line_timing(rec: dict[str, Any], new_start: float) -> None:
        old_start = float(rec.get("start", 0.0) or 0.0)
        old_end = float(rec.get("end", old_start) or old_start)
        shift = new_start - old_start
        rec["start"] = snap(new_start)
        rec["end"] = snap(max(new_start + 0.1, old_end + shift))
        for w in rec.get("words", []) or []:
            if "start" in w:
                w["start"] = snap(float(w["start"]) + shift)
            if "end" in w:
                w["end"] = snap(float(w["end"]) + shift)

    for block in blocks:
        if len(block) == 1:
            continue
        y_values = [rec[3] for rec in block]
        if max(y_values) - min(y_values) < 20.0:
            continue
        block_by_y = sorted(block, key=lambda t: (t[3], t[1], t[4], t[0]))
        block_vs = [rec[1] for rec in block]
        block_ve = [rec[2] for rec in block]
        block_indices = sorted(rec[0] for rec in block)
        is_local_contiguous = block_indices[-1] - block_indices[0] + 1 == len(
            block_indices
        )
        is_tight_screen_block = (
            (max(block_vs) - min(block_vs)) <= 0.35
            and (max(block_ve) - min(block_ve)) <= 10.0
            and 2 <= len(block_by_y) <= 5
            and is_local_contiguous
        )
        # Only reorder when the current starts are clearly inconsistent with
        # top-to-bottom order. This avoids disturbing blocks that are already
        # correctly sequenced by selection timing.
        y_order_starts = [rec[4] for rec in block_by_y]
        has_strong_inversion = any(
            y_order_starts[k] > y_order_starts[k + 1] + 0.75
            for k in range(len(y_order_starts) - 1)
        )
        if not has_strong_inversion:
            continue
        if not is_tight_screen_block:
            continue
        # For the simple base case, also remap line starts monotonically in y-order
        # using the block's observed start times. This fixes obvious screen-order
        # inversions without inventing new timings.
        if 2 <= len(block_by_y) <= 6:
            observed_starts = sorted(rec[4] for rec in block)
            prev_end: Optional[float] = None
            for k, rec in enumerate(block_by_y):
                idx = rec[0]
                line = lines_out[idx]
                target_start = observed_starts[k]
                if prev_end is not None:
                    target_start = max(target_start, prev_end + 0.05)
                vis_end = line.get("_visibility_end")
                if vis_end is not None:
                    target_start = min(
                        float(vis_end) - 0.15,
                        target_start,
                    )
                old_start = float(line.get("start", 0.0) or 0.0)
                if target_start > old_start + 0.15 or target_start < old_start - 0.5:
                    _shift_line_timing(line, target_start)
                prev_end = float(line.get("end", target_start) or target_start)
        # Within a clear inverted block, preserve vertical order.
        anchor = block_indices[0]
        for offset, rec in enumerate(block_by_y):
            desired_positions[rec[0]] = anchor + offset

    if not desired_positions:
        return

    # Stable reorder only lines within targeted blocks; all others keep original order.
    indexed = list(enumerate(lines_out))
    indexed.sort(
        key=lambda pair: (
            desired_positions.get(pair[0], int(pair[1].get("_orig_order", pair[0]))),
            int(pair[1].get("_orig_order", pair[0])),
        )
    )
    lines_out[:] = [ln for _, ln in indexed]
    for i, ln in enumerate(lines_out):
        ln["line_index"] = i + 1


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
        base_block = None
        base_meta = base.get("_reconstruction_meta", {})
        if isinstance(base_meta, dict):
            bf = base_meta.get("block_first")
            if isinstance(bf, dict):
                base_block = bf.get("block_id")
        base_vs = base.get("_visibility_start")
        base_ve = base.get("_visibility_end")
        base_y = float(base.get("y", 0.0) or 0.0)

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
            cand = lines_out[j]
            cand_meta = cand.get("_reconstruction_meta", {})
            cand_block = None
            if isinstance(cand_meta, dict):
                bf = cand_meta.get("block_first")
                if isinstance(bf, dict):
                    cand_block = bf.get("block_id")
            # In block-first mode, distinct rows from the same visibility block are
            # not duplicates, even if their text is similar (e.g., chorus rows).
            if (
                base_block is not None
                and cand_block is not None
                and base_block == cand_block
            ):
                continue
            cand_vs = cand.get("_visibility_start")
            cand_ve = cand.get("_visibility_end")
            cand_y = float(cand.get("y", 0.0) or 0.0)
            if (
                base_vs is not None
                and base_ve is not None
                and cand_vs is not None
                and cand_ve is not None
            ):
                overlap = min(float(base_ve), float(cand_ve)) - max(
                    float(base_vs), float(cand_vs)
                )
                if overlap > 0.5 and abs(base_y - cand_y) >= 24.0:
                    continue
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
        _repair_repeat_cluster_tokenization_variants(lines_out, group)
        for idx in group:
            norms[idx] = normalize_text_basic(str(lines_out[idx].get("text", "")))
            token_lists[idx] = [t for t in norms[idx].split() if t]
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


def _merge_adjacent_words(
    words: list[dict[str, Any]], idx: int, merged_text: str
) -> None:
    a = words[idx]
    b = words[idx + 1]
    a_conf = float(a.get("confidence", 0.0) or 0.0)
    b_conf = float(b.get("confidence", 0.0) or 0.0)
    merged = {
        **a,
        "text": merged_text,
        "start": a.get("start"),
        "end": b.get("end", a.get("end")),
        "confidence": round((a_conf + b_conf) / 2.0, 3),
    }
    words[idx : idx + 2] = [merged]
    for j, w in enumerate(words, start=1):
        w["word_index"] = j


def _plural_family_key(tok: str) -> str:
    if tok.endswith("s") and len(tok) >= 4:
        return tok[:-1]
    return tok


def _ocr_confusion_family_key(tok: str) -> str:
    chars = []
    for ch in tok:
        if ch in {"l", "1", "|"}:
            chars.append("i")
        else:
            chars.append(ch)
    return "".join(chars)


def _line_tokens_from_output(line: dict[str, Any]) -> list[str]:
    return [t for t in normalize_text_basic(str(line.get("text", ""))).split() if t]


def _best_cluster_token_match_for_merge(
    merged_norm: str, cluster_tokens: list[str]
) -> str | None:
    best_token = None
    best_score = 0.0
    for tok in cluster_tokens:
        if len(tok) < len(merged_norm):
            continue
        if tok == merged_norm:
            return tok
        score = SequenceMatcher(None, merged_norm, tok).ratio()
        if score >= 0.84 and score > best_score:
            best_token = tok
            best_score = score
    return best_token


def _merge_repeat_cluster_split_fragments(
    lines_out: list[dict[str, Any]], group: list[int], cluster_tokens: list[str]
) -> None:
    for idx in group:
        line = lines_out[idx]
        words = line.get("words", [])
        if len(words) < 2:
            continue
        j = 0
        changed = False
        while j < len(words) - 1:
            a = str(words[j].get("text", ""))
            b = str(words[j + 1].get("text", ""))
            a_n = normalize_text_basic(a)
            b_n = normalize_text_basic(b)
            merged_n = a_n + b_n
            if not (1 <= len(a_n) <= 4 and 1 <= len(b_n) <= 4 and len(merged_n) >= 5):
                j += 1
                continue
            merged_text = _best_cluster_token_match_for_merge(merged_n, cluster_tokens)
            if not merged_text:
                j += 1
                continue
            _merge_adjacent_words(words, j, merged_text)
            changed = True
            continue
        if changed:
            line["text"] = " ".join(str(w.get("text", "")) for w in words)


def _harmonize_repeat_cluster_plural_variants(
    lines_out: list[dict[str, Any]], group: list[int]
) -> None:
    token_lists = [_line_tokens_from_output(lines_out[idx]) for idx in group]
    if not token_lists:
        return

    lengths = [len(toks) for toks in token_lists]
    target_len = max(set(lengths), key=lambda n: (sum(1 for x in lengths if x == n), n))
    aligned = [pos for pos, toks in enumerate(token_lists) if len(toks) == target_len]
    if len(aligned) < 2:
        return

    consensus_by_pos: dict[int, str] = {}
    for pos in range(target_len):
        chosen = _repeat_cluster_consensus_token_at_pos(token_lists, aligned, pos)
        if chosen is not None:
            consensus_by_pos[pos] = chosen

    if not consensus_by_pos:
        return
    for rel_idx in aligned:
        line = lines_out[group[rel_idx]]
        words = line.get("words", [])
        if len(words) != target_len:
            continue
        changed = False
        for pos, chosen_norm in consensus_by_pos.items():
            cur = str(words[pos].get("text", ""))
            cur_norm = normalize_text_basic(cur).replace(" ", "")
            same_plural_family = _plural_family_key(cur_norm) == _plural_family_key(
                chosen_norm
            )
            same_ocr_family = _ocr_confusion_family_key(
                cur_norm
            ) == _ocr_confusion_family_key(chosen_norm)
            if not (same_plural_family or same_ocr_family):
                continue
            if cur_norm == chosen_norm:
                continue
            words[pos]["text"] = chosen_norm
            changed = True
        if changed:
            line["text"] = " ".join(str(w.get("text", "")) for w in words)


def _repeat_cluster_consensus_token_at_pos(
    token_lists: list[list[str]], aligned: list[int], pos: int
) -> str | None:
    families: dict[str, dict[str, int]] = {}
    ocr_families: dict[str, dict[str, int]] = {}
    for rel_idx in aligned:
        tok = token_lists[rel_idx][pos]
        fam = _plural_family_key(tok)
        families.setdefault(fam, {})
        families[fam][tok] = families[fam].get(tok, 0) + 1
        ocr_fam = _ocr_confusion_family_key(tok)
        ocr_families.setdefault(ocr_fam, {})
        ocr_families[ocr_fam][tok] = ocr_families[ocr_fam].get(tok, 0) + 1

    if len(families) == 1:
        variants = next(iter(families.values()))
        if len(variants) < 2:
            return None
        return sorted(
            variants.items(), key=lambda item: (item[1], len(item[0])), reverse=True
        )[0][0]

    # Allow OCR-confusable token families (e.g. "i've" vs "l've") but avoid semantic alternations.
    if len(ocr_families) != 1:
        return None
    variants = next(iter(ocr_families.values()))
    if len(variants) < 2:
        return None
    return sorted(
        variants.items(),
        key=lambda item: (
            item[1],
            item[0].count("i") - item[0].count("l"),
            len(item[0]),
        ),
        reverse=True,
    )[0][0]


def _repair_repeat_cluster_tokenization_variants(
    lines_out: list[dict[str, Any]], group: list[int]
) -> None:
    if len(group) < 2:
        return
    cluster_tokens: list[str] = []
    for idx in group:
        cluster_tokens.extend(_line_tokens_from_output(lines_out[idx]))
    _merge_repeat_cluster_split_fragments(lines_out, group, cluster_tokens)
    _harmonize_repeat_cluster_plural_variants(lines_out, group)


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


def _tokens_contiguous_subphrase(needle: list[str], haystack: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    n = len(needle)
    for i in range(0, len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return True
    return False


def _neighbor_supports_fragment_tokens(
    fragment_tokens: list[str], neighbor_tokens: list[str]
) -> bool:
    if _tokens_contiguous_subphrase(fragment_tokens, neighbor_tokens):
        return True

    # OCR often splits a single word across 2-3 tiny tokens ("con ting").
    if 2 <= len(fragment_tokens) <= 3 and all(
        1 <= len(t) <= 4 for t in fragment_tokens
    ):
        merged = "".join(fragment_tokens)
        if len(merged) >= 5:
            for tok in neighbor_tokens:
                if tok == merged:
                    return True
                if (
                    len(tok) >= len(merged)
                    and SequenceMatcher(None, merged, tok).ratio() >= 0.84
                ):
                    return True

    # Singular/plural near-fragments are common in noisy repeated lines ("dollar" vs "dollars").
    if len(fragment_tokens) == 1:
        tok = fragment_tokens[0]
        if len(tok) >= 4:
            singular = tok[:-1] if tok.endswith("s") else tok
            plural = tok if tok.endswith("s") else f"{tok}s"
            for n_tok in neighbor_tokens:
                if n_tok == singular or n_tok == plural:
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
        avg_conf = frag_conf_sums[key] / max(count, 1)
        min_count = 3
        if count >= 2 and len(key) <= 2 and avg_conf <= 0.25:
            min_count = 2
        allow_high_conf_refrain_frag = count >= 5 and len(key) <= 3 and avg_conf <= 0.9
        if count < min_count:
            continue
        if avg_conf > 0.4 and not allow_high_conf_refrain_frag:
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
                if _neighbor_supports_fragment_tokens(list(key), neigh_toks):
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


def _case_like_token(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper() and source[1:].islower():
        return replacement.capitalize()
    if source.islower():
        return replacement.lower()
    return replacement


def _canonicalize_local_chant_token_variants(lines_out: list[dict[str, Any]]) -> None:
    """Normalize OCR variants inside chant-like repeated-token lines (e.g. dohi -> doh)."""
    if not lines_out:
        return
    protected_vocab = _VOCALIZATION_NOISE_TOKENS | {"hey", "yeah", "yea", "yo", "no"}
    for ln in lines_out:
        words = ln.get("words", [])
        if len(words) < 2 or len(words) > 8:
            continue
        norm_tokens = [normalize_text_basic(str(w.get("text", ""))) for w in words]
        norm_tokens = [t for t in norm_tokens if t]
        if len(norm_tokens) != len(words):
            continue
        if len(set(norm_tokens)) < 2:
            continue
        if any(len(t) < 2 or len(t) > 5 or not t.isalpha() for t in norm_tokens):
            continue
        if set(norm_tokens).issubset(protected_vocab):
            continue

        shortest = min(norm_tokens, key=len)
        if len(shortest) < 3:
            continue
        if not all(
            SequenceMatcher(None, shortest, t).ratio() >= 0.72 for t in norm_tokens
        ):
            continue

        canon = min(
            set(norm_tokens),
            key=lambda t: (
                -norm_tokens.count(t),
                len(t),
                t,
            ),
        )
        changed = False
        for w, norm in zip(words, norm_tokens):
            if norm == canon:
                continue
            if SequenceMatcher(None, canon, norm).ratio() < 0.72:
                continue
            w["text"] = _case_like_token(str(w.get("text", "")), canon)
            changed = True
        if changed:
            ln["text"] = " ".join(
                str(w.get("text", "")) for w in words if w.get("text")
            )


def _trim_short_adlib_tails(lines_out: list[dict[str, Any]]) -> None:  # noqa: C901
    """Trim short ad-lib tails on fragment lines when a stronger neighbor already covers the lyric."""
    if len(lines_out) < 3:
        return

    norm_lines = [normalize_text_basic(str(ln.get("text", ""))) for ln in lines_out]
    token_lines = [[t for t in n.split() if t] for n in norm_lines]

    for i, ln in enumerate(lines_out):
        words = ln.get("words", [])
        toks = token_lines[i]
        if len(words) != len(toks) or len(toks) < 3 or len(toks) > 5:
            continue
        start = float(ln.get("start", 0.0) or 0.0)
        end = float(ln.get("end", start) or start)
        dur = max(0.0, end - start)
        if dur > 1.8:
            continue

        split_idx = None
        for j in range(1, len(toks)):
            tail = toks[j:]
            if not tail or len(tail) > 2:
                continue
            if not set(tail).issubset(_ADLIB_TAIL_TOKENS):
                continue
            if len(toks[:j]) < 2:
                continue
            split_idx = j
            break
        if split_idx is None:
            continue

        prefix = toks[:split_idx]
        if set(prefix).issubset(_ADLIB_TAIL_TOKENS | _VOCALIZATION_NOISE_TOKENS):
            continue
        prefix_supported = False
        base_quality = _line_duplicate_quality_score(ln)
        for k in range(max(0, i - 2), min(len(lines_out), i + 3)):
            if k == i:
                continue
            other = lines_out[k]
            other_toks = token_lines[k]
            if len(other_toks) <= len(prefix):
                continue
            other_start = float(other.get("start", 0.0) or 0.0)
            other_end = float(other.get("end", other_start) or other_start)
            if other_end < start - 1.0 or other_start > end + 1.0:
                continue
            if not _tokens_contiguous_subphrase(prefix, other_toks):
                continue
            if _line_duplicate_quality_score(other) + 0.05 < base_quality:
                continue
            prefix_supported = True
            break
        if not prefix_supported:
            continue

        kept_words = words[:split_idx]
        if len(kept_words) < 2:
            continue
        ln["words"] = kept_words
        ln["text"] = " ".join(
            str(w.get("text", "")) for w in kept_words if w.get("text")
        )
        if kept_words:
            ln["end"] = float(kept_words[-1].get("end", end) or end)
            for wi, w in enumerate(kept_words):
                w["word_index"] = wi + 1


# Backward-compatible re-exports for tests/import sites after file split.
from .bootstrap_postprocess_block_cycle_passes import (  # noqa: E402,F401
    _consolidate_block_first_fragment_rows,
    _dedupe_block_first_cycle_rows,
    _filter_singer_label_prefixes,
    _normalize_block_first_repeated_cycles,
    _normalize_block_first_row_timings,
    _rebalance_compressed_middle_four_line_sequences,
    _remove_vocalization_noise_runs,
    _repair_large_adjacent_time_inversions,
    _repair_strong_local_chronology_inversions,
    _retime_short_interstitial_output_lines,
    _trim_leading_vocalization_in_block_first_cycle_rows,
    _trim_leading_vocalization_prefixes,
)
