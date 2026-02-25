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


def _trim_short_adlib_tails(lines_out: list[dict[str, Any]]) -> None:
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
            inversion = sa - sb
            if inversion < 0.5:
                continue
            ea = float(a.get("end", sa) or sa)
            eb = float(b.get("end", sb) or sb)
            # Avoid swapping heavily overlapping near-simultaneous lanes.
            overlap = max(0.0, min(ea, eb) - max(sa, sb))
            dur_a = max(0.1, ea - sa)
            dur_b = max(0.1, eb - sb)
            conf_b = float(b.get("confidence", 0.0) or 0.0)

            # Strong no-overlap inversion: almost certainly chronology damage.
            strong_no_overlap = overlap <= 0.05 and eb <= sa - 0.2
            # Small-overlap fragment inversion: short low-confidence fragment got pushed late.
            short_fragment_inversion = (
                overlap <= 0.3
                and dur_b <= 1.4
                and (conf_b <= 0.35 or dur_a >= dur_b * 2.0)
                and sb + 0.4 < sa
            )
            if overlap > 0.8 and not (strong_no_overlap or short_fragment_inversion):
                continue
            if inversion < 0.9 and not short_fragment_inversion:
                continue
            if not (strong_no_overlap or short_fragment_inversion):
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
        if (
            prefix in LYRIC_FUNCTION_WORDS
            or prefix.replace("'", "") in LYRIC_FUNCTION_WORDS
        ):
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
