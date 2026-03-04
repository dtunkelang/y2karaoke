"""Repeat-cluster token normalization helpers for bootstrap postprocessing."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from ..text_utils import normalize_text_basic


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
