"""Filtering helpers for block-cycle postprocessing passes."""

from __future__ import annotations

from typing import Optional

from ..text_utils import LYRIC_FUNCTION_WORDS, normalize_text_basic


def vocalization_noise_tokens(
    line: dict[str, object], *, vocalization_noise_tokens_set: set[str]
) -> list[str] | None:
    words = line.get("words", [])
    if not isinstance(words, list) or len(words) < 2:
        return None
    toks = [
        normalize_text_basic(str(w.get("text", "")))
        for w in words
        if isinstance(w, dict)
    ]
    toks = [t for t in toks if t]
    if len(toks) < 2:
        return None
    uniq = set(toks)
    if len(uniq) > 2:
        return None
    if not uniq.issubset(vocalization_noise_tokens_set):
        return None
    return toks


def remove_vocalization_noise_runs(
    lines_out: list[dict[str, object]],
    *,
    vocalization_noise_tokens_set: set[str],
    hum_noise_tokens_set: set[str],
) -> None:
    if not lines_out:
        return
    keep: list[dict[str, object]] = []
    i = 0
    while i < len(lines_out):
        toks = vocalization_noise_tokens(
            lines_out[i], vocalization_noise_tokens_set=vocalization_noise_tokens_set
        )
        if toks is None:
            keep.append(lines_out[i])
            i += 1
            continue

        j = i
        run: list[dict[str, object]] = []
        run_token_count = 0
        run_vocab: set[str] = set()
        while j < len(lines_out):
            jtoks = vocalization_noise_tokens(
                lines_out[j],
                vocalization_noise_tokens_set=vocalization_noise_tokens_set,
            )
            if jtoks is None:
                break
            run.append(lines_out[j])
            run_token_count += len(jtoks)
            run_vocab.update(jtoks)
            j += 1

        min_tokens = 2 if run_vocab and run_vocab.issubset(hum_noise_tokens_set) else 10
        if run_token_count >= min_tokens and len(run_vocab) <= 2:
            i = j
            continue

        keep.extend(run)
        i = j

    lines_out[:] = keep
    for i, ln in enumerate(lines_out):
        ln["line_index"] = i + 1


def filter_singer_label_prefixes(
    lines_out: list[dict[str, object]], artist: Optional[str]
) -> None:
    if not lines_out:
        return

    banned_prefixes = identify_banned_prefixes(lines_out, artist)
    if not banned_prefixes:
        return

    for ln in lines_out:
        remove_prefix_from_line(ln, banned_prefixes)

    lines_out[:] = [ln for ln in lines_out if ln.get("words")]
    for i, ln in enumerate(lines_out):
        ln["line_index"] = i + 1


def identify_banned_prefixes(
    lines_out: list[dict[str, object]], artist: Optional[str]
) -> set[str]:
    counts: dict[str, int] = {}
    for ln in lines_out:
        words = ln.get("words", [])
        if isinstance(words, list) and words:
            first = words[0]
            if isinstance(first, dict):
                prefix = normalize_text_basic(str(first.get("text", "")))
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
        if count > 0.1 * total and count >= 3:
            banned_prefixes.add(prefix)
        elif artist_norm and prefix in artist_norm:
            banned_prefixes.add(prefix)
    return banned_prefixes


def remove_prefix_from_line(line: dict[str, object], banned_prefixes: set[str]) -> None:
    words = line.get("words", [])
    if not isinstance(words, list) or not words:
        return
    first = words[0]
    if not isinstance(first, dict):
        return
    prefix = normalize_text_basic(str(first.get("text", "")))
    if prefix not in banned_prefixes:
        return
    words.pop(0)
    if not words:
        line["words"] = []
        line["text"] = ""
        return
    line["words"] = words
    line["text"] = " ".join(
        str(w.get("text", "")) for w in words if isinstance(w, dict)
    )
    first_word = words[0]
    if isinstance(first_word, dict):
        line["start"] = first_word.get("start")
    for i, w in enumerate(words):
        if isinstance(w, dict):
            w["word_index"] = i + 1
