"""Line-level postprocessing passes for visual bootstrap outputs."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Optional

from ..text_utils import LYRIC_FUNCTION_WORDS, normalize_text_basic
from .bootstrap_postprocess_dedup_passes import (
    canonicalize_repeated_line_text_variants as _canonicalize_repeated_line_text_variants_impl,
    line_duplicate_quality_score as _line_duplicate_quality_score_impl,
    line_uncertainty as _line_uncertainty_impl,
    remove_weaker_near_duplicate_lines as _remove_weaker_near_duplicate_lines_impl,
)
from .bootstrap_postprocess_overlay import (
    overlay_line_signal_score as _overlay_line_signal_score_impl,
    remove_overlay_credit_lines as _remove_overlay_credit_lines_impl,
)
from .bootstrap_postprocess_fragment_noise import (
    neighbor_supports_fragment_tokens as _neighbor_supports_fragment_tokens_impl,
    remove_repeated_fragment_noise_lines as _remove_repeated_fragment_noise_lines_impl,
    tokens_contiguous_subphrase as _tokens_contiguous_subphrase_impl,
)
from .bootstrap_postprocess_block_order import (
    reorder_clean_visibility_blocks as _reorder_clean_visibility_blocks_impl,
)
from .bootstrap_postprocess_constants import ADLIB_TAIL_TOKENS as _ADLIB_TAIL_TOKENS
from .bootstrap_postprocess_constants import (
    HUM_NOISE_TOKENS as _HUM_NOISE_TOKENS_SHARED,
)
from .bootstrap_postprocess_constants import (
    OVERLAY_BRAND_TOKENS as _OVERLAY_BRAND_TOKENS,
)
from .bootstrap_postprocess_constants import (
    OVERLAY_CTA_TOKENS as _OVERLAY_CTA_TOKENS,
)
from .bootstrap_postprocess_constants import (
    OVERLAY_LEGAL_TOKENS as _OVERLAY_LEGAL_TOKENS,
)
from .bootstrap_postprocess_constants import (
    OVERLAY_PLATFORM_TOKENS as _OVERLAY_PLATFORM_TOKENS,
)
from .bootstrap_postprocess_constants import (
    VOCALIZATION_NOISE_TOKENS as _VOCALIZATION_NOISE_TOKENS,
)

# Backward-compatible symbol for existing imports in neighboring modules.
_HUM_NOISE_TOKENS = _HUM_NOISE_TOKENS_SHARED


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
    _reorder_clean_visibility_blocks_impl(lines_out)


def _overlay_line_signal_score(line: dict[str, Any]) -> int:  # noqa: C901
    return _overlay_line_signal_score_impl(
        line,
        normalize_text_basic_fn=normalize_text_basic,
        overlay_platform_tokens=_OVERLAY_PLATFORM_TOKENS,
        overlay_cta_tokens=_OVERLAY_CTA_TOKENS,
        overlay_legal_tokens=_OVERLAY_LEGAL_TOKENS,
        overlay_brand_tokens=_OVERLAY_BRAND_TOKENS,
    )


def _remove_overlay_credit_lines(lines_out: list[dict[str, Any]]) -> None:
    _remove_overlay_credit_lines_impl(
        lines_out,
        overlay_line_signal_score_fn=_overlay_line_signal_score,
    )


def _line_duplicate_quality_score(line: dict[str, Any]) -> float:
    return _line_duplicate_quality_score_impl(line)


def _line_uncertainty(line: dict[str, Any]) -> float:
    return _line_uncertainty_impl(line)


def _remove_weaker_near_duplicate_lines(  # noqa: C901
    lines_out: list[dict[str, Any]],
) -> None:
    _remove_weaker_near_duplicate_lines_impl(lines_out)


def _canonicalize_repeated_line_text_variants(  # noqa: C901
    lines_out: list[dict[str, Any]],
) -> None:
    _canonicalize_repeated_line_text_variants_impl(lines_out)


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
    return _tokens_contiguous_subphrase_impl(needle, haystack)


def _neighbor_supports_fragment_tokens(
    fragment_tokens: list[str], neighbor_tokens: list[str]
) -> bool:
    return _neighbor_supports_fragment_tokens_impl(
        fragment_tokens,
        neighbor_tokens,
        tokens_contiguous_subphrase_fn=_tokens_contiguous_subphrase,
    )


def _remove_repeated_fragment_noise_lines(  # noqa: C901
    lines_out: list[dict[str, Any]], artist: Optional[str], title: Optional[str]
) -> None:
    _remove_repeated_fragment_noise_lines_impl(
        lines_out,
        artist=artist,
        title=title,
        normalize_text_basic_fn=normalize_text_basic,
        lyric_function_words=LYRIC_FUNCTION_WORDS,
        vocalization_noise_tokens=_VOCALIZATION_NOISE_TOKENS,
        neighbor_supports_fragment_tokens_fn=_neighbor_supports_fragment_tokens,
    )


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
from .bootstrap_postprocess_repeat_cluster_helpers import (  # noqa: E402,F401
    _repair_repeat_cluster_tokenization_variants,
)
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
