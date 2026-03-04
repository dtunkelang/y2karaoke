"""Post-processing helpers for karaoke visual bootstrap outputs."""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Any, List, Optional

from ..models import TargetLine
from ..text_utils import (
    LYRIC_FUNCTION_WORDS,
    normalize_ocr_line,
    normalize_text_basic,
)
from .reconstruction import snap
from .reconstruction_intro_filters import filter_intro_non_lyrics
from .bootstrap_postprocess_line_passes import (
    _canonicalize_local_chant_token_variants,
    _canonicalize_repeated_line_text_variants,
    _normalize_output_casing,
    _remove_overlay_credit_lines,
    _remove_repeated_fragment_noise_lines,
    _remove_repeated_singleton_noise_lines,
    _remove_weaker_near_duplicate_lines,
    _strip_internal_line_metadata,
    _trim_short_adlib_tails,
)

from .bootstrap_postprocess_block_cycle_passes import (
    _consolidate_block_first_fragment_rows,
    _normalize_block_first_repeated_cycles,
    _filter_singer_label_prefixes,
    _rebalance_compressed_middle_four_line_sequences,
    _remove_vocalization_noise_runs,
    _normalize_block_first_row_timings,
    _dedupe_block_first_cycle_rows,
    _repair_large_adjacent_time_inversions,
    _repair_strong_local_chronology_inversions,
    _retime_short_interstitial_output_lines,
    _trim_leading_vocalization_prefixes,
)
from .bootstrap_postprocess_token_ocr import (
    OCR_SUB_CHAR_MAP as _OCR_SUB_CHAR_MAP,
    best_ocr_substitution as _best_ocr_substitution_impl,
    fallback_spell_checker as _fallback_spell_checker_impl,
    fallback_spell_guess as _fallback_spell_guess_impl,
    is_safe_spell_guess_correction as _is_safe_spell_guess_correction_impl,
    ocr_insertion_candidates as _ocr_insertion_candidates_impl,
    ocr_substitution_candidates as _ocr_substitution_candidates_impl,
)
from .bootstrap_postprocess_token_rules import (
    contextual_compound_split as _contextual_compound_split_impl,
    fallback_spell_validated_split as _fallback_spell_validated_split_impl,
    fallback_split_fused_token as _fallback_split_fused_token_impl,
    looks_fused_prefix_candidate as _looks_fused_prefix_candidate_impl,
    maybe_contextual_inflection_token as _maybe_contextual_inflection_token_impl,
    maybe_expand_colloquial_token as _maybe_expand_colloquial_token_impl,
    maybe_restore_contraction_token as _maybe_restore_contraction_token_impl,
    maybe_restore_contextual_contraction_token as _maybe_restore_contextual_contraction_token_impl,
    maybe_split_fused_contraction_token as _maybe_split_fused_contraction_token_impl,
    repair_fallback_token as _repair_fallback_token_impl,
)
from .bootstrap_postprocess_token_pipeline import (
    split_fused_output_words as _split_fused_output_words_impl,
)
from .bootstrap_postprocess_chant_noise import (
    collect_chant_signature_stats as _collect_chant_signature_stats_impl,
    is_chant_noise_signature as _is_chant_noise_signature_impl,
    line_has_neighbor_chant_token_support as _line_has_neighbor_chant_token_support_impl,
    remove_high_repeat_nonlexical_chant_noise_lines as _remove_high_repeat_nonlexical_chant_noise_lines_impl,
    remove_repeated_chant_noise_lines as _remove_repeated_chant_noise_lines_impl,
    should_drop_high_repeat_chant_signature as _should_drop_high_repeat_chant_signature_impl,
)
from .bootstrap_postprocess_pipeline import (
    build_initial_lines_output as _build_initial_lines_output_impl,
    nearest_known_word_indices as _nearest_known_word_indices_impl,
)

_FUSED_FALLBACK_PREFIX_ANCHORS = (
    "i",
    "my",
)
_FUSED_FALLBACK_SHORT_FUNCTIONS = ("i", "a", "in", "on", "to", "of", "my")
_FUSED_FALLBACK_SUFFIX_ANCHORS = ("in",)
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
_COMMON_LYRIC_CHANT_TOKENS = {
    "hey",
    "yeah",
    "yea",
    "yo",
    "no",
    "na",
    "la",
    "oh",
    "ah",
    "woo",
    "ooh",
}
_COLLOQUIAL_EXPANSIONS = {
    "wanna": ("want", "to"),
    "gonna": ("going", "to"),
    "gotta": ("got", "to"),
    "lemme": ("let", "me"),
    "gimme": ("give", "me"),
}
_CONTRACTION_RESTORES = {
    "wont": "won't",
    "cant": "can't",
    "dont": "don't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "isnt": "isn't",
    "arent": "aren't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "couldnt": "couldn't",
    "wouldnt": "wouldn't",
    "shouldnt": "shouldn't",
    "im": "I'm",
    "ive": "I've",
    "ill": "I'll",
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


def _trace_postprocess_snapshot(label: str, lines_out: list[dict[str, Any]]) -> None:
    if os.environ.get("Y2K_VISUAL_POST_TRACE", "0") != "1":
        return
    preview: list[dict[str, Any]] = []
    for ln in lines_out[:15]:
        preview.append(
            {
                "i": int(ln.get("line_index", 0) or 0),
                "s": round(float(ln.get("start", 0.0) or 0.0), 2),
                "e": round(float(ln.get("end", 0.0) or 0.0), 2),
                "y": round(float(ln.get("y", 0.0) or 0.0), 1),
                "vs": (
                    round(float(ln.get("_visibility_start") or 0.0), 2)
                    if ln.get("_visibility_start") is not None
                    else None
                ),
                "ve": (
                    round(float(ln.get("_visibility_end") or 0.0), 2)
                    if ln.get("_visibility_end") is not None
                    else None
                ),
                "t": str(ln.get("text", ""))[:60],
            }
        )
    import logging

    logging.getLogger(__name__).info("POST_TRACE %s first15=%s", label, preview)


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


def _clamp_confidence(value: Optional[float], default: float = 0.0) -> float:
    if value is None:
        value = default
    return max(0.0, min(1.0, float(value)))


def nearest_known_word_indices(
    known_indices: List[int], n_words: int
) -> tuple[List[int], List[int]]:
    return _nearest_known_word_indices_impl(known_indices, n_words)


def build_refined_lines_output(  # noqa: C901
    t_lines: list[TargetLine], artist: Optional[str], title: Optional[str]
) -> list[dict[str, Any]]:
    lines_out = _build_initial_lines_output_impl(
        t_lines,
        artist=artist,
        title=title,
        split_fused_output_words_fn=_split_fused_output_words,
        clamp_confidence_fn=_clamp_confidence,
        nearest_known_word_indices_fn=nearest_known_word_indices,
    )
    _trace_postprocess_snapshot("initial", lines_out)
    _retime_short_interstitial_output_lines(lines_out)
    _trace_postprocess_snapshot("after_short_interstitial", lines_out)
    if os.environ.get("Y2K_VISUAL_DISABLE_POST_REBALANCE_FOUR", "0") != "1":
        _rebalance_compressed_middle_four_line_sequences(lines_out)
    _trace_postprocess_snapshot("after_rebalance_four", lines_out)
    _filter_singer_label_prefixes(lines_out, artist=artist)
    _trace_postprocess_snapshot("after_singer_prefix", lines_out)
    lines_out = filter_intro_non_lyrics(lines_out, artist=artist)
    _trace_postprocess_snapshot("after_intro_filter", lines_out)
    _remove_overlay_credit_lines(lines_out)
    _trace_postprocess_snapshot("after_overlay", lines_out)
    _remove_weaker_near_duplicate_lines(lines_out)
    _trace_postprocess_snapshot("after_near_dupe", lines_out)
    _canonicalize_repeated_line_text_variants(lines_out)
    _canonicalize_local_chant_token_variants(lines_out)
    if os.environ.get("Y2K_VISUAL_ENABLE_LEADING_VOCAL_PREFIX_TRIM", "0") == "1":
        _trim_leading_vocalization_prefixes(lines_out)
    _trim_short_adlib_tails(lines_out)
    _remove_repeated_singleton_noise_lines(lines_out, artist=artist, title=title)
    _remove_high_repeat_nonlexical_chant_noise_lines(lines_out)
    if os.environ.get("Y2K_VISUAL_ENABLE_CHANT_NOISE_FILTER", "0") == "1":
        _remove_repeated_chant_noise_lines(lines_out)
    _remove_repeated_fragment_noise_lines(lines_out, artist=artist, title=title)
    _trace_postprocess_snapshot("after_fragment_noise", lines_out)
    has_multi_cycle_block_first = any(
        isinstance((ln.get("_reconstruction_meta") or {}).get("block_first"), dict)
        and int(
            ((ln.get("_reconstruction_meta") or {}).get("block_first") or {}).get(
                "cycle_count", 1
            )
            or 1
        )
        > 1
        for ln in lines_out
    )
    _consolidate_block_first_fragment_rows(lines_out)
    _trace_postprocess_snapshot("after_block_first_consolidation", lines_out)
    _normalize_block_first_row_timings(lines_out)
    _trace_postprocess_snapshot("after_block_first_timing", lines_out)
    _normalize_block_first_repeated_cycles(lines_out)
    _trace_postprocess_snapshot("after_block_first_repeat_cycles", lines_out)
    if has_multi_cycle_block_first:
        _dedupe_block_first_cycle_rows(lines_out)
        _trace_postprocess_snapshot("after_block_first_cycle_dedupe", lines_out)
    _repair_strong_local_chronology_inversions(lines_out)
    _repair_large_adjacent_time_inversions(lines_out)
    _trace_postprocess_snapshot("after_chronology_repairs", lines_out)
    _remove_vocalization_noise_runs(lines_out)
    _trace_postprocess_snapshot("after_vocal_noise", lines_out)
    _normalize_output_casing(lines_out)
    # Block-aware ordering/timing belongs in refinement, where TargetLine visibility and
    # selection timing can still be adjusted safely. A postprocess-only reorder here can
    # improve local order while harming global token sequence alignment.
    # Keep this disabled by default until a refinement-stage block model replaces it.
    # _reorder_clean_visibility_blocks(lines_out)
    _strip_internal_line_metadata(lines_out)
    return lines_out


def _is_chant_noise_signature(tokens: list[str]) -> bool:
    return _is_chant_noise_signature_impl(
        tokens,
        vocalization_noise_tokens=_VOCALIZATION_NOISE_TOKENS,
        common_lyric_chant_tokens=_COMMON_LYRIC_CHANT_TOKENS,
    )


def _remove_repeated_chant_noise_lines(  # noqa: C901
    lines_out: list[dict[str, Any]],
) -> None:
    _remove_repeated_chant_noise_lines_impl(
        lines_out, is_chant_noise_signature_fn=_is_chant_noise_signature
    )


def _collect_chant_signature_stats(
    lines_out: list[dict[str, Any]], token_lists: list[list[str]]
) -> tuple[
    dict[tuple[str, ...], list[int]],
    dict[tuple[str, ...], float],
    dict[tuple[str, ...], float],
]:
    return _collect_chant_signature_stats_impl(
        lines_out, token_lists, is_chant_noise_signature_fn=_is_chant_noise_signature
    )


def _line_has_neighbor_chant_token_support(
    idx: int,
    root: str,
    lines_out: list[dict[str, Any]],
    token_lists: list[list[str]],
) -> bool:
    return _line_has_neighbor_chant_token_support_impl(
        idx,
        root,
        lines_out,
        token_lists,
        is_chant_noise_signature_fn=_is_chant_noise_signature,
    )


def _should_drop_high_repeat_chant_signature(
    sig: tuple[str, ...],
    idxs: list[int],
    sig_conf_sums: dict[tuple[str, ...], float],
    sig_dur_sums: dict[tuple[str, ...], float],
) -> bool:
    return _should_drop_high_repeat_chant_signature_impl(
        sig,
        idxs,
        sig_conf_sums,
        sig_dur_sums,
        is_spelled_word_fn=_is_spelled_word,
    )


def _remove_high_repeat_nonlexical_chant_noise_lines(
    lines_out: list[dict[str, Any]],
) -> None:
    _remove_high_repeat_nonlexical_chant_noise_lines_impl(
        lines_out,
        collect_chant_signature_stats_fn=_collect_chant_signature_stats,
        should_drop_high_repeat_chant_signature_fn=_should_drop_high_repeat_chant_signature,
        line_has_neighbor_chant_token_support_fn=_line_has_neighbor_chant_token_support,
    )


def _is_high_uncertainty_reconstruction(meta: Optional[dict[str, Any]]) -> bool:
    if not isinstance(meta, dict):
        return False
    uncertainty = float(meta.get("uncertainty_score", 0.0) or 0.0)
    support_ratio = float(meta.get("selected_text_support_ratio", 1.0) or 0.0)
    text_variants = int(meta.get("text_variant_count", 1) or 1)
    weak_votes = int(meta.get("weak_vote_positions", 0) or 0)
    return (
        uncertainty >= 0.2
        or support_ratio < 0.7
        or text_variants >= 4
        or weak_votes >= 2
        or bool(meta.get("used_observed_fallback"))
    )


def _allow_token_level_ocr_fallback(token: str) -> bool:
    low = token.strip().lower()
    if not low.isalpha() or len(low) < 6:
        return False
    if not (low.endswith("in") or low.endswith("i")):
        return False
    return not _is_spelled_word(low)


def _looks_fused_prefix_candidate(token: str) -> bool:
    return _looks_fused_prefix_candidate_impl(
        token, fused_fallback_prefix_anchors=_FUSED_FALLBACK_PREFIX_ANCHORS
    )


def _fallback_split_fused_token(token: str) -> list[str] | None:  # noqa: C901
    return _fallback_split_fused_token_impl(
        token,
        fused_fallback_short_functions=_FUSED_FALLBACK_SHORT_FUNCTIONS,
        fused_fallback_prefix_anchors=_FUSED_FALLBACK_PREFIX_ANCHORS,
        fused_fallback_suffix_anchors=_FUSED_FALLBACK_SUFFIX_ANCHORS,
        fallback_spell_validated_split_fn=_fallback_spell_validated_split,
        repair_fallback_token_fn=_repair_fallback_token,
    )


def _repair_fallback_token(token: str) -> str:
    return _repair_fallback_token_impl(token)


def _case_like(source: str, token: str) -> str:
    if token.lower() == "i":
        return "I"
    if source.isupper():
        return token.upper()
    if source[:1].isupper() and source[1:].islower():
        return token.capitalize()
    if source.islower():
        return token.lower()
    return token


def _fallback_spell_checker() -> Any:
    return _fallback_spell_checker_impl()


@lru_cache(maxsize=4096)
def _is_spelled_word(token: str) -> bool:
    low = token.lower()
    if low in {"i", "a"}:
        return True
    if not low.isalpha() or len(low) < 2:
        return False
    checker = _fallback_spell_checker()
    if checker is None:
        return False
    try:
        return checker.checkSpellingOfString_startingAt_(low, 0).length == 0
    except Exception:
        return False


def _fallback_spell_validated_split(token: str) -> list[str] | None:
    return _fallback_spell_validated_split_impl(
        token,
        is_spelled_word_fn=_is_spelled_word,
        fused_fallback_short_functions=_FUSED_FALLBACK_SHORT_FUNCTIONS,
    )


def _contextual_compound_split(
    token: str, next_token: str, confidence: float
) -> list[str] | None:
    return _contextual_compound_split_impl(
        token,
        next_token,
        confidence,
        normalize_text_basic_fn=normalize_text_basic,
        lyric_function_words=LYRIC_FUNCTION_WORDS,
        is_spelled_word_fn=_is_spelled_word,
        case_like_fn=_case_like,
    )


def _fallback_spell_guess(token: str) -> str | None:
    return _fallback_spell_guess_impl(
        token, fallback_spell_checker_fn=_fallback_spell_checker
    )


def _is_safe_spell_guess_correction(source: str, guess: str) -> bool:
    return _is_safe_spell_guess_correction_impl(source, guess)


def _ocr_substitution_candidates(token: str) -> list[str]:
    return _ocr_substitution_candidates_impl(
        token, is_spelled_word_fn=_is_spelled_word, sub_char_map=_OCR_SUB_CHAR_MAP
    )


def _ocr_insertion_candidates(token: str) -> list[str]:
    return _ocr_insertion_candidates_impl(token, is_spelled_word_fn=_is_spelled_word)


def _best_ocr_substitution(token: str) -> str | None:
    return _best_ocr_substitution_impl(
        token,
        is_spelled_word_fn=_is_spelled_word,
        case_like_fn=_case_like,
        ocr_substitution_candidates_fn=_ocr_substitution_candidates,
        ocr_insertion_candidates_fn=_ocr_insertion_candidates,
    )


def _maybe_repair_output_token(text: str, confidence: float) -> str:
    token = text.strip()
    if not token or " " in token:
        return token
    low = token.lower()
    if not low.isalpha():
        return token
    if _is_spelled_word(low):
        return token
    if _looks_fused_prefix_candidate(token):
        return token
    if confidence > 0.55 and not _allow_token_level_ocr_fallback(token):
        return token

    ocr_candidate = _best_ocr_substitution(token)
    if ocr_candidate and ocr_candidate != token:
        return ocr_candidate

    guess = _fallback_spell_guess(token)
    if not guess:
        return token
    if not _is_safe_spell_guess_correction(token, guess):
        return token
    return _case_like(token, guess)


def _maybe_expand_colloquial_token(text: str, confidence: float) -> list[str] | None:
    return _maybe_expand_colloquial_token_impl(
        text,
        confidence,
        colloquial_expansions=_COLLOQUIAL_EXPANSIONS,
        case_like_fn=_case_like,
    )


def _maybe_restore_contraction_token(text: str, confidence: float) -> str | None:
    return _maybe_restore_contraction_token_impl(
        text,
        confidence,
        contraction_restores=_CONTRACTION_RESTORES,
        case_like_fn=_case_like,
    )


def _maybe_restore_contextual_contraction_token(
    text: str, confidence: float, prev_token: str, next_token: str
) -> str | None:
    return _maybe_restore_contextual_contraction_token_impl(
        text,
        confidence,
        prev_token,
        next_token,
        normalize_text_basic_fn=normalize_text_basic,
        case_like_fn=_case_like,
    )


def _maybe_split_fused_contraction_token(token: str) -> list[str] | None:
    return _maybe_split_fused_contraction_token_impl(token)


def _maybe_contextual_inflection_token(
    text: str, confidence: float, prev_token: str, next_token: str
) -> str | None:
    return _maybe_contextual_inflection_token_impl(
        text,
        confidence,
        prev_token,
        next_token,
        normalize_text_basic_fn=normalize_text_basic,
        is_spelled_word_fn=_is_spelled_word,
        case_like_fn=_case_like,
    )


def _split_fused_output_words(  # noqa: C901
    words: list[dict[str, Any]],
    *,
    reconstruction_meta: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    return _split_fused_output_words_impl(
        words,
        reconstruction_meta=reconstruction_meta,
        normalize_ocr_line_fn=normalize_ocr_line,
        normalize_text_basic_fn=normalize_text_basic,
        snap_fn=snap,
        is_high_uncertainty_reconstruction_fn=_is_high_uncertainty_reconstruction,
        allow_token_level_ocr_fallback_fn=_allow_token_level_ocr_fallback,
        maybe_split_fused_contraction_token_fn=_maybe_split_fused_contraction_token,
        fallback_split_fused_token_fn=_fallback_split_fused_token,
        fallback_spell_validated_split_fn=_fallback_spell_validated_split,
        contextual_compound_split_fn=_contextual_compound_split,
        maybe_expand_colloquial_token_fn=_maybe_expand_colloquial_token,
        maybe_restore_contraction_token_fn=_maybe_restore_contraction_token,
        maybe_restore_contextual_contraction_token_fn=_maybe_restore_contextual_contraction_token,
        maybe_contextual_inflection_token_fn=_maybe_contextual_inflection_token,
        maybe_repair_output_token_fn=_maybe_repair_output_token,
    )
