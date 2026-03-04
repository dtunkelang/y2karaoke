"""Token-level output splitting pipeline for bootstrap postprocess."""

from __future__ import annotations

from typing import Any, Callable, Optional


def split_fused_output_words(  # noqa: C901
    words: list[dict[str, Any]],
    *,
    reconstruction_meta: Optional[dict[str, Any]],
    normalize_ocr_line_fn: Callable[[str], str],
    normalize_text_basic_fn: Callable[[str], str],
    snap_fn: Callable[[float], float],
    is_high_uncertainty_reconstruction_fn: Callable[[Optional[dict[str, Any]]], bool],
    allow_token_level_ocr_fallback_fn: Callable[[str], bool],
    maybe_split_fused_contraction_token_fn: Callable[[str], list[str] | None],
    fallback_split_fused_token_fn: Callable[[str], list[str] | None],
    fallback_spell_validated_split_fn: Callable[[str], list[str] | None],
    contextual_compound_split_fn: Callable[[str, str, float], list[str] | None],
    maybe_expand_colloquial_token_fn: Callable[[str, float], list[str] | None],
    maybe_restore_contraction_token_fn: Callable[[str, float], str | None],
    maybe_restore_contextual_contraction_token_fn: Callable[
        [str, float, str, str], str | None
    ],
    maybe_contextual_inflection_token_fn: Callable[[str, float, str, str], str | None],
    maybe_repair_output_token_fn: Callable[[str, float], str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    allow_fallback = is_high_uncertainty_reconstruction_fn(reconstruction_meta)
    block_first_meta = (
        reconstruction_meta.get("block_first")
        if isinstance(reconstruction_meta, dict)
        and isinstance(reconstruction_meta.get("block_first"), dict)
        else None
    )
    conservative_block_first = bool(block_first_meta) and not allow_fallback
    for rec_idx, rec in enumerate(words):
        txt = str(rec.get("text", "")).strip()
        conf = float(rec.get("confidence", 0.0))
        prev_txt = (
            str(words[rec_idx - 1].get("text", "")).strip() if rec_idx - 1 >= 0 else ""
        )
        next_txt = (
            str(words[rec_idx + 1].get("text", "")).strip()
            if rec_idx + 1 < len(words)
            else ""
        )
        normalized = normalize_ocr_line_fn(txt).strip()
        parts = [p for p in normalized.split() if p]
        if len(parts) <= 1:
            contraction_parts = maybe_split_fused_contraction_token_fn(txt)
            if contraction_parts:
                parts = contraction_parts
        if len(parts) <= 1 and (
            allow_fallback
            or (
                allow_token_level_ocr_fallback_fn(txt)
                and not (conservative_block_first and conf >= 0.45)
            )
        ):
            fallback_parts = fallback_split_fused_token_fn(txt)
            if fallback_parts:
                parts = fallback_parts
        if len(parts) <= 1 and not allow_fallback and not conservative_block_first:
            fallback_parts = fallback_spell_validated_split_fn(txt.lower())
            if fallback_parts:
                parts = fallback_parts
        if len(parts) <= 1 and not (conservative_block_first and conf >= 0.45):
            fallback_parts = contextual_compound_split_fn(txt, next_txt, conf)
            if fallback_parts:
                parts = fallback_parts
        if len(parts) <= 1:
            single = parts[0] if parts else txt
            expanded_single = maybe_expand_colloquial_token_fn(single, conf)
            if expanded_single:
                parts = expanded_single
            else:
                contraction_restored = maybe_restore_contraction_token_fn(single, conf)
                if contraction_restored:
                    patched = dict(rec)
                    patched["text"] = contraction_restored
                    out.append(patched)
                    continue
                contextual_contraction = maybe_restore_contextual_contraction_token_fn(
                    single, conf, prev_txt, next_txt
                )
                if contextual_contraction:
                    patched = dict(rec)
                    patched["text"] = contextual_contraction
                    out.append(patched)
                    continue
                inflected = maybe_contextual_inflection_token_fn(
                    single, conf, prev_txt, next_txt
                )
                if inflected and inflected != single:
                    patched = dict(rec)
                    patched["text"] = inflected
                    out.append(patched)
                    continue
                if conservative_block_first and conf >= 0.45:
                    out.append(rec)
                    continue
                repaired_single = maybe_repair_output_token_fn(single, conf)
                if repaired_single != txt and repaired_single:
                    patched = dict(rec)
                    patched["text"] = repaired_single
                    out.append(patched)
                else:
                    out.append(rec)
                continue

        start = float(rec.get("start", 0.0))
        end = float(rec.get("end", start + 0.1))
        span = max(0.1, end - start)
        expanded_parts: list[str] = []
        for p in parts:
            expanded = maybe_expand_colloquial_token_fn(p, conf)
            if expanded:
                expanded_parts.extend(expanded)
            else:
                expanded_parts.append(maybe_repair_output_token_fn(p, conf))
        parts = expanded_parts

        weights = [max(len(p), 1) for p in parts]
        total = float(sum(weights))
        cursor = start
        for idx, (part, w) in enumerate(zip(parts, weights)):
            dur = span * (float(w) / total)
            seg_end = end if idx == len(parts) - 1 else cursor + dur
            out.append(
                {
                    "word_index": 0,
                    "text": part,
                    "start": snap_fn(cursor),
                    "end": snap_fn(seg_end),
                    "confidence": conf,
                }
            )
            cursor = seg_end

    for i, rec in enumerate(out):
        rec["word_index"] = i + 1
    return out
