"""Sequencing helpers for visual bootstrap postprocessing passes."""

from __future__ import annotations

import os
from typing import Any, Callable, Optional


def run_postprocess_sequence(
    lines_out: list[dict[str, Any]],
    *,
    artist: Optional[str],
    title: Optional[str],
    trace_postprocess_snapshot_fn: Callable[[str, list[dict[str, Any]]], None],
    retime_short_interstitial_output_lines_fn: Callable[[list[dict[str, Any]]], None],
    rebalance_compressed_middle_four_line_sequences_fn: Callable[
        [list[dict[str, Any]]], None
    ],
    filter_singer_label_prefixes_fn: Callable[..., None],
    filter_intro_non_lyrics_fn: Callable[..., list[dict[str, Any]]],
    remove_overlay_credit_lines_fn: Callable[[list[dict[str, Any]]], None],
    remove_weaker_near_duplicate_lines_fn: Callable[[list[dict[str, Any]]], None],
    canonicalize_repeated_line_text_variants_fn: Callable[[list[dict[str, Any]]], None],
    canonicalize_local_chant_token_variants_fn: Callable[[list[dict[str, Any]]], None],
    trim_leading_vocalization_prefixes_fn: Callable[[list[dict[str, Any]]], None],
    trim_short_adlib_tails_fn: Callable[[list[dict[str, Any]]], None],
    remove_repeated_singleton_noise_lines_fn: Callable[..., None],
    remove_high_repeat_nonlexical_chant_noise_lines_fn: Callable[
        [list[dict[str, Any]]], None
    ],
    remove_repeated_chant_noise_lines_fn: Callable[[list[dict[str, Any]]], None],
    remove_repeated_fragment_noise_lines_fn: Callable[..., None],
    consolidate_block_first_fragment_rows_fn: Callable[[list[dict[str, Any]]], None],
    normalize_block_first_row_timings_fn: Callable[[list[dict[str, Any]]], None],
    normalize_block_first_repeated_cycles_fn: Callable[[list[dict[str, Any]]], None],
    dedupe_block_first_cycle_rows_fn: Callable[[list[dict[str, Any]]], None],
    repair_strong_local_chronology_inversions_fn: Callable[
        [list[dict[str, Any]]], None
    ],
    repair_large_adjacent_time_inversions_fn: Callable[[list[dict[str, Any]]], None],
    remove_vocalization_noise_runs_fn: Callable[[list[dict[str, Any]]], None],
    normalize_output_casing_fn: Callable[[list[dict[str, Any]]], None],
    strip_internal_line_metadata_fn: Callable[[list[dict[str, Any]]], None],
) -> list[dict[str, Any]]:
    trace_postprocess_snapshot_fn("initial", lines_out)
    retime_short_interstitial_output_lines_fn(lines_out)
    trace_postprocess_snapshot_fn("after_short_interstitial", lines_out)
    if os.environ.get("Y2K_VISUAL_DISABLE_POST_REBALANCE_FOUR", "0") != "1":
        rebalance_compressed_middle_four_line_sequences_fn(lines_out)
    trace_postprocess_snapshot_fn("after_rebalance_four", lines_out)
    filter_singer_label_prefixes_fn(lines_out, artist=artist)
    trace_postprocess_snapshot_fn("after_singer_prefix", lines_out)
    lines_out = filter_intro_non_lyrics_fn(lines_out, artist=artist)
    trace_postprocess_snapshot_fn("after_intro_filter", lines_out)
    remove_overlay_credit_lines_fn(lines_out)
    trace_postprocess_snapshot_fn("after_overlay", lines_out)
    remove_weaker_near_duplicate_lines_fn(lines_out)
    trace_postprocess_snapshot_fn("after_near_dupe", lines_out)
    canonicalize_repeated_line_text_variants_fn(lines_out)
    canonicalize_local_chant_token_variants_fn(lines_out)
    if os.environ.get("Y2K_VISUAL_ENABLE_LEADING_VOCAL_PREFIX_TRIM", "0") == "1":
        trim_leading_vocalization_prefixes_fn(lines_out)
    trim_short_adlib_tails_fn(lines_out)
    remove_repeated_singleton_noise_lines_fn(lines_out, artist=artist, title=title)
    remove_high_repeat_nonlexical_chant_noise_lines_fn(lines_out)
    if os.environ.get("Y2K_VISUAL_ENABLE_CHANT_NOISE_FILTER", "0") == "1":
        remove_repeated_chant_noise_lines_fn(lines_out)
    remove_repeated_fragment_noise_lines_fn(lines_out, artist=artist, title=title)
    trace_postprocess_snapshot_fn("after_fragment_noise", lines_out)
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
    consolidate_block_first_fragment_rows_fn(lines_out)
    trace_postprocess_snapshot_fn("after_block_first_consolidation", lines_out)
    normalize_block_first_row_timings_fn(lines_out)
    trace_postprocess_snapshot_fn("after_block_first_timing", lines_out)
    normalize_block_first_repeated_cycles_fn(lines_out)
    trace_postprocess_snapshot_fn("after_block_first_repeat_cycles", lines_out)
    if has_multi_cycle_block_first:
        dedupe_block_first_cycle_rows_fn(lines_out)
        trace_postprocess_snapshot_fn("after_block_first_cycle_dedupe", lines_out)
    repair_strong_local_chronology_inversions_fn(lines_out)
    repair_large_adjacent_time_inversions_fn(lines_out)
    trace_postprocess_snapshot_fn("after_chronology_repairs", lines_out)
    remove_vocalization_noise_runs_fn(lines_out)
    trace_postprocess_snapshot_fn("after_vocal_noise", lines_out)
    normalize_output_casing_fn(lines_out)
    # Block-aware ordering/timing belongs in refinement, where TargetLine visibility and
    # selection timing can still be adjusted safely. A postprocess-only reorder here can
    # improve local order while harming global token sequence alignment.
    # Keep this disabled by default until a refinement-stage block model replaces it.
    # _reorder_clean_visibility_blocks(lines_out)
    strip_internal_line_metadata_fn(lines_out)
    return lines_out
