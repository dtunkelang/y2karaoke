"""Block/cycle-oriented postprocessing passes for visual bootstrap outputs."""

from __future__ import annotations

from typing import Any

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
from .bootstrap_postprocess_block_cycle_consolidation import (
    consolidate_block_first_fragment_rows as _consolidate_block_first_fragment_rows_impl,
)
from .bootstrap_postprocess_interstitial_retime import (
    retime_short_interstitial_output_lines as _retime_short_interstitial_output_lines_impl,
)
from .bootstrap_postprocess_block_cycle_row_timing import (
    normalize_block_first_row_timings as _normalize_block_first_row_timings_impl,
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
    _consolidate_block_first_fragment_rows_impl(
        lines_out,
        normalize_text_basic_fn=normalize_text_basic,
    )


def _normalize_block_first_row_timings(  # noqa: C901
    lines_out: list[dict[str, Any]],
) -> None:
    _normalize_block_first_row_timings_impl(lines_out, snap_fn=snap)


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
