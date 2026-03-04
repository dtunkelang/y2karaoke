"""Dynamic Time Warping (DTW) alignment for Whisper and LRC."""

from contextlib import contextmanager
import logging
from typing import Callable, Iterator, List, Tuple, Dict, Optional

import numpy as np

from ...models import Line
from ..alignment.timing_models import TranscriptionWord
from ..alignment import timing_models
from ... import phonetic_utils
from ...phonetic_utils import (
    _get_ipa,
    _phonetic_similarity,
)
from .whisper_dtw_alignment_ops import (
    apply_dtw_alignments_base as _apply_dtw_alignments_base_impl,
    compute_dtw_alignment_metrics as _compute_dtw_alignment_metrics_impl,
    compute_phonetic_costs_base as _compute_phonetic_costs_base_impl,
    empty_dtw_metrics as _empty_dtw_metrics_impl,
    extract_alignments_from_path_base as _extract_alignments_from_path_base_impl,
    extract_lrc_words_base as _extract_lrc_words_base_impl,
    retime_lines_from_dtw_alignments as _retime_lines_from_dtw_alignments_impl,
)
from .whisper_dtw_fallback import (
    dtw_fallback_path as _dtw_fallback_path_impl,
)
from .whisper_dtw_tokens import _LineMappingContext  # noqa: F401

logger = logging.getLogger(__name__)

_ACTIVE_PHONETIC_SIMILARITY: Optional[Callable[..., float]] = None
_ACTIVE_GET_IPA: Optional[Callable[..., Optional[str]]] = None
_ACTIVE_LOAD_FASTDTW: Optional[Callable[..., Callable[..., tuple]]] = None


@contextmanager
def use_whisper_dtw_hooks(
    *,
    phonetic_similarity_fn: Optional[Callable[..., float]] = None,
    get_ipa_fn: Optional[Callable[..., Optional[str]]] = None,
    load_fastdtw_fn: Optional[Callable[..., Callable[..., tuple]]] = None,
) -> Iterator[None]:
    """Temporarily override DTW collaborators for tests."""
    global _ACTIVE_PHONETIC_SIMILARITY, _ACTIVE_GET_IPA, _ACTIVE_LOAD_FASTDTW

    prev_similarity = _ACTIVE_PHONETIC_SIMILARITY
    prev_get_ipa = _ACTIVE_GET_IPA
    prev_load_fastdtw = _ACTIVE_LOAD_FASTDTW
    if phonetic_similarity_fn is not None:
        _ACTIVE_PHONETIC_SIMILARITY = phonetic_similarity_fn
    if get_ipa_fn is not None:
        _ACTIVE_GET_IPA = get_ipa_fn
    if load_fastdtw_fn is not None:
        _ACTIVE_LOAD_FASTDTW = load_fastdtw_fn
    try:
        yield
    finally:
        _ACTIVE_PHONETIC_SIMILARITY = prev_similarity
        _ACTIVE_GET_IPA = prev_get_ipa
        _ACTIVE_LOAD_FASTDTW = prev_load_fastdtw


def _phonetic_similarity_for_state(*args, **kwargs) -> float:
    fn = _ACTIVE_PHONETIC_SIMILARITY or _phonetic_similarity
    return fn(*args, **kwargs)


def _get_ipa_for_state(*args, **kwargs) -> Optional[str]:
    fn = _ACTIVE_GET_IPA or _get_ipa
    return fn(*args, **kwargs)


def _load_fastdtw_for_state():
    fn = _ACTIVE_LOAD_FASTDTW or _load_fastdtw
    return fn()


def _load_fastdtw():
    from fastdtw import fastdtw  # type: ignore

    return fastdtw


def _dtw_fallback_path(
    lrc_seq: np.ndarray,
    whisper_seq: np.ndarray,
    dist: Callable[..., float],
    *,
    window: Optional[int] = None,
) -> Tuple[float, List[Tuple[int, int]]]:
    return _dtw_fallback_path_impl(lrc_seq, whisper_seq, dist, window=window)


def _dtw_fallback_with_runtime_guard(
    lrc_seq: np.ndarray, whisper_seq: np.ndarray, dist: Callable[..., float]
) -> Tuple[float, List[Tuple[int, int]]]:
    m = int(lrc_seq.shape[0])
    n = int(whisper_seq.shape[0])
    cell_budget = m * n
    if cell_budget <= 250_000:
        return _dtw_fallback_path(lrc_seq, whisper_seq, dist)

    window = max(96, abs(n - m) + 96)
    logger.info("Using banded exact DTW fallback (m=%d, n=%d, window=%d)", m, n, window)
    distance, path = _dtw_fallback_path(lrc_seq, whisper_seq, dist, window=window)
    if np.isfinite(distance):
        return distance, path

    logger.warning(
        "Banded exact DTW fallback found no path (window=%d); retrying full exact DTW",
        window,
    )
    return _dtw_fallback_path(lrc_seq, whisper_seq, dist)


def _empty_dtw_metrics() -> Dict[str, float]:
    return _empty_dtw_metrics_impl()


def _extract_lrc_words_base(lines: List[Line]) -> List[Dict]:
    return _extract_lrc_words_base_impl(lines)


def _compute_phonetic_costs_base(
    lrc_words: List[Dict],
    whisper_words: List[TranscriptionWord],
    language: str,
    min_similarity: float,
) -> Dict[Tuple[int, int], float]:
    return _compute_phonetic_costs_base_impl(
        lrc_words,
        whisper_words,
        language,
        min_similarity,
        _phonetic_similarity_for_state,
    )


def _extract_alignments_from_path_base(
    path: List[Tuple[int, int]],
    lrc_words: List[Dict],
    whisper_words: List[TranscriptionWord],
    language: str,
    min_similarity: float,
    precomputed_similarity: Optional[Dict[Tuple[int, int], float]] = None,
) -> Dict[int, Tuple[TranscriptionWord, float]]:
    return _extract_alignments_from_path_base_impl(
        path,
        lrc_words,
        whisper_words,
        language,
        min_similarity,
        _phonetic_similarity_for_state,
        precomputed_similarity=precomputed_similarity,
    )


def _apply_dtw_alignments_base(
    lines: List[Line],
    lrc_words: List[Dict],
    alignments_map: Dict[int, Tuple[TranscriptionWord, float]],
) -> Tuple[List[Line], List[str]]:
    return _apply_dtw_alignments_base_impl(lines, lrc_words, alignments_map)


def align_dtw_whisper_base(
    lines: List[Line],
    whisper_words: List[TranscriptionWord],
    language: str = "fra-Latn",
    min_similarity: float = 0.4,
) -> Tuple[List[Line], List[str]]:
    """Align LRC to Whisper using Dynamic Time Warping (base version)."""
    if not lines or not whisper_words:
        return lines, []

    lrc_words = _extract_lrc_words_base(lines)
    if not lrc_words:
        return lines, []

    # Pre-compute IPA
    logger.debug(f"DTW: Pre-computing IPA for {len(whisper_words)} Whisper words...")
    for ww in whisper_words:
        _get_ipa_for_state(ww.text, language)
    for lw in lrc_words:
        _get_ipa_for_state(lw["text"], language)

    logger.debug(
        f"DTW: Building cost matrix ({len(lrc_words)} x {len(whisper_words)})..."
    )
    phonetic_costs = _compute_phonetic_costs_base(
        lrc_words, whisper_words, language, min_similarity
    )
    lrc_times = np.array([lw["start"] for lw in lrc_words])
    whisper_times = np.array([ww.start for ww in whisper_words])
    lrc_seq = np.column_stack([np.arange(len(lrc_words)), lrc_times])
    whisper_seq = np.column_stack([np.arange(len(whisper_words)), whisper_times])

    def dtw_dist(a, b):
        i, lrc_t = int(a[0]), a[1]
        j, whisper_t = int(b[0]), b[1]
        phon_cost = phonetic_costs[(i, j)]
        time_diff = abs(whisper_t - lrc_t)
        time_penalty = min(time_diff / 20.0, 1.0)
        return 0.7 * phon_cost + 0.3 * time_penalty

    # Run DTW
    logger.debug("DTW: Running alignment...")
    try:
        fastdtw = _load_fastdtw_for_state()

        _distance, path = fastdtw(lrc_seq, whisper_seq, dist=dtw_dist)

    except ImportError:
        logger.warning("fastdtw not available, falling back to exact DTW")
        _distance, path = _dtw_fallback_with_runtime_guard(
            lrc_seq, whisper_seq, dtw_dist
        )

    precomputed_similarity = {
        key: max(0.0, 1.0 - float(cost)) for key, cost in phonetic_costs.items()
    }
    alignments_map = _extract_alignments_from_path_base(
        path,
        lrc_words,
        whisper_words,
        language,
        min_similarity,
        precomputed_similarity=precomputed_similarity,
    )

    aligned_lines, corrections = _apply_dtw_alignments_base(
        lines, lrc_words, alignments_map
    )

    logger.info(f"DTW alignment complete: {len(corrections)} lines modified")
    return aligned_lines, corrections


def _compute_dtw_alignment_metrics(
    lines: List[Line],
    lrc_words: List[Dict],
    alignments_map: Dict[int, Tuple[TranscriptionWord, float]],
) -> Dict[str, float]:
    return _compute_dtw_alignment_metrics_impl(lines, lrc_words, alignments_map)


def _retime_lines_from_dtw_alignments(
    lines: List[Line],
    lrc_words: List[Dict],
    alignments_map: Dict[int, Tuple[TranscriptionWord, float]],
    min_word_duration: float = 0.05,
) -> Tuple[List[Line], List[str]]:
    return _retime_lines_from_dtw_alignments_impl(
        lines, lrc_words, alignments_map, min_word_duration=min_word_duration
    )


def _align_dtw_whisper_with_data(
    lines: List[Line],
    whisper_words: List[TranscriptionWord],
    language: str = "fra-Latn",
    min_similarity: float = 0.4,
    silence_regions: Optional[List[Tuple[float, float]]] = None,
    audio_features: Optional[timing_models.AudioFeatures] = None,
    lenient_vocal_activity_threshold: float = 0.3,
    lenient_activity_bonus: float = 0.4,
    low_word_confidence_threshold: float = 0.5,
) -> Tuple[
    List[Line],
    List[str],
    Dict[str, float],
    List[Dict],
    Dict[int, Tuple[TranscriptionWord, float]],
]:
    """Align LRC to Whisper using DTW and return alignment data for confidence gating."""
    from .whisper_integration import _fill_vocal_activity_gaps
    from ...audio_analysis import (
        _check_vocal_activity_in_range,
        _compute_silence_overlap,
        _is_time_in_silence,
    )
    from . import whisper_phonetic_dtw

    if not lines or not whisper_words:
        return (
            lines,
            [],
            _empty_dtw_metrics(),
            [],
            {},
        )

    if audio_features:
        whisper_words, _ = _fill_vocal_activity_gaps(
            whisper_words, audio_features, lenient_vocal_activity_threshold
        )

    lrc_words = whisper_phonetic_dtw._extract_lrc_words(lines)
    if not lrc_words:
        return (
            lines,
            [],
            _empty_dtw_metrics(),
            [],
            {},
        )

    # Pre-compute IPA
    logger.debug(f"DTW: Pre-computing IPA for {len(whisper_words)} Whisper words...")
    phonetic_utils._prewarm_ipa_cache(
        [ww.text for ww in whisper_words] + [lw["text"] for lw in lrc_words],
        language,
    )

    logger.debug(
        f"DTW: Building cost matrix ({len(lrc_words)} x {len(whisper_words)})..."
    )
    phonetic_costs = whisper_phonetic_dtw._compute_phonetic_costs(
        lrc_words, whisper_words, language, min_similarity
    )
    lrc_times = np.array([lw["start"] for lw in lrc_words])
    whisper_times = np.array([ww.start for ww in whisper_words])
    lrc_seq = np.column_stack([np.arange(len(lrc_words)), lrc_times])
    whisper_seq = np.column_stack([np.arange(len(whisper_words)), whisper_times])
    use_silence = silence_regions or []

    def dtw_dist(a, b):
        i, lrc_t = int(a[0]), a[1]
        j, whisper_t = int(b[0]), b[1]
        phon_cost = phonetic_costs[(i, j)]

        # Leniency mechanism: if Whisper word has low confidence but there is vocal activity,
        # be more lenient about phonetic mismatch.
        if (
            audio_features
            and whisper_words[j].probability < low_word_confidence_threshold
        ):
            # Check activity around the whisper word
            w_start = whisper_words[j].start
            w_end = whisper_words[j].end
            vocal_activity = _check_vocal_activity_in_range(
                w_start, w_end, audio_features
            )
            if vocal_activity > lenient_vocal_activity_threshold:
                phon_cost = max(0.0, phon_cost - lenient_activity_bonus)

        time_diff = abs(whisper_t - lrc_t)
        time_penalty = min(time_diff / 20.0, 1.0)
        gap_start = min(lrc_t, whisper_t)
        gap_end = max(lrc_t, whisper_t)
        silence_overlap = _compute_silence_overlap(gap_start, gap_end, use_silence)
        silence_penalty = min(silence_overlap / 2.0, 1.0)
        if _is_time_in_silence(whisper_t, use_silence):
            silence_penalty = max(silence_penalty, 0.8)
        activity_penalty = 0.0
        if audio_features and gap_end - gap_start > 0.5:
            activity = _check_vocal_activity_in_range(
                gap_start, gap_end, audio_features
            )
            non_silent = max(gap_end - gap_start - silence_overlap, 0.0)
            if activity > 0.5 and non_silent > 0.5:
                activity_penalty = min(activity, 1.0)
        return (
            0.5 * phon_cost
            + 0.2 * time_penalty
            + 0.2 * silence_penalty
            + 0.1 * activity_penalty
        )

    # Run DTW
    logger.debug("DTW: Running alignment...")
    try:
        fastdtw = _load_fastdtw_for_state()

        _distance, path = fastdtw(lrc_seq, whisper_seq, dist=dtw_dist)

    except ImportError:
        logger.warning("fastdtw not available, falling back to exact DTW")
        _distance, path = _dtw_fallback_with_runtime_guard(
            lrc_seq, whisper_seq, dtw_dist
        )

    precomputed_similarity = {
        key: max(0.0, 1.0 - float(cost)) for key, cost in phonetic_costs.items()
    }
    alignments_map = _extract_alignments_from_path_base(
        path,
        lrc_words,
        whisper_words,
        language,
        min_similarity,
        precomputed_similarity=precomputed_similarity,
    )

    aligned_lines, corrections = _apply_dtw_alignments_base(
        lines, lrc_words, alignments_map
    )
    metrics = _compute_dtw_alignment_metrics(lines, lrc_words, alignments_map)

    logger.info(f"DTW alignment complete: {len(corrections)} lines modified")
    return aligned_lines, corrections, metrics, lrc_words, alignments_map


def align_dtw_whisper(
    lines: List[Line],
    whisper_words: List[TranscriptionWord],
    language: str = "fra-Latn",
    min_similarity: float = 0.4,
) -> Tuple[List[Line], List[str], Dict[str, float]]:
    """Align LRC to Whisper using Dynamic Time Warping."""
    lrc_words = _extract_lrc_words_base(lines)
    if not lrc_words:
        return (
            lines,
            [],
            _empty_dtw_metrics(),
        )

    # Pre-compute IPA
    phonetic_utils._prewarm_ipa_cache(
        [ww.text for ww in whisper_words] + [lw["text"] for lw in lrc_words],
        language,
    )

    phonetic_costs = _compute_phonetic_costs_base(
        lrc_words, whisper_words, language, min_similarity
    )

    lrc_times = np.array([lw["start"] for lw in lrc_words])
    whisper_times = np.array([ww.start for ww in whisper_words])
    lrc_seq = np.column_stack([np.arange(len(lrc_words)), lrc_times])
    whisper_seq = np.column_stack([np.arange(len(whisper_words)), whisper_times])

    def dtw_dist(a, b):
        i, lrc_t = int(a[0]), a[1]
        j, whisper_t = int(b[0]), b[1]
        phon_cost = phonetic_costs[(i, j)]
        time_diff = abs(whisper_t - lrc_t)
        time_penalty = min(time_diff / 20.0, 1.0)
        return 0.7 * phon_cost + 0.3 * time_penalty

    try:
        fastdtw = _load_fastdtw_for_state()
        _distance, path = fastdtw(lrc_seq, whisper_seq, dist=dtw_dist)
    except ImportError:
        logger.warning("fastdtw not available, falling back to exact DTW")
        _distance, path = _dtw_fallback_path(lrc_seq, whisper_seq, dtw_dist)

    alignments_map = _extract_alignments_from_path_base(
        path, lrc_words, whisper_words, language, min_similarity
    )

    aligned_lines, corrections = _apply_dtw_alignments_base(
        lines, lrc_words, alignments_map
    )
    metrics = _compute_dtw_alignment_metrics(lines, lrc_words, alignments_map)

    return aligned_lines, corrections, metrics
