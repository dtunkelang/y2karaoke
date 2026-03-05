"""Fallback DTW implementations used when fastdtw is unavailable."""

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def dtw_fallback_path(
    lrc_seq: np.ndarray,
    whisper_seq: np.ndarray,
    dist: Callable[..., float],
    *,
    window: Optional[int] = None,
) -> Tuple[float, List[Tuple[int, int]]]:
    """Compute an exact DTW path when fastdtw is unavailable."""
    m = int(lrc_seq.shape[0])
    n = int(whisper_seq.shape[0])
    if m == 0 or n == 0:
        return float("inf"), []

    costs = np.full((m + 1, n + 1), np.inf, dtype=np.float64)
    prev = np.zeros((m + 1, n + 1), dtype=np.uint8)
    costs[0, 0] = 0.0

    for i in range(1, m + 1):
        a = lrc_seq[i - 1]
        j_lo, j_hi = _dtw_window_bounds(i, m=m, n=n, window=window)
        for j in range(j_lo, j_hi + 1):
            b = whisper_seq[j - 1]
            step_cost = float(dist(a, b))
            best, direction = _dtw_best_predecessor(costs, i, j)
            costs[i, j] = step_cost + best
            prev[i, j] = direction

    path = _dtw_backtrack_path(prev, m=m, n=n)
    return float(costs[m, n]), path


def _dtw_window_bounds(
    i: int, *, m: int, n: int, window: Optional[int]
) -> tuple[int, int]:
    if window is None:
        return 1, n
    center = int(round((i - 1) * (n / max(m, 1)))) + 1
    j_lo = max(1, center - window)
    j_hi = min(n, center + window)
    return j_lo, j_hi


def _dtw_best_predecessor(costs: np.ndarray, i: int, j: int) -> tuple[float, int]:
    diag = costs[i - 1, j - 1]
    up = costs[i - 1, j]
    left = costs[i, j - 1]
    best = diag
    direction = 1  # diagonal
    if up < best:
        best = up
        direction = 2  # up
    if left < best:
        best = left
        direction = 3  # left
    return best, direction


def _dtw_backtrack_path(prev: np.ndarray, *, m: int, n: int) -> List[Tuple[int, int]]:
    i, j = m, n
    path: List[Tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        direction = int(prev[i, j])
        if direction == 1:
            i -= 1
            j -= 1
        elif direction == 2:
            i -= 1
        elif direction == 3:
            j -= 1
        else:
            break
    path.reverse()
    return path


def dtw_fallback_with_runtime_guard(
    lrc_seq: np.ndarray, whisper_seq: np.ndarray, dist: Callable[..., float]
) -> Tuple[float, List[Tuple[int, int]]]:
    m = int(lrc_seq.shape[0])
    n = int(whisper_seq.shape[0])
    cell_budget = m * n
    if cell_budget <= 250_000:
        return dtw_fallback_path(lrc_seq, whisper_seq, dist)

    window = max(96, abs(n - m) + 96)
    logger.info("Using banded exact DTW fallback (m=%d, n=%d, window=%d)", m, n, window)
    distance, path = dtw_fallback_path(lrc_seq, whisper_seq, dist, window=window)
    if np.isfinite(distance):
        return distance, path

    logger.warning(
        "Banded exact DTW fallback found no path (window=%d); retrying full exact DTW",
        window,
    )
    return dtw_fallback_path(lrc_seq, whisper_seq, dist)
