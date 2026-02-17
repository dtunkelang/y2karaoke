import numpy as np
import pytest
from typing import List, Dict, Any

from y2karaoke.core.refine_visual import _detect_highlight_times


def _create_word_vals(
    times: List[float], colors: List[List[float]]
) -> List[Dict[str, Any]]:
    return [
        {"t": t, "avg": np.array(c, dtype=np.float32), "mask": None, "lab": None}
        for t, c in zip(times, colors)
    ]


def test_detect_highlight_times_returns_none_for_short_sequence():
    vals = _create_word_vals([0.1] * 5, [[0, 0, 0]] * 5)
    s, e = _detect_highlight_times(vals)
    assert s is None
    assert e is None


def test_detect_highlight_times_returns_none_for_flat_color():
    times = np.linspace(0, 2, 20)
    colors = [[10.0, 128.0, 128.0]] * 20  # Constant color
    vals = _create_word_vals(times, colors)
    s, e = _detect_highlight_times(vals)
    assert s is None
    assert e is None


def test_detect_highlight_times_detects_simple_transition():
    # Simulate:
    # 0.0 - 1.0: Inactive (Color A - Bright)
    # 1.0 - 1.5: Transition
    # 1.5 - 2.0: Active (Color B - Dark/Color)

    times = np.linspace(0, 2.0, 41)  # 0.05s intervals
    colors = []

    # Logic assumes Peak (Bright) -> Valley (Dark)
    color_a = np.array([90.0, 128.0, 128.0])  # Bright
    color_b = np.array([10.0, 128.0, 128.0])  # Darker

    for t in times:
        if t <= 1.0:
            c = color_a
        elif t >= 1.5:
            c = color_b
        else:
            # Linear interp
            ratio = (t - 1.0) / 0.5
            c = color_a + (color_b - color_a) * ratio
        colors.append(c)

    vals = _create_word_vals(times, colors)

    s, e = _detect_highlight_times(vals)

    assert s is not None
    assert e is not None

    # Start trigger: Consistent departure from noise floor
    # Noise is 0 here. Departure from initial state happens at 1.05.
    assert 1.0 <= s <= 1.2

    # End trigger: When color becomes closer to final than initial (Crossover)
    # Transition 1.0 -> 1.5. Midpoint is 1.25.
    # So e should be around 1.25 or 1.3.
    assert 1.2 <= e <= 1.4


def test_detect_highlight_times_handles_noisy_data():
    np.random.seed(42)
    times = np.linspace(0, 2.0, 41)
    colors = []
    color_a = np.array([90.0, 128.0, 128.0])
    color_b = np.array([10.0, 128.0, 128.0])

    for t in times:
        if t <= 1.0:
            c = color_a
        elif t >= 1.5:
            c = color_b
        else:
            ratio = (t - 1.0) / 0.5
            c = color_a + (color_b - color_a) * ratio

        # Add noise
        noise = np.random.normal(0, 0.5, 3)
        colors.append(c + noise)

    vals = _create_word_vals(times, colors)
    s, e = _detect_highlight_times(vals)

    assert s is not None
    assert e is not None
    assert 1.0 <= s <= 1.25
    assert 1.2 <= e <= 1.45
