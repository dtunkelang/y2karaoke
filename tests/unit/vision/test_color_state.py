import numpy as np

from y2karaoke.vision.color import classify_word_state


def test_classify_word_state_ignores_dark_background() -> None:
    roi = np.zeros((24, 48, 3), dtype=np.uint8)
    roi[8:16, 14:34] = np.array([245, 245, 245], dtype=np.uint8)

    c_un = np.array([250, 250, 250], dtype=np.float32)
    c_sel = np.array([190, 80, 200], dtype=np.float32)
    state, ratio, _ = classify_word_state(roi, c_un, c_sel)

    assert state == "unselected"
    assert ratio < 0.2


def test_classify_word_state_detects_mixed_foreground() -> None:
    roi = np.zeros((24, 48, 3), dtype=np.uint8)
    roi[8:16, 10:24] = np.array([248, 248, 248], dtype=np.uint8)
    roi[8:16, 24:38] = np.array([188, 84, 205], dtype=np.uint8)

    c_un = np.array([250, 250, 250], dtype=np.float32)
    c_sel = np.array([190, 80, 200], dtype=np.float32)
    state, ratio, _ = classify_word_state(roi, c_un, c_sel)

    assert state == "mixed"
    assert 0.2 < ratio < 0.8
