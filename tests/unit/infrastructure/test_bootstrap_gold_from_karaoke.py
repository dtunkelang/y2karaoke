import importlib.util
import sys
from pathlib import Path
import numpy as np

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "bootstrap_gold_from_karaoke.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "bootstrap_gold_from_karaoke_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

TargetLine = _MODULE.TargetLine
parse_lrc_lines = _MODULE.parse_lrc_lines
score_candidate = _MODULE.score_candidate
text_similarity = _MODULE._text_similarity
cluster_colors = _MODULE._cluster_colors
classify_word_state = _MODULE._classify_word_state


def test_parse_lrc_lines_extracts_timestamps_and_words() -> None:
    lrc = (
        "[00:14.2]White shirt now red, my bloody nose\n"
        "[00:17.8]Sleepin', you're on your tippy toes"
    )
    lines = parse_lrc_lines(lrc)
    assert len(lines) == 2
    assert lines[0].start == 14.2
    assert lines[0].words[0] == "White"
    assert lines[1].start == 17.8


def test_score_candidate_prefers_karaoke_signals() -> None:
    strong = {
        "title": "Song Name (Karaoke Version)",
        "channel": "Sing King",
        "view_count": 3_000_000,
        "duration": 200,
    }
    weak = {
        "title": "Song Name (Official Video)",
        "channel": "ArtistVEVO",
        "view_count": 3_000_000,
        "duration": 200,
    }
    assert score_candidate(strong) > score_candidate(weak)


def test_text_similarity_is_case_and_punctuation_tolerant() -> None:
    s = text_similarity(
        "White shirt now red, my bloody nose", "white shirt now red my bloody nose"
    )
    assert s > 0.95


def test_cluster_colors_finds_distinct_centers() -> None:
    # Simulate some white-ish and red-ish pixels
    whites = [np.array([250, 250, 250]) for _ in range(20)]
    reds = [np.array([20, 20, 240]) for _ in range(20)]
    samples = whites + reds
    c1, c2 = cluster_colors(samples)

    # Ensure centers are found and distinct
    assert np.linalg.norm(c1 - c2) > 100
    # One should be white-ish, one red-ish
    centers = [c1, c2]
    assert any(np.mean(c) > 200 for c in centers)
    assert any(c[2] > 200 and c[0] < 50 for c in centers)


def test_classify_word_state() -> None:
    c_un = np.array([255, 255, 255])  # White unselected
    c_sel = np.array([0, 0, 255])  # Red selected

    # All white ROI
    roi_white = np.full((10, 10, 3), 255, dtype=np.uint8)
    assert classify_word_state(roi_white, c_un, c_sel) == "unselected"

    # All red ROI
    roi_red = np.zeros((10, 10, 3), dtype=np.uint8)
    roi_red[:, :, 2] = 255
    assert classify_word_state(roi_red, c_un, c_sel) == "selected"

    # Half and half mixed ROI
    roi_mixed = np.full((10, 10, 3), 255, dtype=np.uint8)
    roi_mixed[:, 5:, 0:2] = 0  # Right half is red
    assert classify_word_state(roi_mixed, c_un, c_sel) == "mixed"
