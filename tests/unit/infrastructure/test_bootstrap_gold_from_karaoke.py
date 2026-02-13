import importlib.util
import sys
from pathlib import Path

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
progress_to_word_candidates = _MODULE._progress_to_word_candidates
fit_line_word_times = _MODULE._fit_line_word_times
interp_cross_time = _MODULE._interp_cross_time


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


def test_progress_to_word_candidates_and_fit_respect_line_start() -> None:
    line = TargetLine(1, 10.0, None, "hello world", ["hello", "world"])
    times = [10.0, 10.2, 10.4, 10.6, 10.8, 11.0]
    progress = [0.0, 0.1, 0.45, 0.55, 0.8, 1.0]
    cands = progress_to_word_candidates(line, 10.0, 12.0, times, progress)
    fitted = fit_line_word_times(line, cands, 12.0)
    assert fitted[0][0] == 10.0
    assert fitted[0][1] <= fitted[1][0]
    assert fitted[1][1] <= 12.0


def test_interp_cross_time_is_subframe() -> None:
    t = [0.0, 0.5, 1.0]
    p = [0.0, 0.4, 1.0]
    crossed = interp_cross_time(t, p, 0.7)
    assert crossed is not None
    assert 0.70 < crossed < 0.85
