from pathlib import Path

import pytest

from y2karaoke.core.visual.bootstrap_selection import select_candidate_with_rankings


def test_select_candidate_with_rankings_explicit_url():
    out = select_candidate_with_rankings(
        candidate_url="https://youtube.com/watch?v=abc",
        artist=None,
        title=None,
        max_candidates=5,
        suitability_fps=1.0,
        show_candidates=False,
        allow_low_suitability=False,
        min_detectability=0.45,
        min_word_level_score=0.15,
        downloader=object(),
        song_dir=Path("/tmp/demo"),
        search_fn=lambda *a: [],
        rank_fn=lambda *a: [],
        suitability_check_fn=lambda *a: True,
    )
    assert out[0].endswith("abc")
    assert out[1] is None
    assert out[2] == {}
    assert out[3] == []


def test_select_candidate_with_rankings_rejects_low_quality(tmp_path):
    with pytest.raises(ValueError):
        select_candidate_with_rankings(
            candidate_url=None,
            artist="Artist",
            title="Song",
            max_candidates=5,
            suitability_fps=1.0,
            show_candidates=False,
            allow_low_suitability=False,
            min_detectability=0.45,
            min_word_level_score=0.15,
            downloader=object(),
            song_dir=tmp_path,
            search_fn=lambda *a: [{"url": "u"}],
            rank_fn=lambda *a: [
                {
                    "url": "u",
                    "video_path": str(tmp_path / "v.mp4"),
                    "metrics": {"detectability_score": 0.2, "word_level_score": 0.1},
                }
            ],
            suitability_check_fn=lambda *a: False,
        )
