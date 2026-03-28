from pathlib import Path

from tools import analyze_alternating_hook_lyrics_hits as tool


def test_analyze_alternating_hook_lyrics_hits_counts_distinct_pairs(tmp_path: Path):
    first = tmp_path / "run1"
    first.mkdir()
    (first / "01_song_clip_lyrics.txt").write_text(
        "Take on me\nTake me on\nI'll be gone\n",
        encoding="utf-8",
    )

    second = tmp_path / "run2"
    second.mkdir()
    (second / "02_song_clip_lyrics.txt").write_text(
        "Take on me\nTake me on\n",
        encoding="utf-8",
    )

    result = tool.analyze(root=tmp_path)

    assert result["hit_count"] == 2
    assert result["distinct_pairs"] == [
        {"first_text": "Take on me", "second_text": "Take me on", "count": 2}
    ]
    assert result["distinct_clip_stems"] == [
        {"clip_stem": "01_song", "count": 1},
        {"clip_stem": "02_song", "count": 1},
    ]


def test_analyze_alternating_hook_lyrics_hits_skips_non_alternating(tmp_path: Path):
    (tmp_path / "01_song_clip_lyrics.txt").write_text(
        "Take on me\nTake on me\n",
        encoding="utf-8",
    )

    result = tool.analyze(root=tmp_path)

    assert result["hit_count"] == 0
