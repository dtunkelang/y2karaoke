import pytest

from y2karaoke.core.components.lyrics import helpers as lh
from y2karaoke.core import lyrics_whisper as lw
from y2karaoke.core.models import SongMetadata


def test_detect_and_apply_offset_applies_delta(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda *_: 1.0,
    )
    line_timings = [(0.0, "line one"), (2.0, "line two")]

    updated, offset = lh._detect_and_apply_offset(
        "vocals.wav", line_timings, lyrics_offset=None
    )

    assert offset == 1.0
    assert updated[0][0] == pytest.approx(1.0)
    assert updated[1][0] == pytest.approx(3.0)


def test_detect_and_apply_offset_respects_manual(monkeypatch):
    monkeypatch.setattr(
        "y2karaoke.core.components.alignment.alignment.detect_song_start",
        lambda *_: 2.0,
    )
    line_timings = [(1.0, "line one")]

    updated, offset = lh._detect_and_apply_offset(
        "vocals.wav", line_timings, lyrics_offset=5.0
    )

    assert offset == 5.0
    assert updated[0][0] == pytest.approx(6.0)


def test_get_lyrics_simple_uses_lrc_and_singer_info(monkeypatch):
    lrc_text = "[00:02.00]First\n[00:04.00]Second"
    line_timings = [(2.0, "First"), (4.0, "Second")]

    def fake_fetch_lrc(*args, **kwargs):
        return lrc_text, line_timings, "test"

    def fake_genius(*args, **kwargs):
        metadata = SongMetadata(
            singers=["A", "B"], is_duet=True, title="Hello", artist="Tester"
        )
        return [("First", "A"), ("Second", "B")], metadata

    with lw.use_lyrics_whisper_hooks(
        fetch_lrc_text_and_timings_fn=fake_fetch_lrc,
        fetch_genius_lyrics_with_singers_fn=fake_genius,
    ):
        lines, metadata = lw.get_lyrics_simple(
            title="Hello",
            artist="Tester",
            vocals_path=None,
            romanize=False,
            filter_promos=False,
        )

    assert metadata is not None
    assert metadata.is_duet is True
    assert len(lines) == 2
    assert lines[0].singer == metadata.get_singer_id("A")
    assert lines[1].singer == metadata.get_singer_id("B")
    assert all(word.singer == lines[0].singer for word in lines[0].words)


def test_get_lyrics_simple_fallback_placeholder(monkeypatch):
    with lw.use_lyrics_whisper_hooks(
        fetch_lrc_text_and_timings_fn=lambda *a, **k: (None, None, ""),
        fetch_genius_lyrics_with_singers_fn=lambda *a, **k: (None, None),
    ):
        lines, metadata = lw.get_lyrics_simple(
            title="Missing",
            artist="Artist",
            vocals_path=None,
        )

    assert metadata is not None
    assert len(lines) == 1
    assert "Lyrics not available" in " ".join(word.text for word in lines[0].words)


def test_calculate_quality_score_bounds():
    report = {
        "lyrics_quality": {"quality_score": 80},
        "alignment_method": "lrc_only",
        "issues": [],
    }
    assert lw._calculate_quality_score(report) == 80

    report = {
        "lyrics_quality": {"quality_score": 10},
        "alignment_method": "none",
        "issues": ["a", "b"],
    }
    assert lw._calculate_quality_score(report) == 0.0


def test_calculate_quality_score_from_dtw_metrics():
    report = {
        "lyrics_quality": {},
        "alignment_method": "whisper_hybrid",
        "dtw_metrics": {
            "matched_ratio": 1.0,
            "avg_similarity": 0.85,
            "line_coverage": 1.0,
        },
        "issues": [],
    }
    assert lw._calculate_quality_score(report) == 100.0
