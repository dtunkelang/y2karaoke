import y2karaoke.core.timing_evaluator as te


def test_select_best_source_returns_none_when_no_reports(monkeypatch):
    monkeypatch.setattr(te, "compare_sources", lambda *a, **k: {})

    result = te.select_best_source("Title", "Artist", "vocals.wav")

    assert result == (None, None, None)


def test_select_best_source_applies_duration_bonus(monkeypatch):
    report_a = te.TimingReport(
        source_name="A",
        overall_score=80.0,
        line_alignment_score=80.0,
        pause_alignment_score=80.0,
    )
    report_b = te.TimingReport(
        source_name="B",
        overall_score=82.0,
        line_alignment_score=82.0,
        pause_alignment_score=82.0,
    )

    monkeypatch.setattr(
        te, "compare_sources", lambda *a, **k: {"A": report_a, "B": report_b}
    )
    monkeypatch.setattr(
        "y2karaoke.core.sync.fetch_from_all_sources",
        lambda *a, **k: {"A": ("lrcA", 175), "B": ("lrcB", 200)},
    )

    lrc_text, source, report = te.select_best_source(
        "Title", "Artist", "vocals.wav", target_duration=180
    )

    assert lrc_text == "lrcA"
    assert source == "A"
    assert report is report_a


def test_select_best_source_returns_none_when_best_source_missing(monkeypatch):
    report = te.TimingReport(
        source_name="A",
        overall_score=80.0,
        line_alignment_score=80.0,
        pause_alignment_score=80.0,
    )
    monkeypatch.setattr(te, "compare_sources", lambda *a, **k: {"A": report})
    monkeypatch.setattr(
        "y2karaoke.core.sync.fetch_from_all_sources", lambda *a, **k: {}
    )

    result = te.select_best_source("Title", "Artist", "vocals.wav")

    assert result == (None, None, None)
