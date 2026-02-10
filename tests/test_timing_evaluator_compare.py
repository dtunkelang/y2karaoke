from y2karaoke.core.models import Line, Word
from y2karaoke.core.timing_models import AudioFeatures, TimingReport
import y2karaoke.core.timing_evaluator as te
import y2karaoke.core.timing_evaluator_comparison as te_comp


def test_compare_sources_returns_empty_on_audio_failure(monkeypatch):
    monkeypatch.setattr(te_comp, "extract_audio_features", lambda _path: None)
    assert te.compare_sources("Song", "Artist", "vocals.wav") == {}


def test_compare_sources_builds_reports(monkeypatch):
    features = AudioFeatures(
        onset_times=[],
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=1.0,
        duration=1.0,
        energy_envelope=[],
        energy_times=[],
    )
    monkeypatch.setattr(te_comp, "extract_audio_features", lambda _path: features)

    monkeypatch.setattr(
        "y2karaoke.core.sync.fetch_from_all_sources",
        lambda _title, _artist: {
            "empty": ("", None),
            "source1": ("[00:00.00]hello world", 3.0),
        },
    )

    monkeypatch.setattr(
        "y2karaoke.core.lrc.create_lines_from_lrc",
        lambda _text, romanize=False, title=None, artist=None: [
            Line(
                words=[
                    Word(text="hello", start_time=0.0, end_time=0.1),
                    Word(text="world", start_time=0.1, end_time=0.2),
                ]
            )
        ],
    )
    monkeypatch.setattr(
        "y2karaoke.core.lrc.parse_lrc_with_timing",
        lambda _text, _title, _artist: [(0.0, "hello world")],
    )

    def fake_evaluate_timing(lines, _features, source_name):
        return TimingReport(
            source_name=source_name,
            overall_score=80.0,
            line_alignment_score=80.0,
            pause_alignment_score=90.0,
            summary="ok",
        )

    monkeypatch.setattr(te_comp, "evaluate_timing", fake_evaluate_timing)

    reports = te.compare_sources("Song", "Artist", "vocals.wav")
    assert "source1" in reports
    assert reports["source1"].summary == "ok"


def test_compare_sources_skips_source_on_error(monkeypatch):
    features = AudioFeatures(
        onset_times=[],
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=1.0,
        duration=1.0,
        energy_envelope=[],
        energy_times=[],
    )
    monkeypatch.setattr(te_comp, "extract_audio_features", lambda _path: features)

    monkeypatch.setattr(
        "y2karaoke.core.sync.fetch_from_all_sources",
        lambda _title, _artist: {"bad": ("[00:00.00]hello", 3.0)},
    )

    monkeypatch.setattr(
        "y2karaoke.core.lrc.create_lines_from_lrc",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad lrc")),
    )

    reports = te.compare_sources("Song", "Artist", "vocals.wav")
    assert reports == {}


def test_compare_sources_handles_empty_timings(monkeypatch):
    features = AudioFeatures(
        onset_times=[],
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=1.0,
        duration=1.0,
        energy_envelope=[],
        energy_times=[],
    )
    monkeypatch.setattr(te_comp, "extract_audio_features", lambda _path: features)

    monkeypatch.setattr(
        "y2karaoke.core.sync.fetch_from_all_sources",
        lambda _title, _artist: {"source": ("[00:00.00]hello", 3.0)},
    )

    monkeypatch.setattr(
        "y2karaoke.core.lrc.create_lines_from_lrc",
        lambda *_args, **_kwargs: [Line(words=[])],
    )
    monkeypatch.setattr(
        "y2karaoke.core.lrc.parse_lrc_with_timing",
        lambda *_args, **_kwargs: [],
    )

    monkeypatch.setattr(
        te_comp,
        "evaluate_timing",
        lambda _lines, _features, source_name: TimingReport(
            source_name=source_name,
            overall_score=50.0,
            line_alignment_score=50.0,
            pause_alignment_score=50.0,
            summary="ok",
        ),
    )

    reports = te.compare_sources("Song", "Artist", "vocals.wav")
    assert "source" in reports


def test_select_best_source_prefers_highest_score(monkeypatch):
    reports = {
        "a": TimingReport(
            source_name="a",
            overall_score=50.0,
            line_alignment_score=50.0,
            pause_alignment_score=50.0,
            summary="a",
        ),
        "b": TimingReport(
            source_name="b",
            overall_score=70.0,
            line_alignment_score=70.0,
            pause_alignment_score=70.0,
            summary="b",
        ),
    }
    monkeypatch.setattr(te_comp, "compare_sources", lambda *_args: reports)
    monkeypatch.setattr(
        "y2karaoke.core.sync.fetch_from_all_sources",
        lambda _title, _artist: {
            "a": ("lrc a", 100),
            "b": ("lrc b", 100),
        },
    )

    lrc_text, source, report = te.select_best_source(
        "Song", "Artist", "vocals.wav", target_duration=100
    )
    assert source == "b"
    assert lrc_text == "lrc b"
    assert report is reports["b"]


def test_select_best_source_handles_no_reports(monkeypatch):
    monkeypatch.setattr(te_comp, "compare_sources", lambda *_args: {})
    lrc_text, source, report = te.select_best_source("Song", "Artist", "vocals.wav")
    assert lrc_text is None
    assert source is None
    assert report is None
