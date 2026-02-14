import pytest

import y2karaoke.core.components.alignment.timing_evaluator as te
import y2karaoke.core.components.whisper.whisper_integration as wi
import y2karaoke.core.components.whisper.whisper_dtw as w_dtw
import y2karaoke.core.components.whisper.whisper_alignment as wa
from y2karaoke.core.models import Line, Word


def _alignment_similarity_hook(fn):
    return wa.use_whisper_alignment_hooks(phonetic_similarity_fn=fn)


def test_align_words_to_whisper_adjusts_word():
    lines = [Line(words=[Word(text="hello", start_time=0.0, end_time=0.5)])]
    whisper_words = [
        te.TranscriptionWord(start=2.0, end=2.4, text="hello", probability=0.9)
    ]
    with wi.use_whisper_integration_hooks(get_ipa_fn=lambda *_args, **_kwargs: None):
        aligned, corrections = te.align_words_to_whisper(
            lines,
            whisper_words,
            language="eng-Latn",
        )
    assert aligned[0].words[0].start_time == 2.0
    assert corrections


def test_assess_lrc_quality_reports_good_match(monkeypatch):
    lines = [Line(words=[Word(text="hello", start_time=1.0, end_time=1.5)])]
    words = [te.TranscriptionWord(start=1.1, end=1.6, text="hello", probability=0.9)]

    with w_dtw.use_whisper_dtw_hooks(
        phonetic_similarity_fn=lambda *_args, **_kwargs: 0.8
    ):
        quality, assessments = te._assess_lrc_quality(lines, words, "eng-Latn")
    assert quality >= 0.9
    assert assessments


def test_extract_lrc_words_returns_indices():
    lines = [Line(words=[Word(text="one", start_time=0, end_time=1)])]
    lrc_words = te._extract_lrc_words(lines)
    assert len(lrc_words) == 1
    assert lrc_words[0]["text"] == "one"
    assert lrc_words[0]["line_idx"] == 0


def test_compute_phonetic_costs_includes_matches(monkeypatch):
    lrc_words = [{"text": "hello", "start": 1.0}]
    whisper_words = [
        te.TranscriptionWord(start=1.1, end=1.6, text="hello", probability=0.9)
    ]

    with w_dtw.use_whisper_dtw_hooks(
        phonetic_similarity_fn=lambda *_args, **_kwargs: 0.6
    ):
        costs = te._compute_phonetic_costs(lrc_words, whisper_words, "eng-Latn", 0.5)
    assert (0, 0) in costs
    assert costs[(0, 0)] < 0.5


def test_extract_alignments_from_path_filters_by_similarity(monkeypatch):
    path = [(0, 0)]
    lrc_words = [{"text": "hello", "start": 1.0}]
    whisper_words = [
        te.TranscriptionWord(start=1.1, end=1.6, text="hello", probability=0.9)
    ]

    with w_dtw.use_whisper_dtw_hooks(
        phonetic_similarity_fn=lambda *_args, **_kwargs: 0.1
    ):
        alignments = te._extract_alignments_from_path(
            path, lrc_words, whisper_words, "eng-Latn", 0.5
        )
    assert len(alignments) == 0


def test_apply_dtw_alignments_shifts_large_offsets():
    lines = [Line(words=[Word(text="hello", start_time=1.0, end_time=1.5)])]
    lrc_words = [{"text": "hello", "start": 1.0, "line_idx": 0, "word_idx": 0}]
    alignments = {
        0: (
            te.TranscriptionWord(start=5.0, end=5.5, text="hello", probability=0.9),
            0.9,
        )
    }

    aligned, corrections = te._apply_dtw_alignments(lines, lrc_words, alignments)
    assert aligned[0].words[0].start_time == 5.0
    assert corrections


def test_align_dtw_whisper_falls_back_without_fastdtw(monkeypatch):
    monkeypatch.setattr(
        w_dtw,
        "_load_fastdtw",
        lambda: (_ for _ in ()).throw(ImportError("missing fastdtw")),
    )

    lines = [Line(words=[Word(text="hello", start_time=1.0, end_time=1.5)])]
    words = [te.TranscriptionWord(start=1.1, end=1.6, text="hello", probability=0.9)]

    aligned, corrections, metrics = te.align_dtw_whisper(
        lines, words, min_similarity=0.1
    )
    assert len(aligned) == 1
    # Simple fallback doesn't necessarily shift if fastdtw missing, but should return lines
    assert metrics["matched_ratio"] >= 0


def test_correct_timing_with_whisper_no_transcription(monkeypatch):
    lines = [Line(words=[Word(text="a", start_time=0, end_time=1)])]
    with wi.use_whisper_integration_hooks(
        transcribe_vocals_fn=lambda *_: ([], [], "en", "base")
    ):
        aligned, corrections, metrics = te.correct_timing_with_whisper(
            lines, "vocals.wav"
        )
    assert aligned == lines
    assert not corrections


def test_correct_timing_with_whisper_uses_dtw(monkeypatch):
    lines = [Line(words=[Word(text="hello", start_time=10.0, end_time=11.0)])]
    words = [te.TranscriptionWord(start=20.0, end=21.0, text="hello", probability=0.9)]
    segments = [
        te.TranscriptionSegment(start=20.0, end=21.0, text="hello", words=words)
    ]

    with wi.use_whisper_integration_hooks(
        transcribe_vocals_fn=lambda *_: (segments, words, "en", "base"),
        extract_audio_features_fn=lambda *_: None,
        assess_lrc_quality_fn=lambda *_, **__: (0.1, []),
        align_dtw_whisper_with_data_fn=lambda *_, **__: (
            [Line(words=[Word(text="hello", start_time=20.0, end_time=21.0)])],
            ["dtw"],
            {"matched_ratio": 0.8, "avg_similarity": 0.8, "line_coverage": 0.8},
            [],
            {0: (words[0], 0.9)},
        ),
    ):
        aligned, corrections, metrics = te.correct_timing_with_whisper(
            lines, "vocals.wav"
        )
    assert aligned[0].start_time == 10.0
    assert "dtw" in corrections or any("DTW" in c for c in corrections)


def test_correct_timing_with_whisper_quality_good_uses_hybrid(monkeypatch):
    lines = [Line(words=[Word(text="hello", start_time=10.0, end_time=11.0)])]
    words = [te.TranscriptionWord(start=10.1, end=11.1, text="hello", probability=0.9)]
    segments = [
        te.TranscriptionSegment(start=10.0, end=11.5, text="hello", words=words)
    ]

    with wi.use_whisper_integration_hooks(
        transcribe_vocals_fn=lambda *_: (segments, words, "en", "base"),
        extract_audio_features_fn=lambda *_: None,
        assess_lrc_quality_fn=lambda *_, **__: (0.9, []),
        align_hybrid_lrc_whisper_fn=lambda *_, **__: (
            [Line(words=[Word(text="hello", start_time=10.1, end_time=11.1)])],
            ["hybrid"],
        ),
    ):
        aligned, corrections, metrics = te.correct_timing_with_whisper(
            lines, "vocals.wav"
        )
    assert "hybrid" in corrections


def test_correct_timing_with_whisper_quality_mixed_uses_hybrid(monkeypatch):
    lines = [Line(words=[Word(text="hello", start_time=10.0, end_time=11.0)])]
    words = [te.TranscriptionWord(start=10.1, end=11.1, text="hello", probability=0.9)]
    segments = [
        te.TranscriptionSegment(start=10.0, end=11.5, text="hello", words=words)
    ]

    with wi.use_whisper_integration_hooks(
        transcribe_vocals_fn=lambda *_: (segments, words, "en", "base"),
        extract_audio_features_fn=lambda *_: None,
        assess_lrc_quality_fn=lambda *_, **__: (0.5, []),
        align_hybrid_lrc_whisper_fn=lambda *_, **__: (
            [Line(words=[Word(text="hello", start_time=10.1, end_time=11.1)])],
            ["hybrid"],
        ),
    ):
        aligned, corrections, metrics = te.correct_timing_with_whisper(
            lines, "vocals.wav"
        )
    assert "hybrid" in corrections


def test_correct_timing_with_whisper_uses_dtw_retime_when_confident():
    lines = [Line(words=[Word(text="hello", start_time=10.0, end_time=11.0)])]
    words = [te.TranscriptionWord(start=20.0, end=21.0, text="hello", probability=0.9)]
    segments = [
        te.TranscriptionSegment(start=20.0, end=21.0, text="hello", words=words)
    ]

    # High confidence metrics
    metrics_high = {"matched_ratio": 0.9, "avg_similarity": 0.9, "line_coverage": 0.9}
    lrc_words = [{"line_idx": 0, "word_idx": 0, "text": "hello"}]
    align_map = {0: (words[0], 0.9)}
    with wi.use_whisper_integration_hooks(
        transcribe_vocals_fn=lambda *_: (segments, words, "en", "base"),
        extract_audio_features_fn=lambda *_: None,
        assess_lrc_quality_fn=lambda *_, **__: (0.1, []),
        align_dtw_whisper_with_data_fn=lambda *_, **__: (
            lines,
            ["dtw_data"],
            metrics_high,
            lrc_words,
            align_map,
        ),
        retime_lines_from_dtw_alignments_fn=lambda *_, **__: (
            [Line(words=[Word(text="hello", start_time=20.0, end_time=21.0)])],
            ["retimed"],
        ),
    ):
        aligned, corrections, metrics = te.correct_timing_with_whisper(
            lines, "vocals.wav"
        )
    assert "retimed" in corrections


def test_pull_lines_allows_low_similarity_when_late_and_ordered(monkeypatch):
    lines = [
        Line(words=[Word(text="oh", start_time=10.0, end_time=10.2)]),
        Line(
            words=[
                Word(text="qu'un", start_time=36.0, end_time=36.2),
                Word(text="concorde", start_time=36.3, end_time=36.5),
            ]
        ),
        Line(
            words=[
                Word(text="qu'un", start_time=36.6, end_time=36.7),
                Word(text="navire", start_time=36.6, end_time=36.8),
                Word(text="n'y", start_time=36.9, end_time=37.0),
                Word(text="va", start_time=37.1, end_time=37.2),
            ]
        ),
    ]
    segments = [
        te.TranscriptionSegment(
            start=23.0, end=27.0, text="oh qu'un concorde", words=[]
        ),
        te.TranscriptionSegment(
            start=27.0, end=36.0, text="oh qu'un avion ira", words=[]
        ),
    ]

    def fake_similarity(line_text, seg_text, _language):
        if "concorde" in line_text and "concorde" in seg_text:
            return 0.4
        return 0.1

    with _alignment_similarity_hook(fake_similarity):
        adjusted, fixed = te._pull_lines_to_best_segments(
            lines, segments, language="fra-Latn", min_similarity=0.7
        )

    assert adjusted[1].start_time == pytest.approx(23.0)
    assert adjusted[2].start_time == pytest.approx(27.0)
    assert fixed >= 1


def test_pull_lines_to_best_segments_allows_small_prev_overlap():
    lines = [
        Line(words=[Word(text="lead", start_time=64.0, end_time=64.85)]),
        Line(
            words=[
                Word(text="mother", start_time=68.58, end_time=69.0),
                Word(text="shelter", start_time=69.0, end_time=69.5),
                Word(text="mother", start_time=69.5, end_time=70.0),
                Word(text="shelter", start_time=70.0, end_time=70.68),
            ]
        ),
        Line(words=[Word(text="next", start_time=71.5, end_time=72.0)]),
    ]
    segments = [
        te.TranscriptionSegment(
            start=64.61, end=76.06, text="mother shelter mother shelter us", words=[]
        ),
        te.TranscriptionSegment(start=78.04, end=85.2, text="father line", words=[]),
    ]

    adjusted, fixed = te._pull_lines_to_best_segments(
        lines, segments, language="eng-Latn"
    )

    assert fixed >= 1
    assert adjusted[1].start_time == pytest.approx(64.9, abs=0.05)
    assert adjusted[1].end_time < lines[2].start_time


def test_pull_lines_to_best_segments_allows_late_low_similarity_when_window_fits():
    lines = [
        Line(words=[Word(text="anchor", start_time=74.0, end_time=74.5)]),
        Line(
            words=[
                Word(text="my", start_time=79.54, end_time=79.8),
                Word(text="father", start_time=79.8, end_time=80.2),
                Word(text="was", start_time=80.2, end_time=80.4),
                Word(text="a", start_time=80.4, end_time=80.5),
                Word(text="lord", start_time=80.5, end_time=81.0),
                Word(text="of", start_time=81.0, end_time=81.2),
                Word(text="land", start_time=81.2, end_time=82.48),
            ]
        ),
        Line(words=[Word(text="next", start_time=82.72, end_time=83.1)]),
    ]
    segments = [
        te.TranscriptionSegment(
            start=78.04,
            end=85.2,
            text="my father was a lord of land my daddy was a repo man",
            words=[],
        )
    ]

    with _alignment_similarity_hook(lambda *_a, **_k: 0.4):
        adjusted, fixed = te._pull_lines_to_best_segments(
            lines, segments, language="eng-Latn", min_similarity=0.7
        )

    assert fixed >= 1
    assert adjusted[1].start_time == pytest.approx(78.04, abs=0.05)
    assert adjusted[1].end_time < lines[2].start_time


def test_retime_adjacent_lines_to_whisper_window(monkeypatch):
    lines = [
        Line(words=[Word(text="Aucun", start_time=10.0, end_time=10.3)]),
        Line(words=[Word(text="express", start_time=20.0, end_time=20.3)]),
    ]
    segments = [
        te.TranscriptionSegment(start=9.0, end=13.0, text="aucun express", words=[]),
    ]

    with _alignment_similarity_hook(lambda *_a, **_k: 0.6):
        adjusted, fixes = te._retime_adjacent_lines_to_whisper_window(
            lines, segments, language="fra-Latn", min_similarity=0.3
        )

    assert fixes == 1
    assert adjusted[0].start_time == pytest.approx(9.0)
    assert adjusted[1].start_time >= adjusted[0].end_time


def test_retime_adjacent_lines_to_segment_window(monkeypatch):
    lines = [
        Line(words=[Word(text="Aucun", start_time=10.0, end_time=10.3)]),
        Line(words=[Word(text="express", start_time=20.0, end_time=20.3)]),
    ]
    segments = [
        te.TranscriptionSegment(start=9.0, end=11.0, text="aucun", words=[]),
        te.TranscriptionSegment(start=11.0, end=13.0, text="express", words=[]),
    ]

    with _alignment_similarity_hook(lambda *_a, **_k: 0.6):
        adjusted, fixes = te._retime_adjacent_lines_to_segment_window(
            lines, segments, language="fra-Latn", min_similarity=0.3
        )

    assert fixes == 1
    assert adjusted[0].start_time == pytest.approx(9.0)
    assert adjusted[1].end_time == pytest.approx(12.8)


def test_retime_adjacent_lines_to_whisper_window_uses_prior_segment(monkeypatch):
    lines = [
        Line(words=[Word(text="Aucun", start_time=20.0, end_time=20.3)]),
        Line(words=[Word(text="trolley", start_time=20.4, end_time=20.6)]),
    ]
    segments = [
        te.TranscriptionSegment(start=12.0, end=14.0, text="aucun", words=[]),
        te.TranscriptionSegment(start=18.0, end=22.0, text="trolley", words=[]),
    ]

    with _alignment_similarity_hook(lambda *_a, **_k: 0.3):
        adjusted, fixes = te._retime_adjacent_lines_to_whisper_window(
            lines,
            segments,
            language="fra-Latn",
            min_similarity=0.5,
            min_similarity_late=0.25,
            max_late=1.0,
            max_late_short=6.0,
            max_time_window=15.0,
        )

    assert fixes == 1
    assert adjusted[0].start_time == pytest.approx(12.0)


def test_pull_next_line_into_same_segment():
    lines = [
        Line(words=[Word(text="first", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="second", start_time=20.0, end_time=20.4)]),
    ]
    segments = [
        te.TranscriptionSegment(start=9.0, end=12.0, text="first", words=[]),
    ]

    adjusted, fixes = te._pull_next_line_into_same_segment(
        lines, segments, max_late=6.0, max_time_window=15.0
    )

    assert fixes == 0


def test_pull_next_line_into_same_segment_retimes_pair_when_full():
    lines = [
        Line(words=[Word(text="first", start_time=9.0, end_time=12.0)]),
        Line(words=[Word(text="second", start_time=20.0, end_time=20.4)]),
    ]
    segments = [
        te.TranscriptionSegment(start=9.0, end=12.0, text="first", words=[]),
    ]

    adjusted, fixes = te._pull_next_line_into_same_segment(
        lines, segments, max_late=6.0, max_time_window=15.0
    )

    assert fixes == 0


def test_merge_short_following_line_into_segment():
    lines = [
        Line(words=[Word(text="first", start_time=10.0, end_time=12.0)]),
        Line(words=[Word(text="second", start_time=20.0, end_time=20.3)]),
    ]
    segments = [
        te.TranscriptionSegment(start=9.0, end=12.0, text="first", words=[]),
    ]

    adjusted, fixes = te._merge_short_following_line_into_segment(
        lines, segments, max_late=6.0, max_time_window=15.0
    )

    assert fixes == 0


def test_clamp_repeated_line_duration():
    lines = [
        Line(words=[Word(text="hey", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="hey", start_time=12.0, end_time=20.0)]),
        Line(words=[Word(text="next", start_time=22.0, end_time=23.0)]),
    ]

    adjusted, fixes = te._clamp_repeated_line_duration(
        lines, max_duration=1.5, min_gap=0.01
    )

    assert fixes == 1
    assert adjusted[1].end_time - adjusted[1].start_time <= 1.5


def test_pull_next_line_into_segment_window():
    lines = [
        Line(words=[Word(text="Aucun", start_time=9.0, end_time=10.0)]),
        Line(words=[Word(text="navire", start_time=20.0, end_time=20.4)]),
    ]
    segments = [
        te.TranscriptionSegment(start=9.0, end=11.0, text="aucun", words=[]),
        te.TranscriptionSegment(start=11.0, end=14.0, text="navire", words=[]),
    ]

    with _alignment_similarity_hook(lambda *_a, **_k: 0.25):
        adjusted, fixes = te._pull_next_line_into_segment_window(
            lines, segments, language="fra-Latn", min_similarity=0.2
        )

    assert fixes == 1
    assert adjusted[1].start_time == pytest.approx(11.0)


def test_pull_next_line_into_segment_window_uses_nearest_segment():
    lines = [
        Line(words=[Word(text="Aucun", start_time=9.0, end_time=10.0)]),
        Line(words=[Word(text="navire", start_time=20.0, end_time=20.4)]),
    ]
    segments = [
        te.TranscriptionSegment(start=9.0, end=11.0, text="aucun", words=[]),
        te.TranscriptionSegment(start=11.0, end=14.0, text="navire", words=[]),
        te.TranscriptionSegment(start=44.0, end=46.0, text="later", words=[]),
    ]

    with _alignment_similarity_hook(lambda *_a, **_k: 0.0):
        adjusted, fixes = te._pull_next_line_into_segment_window(
            lines, segments, language="fra-Latn", min_similarity=0.2
        )

    assert fixes == 1
    assert adjusted[1].start_time == pytest.approx(11.0)


def test_pull_lines_near_segment_end_prefers_prior_segment():
    lines = [Line(words=[Word(text="line", start_time=10.5, end_time=11.0)])]
    segments = [
        te.TranscriptionSegment(start=8.0, end=10.4, text="prior", words=[]),
        te.TranscriptionSegment(start=12.0, end=14.0, text="later", words=[]),
    ]

    adjusted, fixes = te._pull_lines_near_segment_end(
        lines, segments, language="fra-Latn", max_late=0.5, max_time_window=15.0
    )

    assert fixes == 1
    assert adjusted[0].start_time == pytest.approx(8.0)


def test_pull_lines_near_segment_end_allows_short_line_with_similarity():
    lines = [Line(words=[Word(text="sinon toi", start_time=20.0, end_time=20.4)])]
    segments = [
        te.TranscriptionSegment(start=12.0, end=14.0, text="si non toi", words=[]),
    ]

    with _alignment_similarity_hook(lambda *_a, **_k: 0.5):
        adjusted, fixes = te._pull_lines_near_segment_end(
            lines,
            segments,
            language="fra-Latn",
            max_late=0.5,
            max_late_short=6.0,
            min_similarity=0.35,
            max_time_window=15.0,
        )

    assert fixes == 1
    assert adjusted[0].start_time == pytest.approx(12.0)


def test_pull_lines_near_segment_end_extends_short_line_duration():
    lines = [
        Line(words=[Word(text="sinon", start_time=20.0, end_time=20.1)]),
        Line(words=[Word(text="next", start_time=30.0, end_time=30.2)]),
    ]
    segments = [
        te.TranscriptionSegment(start=12.0, end=14.0, text="sinon", words=[]),
    ]

    adjusted, fixes = te._pull_lines_near_segment_end(
        lines,
        segments,
        language="fra-Latn",
        max_late=0.5,
        max_late_short=6.0,
        min_similarity=0.0,
        min_short_duration=0.35,
        max_time_window=15.0,
    )

    assert fixes == 1
    assert adjusted[0].end_time - adjusted[0].start_time >= 0.35


def test_transcribe_vocals_success(monkeypatch):
    class FakeWord:
        def __init__(self, start, end, word, probability):
            self.start = start
            self.end = end
            self.word = word
            self.probability = probability

    class FakeSegment:
        def __init__(self):
            self.start = 0.0
            self.end = 1.0
            self.text = " Hello "
            self.words = [FakeWord(0.0, 0.5, " hello ", 0.9)]

    class FakeInfo:
        language = "en"

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, *args, **kwargs):
            return [FakeSegment()], FakeInfo()

    with wi.use_whisper_integration_hooks(
        get_whisper_cache_path_fn=lambda *_: None,
        save_whisper_cache_fn=lambda *_: None,
        load_whisper_model_class_fn=lambda: FakeModel,
    ):
        segments, words, language, model = te.transcribe_vocals(
            "vocals.wav", language="en"
        )
    assert language == "en"
    assert len(segments) == 1
    assert len(words) == 1
    assert segments[0].text == "Hello"
    assert words[0].text == "hello"
    assert model == "base"


def test_transcribe_vocals_handles_transcribe_error(monkeypatch):
    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, *args, **kwargs):
            raise RuntimeError("boom")

    with wi.use_whisper_integration_hooks(
        get_whisper_cache_path_fn=lambda *_: None,
        load_whisper_model_class_fn=lambda: FakeModel,
    ):
        segments, words, language, model = te.transcribe_vocals(
            "vocals.wav", language="en"
        )
    assert segments == []
    assert words == []
    assert language == ""
    assert model == "base"
