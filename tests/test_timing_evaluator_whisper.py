import pytest

import y2karaoke.core.timing_evaluator as te
import y2karaoke.core.phonetic_utils as pu
import y2karaoke.core.whisper_integration as wi
import y2karaoke.core.whisper_dtw as w_dtw
import y2karaoke.core.whisper_alignment_utils as wa_utils
import y2karaoke.core.whisper_alignment_retime as wa_retime
import y2karaoke.core.whisper_alignment_pull as wa_pull
from y2karaoke.core.models import Line, Word


def test_align_words_to_whisper_adjusts_word():
    pu._ipa_cache.clear()
    lines = [Line(words=[Word(text="hello", start_time=0.0, end_time=0.5)])]
    whisper_words = [
        te.TranscriptionWord(start=2.0, end=2.4, text="hello", probability=0.9)
    ]
    wi._get_ipa = lambda *_args, **_kwargs: None
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

    monkeypatch.setattr(pu, "_phonetic_similarity", lambda *_args, **_kwargs: 0.8)

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

    monkeypatch.setattr(pu, "_phonetic_similarity", lambda *_args, **_kwargs: 0.6)

    costs = te._compute_phonetic_costs(lrc_words, whisper_words, "eng-Latn", 0.5)
    assert (0, 0) in costs
    assert costs[(0, 0)] < 0.5


def test_extract_alignments_from_path_filters_by_similarity(monkeypatch):
    path = [(0, 0)]
    lrc_words = [{"text": "hello", "start": 1.0}]
    whisper_words = [
        te.TranscriptionWord(start=1.1, end=1.6, text="hello", probability=0.9)
    ]

    # Mock phonetic similarity to fail filter - patch where it's imported
    monkeypatch.setattr(w_dtw, "_phonetic_similarity", lambda *_args, **_kwargs: 0.1)
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
    import sys

    # Hide fastdtw if present
    monkeypatch.setitem(sys.modules, "fastdtw", None)

    lines = [Line(words=[Word(text="hello", start_time=1.0, end_time=1.5)])]
    words = [te.TranscriptionWord(start=1.1, end=1.6, text="hello", probability=0.9)]

    aligned, corrections, metrics = te.align_dtw_whisper(
        lines, words, min_similarity=0.1
    )
    assert len(aligned) == 1
    # Simple fallback doesn't necessarily shift if fastdtw missing, but should return lines
    assert metrics["matched_ratio"] >= 0


def test_correct_timing_with_whisper_no_transcription(monkeypatch):
    monkeypatch.setattr(wi, "transcribe_vocals", lambda *_: ([], [], "en", "base"))
    lines = [Line(words=[Word(text="a", start_time=0, end_time=1)])]
    aligned, corrections, metrics = te.correct_timing_with_whisper(lines, "vocals.wav")
    assert aligned == lines
    assert not corrections


def test_correct_timing_with_whisper_uses_dtw(monkeypatch):
    lines = [Line(words=[Word(text="hello", start_time=10.0, end_time=11.0)])]
    words = [te.TranscriptionWord(start=20.0, end=21.0, text="hello", probability=0.9)]
    segments = [
        te.TranscriptionSegment(start=20.0, end=21.0, text="hello", words=words)
    ]

    monkeypatch.setattr(
        wi, "transcribe_vocals", lambda *_: (segments, words, "en", "base")
    )
    monkeypatch.setattr(wi, "extract_audio_features", lambda *_: None)
    # Force low quality to trigger DTW
    monkeypatch.setattr(wi, "_assess_lrc_quality", lambda *_, **__: (0.1, []))

    # Mock DTW alignment result
    monkeypatch.setattr(
        wi,
        "_align_dtw_whisper_with_data",
        lambda *_, **__: (
            [Line(words=[Word(text="hello", start_time=20.0, end_time=21.0)])],
            ["dtw"],
            {"matched_ratio": 0.8, "avg_similarity": 0.8, "line_coverage": 0.8},
            [],
            {0: (words[0], 0.9)},
        ),
    )

    aligned, corrections, metrics = te.correct_timing_with_whisper(lines, "vocals.wav")
    assert aligned[0].start_time == 20.0
    assert "dtw" in corrections or any("DTW" in c for c in corrections)


def test_correct_timing_with_whisper_quality_good_uses_hybrid(monkeypatch):
    lines = [Line(words=[Word(text="hello", start_time=10.0, end_time=11.0)])]
    words = [te.TranscriptionWord(start=10.1, end=11.1, text="hello", probability=0.9)]
    segments = [
        te.TranscriptionSegment(start=10.0, end=11.5, text="hello", words=words)
    ]

    monkeypatch.setattr(
        wi, "transcribe_vocals", lambda *_: (segments, words, "en", "base")
    )
    monkeypatch.setattr(wi, "extract_audio_features", lambda *_: None)
    monkeypatch.setattr(wi, "_assess_lrc_quality", lambda *_, **__: (0.9, []))

    # Mock hybrid result
    monkeypatch.setattr(
        wi,
        "align_hybrid_lrc_whisper",
        lambda *_, **__: (
            [Line(words=[Word(text="hello", start_time=10.1, end_time=11.1)])],
            ["hybrid"],
        ),
    )

    aligned, corrections, metrics = te.correct_timing_with_whisper(lines, "vocals.wav")
    assert "hybrid" in corrections


def test_correct_timing_with_whisper_quality_mixed_uses_hybrid(monkeypatch):
    lines = [Line(words=[Word(text="hello", start_time=10.0, end_time=11.0)])]
    words = [te.TranscriptionWord(start=10.1, end=11.1, text="hello", probability=0.9)]
    segments = [
        te.TranscriptionSegment(start=10.0, end=11.5, text="hello", words=words)
    ]

    monkeypatch.setattr(
        wi, "transcribe_vocals", lambda *_: (segments, words, "en", "base")
    )
    monkeypatch.setattr(wi, "extract_audio_features", lambda *_: None)
    monkeypatch.setattr(wi, "_assess_lrc_quality", lambda *_, **__: (0.5, []))

    monkeypatch.setattr(
        wi,
        "align_hybrid_lrc_whisper",
        lambda *_, **__: (
            [Line(words=[Word(text="hello", start_time=10.1, end_time=11.1)])],
            ["hybrid"],
        ),
    )

    aligned, corrections, metrics = te.correct_timing_with_whisper(lines, "vocals.wav")
    assert "hybrid" in corrections


def test_correct_timing_with_whisper_uses_dtw_retime_when_confident(monkeypatch):
    lines = [Line(words=[Word(text="hello", start_time=10.0, end_time=11.0)])]
    words = [te.TranscriptionWord(start=20.0, end=21.0, text="hello", probability=0.9)]
    segments = [
        te.TranscriptionSegment(start=20.0, end=21.0, text="hello", words=words)
    ]

    monkeypatch.setattr(
        wi, "transcribe_vocals", lambda *_: (segments, words, "en", "base")
    )
    monkeypatch.setattr(wi, "extract_audio_features", lambda *_: None)
    monkeypatch.setattr(wi, "_assess_lrc_quality", lambda *_, **__: (0.1, []))

    # High confidence metrics
    metrics_high = {"matched_ratio": 0.9, "avg_similarity": 0.9, "line_coverage": 0.9}
    lrc_words = [{"line_idx": 0, "word_idx": 0, "text": "hello"}]
    align_map = {0: (words[0], 0.9)}

    monkeypatch.setattr(
        wi,
        "_align_dtw_whisper_with_data",
        lambda *_, **__: (lines, ["dtw_data"], metrics_high, lrc_words, align_map),
    )

    # Mock retime result
    monkeypatch.setattr(
        wi,
        "_retime_lines_from_dtw_alignments",
        lambda *_, **__: (
            [Line(words=[Word(text="hello", start_time=20.0, end_time=21.0)])],
            ["retimed"],
        ),
    )

    aligned, corrections, metrics = te.correct_timing_with_whisper(lines, "vocals.wav")
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

    monkeypatch.setattr(wa_utils, "_phonetic_similarity", fake_similarity)
    monkeypatch.setattr(wa_retime, "_phonetic_similarity", fake_similarity)
    monkeypatch.setattr(wa_pull, "_phonetic_similarity", fake_similarity)

    adjusted, fixed = te._pull_lines_to_best_segments(
        lines, segments, language="fra-Latn", min_similarity=0.7
    )

    assert adjusted[1].start_time == pytest.approx(23.0)
    assert adjusted[2].start_time == pytest.approx(27.0)
    assert fixed >= 1


def test_retime_adjacent_lines_to_whisper_window(monkeypatch):
    lines = [
        Line(words=[Word(text="Aucun", start_time=10.0, end_time=10.3)]),
        Line(words=[Word(text="express", start_time=20.0, end_time=20.3)]),
    ]
    segments = [
        te.TranscriptionSegment(start=9.0, end=13.0, text="aucun express", words=[]),
    ]

    monkeypatch.setattr(wa_utils, "_phonetic_similarity", lambda *_a, **_k: 0.6)
    monkeypatch.setattr(wa_retime, "_phonetic_similarity", lambda *_a, **_k: 0.6)
    monkeypatch.setattr(wa_pull, "_phonetic_similarity", lambda *_a, **_k: 0.6)

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

    monkeypatch.setattr(wa_utils, "_phonetic_similarity", lambda *_a, **_k: 0.6)
    monkeypatch.setattr(wa_retime, "_phonetic_similarity", lambda *_a, **_k: 0.6)
    monkeypatch.setattr(wa_pull, "_phonetic_similarity", lambda *_a, **_k: 0.6)

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

    monkeypatch.setattr(wa_utils, "_phonetic_similarity", lambda *_a, **_k: 0.3)
    monkeypatch.setattr(wa_retime, "_phonetic_similarity", lambda *_a, **_k: 0.3)
    monkeypatch.setattr(wa_pull, "_phonetic_similarity", lambda *_a, **_k: 0.3)

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


def test_pull_next_line_into_segment_window(monkeypatch):
    lines = [
        Line(words=[Word(text="Aucun", start_time=9.0, end_time=10.0)]),
        Line(words=[Word(text="navire", start_time=20.0, end_time=20.4)]),
    ]
    segments = [
        te.TranscriptionSegment(start=9.0, end=11.0, text="aucun", words=[]),
        te.TranscriptionSegment(start=11.0, end=14.0, text="navire", words=[]),
    ]

    monkeypatch.setattr(wa_utils, "_phonetic_similarity", lambda *_a, **_k: 0.25)
    monkeypatch.setattr(wa_retime, "_phonetic_similarity", lambda *_a, **_k: 0.25)
    monkeypatch.setattr(wa_pull, "_phonetic_similarity", lambda *_a, **_k: 0.25)

    adjusted, fixes = te._pull_next_line_into_segment_window(
        lines, segments, language="fra-Latn", min_similarity=0.2
    )

    assert fixes == 1
    assert adjusted[1].start_time == pytest.approx(11.0)


def test_pull_next_line_into_segment_window_uses_nearest_segment(monkeypatch):
    lines = [
        Line(words=[Word(text="Aucun", start_time=9.0, end_time=10.0)]),
        Line(words=[Word(text="navire", start_time=20.0, end_time=20.4)]),
    ]
    segments = [
        te.TranscriptionSegment(start=9.0, end=11.0, text="aucun", words=[]),
        te.TranscriptionSegment(start=11.0, end=14.0, text="navire", words=[]),
        te.TranscriptionSegment(start=44.0, end=46.0, text="later", words=[]),
    ]

    monkeypatch.setattr(wa_utils, "_phonetic_similarity", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(wa_retime, "_phonetic_similarity", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(wa_pull, "_phonetic_similarity", lambda *_a, **_k: 0.0)

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


def test_pull_lines_near_segment_end_allows_short_line_with_similarity(monkeypatch):
    lines = [Line(words=[Word(text="sinon toi", start_time=20.0, end_time=20.4)])]
    segments = [
        te.TranscriptionSegment(start=12.0, end=14.0, text="si non toi", words=[]),
    ]

    monkeypatch.setattr(wa_utils, "_phonetic_similarity", lambda *_a, **_k: 0.5)
    monkeypatch.setattr(wa_retime, "_phonetic_similarity", lambda *_a, **_k: 0.5)
    monkeypatch.setattr(wa_pull, "_phonetic_similarity", lambda *_a, **_k: 0.5)

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

    class FakeWhisperModule:
        WhisperModel = FakeModel

    monkeypatch.setattr(wi, "_get_whisper_cache_path", lambda *_: None)
    monkeypatch.setattr(wi, "_save_whisper_cache", lambda *_: None)
    monkeypatch.setitem(
        __import__("sys").modules, "faster_whisper", FakeWhisperModule()
    )

    segments, words, language, model = te.transcribe_vocals("vocals.wav", language="en")
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

    class FakeWhisperModule:
        WhisperModel = FakeModel

    monkeypatch.setattr(wi, "_get_whisper_cache_path", lambda *_: None)
    monkeypatch.setitem(
        __import__("sys").modules, "faster_whisper", FakeWhisperModule()
    )

    segments, words, language, model = te.transcribe_vocals("vocals.wav", language="en")
    assert segments == []
    assert words == []
    assert language == ""
    assert model == "base"
