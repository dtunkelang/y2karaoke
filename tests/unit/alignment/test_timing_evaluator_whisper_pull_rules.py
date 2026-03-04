import pytest

import y2karaoke.core.components.alignment.timing_evaluator as te
import y2karaoke.core.components.whisper.whisper_integration as wi
import y2karaoke.core.components.whisper.whisper_integration_transcribe as wit
import y2karaoke.core.components.whisper.whisper_alignment as wa
from y2karaoke.core.models import Line, Word


def _alignment_similarity_hook(fn):
    return wa.use_whisper_alignment_hooks(phonetic_similarity_fn=fn)


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
        adjusted, fixed = wi._pull_lines_to_best_segments(
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

    adjusted, fixed = wi._pull_lines_to_best_segments(
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
        adjusted, fixed = wi._pull_lines_to_best_segments(
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
        adjusted, fixes = wi._retime_adjacent_lines_to_whisper_window(
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
        adjusted, fixes = wi._retime_adjacent_lines_to_segment_window(
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
        adjusted, fixes = wi._retime_adjacent_lines_to_whisper_window(
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

    adjusted, fixes = wi._pull_next_line_into_same_segment(
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

    adjusted, fixes = wi._pull_next_line_into_same_segment(
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

    adjusted, fixes = wi._merge_short_following_line_into_segment(
        lines, segments, max_late=6.0, max_time_window=15.0
    )

    assert fixes == 0


def test_clamp_repeated_line_duration():
    lines = [
        Line(words=[Word(text="hey", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="hey", start_time=12.0, end_time=20.0)]),
        Line(words=[Word(text="next", start_time=22.0, end_time=23.0)]),
    ]

    adjusted, fixes = wi._clamp_repeated_line_duration(
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
        adjusted, fixes = wi._pull_next_line_into_segment_window(
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
        adjusted, fixes = wi._pull_next_line_into_segment_window(
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

    adjusted, fixes = wi._pull_lines_near_segment_end(
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
        adjusted, fixes = wi._pull_lines_near_segment_end(
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

    adjusted, fixes = wi._pull_lines_near_segment_end(
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


def test_transcribe_vocals_cached_sparse_upgrade_persists_cache(monkeypatch):
    sparse_segments = [te.TranscriptionSegment(start=0.0, end=1.0, text="a", words=[])]
    sparse_words = [te.TranscriptionWord(start=0.1, end=0.2, text="a", probability=0.9)]
    upgraded_segments = [
        te.TranscriptionSegment(
            start=0.0,
            end=2.0,
            text="upgraded",
            words=[
                te.TranscriptionWord(start=0.0, end=0.5, text="up", probability=0.9)
            ],
        )
    ]
    upgraded_words = [
        te.TranscriptionWord(start=0.0, end=0.5, text="up", probability=0.9)
    ]
    saved = {"count": 0, "words": 0}

    monkeypatch.setattr(
        wit,
        "_maybe_upgrade_sparse_transcription_with_whisperx",
        lambda **_k: (upgraded_segments, upgraded_words, "en", True),
    )
    monkeypatch.setattr(
        wi,
        "_find_best_cached_whisper_model",
        lambda *_: ("fake_cache.json", "base"),
    )

    with wi.use_whisper_integration_hooks(
        get_whisper_cache_path_fn=lambda *_: "fake_cache.json",
        load_whisper_cache_fn=lambda *_: (sparse_segments, sparse_words, "en"),
        save_whisper_cache_fn=lambda _path, _segs, words, *_: saved.update(
            {"count": saved["count"] + 1, "words": len(words)}
        ),
        load_whisper_model_class_fn=lambda: None,
    ):
        segments, words, language, model = te.transcribe_vocals(
            "vocals.wav", language="en"
        )

    assert len(segments) == len(upgraded_segments)
    assert len(words) == len(upgraded_words)
    assert language == "en"
    assert model == "base"
    assert saved["count"] == 1
    assert saved["words"] == len(upgraded_words)
