import y2karaoke.core.components.alignment.timing_evaluator as te
import y2karaoke.core.phonetic_utils as pu
import y2karaoke.core.components.whisper.whisper_integration as wi
from y2karaoke.core.models import Line, Word


def test_align_lyrics_to_transcription_adjusts_times():
    lines = [
        Line(
            words=[
                Word(text="hello", start_time=0.0, end_time=0.5),
                Word(text="world", start_time=0.5, end_time=1.0),
            ]
        )
    ]
    transcription = [
        te.TranscriptionSegment(start=5.0, end=7.0, text="hello world", words=[])
    ]

    aligned, notes = te.align_lyrics_to_transcription(
        lines,
        transcription,
        min_similarity=0.1,
        max_time_shift=10.0,
        language="eng-Latn",
    )
    assert aligned[0].words[0].start_time == 5.0
    assert aligned[0].words[1].start_time == 6.0
    assert notes


def test_align_lyrics_to_transcription_skips_when_out_of_range():
    lines = [Line(words=[Word(text="hello", start_time=0.0, end_time=0.5)])]
    transcription = [
        te.TranscriptionSegment(start=20.0, end=21.0, text="hello", words=[])
    ]
    aligned, notes = te.align_lyrics_to_transcription(
        lines,
        transcription,
        min_similarity=0.1,
        max_time_shift=5.0,
        language="eng-Latn",
    )
    assert aligned[0].words[0].start_time == 0.0
    assert notes == []


def test_transcribe_vocals_returns_empty_without_dependency(monkeypatch):
    monkeypatch.setattr(
        wi,
        "_load_whisper_model_class",
        lambda: (_ for _ in ()).throw(ImportError("missing")),
    )
    segments, words, language, model = te.transcribe_vocals("vocals.wav")
    assert segments == []
    assert words == []
    assert language == ""
    assert model == "base"


def test_text_similarity_phonetic_falls_back():
    with pu.use_phonetic_utils_hooks(get_panphon_distance_fn=lambda: None):
        assert pu._text_similarity("hello", "hello", use_phonetic=True) == 1.0


def test_get_ipa_returns_none_without_epitran():
    with pu.use_phonetic_utils_hooks(get_epitran_fn=lambda _lang="fra-Latn": None):
        assert pu._get_ipa("bonjour", language="fra-Latn") is None
