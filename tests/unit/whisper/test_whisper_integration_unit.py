import pytest
from y2karaoke.core.models import Line, Word
import y2karaoke.core.components.whisper.whisper_integration as wi
from y2karaoke.core.components.alignment.timing_models import (
    TranscriptionWord,
    TranscriptionSegment,
)


def test_whisper_lang_to_epitran():
    assert wi._whisper_lang_to_epitran("en") == "eng-Latn"
    assert wi._whisper_lang_to_epitran("fr") == "fra-Latn"
    assert wi._whisper_lang_to_epitran("zh") == "cmn-Hans"
    assert wi._whisper_lang_to_epitran("unknown") == "eng-Latn"


def test_assess_lrc_quality():
    lines = [
        Line(words=[Word(text="hello", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="world", start_time=20.0, end_time=21.0)]),
    ]
    whisper_words = [
        TranscriptionWord(text="hello", start=10.1, end=11.1, probability=1.0),
        TranscriptionWord(text="world", start=25.0, end=26.0, probability=1.0),
    ]
    # Tolerance 1.5s. 10.0 vs 10.1 is fine. 20.0 vs 25.0 is too far.
    quality, assessments = wi._assess_lrc_quality(
        lines, whisper_words, "eng-Latn", tolerance=1.5
    )
    assert quality == 0.5
    # assessments is List[Tuple[int, float, float]] (line_idx, lrc_time, best_whisper_time)
    assert assessments[0][0] == 0
    assert assessments[1][0] == 1


def test_trim_whisper_transcription_by_lyrics():
    segments = [
        TranscriptionSegment(start=0, end=5, text="intro", words=[]),
        TranscriptionSegment(start=10, end=15, text="first line", words=[]),
        TranscriptionSegment(start=20, end=25, text="outro", words=[]),
    ]
    words = [
        TranscriptionWord(text="intro", start=0, end=5, probability=1.0),
        TranscriptionWord(text="first", start=10, end=12, probability=1.0),
        TranscriptionWord(text="line", start=12, end=15, probability=1.0),
        TranscriptionWord(text="outro", start=20, end=25, probability=1.0),
    ]
    line_texts = ["first line"]

    trimmed_segs, trimmed_words, end_time = wi._trim_whisper_transcription_by_lyrics(
        segments, words, line_texts
    )

    # It should keep segments up to the one matching "first line" + buffer
    # And potentially some after.
    assert end_time > 0
    assert len(trimmed_segs) <= len(segments)


def test_build_word_to_segment_index():
    segments = [
        TranscriptionSegment(
            start=0,
            end=10,
            text="s1",
            words=[
                TranscriptionWord(text="w1", start=1, end=2, probability=1.0),
                TranscriptionWord(text="w2", start=3, end=4, probability=1.0),
            ],
        ),
        TranscriptionSegment(
            start=10,
            end=20,
            text="s2",
            words=[
                TranscriptionWord(text="w3", start=11, end=12, probability=1.0),
            ],
        ),
    ]
    all_words = segments[0].words + segments[1].words
    # wi._build_word_to_segment_index(all_words, segments)
    idx_map = wi._build_word_to_segment_index(all_words, segments)
    assert idx_map[0] == 0
    assert idx_map[1] == 0
    assert idx_map[2] == 1


def test_find_segment_for_time():
    segments = [
        TranscriptionSegment(start=0, end=10, text="s1", words=[]),
        TranscriptionSegment(start=15, end=25, text="s2", words=[]),
    ]
    # wi._find_segment_for_time(time, segments)
    assert wi._find_segment_for_time(5.0, segments) == 0
    assert wi._find_segment_for_time(20.0, segments) == 1
