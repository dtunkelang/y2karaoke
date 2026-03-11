from typing import Any, cast
from y2karaoke.core.models import Line, Word
import y2karaoke.core.components.whisper.whisper_integration as wi
from y2karaoke.core.components.whisper import whisper_integration_align as wialign
from y2karaoke.core.components.whisper import whisper_integration_correct as wicorrect
from y2karaoke.core.components.alignment.timing_models import (
    TranscriptionWord,
    TranscriptionSegment,
)

wi_any = cast(Any, wi)


def test_whisper_lang_to_epitran():
    assert wi_any._whisper_lang_to_epitran("en") == "eng-Latn"
    assert wi_any._whisper_lang_to_epitran("fr") == "fra-Latn"
    assert wi_any._whisper_lang_to_epitran("zh") == "cmn-Hans"
    assert wi_any._whisper_lang_to_epitran("unknown") == "eng-Latn"


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


def test_trim_whisper_transcription_skips_when_match_far_from_tail():
    segments = [
        TranscriptionSegment(start=0, end=5, text="intro", words=[]),
        TranscriptionSegment(start=10, end=15, text="target line", words=[]),
        TranscriptionSegment(start=200, end=205, text="very late words", words=[]),
    ]
    words = [
        TranscriptionWord(text="intro", start=0, end=5, probability=1.0),
        TranscriptionWord(text="target", start=10, end=12, probability=1.0),
        TranscriptionWord(text="line", start=12, end=15, probability=1.0),
        TranscriptionWord(text="late", start=200, end=205, probability=1.0),
    ]
    line_texts = ["target line"]

    trimmed_segs, trimmed_words, end_time = wi._trim_whisper_transcription_by_lyrics(
        segments, words, line_texts
    )

    assert end_time is None
    assert len(trimmed_segs) == len(segments)
    assert len(trimmed_words) == len(words)


def test_should_ignore_trimmed_transcript_when_it_cuts_lyric_tail():
    segments = [
        TranscriptionSegment(start=0, end=20, text="early", words=[]),
        TranscriptionSegment(start=180, end=198, text="late", words=[]),
    ]
    lines = [
        Line(words=[Word(text="a", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="b", start_time=195.0, end_time=196.0)]),
    ]

    assert wialign._should_ignore_trimmed_transcript(
        trimmed_end=184.44,
        original_transcription=segments,
        lines=lines,
    )


def test_should_ignore_trimmed_transcript_keeps_normal_tail_trim():
    segments = [
        TranscriptionSegment(start=0, end=20, text="early", words=[]),
        TranscriptionSegment(start=180, end=190, text="late", words=[]),
    ]
    lines = [
        Line(words=[Word(text="a", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="b", start_time=185.0, end_time=186.0)]),
    ]

    assert not wialign._should_ignore_trimmed_transcript(
        trimmed_end=184.44,
        original_transcription=segments,
        lines=lines,
    )


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
    all_words = (segments[0].words or []) + (segments[1].words or [])
    # wi._build_word_to_segment_index(all_words, segments)
    idx_map = wi_any._build_word_to_segment_index(all_words, segments)
    assert idx_map[0] == 0
    assert idx_map[1] == 0
    assert idx_map[2] == 1


def test_find_segment_for_time():
    segments = [
        TranscriptionSegment(start=0, end=10, text="s1", words=[]),
        TranscriptionSegment(start=15, end=25, text="s2", words=[]),
    ]
    # wi._find_segment_for_time(time, segments)
    assert wi_any._find_segment_for_time(5.0, segments) == 0
    assert wi_any._find_segment_for_time(20.0, segments) == 1


def test_whisper_profile_configs(monkeypatch):
    monkeypatch.delenv("Y2K_WHISPER_PROFILE", raising=False)
    default_map = wialign._default_mapping_decision_config()
    default_correct = wicorrect._default_correction_config()

    monkeypatch.setenv("Y2K_WHISPER_PROFILE", "safe")
    safe_map = wialign._default_mapping_decision_config()
    safe_correct = wicorrect._default_correction_config()

    monkeypatch.setenv("Y2K_WHISPER_PROFILE", "aggressive")
    aggr_map = wialign._default_mapping_decision_config()
    aggr_correct = wicorrect._default_correction_config()

    assert safe_map.sparse_word_threshold > default_map.sparse_word_threshold
    assert aggr_map.sparse_word_threshold < default_map.sparse_word_threshold
    assert safe_correct.quality_good_threshold > default_correct.quality_good_threshold
    assert aggr_correct.quality_good_threshold < default_correct.quality_good_threshold
