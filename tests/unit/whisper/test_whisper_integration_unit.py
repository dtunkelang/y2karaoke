from typing import Any, cast
from y2karaoke.core.models import Line, Word
import y2karaoke.core.components.whisper.whisper_integration as wi
from y2karaoke.core.components.whisper import whisper_integration_align as wialign
from y2karaoke.core.components.whisper import whisper_integration_correct as wicorrect
from y2karaoke.core.components.whisper.whisper_runtime_config import (
    WhisperRuntimeConfig,
)
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


def test_trim_whisper_transcription_skips_ambiguous_repetitive_tail_match():
    segments = [
        TranscriptionSegment(start=0, end=2, text="guess whos back", words=[]),
        TranscriptionSegment(start=2, end=4, text="back again", words=[]),
        TranscriptionSegment(start=4, end=6, text="guess whos back", words=[]),
        TranscriptionSegment(start=6, end=9, text="extra late hook words", words=[]),
    ]
    words = [
        TranscriptionWord(text="guess", start=0, end=0.5, probability=1.0),
        TranscriptionWord(text="back", start=4, end=4.5, probability=1.0),
        TranscriptionWord(text="late", start=8, end=8.5, probability=1.0),
    ]
    line_texts = [
        "Guess who's back",
        "Back again",
        "Guess who's back",
        "Tell a friend",
    ]

    trimmed_segs, trimmed_words, end_time = wi._trim_whisper_transcription_by_lyrics(
        segments, words, line_texts
    )

    assert end_time is None
    assert len(trimmed_segs) == len(segments)
    assert len(trimmed_words) == len(words)


def test_trim_whisper_transcription_skips_ambiguous_repetitive_tail_match_with_longer_repeated_lines():
    segments = [
        TranscriptionSegment(
            start=0, end=3, text="guess whos back back again", words=[]
        ),
        TranscriptionSegment(
            start=3, end=6, text="shadys back tell a friend", words=[]
        ),
        TranscriptionSegment(start=6, end=9.3, text="guess whos back", words=[]),
        TranscriptionSegment(
            start=9.3, end=14.8, text="crowd noise and more vocals", words=[]
        ),
    ]
    words = [
        TranscriptionWord(text="guess", start=0, end=0.5, probability=1.0),
        TranscriptionWord(text="friend", start=5.5, end=6.0, probability=1.0),
        TranscriptionWord(text="guess", start=6.2, end=6.7, probability=1.0),
        TranscriptionWord(text="vocals", start=13.5, end=14.0, probability=1.0),
    ]
    line_texts = [
        "Guess who's back, back again?",
        "Shady's back, tell a friend",
        "Guess who's back? Guess who's back?",
        "Guess who's back? Guess who's back?",
        "Guess who's back? Guess who's back?",
        "Guess who's back?",
    ]

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


def test_correction_should_ignore_trimmed_transcript_when_it_cuts_lyric_tail():
    segments = [
        TranscriptionSegment(start=0, end=20, text="early", words=[]),
        TranscriptionSegment(start=180, end=198, text="late", words=[]),
    ]
    lines = [
        Line(words=[Word(text="a", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="b", start_time=195.0, end_time=196.0)]),
    ]

    assert wicorrect._should_ignore_trimmed_transcript(
        trimmed_end=184.44,
        original_transcription=segments,
        lines=lines,
    )


def test_correction_should_ignore_trimmed_transcript_keeps_normal_tail_trim():
    segments = [
        TranscriptionSegment(start=0, end=20, text="early", words=[]),
        TranscriptionSegment(start=180, end=190, text="late", words=[]),
    ]
    lines = [
        Line(words=[Word(text="a", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="b", start_time=185.0, end_time=186.0)]),
    ]

    assert not wicorrect._should_ignore_trimmed_transcript(
        trimmed_end=184.44,
        original_transcription=segments,
        lines=lines,
    )


def test_should_force_whisperx_for_tail_shortfall():
    words = [
        TranscriptionWord(
            text=f"w{i}",
            start=100.0 + i * 0.7,
            end=100.2 + i * 0.7,
            probability=1.0,
        )
        for i in range(72)
    ] + [
        TranscriptionWord(
            text=f"tail{i}",
            start=183.0 + i * 0.1,
            end=183.05 + i * 0.1,
            probability=1.0,
        )
        for i in range(8)
    ]
    lines = [
        Line(words=[Word(text="a", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="b", start_time=195.0, end_time=196.0)]),
    ]

    assert wialign._should_force_whisperx_for_tail_shortfall(
        all_words=words,
        lines=lines,
        language="fr",
        runtime_config=WhisperRuntimeConfig(tail_shortfall_forced_fallback=True),
    )

    assert wialign._should_force_whisperx_for_tail_shortfall(
        all_words=words,
        lines=lines,
        language=None,
        runtime_config=WhisperRuntimeConfig(tail_shortfall_forced_fallback=True),
        detected_lang="fr",
    )


def test_should_not_force_whisperx_for_tail_shortfall_without_flag():
    words = [
        TranscriptionWord(text="w", start=180.0 + i, end=180.2 + i, probability=1.0)
        for i in range(90)
    ]
    lines = [Line(words=[Word(text="b", start_time=195.0, end_time=196.0)])]

    assert not wialign._should_force_whisperx_for_tail_shortfall(
        all_words=words,
        lines=lines,
        language="fr",
        runtime_config=WhisperRuntimeConfig(),
    )


def test_maybe_force_sparse_weak_alignment_accepts_sparse_low_support_case(monkeypatch):
    lines = [Line(words=[Word(text="x", start_time=1.0, end_time=1.2)])]
    baseline = [Line(words=[Word(text="x", start_time=1.0, end_time=1.2)])]
    forced_lines = [Line(words=[Word(text="x", start_time=1.1, end_time=1.4)])]
    monkeypatch.setattr(
        wialign,
        "align_lines_with_whisperx",
        lambda *_args, **_kwargs: (
            forced_lines,
            {"forced_word_coverage": 0.9, "forced_line_coverage": 1.0},
        ),
    )

    result = wialign._maybe_force_sparse_weak_alignment(
        lines=lines,
        baseline_lines=baseline,
        vocals_path="vocals.wav",
        language="en",
        detected_lang="en",
        used_model="large",
        matched_ratio=0.85,
        phonetic_similarity_coverage=0.25,
        whisper_word_count_before_filter=9,
        whisper_segment_count=1,
        lrc_word_count=40,
        should_rollback_short_line_degradation_fn=lambda *_a, **_k: (False, 0, 0),
        restore_implausibly_short_lines_fn=lambda *_a, **_k: (forced_lines, 0),
        whisper_words=None,
        transcription=None,
        normalize_line_word_timings_fn=lambda lines: lines,
        enforce_monotonic_line_starts_fn=lambda lines: lines,
        enforce_non_overlapping_lines_fn=lambda lines: lines,
        trace_snapshots=[],
        trace_path="",
        logger=type(
            "Logger",
            (),
            {"warning": lambda *_a, **_k: None, "info": lambda *_a, **_k: None},
        )(),
    )

    assert result is not None
    aligned_lines, corrections, metrics = result
    assert aligned_lines == forced_lines
    assert metrics["whisperx_forced"] == 1.0
    assert any(
        "sparse Whisper transcript with weak phonetic support" in c for c in corrections
    )


def test_maybe_force_sparse_weak_alignment_skips_when_phonetic_support_is_good(
    monkeypatch,
):
    lines = [Line(words=[Word(text="x", start_time=1.0, end_time=1.2)])]
    monkeypatch.setattr(
        wialign,
        "align_lines_with_whisperx",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("should not run")
        ),
    )

    result = wialign._maybe_force_sparse_weak_alignment(
        lines=lines,
        baseline_lines=lines,
        vocals_path="vocals.wav",
        language="en",
        detected_lang="en",
        used_model="large",
        matched_ratio=0.85,
        phonetic_similarity_coverage=0.5,
        whisper_word_count_before_filter=9,
        whisper_segment_count=1,
        lrc_word_count=40,
        should_rollback_short_line_degradation_fn=lambda *_a, **_k: (False, 0, 0),
        restore_implausibly_short_lines_fn=lambda *_a, **_k: (lines, 0),
        whisper_words=None,
        transcription=None,
        normalize_line_word_timings_fn=lambda lines: lines,
        enforce_monotonic_line_starts_fn=lambda lines: lines,
        enforce_non_overlapping_lines_fn=lambda lines: lines,
        trace_snapshots=[],
        trace_path="",
        logger=type(
            "Logger",
            (),
            {"warning": lambda *_a, **_k: None, "info": lambda *_a, **_k: None},
        )(),
    )

    assert result is None


def test_maybe_force_sparse_weak_alignment_finalizes_accepted_forced_lines(monkeypatch):
    lines = [Line(words=[Word(text="x", start_time=1.0, end_time=1.2)])]
    forced_lines = [Line(words=[Word(text="x", start_time=1.1, end_time=1.4)])]
    finalized_lines = [Line(words=[Word(text="x", start_time=1.3, end_time=1.6)])]
    calls: list[str] = []
    monkeypatch.setattr(
        wialign,
        "align_lines_with_whisperx",
        lambda *_args, **_kwargs: (
            forced_lines,
            {"forced_word_coverage": 0.9, "forced_line_coverage": 1.0},
        ),
    )

    result = wialign._maybe_force_sparse_weak_alignment(
        lines=lines,
        baseline_lines=lines,
        vocals_path="vocals.wav",
        language="en",
        detected_lang="en",
        used_model="large",
        matched_ratio=0.85,
        phonetic_similarity_coverage=0.25,
        whisper_word_count_before_filter=9,
        whisper_segment_count=1,
        lrc_word_count=40,
        should_rollback_short_line_degradation_fn=lambda *_a, **_k: (False, 0, 0),
        restore_implausibly_short_lines_fn=lambda *_a, **_k: (forced_lines, 0),
        whisper_words=None,
        transcription=None,
        normalize_line_word_timings_fn=lambda input_lines: (
            calls.append("normalize"),
            input_lines,
        )[1],
        enforce_monotonic_line_starts_fn=lambda input_lines: (
            calls.append("monotonic"),
            input_lines,
        )[1],
        enforce_non_overlapping_lines_fn=lambda input_lines: (
            calls.append("non_overlap"),
            finalized_lines,
        )[1],
        trace_snapshots=[],
        trace_path="",
        logger=type(
            "Logger",
            (),
            {"warning": lambda *_a, **_k: None, "info": lambda *_a, **_k: None},
        )(),
    )

    assert result is not None
    aligned_lines, _corrections, _metrics = result
    assert aligned_lines == finalized_lines
    assert calls == ["normalize", "monotonic", "non_overlap"]


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


def test_mapping_profile_config_can_be_selected_explicitly():
    default_map = wialign._default_mapping_decision_config()
    safe_map = wialign._default_mapping_decision_config(
        WhisperRuntimeConfig(profile="safe")
    )
    aggr_map = wialign._default_mapping_decision_config(
        WhisperRuntimeConfig(profile="aggressive")
    )

    assert safe_map.sparse_word_threshold > default_map.sparse_word_threshold
    assert aggr_map.sparse_word_threshold < default_map.sparse_word_threshold


def test_correction_profile_config_can_be_selected_explicitly():
    default_correct = wicorrect._default_correction_config()
    safe_correct = wicorrect._default_correction_config(
        WhisperRuntimeConfig(profile="safe")
    )
    aggr_correct = wicorrect._default_correction_config(
        WhisperRuntimeConfig(profile="aggressive")
    )

    assert safe_correct.quality_good_threshold > default_correct.quality_good_threshold
    assert aggr_correct.quality_good_threshold < default_correct.quality_good_threshold
