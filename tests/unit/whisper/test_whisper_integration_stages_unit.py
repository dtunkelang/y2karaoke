import numpy as np

from y2karaoke.core.components.whisper import whisper_integration_stages as stages
from y2karaoke.core.models import Line, Word
from y2karaoke.core.components.alignment.timing_models import (
    AudioFeatures,
    TranscriptionWord,
)


def test_enforce_mapped_line_stage_invariants_runs_two_cycles():
    lines = [Line(words=[Word(text="a", start_time=1.0, end_time=1.4)])]
    whisper_words = [TranscriptionWord(text="a", start=1.0, end=1.3, probability=0.9)]
    calls: list[str] = []

    def _monotonic(in_lines, _words):
        calls.append("monotonic")
        return in_lines

    def _resolve(in_lines):
        calls.append("resolve")
        return in_lines

    out = stages._enforce_mapped_line_stage_invariants(
        lines,
        whisper_words,
        enforce_monotonic_line_starts_whisper_fn=_monotonic,
        resolve_line_overlaps_fn=_resolve,
    )

    assert out == lines
    assert calls == ["monotonic", "resolve", "monotonic", "resolve"]


def test_run_mapped_line_postpasses_handles_snap_fallback_and_audio_corrections():
    lines = [
        Line(words=[Word(text="one", start_time=10.0, end_time=10.5)]),
        Line(words=[Word(text="two", start_time=11.0, end_time=11.5)]),
    ]
    whisper_words = [
        TranscriptionWord(text="one", start=10.0, end=10.5, probability=0.9),
        TranscriptionWord(text="two", start=11.0, end=11.5, probability=0.9),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=20.0,
        duration=20.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )
    call_log: list[str] = []
    pull_counts = [1, 2]

    def _mark(tag):
        def _fn(*args, **kwargs):
            call_log.append(tag)
            return args[0]

        return _fn

    def _pull_lines_forward(in_lines, _audio_features):
        call_log.append("pull_lines_forward")
        count = pull_counts.pop(0) if pull_counts else 0
        return in_lines, count

    def _snap(lines_in, all_words_in):
        call_log.append("snap_no_kwargs")
        return lines_in

    out, corrections = stages._run_mapped_line_postpasses(
        mapped_lines=lines,
        mapped_lines_set={0, 1},
        all_words=whisper_words,
        transcription=[],
        audio_features=audio_features,
        vocals_path="vocals.wav",
        epitran_lang="eng-Latn",
        corrections=[],
        interpolate_unmatched_lines_fn=_mark("interpolate"),
        refine_unmatched_lines_with_onsets_fn=_mark("refine"),
        shift_repeated_lines_to_next_whisper_fn=_mark("shift_repeat"),
        extend_line_to_trailing_whisper_matches_fn=_mark("extend"),
        pull_late_lines_to_matching_segments_fn=_mark("pull_late"),
        retime_short_interjection_lines_fn=_mark("retime_interjection"),
        snap_first_word_to_whisper_onset_fn=_snap,
        pull_lines_forward_for_continuous_vocals_fn=_pull_lines_forward,
        enforce_monotonic_line_starts_whisper_fn=_mark("monotonic"),
        resolve_line_overlaps_fn=_mark("resolve"),
    )

    assert out == lines
    assert "interpolate" in call_log
    assert "shift_repeat" in call_log
    assert call_log.count("pull_lines_forward") == 2
    assert call_log.count("snap_no_kwargs") >= 3
    assert any(
        "Pulled 1 line(s) forward for continuous vocals" in c for c in corrections
    )
    assert any(
        "Applied 2 late audio onset/silence adjustment(s)" in c for c in corrections
    )
