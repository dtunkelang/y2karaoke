import numpy as np
import pytest

from y2karaoke.core.components.alignment.timing_models import (
    AudioFeatures,
    TranscriptionWord,
)
from y2karaoke.core.components.whisper import (
    whisper_integration_align_experimental as wiaexp,
)
from y2karaoke.core.models import Line, Word


def test_reanchor_low_support_lines_to_later_onset_helper():
    baseline = [
        Line(words=[Word(text="prev", start_time=49.7, end_time=51.2)]),
        Line(
            words=[
                Word(text="No", start_time=53.53, end_time=53.92),
                Word(text="one's", start_time=53.92, end_time=54.31),
                Word(text="around", start_time=54.31, end_time=54.7),
                Word(text="to", start_time=54.7, end_time=55.1),
                Word(text="judge", start_time=55.1, end_time=55.5),
                Word(text="me", start_time=55.5, end_time=55.88),
            ]
        ),
        Line(words=[Word(text="next", start_time=56.29, end_time=60.58)]),
    ]
    mapped = list(baseline)
    whisper_words = [
        TranscriptionWord(text="Gå", start=60.0, end=60.24, probability=0.09),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([54.08, 54.77], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=200.0,
        duration=200.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

    updated, applied = wiaexp.reanchor_low_support_lines_to_later_onset(
        mapped,
        baseline,
        whisper_words,
        audio_features,
    )

    assert applied == 1
    assert updated[1].start_time == pytest.approx(54.08, abs=0.01)


def test_reanchor_low_support_lines_to_later_onset_blocks_when_lexical_support_exists():
    baseline = [
        Line(words=[Word(text="prev", start_time=49.7, end_time=51.2)]),
        Line(
            words=[
                Word(text="No", start_time=53.53, end_time=53.92),
                Word(text="one's", start_time=53.92, end_time=54.31),
                Word(text="around", start_time=54.31, end_time=54.7),
                Word(text="to", start_time=54.7, end_time=55.1),
                Word(text="judge", start_time=55.1, end_time=55.5),
                Word(text="me", start_time=55.5, end_time=55.88),
            ]
        ),
        Line(words=[Word(text="next", start_time=56.29, end_time=60.58)]),
    ]
    mapped = list(baseline)
    whisper_words = [
        TranscriptionWord(text="judge", start=54.9, end=55.2, probability=0.8),
        TranscriptionWord(text="me", start=55.2, end=55.5, probability=0.8),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([54.08, 54.77], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=200.0,
        duration=200.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

    updated, applied = wiaexp.reanchor_low_support_lines_to_later_onset(
        mapped,
        baseline,
        whisper_words,
        audio_features,
    )

    assert applied == 0
    assert updated[1].start_time == pytest.approx(53.53, abs=0.01)


def test_reanchor_low_support_lines_to_later_onset_blocks_already_shifted_line():
    baseline = [
        Line(words=[Word(text="prev", start_time=150.59, end_time=156.41)]),
        Line(
            words=[
                Word(text="No", start_time=156.57, end_time=156.9),
                Word(text="I", start_time=156.9, end_time=157.2),
                Word(text="can't", start_time=157.2, end_time=157.5),
                Word(text="sleep", start_time=157.5, end_time=157.8),
                Word(text="until", start_time=157.8, end_time=158.2),
                Word(text="I", start_time=158.2, end_time=158.5),
            ]
        ),
        Line(words=[Word(text="next", start_time=160.79, end_time=161.2)]),
    ]
    mapped = [
        baseline[0],
        Line(
            words=[
                Word(text="No", start_time=157.11, end_time=157.44),
                Word(text="I", start_time=157.44, end_time=157.74),
                Word(text="can't", start_time=157.74, end_time=158.04),
                Word(text="sleep", start_time=158.04, end_time=158.34),
                Word(text="until", start_time=158.34, end_time=158.74),
                Word(text="I", start_time=158.74, end_time=159.04),
            ]
        ),
        baseline[2],
    ]
    whisper_words = [
        TranscriptionWord(text="[VOCAL]", start=157.11, end=157.2, probability=0.9),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([157.11, 157.43], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=200.0,
        duration=200.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

    updated, applied = wiaexp.reanchor_low_support_lines_to_later_onset(
        mapped,
        baseline,
        whisper_words,
        audio_features,
    )

    assert applied == 0
    assert updated[1].start_time == pytest.approx(157.11, abs=0.01)


def test_reanchor_repeated_cadence_lines_borrows_later_pair_spacing():
    lines = [
        Line(words=[Word(text="No", start_time=53.53, end_time=55.88)]),
        Line(
            words=[
                Word(text="I", start_time=56.29, end_time=56.89),
                Word(text="can't", start_time=56.89, end_time=57.49),
                Word(text="see", start_time=57.49, end_time=58.09),
                Word(text="clearly", start_time=58.09, end_time=58.69),
                Word(text="when", start_time=58.69, end_time=59.29),
                Word(text="you're", start_time=59.29, end_time=59.89),
                Word(text="gone", start_time=59.89, end_time=60.58),
            ]
        ),
        Line(words=[Word(text="bridge", start_time=61.5, end_time=62.0)]),
        Line(words=[Word(text="No", start_time=109.5, end_time=112.39)]),
        Line(
            words=[
                Word(text="I", start_time=112.69, end_time=113.09),
                Word(text="can't", start_time=113.09, end_time=113.49),
                Word(text="see", start_time=113.49, end_time=113.89),
                Word(text="clearly", start_time=113.89, end_time=114.29),
                Word(text="when", start_time=114.29, end_time=114.69),
                Word(text="you're", start_time=114.69, end_time=115.09),
                Word(text="gone", start_time=115.09, end_time=115.4),
            ]
        ),
    ]
    lines[0].words[0].text = "No one's around to judge me"
    lines[1].words[0].text = "I can't see clearly when you're gone"
    lines[3].words[0].text = "No one's around to judge me"
    lines[4].words[0].text = "I can't see clearly when you're gone"

    adjusted, applied = wiaexp.reanchor_repeated_cadence_lines(lines)

    assert applied == 1
    assert adjusted[1].start_time == pytest.approx(56.72, abs=0.01)


def test_reanchor_late_supported_lines_to_earlier_whisper_moves_line_earlier():
    lines = [
        Line(words=[Word(text="prev", start_time=16.43, end_time=19.58)]),
        Line(
            words=[
                Word(text="I", start_time=20.28, end_time=20.48),
                Word(text="like", start_time=20.5, end_time=20.7),
                Word(text="your", start_time=20.72, end_time=20.92),
                Word(text="poom-poom", start_time=20.94, end_time=21.14),
            ]
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="I", start=19.44, end=19.82, probability=1.0),
        TranscriptionWord(text="like", start=19.82, end=20.02, probability=1.0),
        TranscriptionWord(text="you", start=20.02, end=20.16, probability=1.0),
    ]

    adjusted, applied = wiaexp.reanchor_late_supported_lines_to_earlier_whisper(
        lines, whisper_words
    )

    assert applied == 1
    assert adjusted[1].start_time == pytest.approx(19.63, abs=0.01)
    assert adjusted[1].end_time == pytest.approx(21.14, abs=0.01)


def test_reanchor_late_supported_lines_to_earlier_whisper_requires_prefix_support():
    lines = [
        Line(words=[Word(text="prev", start_time=10.0, end_time=10.8)]),
        Line(
            words=[
                Word(text="Ya", start_time=12.0, end_time=12.3),
                Word(text="vi", start_time=12.3, end_time=12.6),
                Word(text="que", start_time=12.6, end_time=12.9),
            ]
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="ya", start=11.2, end=11.5, probability=1.0),
        TranscriptionWord(text="solo", start=11.5, end=11.8, probability=1.0),
    ]

    adjusted, applied = wiaexp.reanchor_late_supported_lines_to_earlier_whisper(
        lines, whisper_words
    )

    assert applied == 0
    assert adjusted[1].start_time == pytest.approx(12.0, abs=0.01)


def test_shift_restored_low_support_runs_to_onset_moves_dense_run_together():
    baseline = [
        Line(words=[Word(text="prev", start_time=50.72, end_time=52.98)]),
        Line(
            words=[
                Word(text="No", start_time=53.53, end_time=53.92),
                Word(text="one's", start_time=53.92, end_time=54.31),
                Word(text="around", start_time=54.31, end_time=54.7),
                Word(text="to", start_time=54.7, end_time=55.1),
                Word(text="judge", start_time=55.1, end_time=55.5),
                Word(text="me", start_time=55.5, end_time=55.88),
            ]
        ),
        Line(
            words=[
                Word(text="I", start_time=56.29, end_time=56.69),
                Word(text="can't", start_time=56.69, end_time=57.09),
                Word(text="see", start_time=57.09, end_time=57.49),
                Word(text="clearly", start_time=57.49, end_time=57.89),
                Word(text="when", start_time=57.89, end_time=58.29),
                Word(text="you're", start_time=58.29, end_time=58.69),
                Word(text="gone", start_time=58.69, end_time=59.0),
            ]
        ),
        Line(
            words=[
                Word(text="I", start_time=60.78, end_time=61.2),
                Word(text="said", start_time=61.2, end_time=61.62),
                Word(text="ooh", start_time=61.62, end_time=62.04),
                Word(text="im", start_time=62.04, end_time=62.46),
                Word(text="blinded", start_time=62.46, end_time=63.4),
                Word(text="by", start_time=63.4, end_time=63.8),
                Word(text="the", start_time=63.8, end_time=64.2),
                Word(text="lights", start_time=64.2, end_time=66.52),
            ]
        ),
        Line(words=[Word(text="next", start_time=67.5, end_time=71.4)]),
    ]
    whisper_words = [
        TranscriptionWord(text="[VOCAL]", start=54.0, end=54.1, probability=0.9),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([54.08, 54.45, 54.63], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=200.0,
        duration=200.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

    adjusted, applied = wiaexp.shift_restored_low_support_runs_to_onset(
        baseline,
        baseline,
        whisper_words,
        audio_features,
    )

    assert applied == 2
    assert adjusted[1].start_time == pytest.approx(54.08, abs=0.01)
    assert adjusted[2].start_time == pytest.approx(56.84, abs=0.01)
    assert adjusted[3].start_time == pytest.approx(60.78, abs=0.01)


def test_shift_restored_low_support_runs_to_onset_allows_late_compact_repetitive_tail_run():
    baseline = [
        Line(words=[Word(text="lead", start_time=0.0, end_time=1.0)]),
        Line(words=[Word(text="bridge", start_time=1.2, end_time=2.2)]),
        Line(words=[Word(text="prep", start_time=2.4, end_time=3.4)]),
        Line(
            words=[
                Word(text="Guess", start_time=8.46, end_time=8.71),
                Word(text="who's", start_time=8.71, end_time=8.95),
                Word(text="back?", start_time=8.95, end_time=9.2),
                Word(text="Guess", start_time=9.2, end_time=9.45),
                Word(text="who's", start_time=9.45, end_time=9.69),
                Word(text="back?", start_time=9.69, end_time=9.94),
            ]
        ),
        Line(
            words=[
                Word(text="Guess", start_time=11.436, end_time=11.686),
                Word(text="who's", start_time=11.686, end_time=11.936),
                Word(text="back?", start_time=11.936, end_time=12.186),
                Word(text="Guess", start_time=12.186, end_time=12.436),
                Word(text="who's", start_time=12.436, end_time=12.686),
                Word(text="back?", start_time=12.686, end_time=12.936),
            ]
        ),
        Line(
            words=[
                Word(text="Guess", start_time=13.387, end_time=13.554),
                Word(text="who's", start_time=13.554, end_time=13.72),
                Word(text="back?", start_time=13.72, end_time=13.887),
            ]
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="[VOCAL]", start=11.0, end=11.1, probability=0.9),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([12.04, 12.3], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=20.0,
        duration=20.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

    adjusted, applied = wiaexp.shift_restored_low_support_runs_to_onset(
        baseline,
        baseline,
        whisper_words,
        audio_features,
    )

    assert applied == 2
    assert adjusted[4].start_time == pytest.approx(12.04, abs=0.01)
    assert adjusted[5].start_time == pytest.approx(13.991, abs=0.01)


def test_reanchor_late_compact_repetitive_tail_lines_to_later_onsets_hits_houdini_suffix():
    baseline = [
        Line(words=[Word(text="lead", start_time=8.46, end_time=9.94)]),
        Line(
            words=[
                Word(text="Guess", start_time=9.948, end_time=10.194),
                Word(text="who's", start_time=10.194, end_time=10.441),
                Word(text="back?", start_time=10.441, end_time=10.687),
                Word(text="Guess", start_time=10.687, end_time=10.934),
                Word(text="who's", start_time=10.934, end_time=11.18),
                Word(text="back?", start_time=11.18, end_time=11.426),
            ]
        ),
        Line(
            words=[
                Word(text="Guess", start_time=11.436, end_time=11.686),
                Word(text="who's", start_time=11.686, end_time=11.936),
                Word(text="back?", start_time=11.936, end_time=12.186),
                Word(text="Guess", start_time=12.186, end_time=12.436),
                Word(text="who's", start_time=12.436, end_time=12.686),
                Word(text="back?", start_time=12.686, end_time=12.936),
            ]
        ),
        Line(
            words=[
                Word(text="Guess", start_time=13.387, end_time=13.858),
                Word(text="who's", start_time=13.858, end_time=14.329),
                Word(text="back?", start_time=14.329, end_time=14.801),
            ]
        ),
    ]
    shifted = [
        baseline[0],
        Line(
            words=[
                Word(text="Guess", start_time=10.356, end_time=10.708),
                Word(text="who's", start_time=10.708, end_time=11.06),
                Word(text="back?", start_time=11.06, end_time=11.412),
                Word(text="Guess", start_time=11.412, end_time=11.763),
                Word(text="who's", start_time=11.763, end_time=11.789),
                Word(text="back?", start_time=11.789, end_time=11.834),
            ]
        ),
        Line(
            words=[
                Word(text="Guess", start_time=11.844, end_time=12.298),
                Word(text="who's", start_time=12.298, end_time=12.753),
                Word(text="back?", start_time=12.753, end_time=13.207),
                Word(text="Guess", start_time=13.207, end_time=13.253),
                Word(text="who's", start_time=13.253, end_time=13.284),
                Word(text="back?", start_time=13.284, end_time=13.344),
            ]
        ),
        Line(
            words=[
                Word(text="Guess", start_time=13.795, end_time=14.234),
                Word(text="who's", start_time=14.283, end_time=14.722),
                Word(text="back?", start_time=14.77, end_time=15.209),
            ]
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="[VOCAL]", start=11.0, end=11.1, probability=0.9)
    ]
    audio_features = AudioFeatures(
        onset_times=np.array(
            [11.749, 11.842, 11.958, 12.399, 12.794, 13.723, 14.257, 14.35, 14.768],
            dtype=float,
        ),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=20.0,
        duration=20.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

    adjusted, applied = (
        wiaexp.reanchor_late_compact_repetitive_tail_lines_to_later_onsets(
            shifted,
            baseline,
            whisper_words,
            audio_features,
        )
    )

    assert applied == 2
    assert adjusted[1].start_time == pytest.approx(10.356, abs=0.01)
    assert adjusted[2].start_time == pytest.approx(12.399, abs=0.01)
    assert adjusted[3].start_time == pytest.approx(14.257, abs=0.01)


def test_reanchor_late_compact_repetitive_tail_lines_to_later_onsets_skips_non_tail_gap():
    baseline = [
        Line(words=[Word(text="lead", start_time=8.46, end_time=9.94)]),
        Line(words=[Word(text="Guess", start_time=11.436, end_time=12.936)]),
        Line(words=[Word(text="Guess", start_time=13.387, end_time=14.801)]),
        Line(words=[Word(text="next", start_time=15.1, end_time=15.6)]),
    ]
    shifted = [
        baseline[0],
        Line(words=[Word(text="Guess", start_time=11.844, end_time=13.344)]),
        Line(words=[Word(text="Guess", start_time=13.795, end_time=15.209)]),
        baseline[3],
    ]
    whisper_words = [
        TranscriptionWord(text="[VOCAL]", start=11.0, end=11.1, probability=0.9)
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([12.399, 14.257], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=20.0,
        duration=20.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

    adjusted, applied = (
        wiaexp.reanchor_late_compact_repetitive_tail_lines_to_later_onsets(
            shifted,
            baseline,
            whisper_words,
            audio_features,
        )
    )

    assert applied == 0
    assert adjusted[1].start_time == pytest.approx(11.844, abs=0.01)
    assert adjusted[2].start_time == pytest.approx(13.795, abs=0.01)


def test_reanchor_light_leading_lines_to_content_words_moves_supported_content_later():
    lines = [
        Line(words=[Word(text="prev", start_time=21.63, end_time=23.82)]),
        Line(
            words=[
                Word(text="La", start_time=24.07, end_time=24.3),
                Word(text="noche", start_time=24.3, end_time=24.7),
                Word(text="es", start_time=24.7, end_time=25.0),
                Word(text="de", start_time=25.0, end_time=25.2),
                Word(text="nosotros", start_time=25.2, end_time=26.37),
            ]
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="acompáñame", start=23.08, end=23.9, probability=0.9),
        TranscriptionWord(text="la", start=24.1, end=24.6, probability=0.9),
        TranscriptionWord(text="noche", start=24.6, end=24.86, probability=0.9),
        TranscriptionWord(text="de", start=24.86, end=25.04, probability=0.9),
        TranscriptionWord(text="nosotros", start=25.04, end=25.34, probability=0.9),
    ]

    adjusted, applied = wiaexp.reanchor_light_leading_lines_to_content_words(
        lines, whisper_words
    )

    assert applied == 1
    assert adjusted[1].start_time == pytest.approx(24.6, abs=0.01)


def test_reanchor_light_leading_lines_to_content_words_allows_light_stem_match():
    lines = [
        Line(words=[Word(text="prev", start_time=24.07, end_time=26.37)]),
        Line(
            words=[
                Word(text="Que", start_time=26.49, end_time=26.7),
                Word(text="ganas", start_time=26.7, end_time=27.0),
                Word(text="me", start_time=27.0, end_time=27.2),
                Word(text="dan-dan-dan", start_time=27.2, end_time=27.74),
            ]
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="que", start=26.62, end=27.12, probability=0.9),
        TranscriptionWord(text="gana", start=27.12, end=27.48, probability=0.9),
        TranscriptionWord(text="me", start=27.48, end=27.58, probability=0.9),
        TranscriptionWord(text="dam", start=27.58, end=27.92, probability=0.9),
    ]

    adjusted, applied = wiaexp.reanchor_light_leading_lines_to_content_words(
        lines, whisper_words
    )

    assert applied == 1
    assert adjusted[1].start_time == pytest.approx(27.12, abs=0.01)


def test_rebalance_short_followup_boundaries_from_whisper_shifts_con_calma_tail():
    lines = [
        Line(words=[Word(text="prev", start_time=22.458, end_time=24.403)]),
        Line(
            words=[
                Word(text="La", start_time=24.68, end_time=24.913),
                Word(text="noche", start_time=24.939, end_time=25.172),
                Word(text="es", start_time=25.198, end_time=25.431),
                Word(text="de", start_time=25.457, end_time=25.69),
                Word(text="nosotros,", start_time=25.716, end_time=25.95),
                Word(text="tú", start_time=25.975, end_time=26.209),
                Word(text="lo", start_time=26.235, end_time=26.468),
                Word(text="sabe'", start_time=26.494, end_time=26.727),
                Word(text="(You", start_time=26.753, end_time=26.986),
                Word(text="know)", start_time=27.012, end_time=27.245),
            ]
        ),
        Line(
            words=[
                Word(text="Que", start_time=27.379, end_time=27.701),
                Word(text="ganas", start_time=27.737, end_time=28.059),
                Word(text="me", start_time=28.095, end_time=28.417),
                Word(text="dan-dan-dan", start_time=28.453, end_time=28.775),
            ]
        ),
        Line(
            words=[
                Word(text="De", start_time=28.889, end_time=29.216),
                Word(text="guayarte,", start_time=29.252, end_time=29.58),
                Word(text="ma...", start_time=29.616, end_time=29.944),
            ]
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="la", start=24.1, end=24.6, probability=0.9),
        TranscriptionWord(text="noche", start=24.6, end=24.86, probability=0.9),
        TranscriptionWord(text="de", start=24.86, end=25.04, probability=0.9),
        TranscriptionWord(text="nosotros", start=25.04, end=25.34, probability=0.9),
        TranscriptionWord(text="tú", start=25.34, end=25.72, probability=0.9),
        TranscriptionWord(text="lo", start=25.72, end=25.86, probability=0.9),
        TranscriptionWord(text="sabes,", start=25.86, end=26.52, probability=0.9),
        TranscriptionWord(text="que", start=26.62, end=27.12, probability=0.9),
        TranscriptionWord(text="gana", start=27.12, end=27.48, probability=0.9),
        TranscriptionWord(text="me", start=27.48, end=27.58, probability=0.9),
        TranscriptionWord(text="dam", start=27.58, end=27.92, probability=0.9),
        TranscriptionWord(text="dam", start=27.92, end=28.22, probability=0.9),
        TranscriptionWord(text="dam,", start=28.22, end=28.56, probability=0.9),
        TranscriptionWord(text="degollarte", start=28.64, end=29.3, probability=0.9),
        TranscriptionWord(text="mami.", start=29.3, end=29.98, probability=0.9),
    ]

    adjusted, applied = wiaexp.rebalance_short_followup_boundaries_from_whisper(
        lines, whisper_words
    )

    assert applied == 1
    assert adjusted[1].end_time == pytest.approx(26.57, abs=0.01)
    assert adjusted[2].start_time == pytest.approx(26.62, abs=0.01)
    assert adjusted[2].end_time == pytest.approx(28.56, abs=0.01)


def test_rebalance_short_followup_boundaries_from_whisper_skips_without_prior_carryover():
    lines = [
        Line(
            words=[
                Word(text="prev", start_time=24.1, end_time=24.8),
                Word(text="line", start_time=24.8, end_time=25.4),
                Word(text="ends", start_time=25.4, end_time=26.0),
                Word(text="here", start_time=26.0, end_time=26.5),
                Word(text="cleanly", start_time=26.5, end_time=26.9),
                Word(text="already", start_time=26.9, end_time=27.1),
                Word(text="today", start_time=27.1, end_time=27.2),
                Word(text="now", start_time=27.2, end_time=27.25),
            ]
        ),
        Line(
            words=[
                Word(text="Que", start_time=27.379, end_time=27.701),
                Word(text="ganas", start_time=27.737, end_time=28.059),
                Word(text="me", start_time=28.095, end_time=28.417),
                Word(text="dan-dan-dan", start_time=28.453, end_time=28.775),
            ]
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="cleanly", start=26.5, end=26.9, probability=0.9),
        TranscriptionWord(text="already", start=26.9, end=27.1, probability=0.9),
        TranscriptionWord(text="que", start=27.3, end=27.6, probability=0.9),
        TranscriptionWord(text="gana", start=27.6, end=27.9, probability=0.9),
        TranscriptionWord(text="me", start=27.9, end=28.0, probability=0.9),
    ]

    adjusted, applied = wiaexp.rebalance_short_followup_boundaries_from_whisper(
        lines, whisper_words
    )

    assert applied == 0
    assert adjusted[0].end_time == pytest.approx(lines[0].end_time, abs=0.01)
    assert adjusted[1].start_time == pytest.approx(lines[1].start_time, abs=0.01)


def test_reanchor_truncated_followup_lines_from_phonetic_variants_moves_con_calma_tail():
    lines = [
        Line(
            words=[
                Word(text="Que", start_time=26.62, end_time=27.068),
                Word(text="ganas", start_time=27.117, end_time=27.565),
                Word(text="me", start_time=27.615, end_time=28.063),
                Word(text="dan-dan-dan", start_time=28.112, end_time=28.56),
            ]
        ),
        Line(
            words=[
                Word(text="De", start_time=28.889, end_time=29.216),
                Word(text="guayarte,", start_time=29.252, end_time=29.58),
                Word(text="ma...", start_time=29.616, end_time=29.944),
            ]
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="dam,", start=28.22, end=28.56, probability=0.9),
        TranscriptionWord(text="degollarte", start=28.64, end=29.3, probability=0.9),
        TranscriptionWord(text="mami.", start=29.3, end=29.98, probability=0.9),
    ]

    adjusted, applied = wiaexp.reanchor_truncated_followup_lines_from_phonetic_variants(
        lines, whisper_words
    )

    assert applied == 1
    assert adjusted[0].end_time == pytest.approx(28.35, abs=0.01)
    assert adjusted[1].start_time == pytest.approx(28.40, abs=0.01)


def test_reanchor_truncated_followup_lines_from_phonetic_variants_skips_non_truncated_lines():
    lines = [
        Line(words=[Word(text="prev", start_time=26.62, end_time=28.56)]),
        Line(
            words=[
                Word(text="De", start_time=28.889, end_time=29.216),
                Word(text="guayarte,", start_time=29.252, end_time=29.58),
                Word(text="mami,", start_time=29.616, end_time=29.944),
            ]
        ),
    ]
    whisper_words = [
        TranscriptionWord(text="degollarte", start=28.64, end=29.3, probability=0.9),
        TranscriptionWord(text="mami.", start=29.3, end=29.98, probability=0.9),
    ]

    adjusted, applied = wiaexp.reanchor_truncated_followup_lines_from_phonetic_variants(
        lines, whisper_words
    )

    assert applied == 0
    assert adjusted[1].start_time == pytest.approx(lines[1].start_time, abs=0.01)
