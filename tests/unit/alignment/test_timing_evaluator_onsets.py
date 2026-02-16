import numpy as np

import y2karaoke.core.components.alignment.timing_evaluator as te
import y2karaoke.core.components.alignment.timing_evaluator_correction as te_corr


def _features():
    return te.AudioFeatures(
        onset_times=np.array([]),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=10.0,
        duration=10.0,
        energy_envelope=np.array([0.0, 0.0, 0.0, 0.0]),
        energy_times=np.array([0.0, 1.0, 2.0, 3.0]),
    )


def test_find_best_onset_for_phrase_end_prefers_first_silent_onset():
    features = _features()
    onset_times = np.array([0.5, 1.1, 2.0, 4.0])

    onset = te_corr._find_best_onset_for_phrase_end(
        onset_times=onset_times,
        line_start=5.0,
        prev_line_audio_end=1.0,
        audio_features=features,
    )

    assert onset == 2.0


def test_find_best_onset_proximity_scores_distance_and_silence():
    features = _features()
    onset_times = np.array([4.0, 5.4])

    onset = te_corr._find_best_onset_proximity(
        onset_times=onset_times,
        line_start=5.0,
        max_correction=2.0,
        audio_features=features,
    )

    assert onset == 5.4


def test_find_best_onset_during_silence_respects_search_window():
    features = _features()
    onset_times = np.array([2.5, 3.5, 6.0])

    onset = te_corr._find_best_onset_during_silence(
        onset_times=onset_times,
        line_start=5.0,
        prev_line_audio_end=2.0,
        max_correction=2.0,
        audio_features=features,
    )

    assert onset == 3.5
