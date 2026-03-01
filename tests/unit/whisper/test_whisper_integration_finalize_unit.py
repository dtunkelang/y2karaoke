import pytest

from y2karaoke.core.components.whisper import whisper_integration_baseline as wib
from y2karaoke.core.components.whisper import whisper_integration_finalize as wifin
from y2karaoke.core.models import Line, Word


def test_constrain_line_starts_to_baseline_skips_large_shift():
    mapped = [
        Line(words=[Word(text="hello", start_time=20.0, end_time=21.5)]),
    ]
    baseline = [
        Line(words=[Word(text="hello", start_time=12.0, end_time=13.5)]),
    ]

    constrained = wib._constrain_line_starts_to_baseline(
        mapped, baseline, max_shift_sec=2.5
    )

    assert constrained[0].start_time == pytest.approx(20.0)
    assert constrained[0].end_time == pytest.approx(21.5)


def test_constrain_line_starts_to_baseline_reverts_large_shift_when_unstable():
    mapped = [
        Line(words=[Word(text="a", start_time=166.78, end_time=167.5)]),
        Line(words=[Word(text="b", start_time=151.42, end_time=152.2)]),
    ]
    baseline = [
        Line(words=[Word(text="a", start_time=149.10, end_time=149.8)]),
        Line(words=[Word(text="b", start_time=151.04, end_time=151.8)]),
    ]

    constrained = wib._constrain_line_starts_to_baseline(
        mapped, baseline, max_shift_sec=2.5
    )

    assert constrained[0].start_time == pytest.approx(149.10)
    assert constrained[1].start_time == pytest.approx(151.04)


def test_constrain_line_starts_to_baseline_applies_small_shift():
    mapped = [
        Line(words=[Word(text="hello", start_time=10.0, end_time=11.0)]),
    ]
    baseline = [
        Line(words=[Word(text="hello", start_time=11.2, end_time=12.2)]),
    ]

    constrained = wib._constrain_line_starts_to_baseline(
        mapped, baseline, max_shift_sec=2.5
    )

    assert constrained[0].start_time == pytest.approx(11.2)


def test_constrain_line_starts_to_baseline_skips_inverting_snap():
    mapped = [
        Line(words=[Word(text="a", start_time=166.78, end_time=167.78)]),
        Line(words=[Word(text="b", start_time=166.79, end_time=167.40)]),
    ]
    baseline = [
        Line(words=[Word(text="a", start_time=149.10, end_time=150.10)]),
        Line(words=[Word(text="b", start_time=151.04, end_time=151.80)]),
    ]

    constrained = wib._constrain_line_starts_to_baseline(
        mapped, baseline, max_shift_sec=2.5
    )

    # First line is too far from baseline and remains aligned.
    assert constrained[0].start_time == pytest.approx(166.78)
    # Second line would invert ordering if snapped to baseline, so keep aligned.
    assert constrained[1].start_time == pytest.approx(166.79)


def test_restore_implausibly_short_lines_restores_newly_compressed():
    baseline = [
        Line(
            words=[
                Word(text="one", start_time=10.0, end_time=10.4),
                Word(text="two", start_time=10.4, end_time=10.8),
                Word(text="three", start_time=10.8, end_time=11.2),
            ]
        )
    ]
    aligned = [
        Line(
            words=[
                Word(text="one", start_time=10.0, end_time=10.05),
                Word(text="two", start_time=10.05, end_time=10.1),
                Word(text="three", start_time=10.1, end_time=10.15),
            ]
        )
    ]

    repaired, restored = wib._restore_implausibly_short_lines(baseline, aligned)

    assert restored == 1
    assert repaired[0].start_time == pytest.approx(baseline[0].start_time)
    assert repaired[0].end_time == pytest.approx(baseline[0].end_time)


def test_restore_implausibly_short_lines_keeps_legitimate_short_baseline():
    baseline = [
        Line(
            words=[
                Word(text="a", start_time=1.0, end_time=1.03),
                Word(text="b", start_time=1.03, end_time=1.06),
                Word(text="c", start_time=1.06, end_time=1.09),
            ]
        )
    ]
    aligned = [
        Line(
            words=[
                Word(text="a", start_time=2.0, end_time=2.03),
                Word(text="b", start_time=2.03, end_time=2.06),
                Word(text="c", start_time=2.06, end_time=2.09),
            ]
        )
    ]

    repaired, restored = wib._restore_implausibly_short_lines(baseline, aligned)

    assert restored == 0
    assert repaired[0].start_time == pytest.approx(aligned[0].start_time)


def test_restore_pairwise_inversions_from_source_restores_outlier():
    source = [
        Line(words=[Word(text="a", start_time=100.0, end_time=101.0)]),
        Line(words=[Word(text="b", start_time=102.0, end_time=103.0)]),
        Line(words=[Word(text="c", start_time=104.0, end_time=105.0)]),
    ]
    aligned = [
        Line(words=[Word(text="a", start_time=130.0, end_time=131.0)]),
        Line(words=[Word(text="b", start_time=102.5, end_time=103.5)]),
        Line(words=[Word(text="c", start_time=104.5, end_time=105.5)]),
    ]

    repaired, restored = wifin._restore_pairwise_inversions_from_source(source, aligned)

    assert restored == 1
    assert repaired[0].start_time == pytest.approx(100.0)
    assert repaired[1].start_time == pytest.approx(102.5)


def test_restore_pairwise_inversions_from_source_keeps_ordered_lines():
    source = [
        Line(words=[Word(text="a", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="b", start_time=12.0, end_time=13.0)]),
    ]
    aligned = [
        Line(words=[Word(text="a", start_time=25.0, end_time=26.0)]),
        Line(words=[Word(text="b", start_time=27.0, end_time=28.0)]),
    ]

    repaired, restored = wifin._restore_pairwise_inversions_from_source(source, aligned)

    assert restored == 0
    assert repaired[0].start_time == pytest.approx(25.0)
    assert repaired[1].start_time == pytest.approx(27.0)


def test_finalize_whisper_line_set_rolls_back_on_text_divergence():
    source_lines = [
        Line(words=[Word(text=f"s{i}", start_time=float(i), end_time=float(i) + 0.5)])
        for i in range(10)
    ]
    aligned_lines = [
        Line(
            words=[
                Word(text=f"x{i}", start_time=float(i) + 1.0, end_time=float(i) + 1.5)
            ]
        )
        for i in range(10)
    ]

    finalized, alignments = wifin._finalize_whisper_line_set(
        source_lines=source_lines,
        aligned_lines=aligned_lines,
        alignments=[],
        transcription=[],
        epitran_lang="eng-Latn",
        force_dtw=False,
        audio_features=None,
        fix_ordering_violations_fn=lambda s, a, al: (a, al),
        normalize_line_word_timings_fn=lambda lines: lines,
        enforce_monotonic_line_starts_fn=lambda lines: lines,
        enforce_non_overlapping_lines_fn=lambda lines: lines,
        pull_lines_near_segment_end_fn=lambda lines, *_a: (lines, 0),
        merge_short_following_line_into_segment_fn=lambda lines, *_a: (lines, 0),
        clamp_repeated_line_duration_fn=lambda lines: (lines, 0),
        drop_duplicate_lines_fn=lambda lines, *_a: (lines, 0),
        drop_duplicate_lines_by_timing_fn=lambda lines, *_a: (lines, 0),
        pull_lines_forward_for_continuous_vocals_fn=lambda lines, *_a: (lines, 0),
    )

    assert finalized[0].text == source_lines[0].text
    assert any("text divergence" in msg for msg in alignments)
