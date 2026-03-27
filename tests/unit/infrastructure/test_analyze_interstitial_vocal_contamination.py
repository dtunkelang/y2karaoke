from y2karaoke.core.components.alignment.timing_models import TranscriptionSegment

from tools import analyze_interstitial_vocal_contamination as tool


def test_classify_gap_marks_echo_fragment_from_neighbor_overlap() -> None:
    result = tool._classify_gap(
        prev_text="Take on me",
        next_text="Take me on",
        default_segments=[],
        aggressive_segments=[
            TranscriptionSegment(start=0.0, end=2.1, text="Take me", words=[])
        ],
    )

    assert result == "echo_fragment"


def test_classify_gap_marks_hallucinated_interstitial() -> None:
    result = tool._classify_gap(
        prev_text="I'll be gone",
        next_text="In a day or two",
        default_segments=[],
        aggressive_segments=[
            TranscriptionSegment(
                start=0.0,
                end=2.76,
                text="I'll see you next time.",
                words=[],
            )
        ],
    )

    assert result == "hallucinated_interstitial"


def test_build_gap_windows_expands_around_boundary() -> None:
    lines = [
        {"text": "Take on me", "start": 1.0, "end": 5.3},
        {"text": "Take me on", "start": 6.85, "end": 10.65},
    ]

    windows = tool._build_gap_windows(lines)

    assert windows[0].start == 4.8
    assert windows[0].end == 7.35
