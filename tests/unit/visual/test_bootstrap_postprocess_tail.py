from y2karaoke.core.models import TargetLine
from y2karaoke.core.visual import bootstrap_postprocess
from y2karaoke.core.visual.bootstrap_postprocess import build_refined_lines_output


def test_build_refined_lines_output_does_not_canonicalize_nearby_sequence_lines():
    lines = [
        TargetLine(
            line_index=1,
            start=10.0,
            end=11.5,
            text="put your hands up",
            words=["put", "your", "hands", "up"],
            y=10,
            word_starts=[10.0, 10.3, 10.7, 11.1],
            word_ends=[10.2, 10.6, 11.0, 11.4],
            word_rois=[(0, 0, 1, 1)] * 4,
            word_confidences=[0.8] * 4,
            reconstruction_meta={"uncertainty_score": 0.05},
        ),
        TargetLine(
            line_index=2,
            start=14.0,
            end=15.5,
            text="put your drinks up",
            words=["put", "your", "drinks", "up"],
            y=10,
            word_starts=[14.0, 14.3, 14.7, 15.1],
            word_ends=[14.2, 14.6, 15.0, 15.4],
            word_rois=[(0, 0, 1, 1)] * 4,
            word_confidences=[0.3] * 4,
            reconstruction_meta={"uncertainty_score": 0.35},
        ),
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [ln["text"] for ln in out] == ["put your hands up", "put your drinks up"]


def test_build_refined_lines_output_removes_repeated_singleton_noise_token_lines():
    lines = [
        TargetLine(
            line_index=1,
            start=10.0,
            end=10.5,
            text="Grit",
            words=["Grit"],
            y=10,
            word_starts=[10.0],
            word_ends=[10.4],
            word_rois=[(0, 0, 1, 1)],
            word_confidences=[0.25],
            reconstruction_meta={"uncertainty_score": 0.3},
        ),
        TargetLine(
            line_index=2,
            start=11.0,
            end=12.5,
            text="this hit that ice cold",
            words=["this", "hit", "that", "ice", "cold"],
            y=20,
            word_starts=[11.0, 11.3, 11.6, 11.9, 12.2],
            word_ends=[11.2, 11.5, 11.8, 12.1, 12.4],
            word_rois=[(0, 0, 1, 1)] * 5,
            word_confidences=[0.8] * 5,
        ),
        TargetLine(
            line_index=3,
            start=13.0,
            end=13.5,
            text="Grit",
            words=["Grit"],
            y=10,
            word_starts=[13.0],
            word_ends=[13.4],
            word_rois=[(0, 0, 1, 1)],
            word_confidences=[0.25],
            reconstruction_meta={"uncertainty_score": 0.3},
        ),
        TargetLine(
            line_index=4,
            start=14.0,
            end=15.5,
            text="stop wait a minute",
            words=["stop", "wait", "a", "minute"],
            y=20,
            word_starts=[14.0, 14.4, 14.8, 15.1],
            word_ends=[14.3, 14.7, 15.0, 15.4],
            word_rois=[(0, 0, 1, 1)] * 4,
            word_confidences=[0.8] * 4,
        ),
        TargetLine(
            line_index=5,
            start=16.0,
            end=16.5,
            text="Grit",
            words=["Grit"],
            y=10,
            word_starts=[16.0],
            word_ends=[16.4],
            word_rois=[(0, 0, 1, 1)],
            word_confidences=[0.25],
            reconstruction_meta={"uncertainty_score": 0.3},
        ),
        TargetLine(
            line_index=6,
            start=17.0,
            end=18.0,
            text="jump on it",
            words=["jump", "on", "it"],
            y=20,
            word_starts=[17.0, 17.4, 17.7],
            word_ends=[17.3, 17.6, 17.9],
            word_rois=[(0, 0, 1, 1)] * 3,
            word_confidences=[0.8] * 3,
        ),
        TargetLine(
            line_index=7,
            start=19.0,
            end=19.5,
            text="Grit",
            words=["Grit"],
            y=10,
            word_starts=[19.0],
            word_ends=[19.4],
            word_rois=[(0, 0, 1, 1)],
            word_confidences=[0.25],
            reconstruction_meta={"uncertainty_score": 0.3},
        ),
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert all(ln["text"] != "Grit" for ln in out)
    assert [ln["text"] for ln in out] == [
        "this hit that ice cold",
        "stop wait a minute",
        "jump on it",
    ]


def test_build_refined_lines_output_keeps_repeated_singleton_when_neighbor_contains_token():
    lines_out = [
        {
            "line_index": 1,
            "text": "say hit your hallelujah",
            "start": 10.0,
            "end": 11.0,
            "confidence": 0.8,
            "words": [{"text": t} for t in ["say", "hit", "your", "hallelujah"]],
        },
        {
            "line_index": 2,
            "text": "hallelujah",
            "start": 11.3,
            "end": 11.8,
            "confidence": 0.25,
            "words": [{"text": "hallelujah"}],
        },
        {
            "line_index": 3,
            "text": "sing hit your hallelujah",
            "start": 20.0,
            "end": 21.0,
            "confidence": 0.8,
            "words": [{"text": t} for t in ["sing", "hit", "your", "hallelujah"]],
        },
        {
            "line_index": 4,
            "text": "hallelujah",
            "start": 21.3,
            "end": 21.8,
            "confidence": 0.25,
            "words": [{"text": "hallelujah"}],
        },
        {
            "line_index": 5,
            "text": "bring hit your hallelujah",
            "start": 30.0,
            "end": 31.0,
            "confidence": 0.8,
            "words": [{"text": t} for t in ["bring", "hit", "your", "hallelujah"]],
        },
        {
            "line_index": 6,
            "text": "hallelujah",
            "start": 31.3,
            "end": 31.8,
            "confidence": 0.25,
            "words": [{"text": "hallelujah"}],
        },
        {
            "line_index": 7,
            "text": "shout hit your hallelujah",
            "start": 40.0,
            "end": 41.0,
            "confidence": 0.8,
            "words": [{"text": t} for t in ["shout", "hit", "your", "hallelujah"]],
        },
        {
            "line_index": 8,
            "text": "hallelujah",
            "start": 41.3,
            "end": 41.8,
            "confidence": 0.25,
            "words": [{"text": "hallelujah"}],
        },
    ]

    bootstrap_postprocess._remove_repeated_singleton_noise_lines(
        lines_out, artist=None, title=None
    )
    assert [ln["text"] for ln in lines_out].count("hallelujah") == 4


def test_build_refined_lines_output_removes_repeated_fragment_noise_when_neighbor_has_superset():
    lines_out = [
        {
            "line_index": 1,
            "text": "don't believe me just watch",
            "start": 10.0,
            "end": 11.5,
            "confidence": 0.7,
            "words": [{"text": t} for t in ["don't", "believe", "me", "just", "watch"]],
        },
        {
            "line_index": 2,
            "text": "believe me",
            "start": 11.6,
            "end": 12.0,
            "confidence": 0.25,
            "words": [{"text": "believe"}, {"text": "me"}],
        },
        {
            "line_index": 3,
            "text": "just watch",
            "start": 12.1,
            "end": 12.6,
            "confidence": 0.25,
            "words": [{"text": "just"}, {"text": "watch"}],
        },
        {
            "line_index": 4,
            "text": "don't believe me just watch",
            "start": 20.0,
            "end": 21.5,
            "confidence": 0.7,
            "words": [{"text": t} for t in ["don't", "believe", "me", "just", "watch"]],
        },
        {
            "line_index": 5,
            "text": "believe me",
            "start": 21.6,
            "end": 22.0,
            "confidence": 0.25,
            "words": [{"text": "believe"}, {"text": "me"}],
        },
        {
            "line_index": 6,
            "text": "just watch",
            "start": 22.1,
            "end": 22.6,
            "confidence": 0.25,
            "words": [{"text": "just"}, {"text": "watch"}],
        },
        {
            "line_index": 7,
            "text": "don't believe me just watch",
            "start": 30.0,
            "end": 31.5,
            "confidence": 0.7,
            "words": [{"text": t} for t in ["don't", "believe", "me", "just", "watch"]],
        },
        {
            "line_index": 8,
            "text": "believe me",
            "start": 31.6,
            "end": 32.0,
            "confidence": 0.25,
            "words": [{"text": "believe"}, {"text": "me"}],
        },
        {
            "line_index": 9,
            "text": "just watch",
            "start": 32.1,
            "end": 32.6,
            "confidence": 0.25,
            "words": [{"text": "just"}, {"text": "watch"}],
        },
    ]

    bootstrap_postprocess._remove_repeated_fragment_noise_lines(
        lines_out, artist=None, title=None
    )
    texts = [ln["text"] for ln in lines_out]
    assert texts.count("believe me") == 0
    assert texts.count("just watch") == 0
    assert texts.count("don't believe me just watch") == 3


def test_build_refined_lines_output_keeps_repeated_fragment_without_neighbor_superset():
    lines_out = []
    for i, start in enumerate([10.0, 20.0, 30.0], start=1):
        lines_out.append(
            {
                "line_index": i,
                "text": "just watch",
                "start": start,
                "end": start + 1.0,
                "confidence": 0.25,
                "words": [{"text": "just"}, {"text": "watch"}],
            }
        )
        lines_out.append(
            {
                "line_index": i + 10,
                "text": "come on dance",
                "start": start + 2.0,
                "end": start + 3.0,
                "confidence": 0.7,
                "words": [{"text": "come"}, {"text": "on"}, {"text": "dance"}],
            }
        )

    bootstrap_postprocess._remove_repeated_fragment_noise_lines(
        lines_out, artist=None, title=None
    )
    assert [ln["text"] for ln in lines_out].count("just watch") == 3


def test_repair_strong_local_chronology_inversions_swaps_inverted_repeat_variant():
    lines_out = [
        {
            "line_index": 1,
            "text": "losing sleep",
            "start": 30.0,
            "end": 31.0,
            "confidence": 0.25,
            "words": [{"text": "losing"}, {"text": "sleep"}],
            "_reconstruction_meta": {"uncertainty_score": 0.3},
        },
        {
            "line_index": 2,
            "text": "losing sleep",
            "start": 20.0,
            "end": 21.0,
            "confidence": 0.6,
            "words": [{"text": "losing"}, {"text": "sleep"}],
            "_reconstruction_meta": {"uncertainty_score": 0.05},
        },
    ]

    bootstrap_postprocess._repair_strong_local_chronology_inversions(lines_out)
    assert [ln["start"] for ln in lines_out] == [20.0, 30.0]
    assert [ln["line_index"] for ln in lines_out] == [1, 2]


def test_repair_strong_local_chronology_inversions_keeps_noncomparable_lines():
    lines_out = [
        {
            "line_index": 1,
            "text": "put your hands up",
            "start": 30.0,
            "end": 31.5,
            "confidence": 0.8,
            "words": [{"text": t} for t in ["put", "your", "hands", "up"]],
        },
        {
            "line_index": 2,
            "text": "stop wait a minute",
            "start": 28.0,
            "end": 29.5,
            "confidence": 0.8,
            "words": [{"text": t} for t in ["stop", "wait", "a", "minute"]],
        },
    ]

    bootstrap_postprocess._repair_strong_local_chronology_inversions(lines_out)
    assert [ln["text"] for ln in lines_out] == [
        "put your hands up",
        "stop wait a minute",
    ]


def test_repair_large_adjacent_time_inversions_swaps_large_misordering():
    lines_out = [
        {
            "line_index": 1,
            "text": "dreaming about the thing that",
            "start": 184.95,
            "end": 185.65,
            "confidence": 0.25,
            "words": [
                {"text": t} for t in ["dreaming", "about", "the", "thing", "that"]
            ],
        },
        {
            "line_index": 2,
            "text": "praying hard",
            "start": 165.8,
            "end": 166.8,
            "confidence": 0.0,
            "words": [{"text": "praying"}, {"text": "hard"}],
        },
    ]

    bootstrap_postprocess._repair_large_adjacent_time_inversions(lines_out)
    assert [ln["text"] for ln in lines_out] == [
        "praying hard",
        "dreaming about the thing that",
    ]


def test_repair_large_adjacent_time_inversions_swaps_short_fragment_small_overlap():
    lines_out = [
        {
            "line_index": 1,
            "text": "makes me want to fly",
            "start": 171.8,
            "end": 181.5,
            "confidence": 0.25,
            "words": [{"text": t} for t in ["makes", "me", "want", "to", "fly"]],
        },
        {
            "line_index": 2,
            "text": "we'll be counting stars",
            "start": 171.2,
            "end": 171.9,
            "confidence": 0.23,
            "words": [{"text": t} for t in ["we'll", "be", "counting", "stars"]],
        },
    ]

    bootstrap_postprocess._repair_large_adjacent_time_inversions(lines_out)
    assert [ln["text"] for ln in lines_out] == [
        "we'll be counting stars",
        "makes me want to fly",
    ]


def test_build_refined_lines_output_removes_repeated_chant_noise_lines(monkeypatch):
    monkeypatch.setattr(
        bootstrap_postprocess,
        "_is_spelled_word",
        lambda token: token.lower() in {"watch"},
    )
    lines_out = [
        {
            "line_index": 1,
            "text": "doh doh doh",
            "start": 10.0,
            "end": 10.8,
            "confidence": 0.25,
            "words": [{"text": "doh"}, {"text": "doh"}, {"text": "doh"}],
        },
        {
            "line_index": 2,
            "text": "Dohi doh doh",
            "start": 20.0,
            "end": 20.8,
            "confidence": 0.25,
            "words": [{"text": "Dohi"}, {"text": "doh"}, {"text": "doh"}],
        },
        {
            "line_index": 3,
            "text": "doh doh",
            "start": 30.0,
            "end": 30.6,
            "confidence": 0.25,
            "words": [{"text": "doh"}, {"text": "doh"}],
        },
        {
            "line_index": 4,
            "text": "just watch",
            "start": 31.0,
            "end": 31.8,
            "confidence": 0.7,
            "words": [{"text": "just"}, {"text": "watch"}],
        },
    ]
    bootstrap_postprocess._remove_repeated_chant_noise_lines(lines_out)
    assert [ln["text"] for ln in lines_out] == ["just watch"]


def test_build_refined_lines_output_keeps_repeated_real_word_chants(monkeypatch):
    monkeypatch.setattr(
        bootstrap_postprocess,
        "_is_spelled_word",
        lambda token: token.lower() in {"hey", "watch"},
    )
    lines_out = [
        {
            "line_index": 1,
            "text": "hey hey hey",
            "start": 10.0,
            "end": 10.8,
            "confidence": 0.25,
            "words": [{"text": "hey"}, {"text": "hey"}, {"text": "hey"}],
        },
        {
            "line_index": 2,
            "text": "hey hey",
            "start": 20.0,
            "end": 20.6,
            "confidence": 0.25,
            "words": [{"text": "hey"}, {"text": "hey"}],
        },
        {
            "line_index": 3,
            "text": "hey hey hey",
            "start": 30.0,
            "end": 30.8,
            "confidence": 0.25,
            "words": [{"text": "hey"}, {"text": "hey"}, {"text": "hey"}],
        },
    ]
    bootstrap_postprocess._remove_repeated_chant_noise_lines(lines_out)
    assert len(lines_out) == 3
