from y2karaoke.core.models import TargetLine
from y2karaoke.core.visual import bootstrap_postprocess
from y2karaoke.core.visual.bootstrap_postprocess import build_refined_lines_output


def test_build_refined_lines_output_removes_long_vocalization_noise_lines():
    lines = [
        TargetLine(
            line_index=1,
            start=10.0,
            end=12.0,
            text="real lyric line",
            words=["real", "lyric", "line"],
            y=10,
            word_starts=[10.0, 10.5, 11.0],
            word_ends=[10.4, 10.9, 11.6],
            word_rois=[(0, 0, 1, 1)] * 3,
            word_confidences=[0.4] * 3,
        ),
        TargetLine(
            line_index=2,
            start=12.5,
            end=16.0,
            text="oh oh oh oh oh oh oh oh oh oh oh oh",
            words=["oh"] * 12,
            y=20,
            word_starts=[12.5 + i * 0.2 for i in range(12)],
            word_ends=[12.6 + i * 0.2 for i in range(12)],
            word_rois=[(0, 0, 1, 1)] * 12,
            word_confidences=[0.2] * 12,
        ),
        TargetLine(
            line_index=3,
            start=16.5,
            end=19.0,
            text="mmm mmm mmm mmm mmm mmm mmm mmm mmm mmm",
            words=["mmm"] * 10,
            y=30,
            word_starts=[16.5 + i * 0.2 for i in range(10)],
            word_ends=[16.6 + i * 0.2 for i in range(10)],
            word_rois=[(0, 0, 1, 1)] * 10,
            word_confidences=[0.2] * 10,
        ),
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert len(out) == 1
    assert out[0]["text"] == "real lyric line"


def test_build_refined_lines_output_keeps_short_vocalization_chant():
    lines = [
        TargetLine(
            line_index=1,
            start=10.0,
            end=11.5,
            text="oh oh oh oh",
            words=["oh", "oh", "oh", "oh"],
            y=10,
            word_starts=[10.0, 10.3, 10.6, 10.9],
            word_ends=[10.2, 10.5, 10.8, 11.2],
            word_rois=[(0, 0, 1, 1)] * 4,
            word_confidences=[0.3] * 4,
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert len(out) == 1
    assert out[0]["text"] == "oh oh oh oh"


def test_build_refined_lines_output_removes_short_hum_noise_line():
    lines = [
        TargetLine(
            line_index=1,
            start=10.0,
            end=11.0,
            text="mmm mmm mmm",
            words=["mmm", "mmm", "mmm"],
            y=10,
            word_starts=[10.0, 10.3, 10.6],
            word_ends=[10.2, 10.5, 10.8],
            word_rois=[(0, 0, 1, 1)] * 3,
            word_confidences=[0.2] * 3,
        ),
        TargetLine(
            line_index=2,
            start=11.5,
            end=12.5,
            text="actual line",
            words=["actual", "line"],
            y=20,
            word_starts=[11.5, 12.0],
            word_ends=[11.9, 12.4],
            word_rois=[(0, 0, 1, 1)] * 2,
            word_confidences=[0.4, 0.4],
        ),
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert len(out) == 1
    assert out[0]["text"] == "actual line"


def test_build_refined_lines_output_removes_two_token_hum_noise_line():
    lines = [
        TargetLine(
            line_index=1,
            start=10.0,
            end=10.8,
            text="mmm mmm",
            words=["mmm", "mmm"],
            y=10,
            word_starts=[10.0, 10.4],
            word_ends=[10.2, 10.6],
            word_rois=[(0, 0, 1, 1)] * 2,
            word_confidences=[0.2, 0.2],
        ),
        TargetLine(
            line_index=2,
            start=11.0,
            end=12.0,
            text="actual line",
            words=["actual", "line"],
            y=20,
            word_starts=[11.0, 11.5],
            word_ends=[11.4, 11.9],
            word_rois=[(0, 0, 1, 1)] * 2,
            word_confidences=[0.4, 0.4],
        ),
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert len(out) == 1
    assert out[0]["text"] == "actual line"


def test_build_refined_lines_output_removes_overlay_credit_line_with_url_and_legal_text():
    lines = [
        TargetLine(
            line_index=1,
            start=12.0,
            end=14.0,
            text="www.mrentertainer.co.uk all rights reserved produced by digitop ltd",
            words=[
                "www.mrentertainer.co.uk",
                "all",
                "rights",
                "reserved",
                "produced",
                "by",
                "digitop",
                "ltd",
            ],
            y=10,
            word_starts=[12.0 + 0.2 * i for i in range(8)],
            word_ends=[12.15 + 0.2 * i for i in range(8)],
            word_rois=[(0, 0, 1, 1)] * 8,
            word_confidences=[0.4] * 8,
        ),
        TargetLine(
            line_index=2,
            start=14.5,
            end=16.0,
            text="where are you now",
            words=["where", "are", "you", "now"],
            y=20,
            word_starts=[14.5, 14.8, 15.1, 15.4],
            word_ends=[14.7, 15.0, 15.3, 15.9],
            word_rois=[(0, 0, 1, 1)] * 4,
            word_confidences=[0.8] * 4,
        ),
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [ln["text"] for ln in out] == ["where are you now"]


def test_build_refined_lines_output_removes_social_subscribe_overlay_line():
    lines = [
        TargetLine(
            line_index=1,
            start=20.0,
            end=22.0,
            text="click youtube subscribe karaoke channel follow us facebook twitter",
            words=[
                "click",
                "youtube",
                "subscribe",
                "karaoke",
                "channel",
                "follow",
                "us",
                "facebook",
                "twitter",
            ],
            y=10,
            word_starts=[20.0 + 0.2 * i for i in range(9)],
            word_ends=[20.15 + 0.2 * i for i in range(9)],
            word_rois=[(0, 0, 1, 1)] * 9,
            word_confidences=[0.3] * 9,
        ),
        TargetLine(
            line_index=2,
            start=22.5,
            end=24.0,
            text="we found love",
            words=["we", "found", "love"],
            y=20,
            word_starts=[22.5, 22.9, 23.4],
            word_ends=[22.8, 23.3, 23.9],
            word_rois=[(0, 0, 1, 1)] * 3,
            word_confidences=[0.8] * 3,
        ),
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [ln["text"] for ln in out] == ["we found love"]


def test_build_refined_lines_output_removes_fragmented_youtube_cta_overlay_line():
    lines = [
        TargetLine(
            line_index=1,
            start=20.0,
            end=21.2,
            text="Tube You SUBSCRIBE CLICK TO",
            words=["Tube", "You", "SUBSCRIBE", "CLICK", "TO"],
            y=10,
            word_starts=[20.0, 20.2, 20.45, 20.75, 20.95],
            word_ends=[20.15, 20.38, 20.7, 20.9, 21.15],
            word_rois=[(0, 0, 1, 1)] * 5,
            word_confidences=[0.3] * 5,
        ),
        TargetLine(
            line_index=2,
            start=21.5,
            end=22.7,
            text="counting stars",
            words=["counting", "stars"],
            y=20,
            word_starts=[21.5, 22.0],
            word_ends=[21.9, 22.5],
            word_rois=[(0, 0, 1, 1)] * 2,
            word_confidences=[0.8, 0.8],
        ),
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [ln["text"] for ln in out] == ["counting stars"]


def test_build_refined_lines_output_keeps_normal_lyric_with_follow_word():
    lines = [
        TargetLine(
            line_index=1,
            start=18.0,
            end=20.0,
            text="follow me into the dark",
            words=["follow", "me", "into", "the", "dark"],
            y=10,
            word_starts=[18.0, 18.4, 18.8, 19.1, 19.4],
            word_ends=[18.3, 18.7, 19.0, 19.3, 19.9],
            word_rois=[(0, 0, 1, 1)] * 5,
            word_confidences=[0.8] * 5,
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [ln["text"] for ln in out] == ["follow me into the dark"]


def test_build_refined_lines_output_removes_weaker_near_duplicate_line():
    lines = [
        TargetLine(
            line_index=1,
            start=30.0,
            end=31.8,
            text="dis moi ou est ton papa",
            words=["dis", "moi", "ou", "est", "ton", "papa"],
            y=10,
            word_starts=[30.0, 30.25, 30.5, 30.75, 31.1, 31.35],
            word_ends=[30.2, 30.45, 30.7, 31.0, 31.3, 31.7],
            word_rois=[(0, 0, 1, 1)] * 6,
            word_confidences=[0.85] * 6,
            reconstruction_meta={
                "uncertainty_score": 0.05,
                "selected_text_support_ratio": 0.95,
            },
        ),
        TargetLine(
            line_index=2,
            start=32.0,
            end=33.8,
            text="dis moi o es tu papa",
            words=["dis", "moi", "o", "es", "tu", "papa"],
            y=10,
            word_starts=[32.0, 32.25, 32.5, 32.75, 33.1, 33.35],
            word_ends=[32.2, 32.45, 32.7, 33.0, 33.3, 33.7],
            word_rois=[(0, 0, 1, 1)] * 6,
            word_confidences=[0.25] * 6,
            reconstruction_meta={
                "uncertainty_score": 0.42,
                "selected_text_support_ratio": 0.45,
            },
        ),
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [ln["text"] for ln in out] == ["dis moi ou est ton papa"]


def test_build_refined_lines_output_keeps_legit_repeated_line_when_far_apart():
    lines = [
        TargetLine(
            line_index=1,
            start=30.0,
            end=31.6,
            text="we found love",
            words=["we", "found", "love"],
            y=10,
            word_starts=[30.0, 30.5, 31.0],
            word_ends=[30.4, 30.9, 31.5],
            word_rois=[(0, 0, 1, 1)] * 3,
            word_confidences=[0.8] * 3,
            reconstruction_meta={"uncertainty_score": 0.05},
        ),
        TargetLine(
            line_index=2,
            start=46.0,
            end=47.6,
            text="we found love",
            words=["we", "found", "love"],
            y=10,
            word_starts=[46.0, 46.5, 47.0],
            word_ends=[46.4, 46.9, 47.5],
            word_rois=[(0, 0, 1, 1)] * 3,
            word_confidences=[0.8] * 3,
            reconstruction_meta={"uncertainty_score": 0.05},
        ),
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [ln["text"] for ln in out] == ["we found love", "we found love"]


def test_build_refined_lines_output_canonicalizes_weaker_distant_repeat_variant(
    monkeypatch,
):
    monkeypatch.setattr(
        bootstrap_postprocess,
        "_maybe_repair_output_token",
        lambda text, confidence: text,
    )
    lines = [
        TargetLine(
            line_index=1,
            start=10.0,
            end=12.0,
            text="we'll be counting stars",
            words=["we'll", "be", "counting", "stars"],
            y=10,
            word_starts=[10.0, 10.4, 10.8, 11.4],
            word_ends=[10.3, 10.7, 11.3, 11.9],
            word_rois=[(0, 0, 1, 1)] * 4,
            word_confidences=[0.85] * 4,
            reconstruction_meta={
                "uncertainty_score": 0.05,
                "selected_text_support_ratio": 0.95,
            },
        ),
        TargetLine(
            line_index=2,
            start=20.0,
            end=21.0,
            text="bridge line",
            words=["bridge", "line"],
            y=20,
            word_starts=[20.0, 20.5],
            word_ends=[20.4, 20.9],
            word_rois=[(0, 0, 1, 1)] * 2,
            word_confidences=[0.8, 0.8],
        ),
        TargetLine(
            line_index=3,
            start=34.0,
            end=36.0,
            text="we'll be counting starz",
            words=["we'll", "be", "counting", "starz"],
            y=10,
            word_starts=[34.0, 34.3, 34.8, 35.35],
            word_ends=[34.2, 34.5, 35.25, 35.8],
            word_rois=[(0, 0, 1, 1)] * 4,
            word_confidences=[0.25] * 4,
            reconstruction_meta={
                "uncertainty_score": 0.4,
                "selected_text_support_ratio": 0.4,
            },
        ),
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [ln["text"] for ln in out if "count" in ln["text"]] == [
        "we'll be counting stars",
        "we'll be counting stars",
    ]
