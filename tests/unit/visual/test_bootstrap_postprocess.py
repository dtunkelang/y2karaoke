from y2karaoke.core.models import TargetLine
from y2karaoke.core.visual import bootstrap_postprocess
from y2karaoke.core.visual.bootstrap_postprocess import (
    build_refined_lines_output,
    nearest_known_word_indices,
)


def test_nearest_known_word_indices_mapping():
    prev_known, next_known = nearest_known_word_indices([1, 4], 6)
    assert prev_known == [-1, 1, 1, 1, 4, 4]
    assert next_known == [1, 1, 4, 4, 4, -1]


def test_build_refined_lines_output_filters_title_artist():
    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=10.0,
            text="Song Title",
            words=["Song", "Title"],
            y=50,
            word_starts=[8.1, 8.7],
            word_ends=[8.5, 9.2],
            word_rois=[(0, 0, 1, 1), (1, 0, 1, 1)],
        ),
        TargetLine(
            line_index=2,
            start=11.0,
            end=13.0,
            text="real lyric",
            words=["real", "lyric"],
            y=60,
            word_starts=[11.1, 11.8],
            word_ends=[11.5, 12.4],
            word_rois=[(0, 0, 1, 1), (1, 0, 1, 1)],
        ),
    ]
    out = build_refined_lines_output(lines, artist="Artist Name", title="Song Title")
    assert len(out) == 1
    assert out[0]["text"] == "real lyric"


def test_build_refined_lines_output_splits_fused_word_tokens():
    lines = [
        TargetLine(
            line_index=1,
            start=45.0,
            end=46.2,
            text="What I want",
            words=["What", "Iwant"],
            y=15,
            word_starts=[45.05, 45.6],
            word_ends=[45.55, 46.15],
            word_rois=[(0, 0, 1, 1), (1, 0, 1, 1)],
            word_confidences=[0.25, 0.25],
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert len(out) == 1
    assert [w["text"] for w in out[0]["words"]] == ["What", "I", "want"]


def test_build_refined_lines_output_delays_short_interstitial_line():
    lines = [
        TargetLine(
            line_index=1,
            start=40.7,
            end=43.1,
            text="Don't say thank you or please",
            words=["Don't", "say", "thank", "you", "or", "please"],
            y=10,
            word_starts=[40.7, 41.1, 41.4, 41.9, 42.4, 42.85],
            word_ends=[41.0, 41.35, 41.75, 42.25, 42.75, 43.1],
            word_rois=[(0, 0, 1, 1)] * 6,
            word_confidences=[0.25] * 6,
        ),
        TargetLine(
            line_index=2,
            start=43.3,
            end=44.1,
            text="I do",
            words=["I", "do"],
            y=20,
            word_starts=[43.3, 43.7],
            word_ends=[43.65, 44.1],
            word_rois=[(0, 0, 1, 1)] * 2,
            word_confidences=[0.25] * 2,
        ),
        TargetLine(
            line_index=3,
            start=45.05,
            end=46.15,
            text="What I want",
            words=["What", "I", "want"],
            y=30,
            word_starts=[45.05, 45.45, 45.8],
            word_ends=[45.35, 45.75, 46.15],
            word_rois=[(0, 0, 1, 1)] * 3,
            word_confidences=[0.25] * 3,
        ),
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert len(out) == 3
    assert out[1]["text"] == "I do"
    assert out[1]["start"] >= 43.7
    assert out[1]["end"] <= 44.95


def test_build_refined_lines_output_rebalances_compressed_middle_four_line_sequence():
    lines = [
        TargetLine(
            line_index=1,
            start=51.2,
            end=52.15,
            text="So you're a tough guy",
            words=["So", "you're", "a", "tough", "guy"],
            y=10,
            word_starts=[51.2, 51.4, 51.55, 51.75, 51.95],
            word_ends=[51.35, 51.52, 51.7, 51.9, 52.15],
            word_rois=[(0, 0, 1, 1)] * 5,
            word_confidences=[0.25] * 5,
        ),
        TargetLine(
            line_index=2,
            start=52.15,
            end=52.85,
            text="Like it really rough guy",
            words=["Like", "it", "really", "rough", "guy"],
            y=20,
            word_starts=[52.15, 52.3, 52.4, 52.55, 52.7],
            word_ends=[52.27, 52.38, 52.53, 52.68, 52.85],
            word_rois=[(0, 0, 1, 1)] * 5,
            word_confidences=[0.25] * 5,
        ),
        TargetLine(
            line_index=3,
            start=52.85,
            end=53.9,
            text="Just can't get enough guy",
            words=["Just", "can't", "get", "enough", "guy"],
            y=30,
            word_starts=[52.85, 53.0, 53.2, 53.4, 53.65],
            word_ends=[52.97, 53.15, 53.35, 53.58, 53.9],
            word_rois=[(0, 0, 1, 1)] * 5,
            word_confidences=[0.25] * 5,
        ),
        TargetLine(
            line_index=4,
            start=55.5,
            end=56.95,
            text="Chest always so puffed guy",
            words=["Chest", "always", "so", "puffed", "guy"],
            y=40,
            word_starts=[55.5, 55.8, 56.1, 56.4, 56.7],
            word_ends=[55.75, 56.0, 56.25, 56.55, 56.95],
            word_rois=[(0, 0, 1, 1)] * 5,
            word_confidences=[0.25] * 5,
        ),
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert len(out) == 4
    assert out[1]["start"] >= 52.5
    assert out[2]["start"] >= 54.0


def test_build_refined_lines_output_uncertainty_gated_fallback_splits_fused_tokens(
    monkeypatch,
):
    monkeypatch.setattr(bootstrap_postprocess, "normalize_ocr_line", lambda s: s)
    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=9.2,
            text="I sleep alone",
            words=["Isleep", "alone"],
            y=10,
            word_starts=[8.0, 8.7],
            word_ends=[8.6, 9.2],
            word_rois=[(0, 0, 1, 1)] * 2,
            word_confidences=[0.4, 0.4],
            reconstruction_meta={
                "uncertainty_score": 0.35,
                "selected_text_support_ratio": 0.5,
                "text_variant_count": 5,
                "weak_vote_positions": 2,
            },
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [w["text"] for w in out[0]["words"]] == ["I", "sleep", "alone"]
    assert out[0]["text"] == "I sleep alone"


def test_build_refined_lines_output_does_not_fallback_split_when_confident(monkeypatch):
    monkeypatch.setattr(bootstrap_postprocess, "normalize_ocr_line", lambda s: s)
    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=9.2,
            text="I sleep alone",
            words=["Isleep", "alone"],
            y=10,
            word_starts=[8.0, 8.7],
            word_ends=[8.6, 9.2],
            word_rois=[(0, 0, 1, 1)] * 2,
            word_confidences=[0.4, 0.4],
            reconstruction_meta={
                "uncertainty_score": 0.02,
                "selected_text_support_ratio": 0.95,
                "text_variant_count": 1,
                "weak_vote_positions": 0,
            },
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [w["text"] for w in out[0]["words"]] == ["Isleep", "alone"]
    assert out[0]["text"] == "Isleep alone"


def test_build_refined_lines_output_fallback_does_not_split_normal_anchor_words(
    monkeypatch,
):
    monkeypatch.setattr(bootstrap_postprocess, "normalize_ocr_line", lambda s: s)
    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=9.2,
            text="in a foreign field",
            words=["foreign", "field"],
            y=10,
            word_starts=[8.0, 8.7],
            word_ends=[8.6, 9.2],
            word_rois=[(0, 0, 1, 1)] * 2,
            word_confidences=[0.4, 0.4],
            reconstruction_meta={"uncertainty_score": 0.4},
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [w["text"] for w in out[0]["words"]] == ["foreign", "field"]
    assert out[0]["text"] == "foreign field"


def test_build_refined_lines_output_fallback_splits_short_function_and_trailing_i(
    monkeypatch,
):
    monkeypatch.setattr(bootstrap_postprocess, "normalize_ocr_line", lambda s: s)
    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=10.0,
            text="test",
            words=["ina", "reasonI"],
            y=10,
            word_starts=[8.0, 8.8],
            word_ends=[8.7, 9.8],
            word_rois=[(0, 0, 1, 1)] * 2,
            word_confidences=[0.4, 0.4],
            reconstruction_meta={"uncertainty_score": 0.4},
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [w["text"] for w in out[0]["words"]] == ["in", "a", "reason", "I"]
    assert out[0]["text"] == "in a reason I"


def test_build_refined_lines_output_fallback_splits_suffix_and_nested_anchor(
    monkeypatch,
):
    monkeypatch.setattr(bootstrap_postprocess, "normalize_ocr_line", lambda s: s)
    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=10.0,
            text="test",
            words=["fearin", "bellsaringin"],
            y=10,
            word_starts=[8.0, 8.8],
            word_ends=[8.7, 9.8],
            word_rois=[(0, 0, 1, 1)] * 2,
            word_confidences=[0.4, 0.4],
            reconstruction_meta={"uncertainty_score": 0.4},
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [w["text"] for w in out[0]["words"]] == [
        "fear",
        "in",
        "bells",
        "a",
        "ringing",
    ]
    assert out[0]["text"] == "fear in bells a ringing"


def test_build_refined_lines_output_fallback_does_not_over_split_lexical_words(
    monkeypatch,
):
    monkeypatch.setattr(bootstrap_postprocess, "normalize_ocr_line", lambda s: s)
    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=10.0,
            text="test",
            words=["missionaries", "explain"],
            y=10,
            word_starts=[8.0, 8.8],
            word_ends=[8.7, 9.8],
            word_rois=[(0, 0, 1, 1)] * 2,
            word_confidences=[0.4, 0.4],
            reconstruction_meta={"uncertainty_score": 0.4},
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [w["text"] for w in out[0]["words"]] == ["missionaries", "explain"]
    assert out[0]["text"] == "missionaries explain"


def test_build_refined_lines_output_fallback_spell_validated_split(monkeypatch):
    monkeypatch.setattr(bootstrap_postprocess, "normalize_ocr_line", lambda s: s)

    def fake_is_spelled(token: str) -> bool:
        return token.lower() in {"long", "live"}

    monkeypatch.setattr(bootstrap_postprocess, "_is_spelled_word", fake_is_spelled)
    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=10.0,
            text="test",
            words=["longlive"],
            y=10,
            word_starts=[8.0],
            word_ends=[9.8],
            word_rois=[(0, 0, 1, 1)],
            word_confidences=[0.4],
            reconstruction_meta={"uncertainty_score": 0.4},
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [w["text"] for w in out[0]["words"]] == ["long", "live"]
    assert out[0]["text"] == "long live"


def test_build_refined_lines_output_spell_validated_split_without_uncertainty(
    monkeypatch,
):
    monkeypatch.setattr(bootstrap_postprocess, "normalize_ocr_line", lambda s: s)

    def fake_is_spelled(token: str) -> bool:
        return token.lower() in {"long", "live"}

    monkeypatch.setattr(bootstrap_postprocess, "_is_spelled_word", fake_is_spelled)
    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=10.0,
            text="test",
            words=["longlive"],
            y=10,
            word_starts=[8.0],
            word_ends=[9.8],
            word_rois=[(0, 0, 1, 1)],
            word_confidences=[0.4],
            reconstruction_meta={
                "uncertainty_score": 0.01,
                "selected_text_support_ratio": 0.95,
                "text_variant_count": 1,
                "weak_vote_positions": 0,
            },
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [w["text"] for w in out[0]["words"]] == ["long", "live"]
    assert out[0]["text"] == "long live"


def test_build_refined_lines_output_token_level_ocr_fallback_without_uncertainty(
    monkeypatch,
):
    monkeypatch.setattr(bootstrap_postprocess, "normalize_ocr_line", lambda s: s)

    def fake_is_spelled(token: str) -> bool:
        return token.lower() in {"bells", "ringing", "explain"}

    monkeypatch.setattr(bootstrap_postprocess, "_is_spelled_word", fake_is_spelled)
    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=10.0,
            text="test",
            words=["bellsaringin", "explain"],
            y=10,
            word_starts=[8.0, 8.8],
            word_ends=[8.7, 9.8],
            word_rois=[(0, 0, 1, 1)] * 2,
            word_confidences=[0.4, 0.4],
            reconstruction_meta={
                "uncertainty_score": 0.01,
                "selected_text_support_ratio": 0.95,
                "text_variant_count": 1,
                "weak_vote_positions": 0,
            },
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [w["text"] for w in out[0]["words"]] == ["bells", "a", "ringing", "explain"]
    assert out[0]["text"] == "bells a ringing explain"


def test_build_refined_lines_output_spell_guess_repairs_invalid_low_conf_token(
    monkeypatch,
):
    monkeypatch.setattr(bootstrap_postprocess, "normalize_ocr_line", lambda s: s)

    def fake_is_spelled(token: str) -> bool:
        return token.lower() in {"would"}

    monkeypatch.setattr(bootstrap_postprocess, "_is_spelled_word", fake_is_spelled)
    monkeypatch.setattr(
        bootstrap_postprocess,
        "_fallback_spell_guess",
        lambda token: "would" if token.lower() == "yould" else None,
    )
    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=10.0,
            text="test",
            words=["yould"],
            y=10,
            word_starts=[8.0],
            word_ends=[9.8],
            word_rois=[(0, 0, 1, 1)],
            word_confidences=[0.3],
            reconstruction_meta={"uncertainty_score": 0.01},
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [w["text"] for w in out[0]["words"]] == ["would"]
    assert out[0]["text"] == "would"


def test_build_refined_lines_output_spell_guess_skips_valid_tokens(monkeypatch):
    monkeypatch.setattr(bootstrap_postprocess, "normalize_ocr_line", lambda s: s)
    monkeypatch.setattr(
        bootstrap_postprocess, "_maybe_expand_colloquial_token", lambda text, conf: None
    )

    def fake_is_spelled(token: str) -> bool:
        return token.lower() in {"wanna"}

    monkeypatch.setattr(bootstrap_postprocess, "_is_spelled_word", fake_is_spelled)
    monkeypatch.setattr(
        bootstrap_postprocess, "_fallback_spell_guess", lambda token: "want"
    )
    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=10.0,
            text="test",
            words=["wanna"],
            y=10,
            word_starts=[8.0],
            word_ends=[9.8],
            word_rois=[(0, 0, 1, 1)],
            word_confidences=[0.3],
            reconstruction_meta={"uncertainty_score": 0.4},
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [w["text"] for w in out[0]["words"]] == ["wanna"]
    assert out[0]["text"] == "wanna"


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


def test_build_refined_lines_output_contextual_compound_split(monkeypatch):
    monkeypatch.setattr(bootstrap_postprocess, "normalize_ocr_line", lambda s: s)

    def fake_is_spelled(token: str) -> bool:
        return token.lower() in {"longlive", "long", "live", "the", "king"}

    monkeypatch.setattr(bootstrap_postprocess, "_is_spelled_word", fake_is_spelled)
    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=10.0,
            text="test",
            words=["Longlive", "the", "king"],
            y=10,
            word_starts=[8.0, 8.8, 9.2],
            word_ends=[8.7, 9.1, 9.8],
            word_rois=[(0, 0, 1, 1)] * 3,
            word_confidences=[0.5, 0.5, 0.5],
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [w["text"] for w in out[0]["words"]] == ["Long", "live", "the", "king"]
    assert out[0]["text"] == "Long live the king"


def test_build_refined_lines_output_contextual_compound_split_does_not_split_nonfunction_followup(
    monkeypatch,
):
    monkeypatch.setattr(bootstrap_postprocess, "normalize_ocr_line", lambda s: s)

    def fake_is_spelled(token: str) -> bool:
        return token.lower() in {"somebody", "some", "body", "knows"}

    monkeypatch.setattr(bootstrap_postprocess, "_is_spelled_word", fake_is_spelled)
    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=10.0,
            text="test",
            words=["Somebody", "knows"],
            y=10,
            word_starts=[8.0, 8.8],
            word_ends=[8.7, 9.8],
            word_rois=[(0, 0, 1, 1)] * 2,
            word_confidences=[0.5, 0.5],
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [w["text"] for w in out[0]["words"]] == ["Somebody", "knows"]
    assert out[0]["text"] == "Somebody knows"


def test_build_refined_lines_output_ocr_substitution_repairs_youid(monkeypatch):
    monkeypatch.setattr(bootstrap_postprocess, "normalize_ocr_line", lambda s: s)

    def fake_is_spelled(token: str) -> bool:
        return token.lower() in {"would"}

    monkeypatch.setattr(bootstrap_postprocess, "_is_spelled_word", fake_is_spelled)
    monkeypatch.setattr(
        bootstrap_postprocess, "_fallback_spell_guess", lambda token: None
    )
    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=10.0,
            text="test",
            words=["youid"],
            y=10,
            word_starts=[8.0],
            word_ends=[9.8],
            word_rois=[(0, 0, 1, 1)],
            word_confidences=[0.4],
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [w["text"] for w in out[0]["words"]] == ["would"]
    assert out[0]["text"] == "would"


def test_build_refined_lines_output_ocr_substitution_skips_valid_token(monkeypatch):
    monkeypatch.setattr(bootstrap_postprocess, "normalize_ocr_line", lambda s: s)
    monkeypatch.setattr(
        bootstrap_postprocess, "_maybe_expand_colloquial_token", lambda text, conf: None
    )

    def fake_is_spelled(token: str) -> bool:
        return token.lower() in {"wanna", "would"}

    monkeypatch.setattr(bootstrap_postprocess, "_is_spelled_word", fake_is_spelled)
    monkeypatch.setattr(
        bootstrap_postprocess, "_fallback_spell_guess", lambda token: None
    )
    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=10.0,
            text="test",
            words=["wanna"],
            y=10,
            word_starts=[8.0],
            word_ends=[9.8],
            word_rois=[(0, 0, 1, 1)],
            word_confidences=[0.4],
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [w["text"] for w in out[0]["words"]] == ["wanna"]
    assert out[0]["text"] == "wanna"


def test_build_refined_lines_output_expands_colloquial_wanna_low_conf(monkeypatch):
    monkeypatch.setattr(bootstrap_postprocess, "normalize_ocr_line", lambda s: s)
    monkeypatch.setattr(bootstrap_postprocess, "_is_spelled_word", lambda token: True)

    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=10.0,
            text="test",
            words=["wanna", "be"],
            y=10,
            word_starts=[8.0, 8.8],
            word_ends=[8.7, 9.8],
            word_rois=[(0, 0, 1, 1)] * 2,
            word_confidences=[0.4, 0.4],
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [w["text"] for w in out[0]["words"]] == ["want", "to", "be"]
    assert out[0]["text"] == "want to be"


def test_build_refined_lines_output_normalizes_dropped_g_gerund_low_conf(monkeypatch):
    monkeypatch.setattr(bootstrap_postprocess, "normalize_ocr_line", lambda s: s)
    monkeypatch.setattr(bootstrap_postprocess, "_is_spelled_word", lambda token: True)
    words = [
        {
            "word_index": 1,
            "text": "singin'",
            "start": 28.0,
            "end": 28.8,
            "confidence": 0.4,
        },
        {
            "word_index": 2,
            "text": "loud",
            "start": 29.0,
            "end": 29.8,
            "confidence": 0.4,
        },
    ]
    out_words = bootstrap_postprocess._split_fused_output_words(words)
    assert [w["text"] for w in out_words] == ["singing", "loud"]


def test_build_refined_lines_output_restores_common_contraction_low_conf(monkeypatch):
    monkeypatch.setattr(bootstrap_postprocess, "normalize_ocr_line", lambda s: s)

    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=10.0,
            text="test",
            words=["wont", "call"],
            y=10,
            word_starts=[8.0, 8.8],
            word_ends=[8.7, 9.8],
            word_rois=[(0, 0, 1, 1)] * 2,
            word_confidences=[0.4, 0.4],
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [w["text"] for w in out[0]["words"]] == ["won't", "call"]
    assert out[0]["text"] == "won't call"


def test_build_refined_lines_output_contextual_plural_inflection_low_conf(monkeypatch):
    monkeypatch.setattr(bootstrap_postprocess, "normalize_ocr_line", lambda s: s)

    def fake_is_spelled(token: str) -> bool:
        return token.lower() in {"wall", "walls", "were", "closed"}

    monkeypatch.setattr(bootstrap_postprocess, "_is_spelled_word", fake_is_spelled)
    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=10.0,
            text="test",
            words=["wall", "were", "closed"],
            y=10,
            word_starts=[8.0, 8.6, 9.1],
            word_ends=[8.5, 9.0, 9.8],
            word_rois=[(0, 0, 1, 1)] * 3,
            word_confidences=[0.4, 0.4, 0.4],
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [w["text"] for w in out[0]["words"]] == ["walls", "were", "closed"]
    assert out[0]["text"] == "walls were closed"


def test_build_refined_lines_output_ocr_insertion_repairs_ble(monkeypatch):
    monkeypatch.setattr(bootstrap_postprocess, "normalize_ocr_line", lambda s: s)

    def fake_is_spelled(token: str) -> bool:
        return token.lower() in {"blew"}

    monkeypatch.setattr(bootstrap_postprocess, "_is_spelled_word", fake_is_spelled)
    monkeypatch.setattr(
        bootstrap_postprocess, "_fallback_spell_guess", lambda token: None
    )
    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=10.0,
            text="test",
            words=["ble"],
            y=10,
            word_starts=[8.0],
            word_ends=[9.8],
            word_rois=[(0, 0, 1, 1)],
            word_confidences=[0.4],
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [w["text"] for w in out[0]["words"]] == ["blew"]
    assert out[0]["text"] == "blew"


def test_build_refined_lines_output_contextual_gerund_and_interjection_and_name_prefix(
    monkeypatch,
):
    monkeypatch.setattr(bootstrap_postprocess, "normalize_ocr_line", lambda s: s)

    def fake_is_spelled(token: str) -> bool:
        return token.lower() in {
            "are",
            "singin",
            "singing",
            "oh",
            "who",
            "saint",
            "peter",
        }

    monkeypatch.setattr(bootstrap_postprocess, "_is_spelled_word", fake_is_spelled)
    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=10.0,
            text="test",
            words=["Aw", "who", "are", "singin", "Saint", "eter"],
            y=10,
            word_starts=[8.0, 8.3, 8.8, 9.1, 9.4, 9.7],
            word_ends=[8.2, 8.6, 9.0, 9.3, 9.6, 9.9],
            word_rois=[(0, 0, 1, 1)] * 6,
            word_confidences=[0.4] * 6,
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [w["text"] for w in out[0]["words"]] == [
        "Oh",
        "who",
        "are",
        "singing",
        "Saint",
        "peter",
    ]


def test_build_refined_lines_output_removes_long_vocalization_noise_run_split_across_lines():
    lines = [
        TargetLine(
            line_index=1,
            start=10.0,
            end=11.0,
            text="oh oh oh oh oh oh",
            words=["oh"] * 6,
            y=10,
            word_starts=[10.0 + i * 0.1 for i in range(6)],
            word_ends=[10.05 + i * 0.1 for i in range(6)],
            word_rois=[(0, 0, 1, 1)] * 6,
            word_confidences=[0.2] * 6,
        ),
        TargetLine(
            line_index=2,
            start=11.1,
            end=12.0,
            text="oh oh oh oh",
            words=["oh"] * 4,
            y=20,
            word_starts=[11.1 + i * 0.1 for i in range(4)],
            word_ends=[11.15 + i * 0.1 for i in range(4)],
            word_rois=[(0, 0, 1, 1)] * 4,
            word_confidences=[0.2] * 4,
        ),
        TargetLine(
            line_index=3,
            start=12.5,
            end=13.5,
            text="actual line",
            words=["actual", "line"],
            y=30,
            word_starts=[12.5, 13.0],
            word_ends=[12.9, 13.4],
            word_rois=[(0, 0, 1, 1)] * 2,
            word_confidences=[0.4, 0.4],
        ),
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert len(out) == 1
    assert out[0]["text"] == "actual line"


def test_build_refined_lines_output_fallback_spell_validated_respects_valid_token(
    monkeypatch,
):
    monkeypatch.setattr(bootstrap_postprocess, "normalize_ocr_line", lambda s: s)

    def fake_is_spelled(token: str) -> bool:
        return token.lower() in {"missionaries", "mission", "aries"}

    monkeypatch.setattr(bootstrap_postprocess, "_is_spelled_word", fake_is_spelled)
    lines = [
        TargetLine(
            line_index=1,
            start=8.0,
            end=10.0,
            text="test",
            words=["missionaries"],
            y=10,
            word_starts=[8.0],
            word_ends=[9.8],
            word_rois=[(0, 0, 1, 1)],
            word_confidences=[0.4],
            reconstruction_meta={"uncertainty_score": 0.4},
        )
    ]

    out = build_refined_lines_output(lines, artist=None, title=None)
    assert [w["text"] for w in out[0]["words"]] == ["missionaries"]
    assert out[0]["text"] == "missionaries"
