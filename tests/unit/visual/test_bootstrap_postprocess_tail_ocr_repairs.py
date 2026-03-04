from y2karaoke.core.models import TargetLine
from y2karaoke.core.visual import bootstrap_postprocess
from y2karaoke.core.visual.bootstrap_postprocess import build_refined_lines_output


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
