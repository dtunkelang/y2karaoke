from y2karaoke.core.visual.word_segmentation import segment_line_tokens_by_visual_gaps


def _w(text: str, x: int, w: int, y: int = 100, h: int = 18) -> dict[str, int | str]:
    return {"text": text, "x": x, "y": y, "w": w, "h": h}


def test_segment_line_tokens_splits_on_large_gaps() -> None:
    ln_w = [
        _w("Oh", 10, 22),
        _w("l", 34, 8),  # tight, same word cluster
        _w("oh", 64, 22),  # large gap => new word
        _w("l", 88, 8),  # tight, same word cluster
    ]
    tokens = segment_line_tokens_by_visual_gaps(ln_w)
    assert tokens == ["Oh", "l", "oh", "l"]


def test_segment_line_tokens_keeps_singletons_with_clear_gaps() -> None:
    ln_w = [
        _w("I'm", 10, 26),
        _w("in", 46, 16),
        _w("love", 76, 30),
        _w("with", 120, 28),
    ]
    tokens = segment_line_tokens_by_visual_gaps(ln_w)
    assert tokens == ["I'm", "in", "love", "with"]


def test_segment_line_tokens_handles_empty_or_single() -> None:
    assert segment_line_tokens_by_visual_gaps([]) == []
    assert segment_line_tokens_by_visual_gaps([_w("Hello", 10, 40)]) == ["Hello"]
