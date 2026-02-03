import numpy as np
from PIL import Image, ImageDraw
from unittest.mock import Mock

from y2karaoke.config import INSTRUMENTAL_BREAK_THRESHOLD, SPLASH_DURATION
from y2karaoke.core.frame_renderer import (
    _draw_cue_indicator,
    _draw_line_text,
    _draw_highlight_sweep,
    render_frame,
)
from y2karaoke.core.models import Line, Word


def _line(text, start, end, singer=None):
    word = Word(text=text, start_time=start, end_time=end, singer=singer)
    return Line(words=[word], singer=singer)


def test_draw_cue_indicator_inactive_dots():
    image = Image.new("RGB", (200, 100), "black")
    draw = ImageDraw.Draw(image)

    _draw_cue_indicator(draw, x=100, y=50, time_until_start=0.2, font_size=24)


def test_draw_line_text_uses_singer_color(monkeypatch):
    line = Line(
        words=[
            Word(text="hi", start_time=0.0, end_time=1.0, singer="A"),
            Word(text="there", start_time=1.0, end_time=2.0, singer="A"),
        ],
        singer="A",
    )
    draw = Mock()
    words_with_spaces = ["hi", " ", "there"]
    word_widths = [10, 5, 20]

    monkeypatch.setattr(
        "y2karaoke.core.frame_renderer.get_singer_colors",
        lambda *_: ((1, 2, 3), (4, 5, 6)),
    )

    _draw_line_text(
        draw=draw,
        line=line,
        y=10,
        line_x=5,
        words_with_spaces=words_with_spaces,
        word_widths=word_widths,
        font=Mock(),
        is_duet=True,
    )

    assert any(call.kwargs.get("fill") == (1, 2, 3) for call in draw.text.call_args_list)


def test_draw_highlight_sweep_partial_word(monkeypatch):
    draw = Mock()
    font = Mock()
    font.getbbox.return_value = (0, 0, 10, 10)
    line = Line(words=[Word(text="hi", start_time=0.0, end_time=1.0)])

    _draw_highlight_sweep(
        draw=draw,
        line=line,
        y=10,
        line_x=0,
        total_width=20,
        words_with_spaces=["hi"],
        word_widths=[20],
        font=font,
        highlight_width=10,
        is_duet=False,
    )

    assert any(call.args[1] == "h" for call in draw.text.call_args_list)


def test_render_frame_draws_splash(monkeypatch):
    called = {"splash": False}

    def fake_splash(*args, **kwargs):
        called["splash"] = True

    monkeypatch.setattr("y2karaoke.core.frame_renderer.draw_splash_screen", fake_splash)
    monkeypatch.setattr("y2karaoke.core.frame_renderer.draw_progress_bar", lambda *a, **k: None)

    lines = [_line("hi", start=1.0, end=2.0)]
    background = np.zeros((1080, 1920, 3), dtype=np.uint8)
    font = Mock()
    font.getbbox.return_value = (0, 0, 10, 10)

    render_frame(
        lines,
        current_time=min(0.5, SPLASH_DURATION - 0.1),
        font=font,
        background=background,
        title="Title",
        artist="Artist",
    )

    assert called["splash"] is True


def test_render_frame_draws_progress_bar(monkeypatch):
    called = {"progress": False}

    def fake_progress(*args, **kwargs):
        called["progress"] = True

    monkeypatch.setattr("y2karaoke.core.frame_renderer.draw_progress_bar", fake_progress)
    monkeypatch.setattr("y2karaoke.core.frame_renderer.draw_splash_screen", lambda *a, **k: None)

    start = INSTRUMENTAL_BREAK_THRESHOLD + 5.0
    lines = [_line("hi", start=start, end=start + 1.0)]
    background = np.zeros((1080, 1920, 3), dtype=np.uint8)
    font = Mock()
    font.getbbox.return_value = (0, 0, 10, 10)

    render_frame(
        lines,
        current_time=0.0,
        font=font,
        background=background,
        title=None,
        artist=None,
    )

    assert called["progress"] is True
