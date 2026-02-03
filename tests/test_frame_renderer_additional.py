import numpy as np
from PIL import Image, ImageDraw
from unittest.mock import Mock

from y2karaoke.config import INSTRUMENTAL_BREAK_THRESHOLD, SPLASH_DURATION
from y2karaoke.core.frame_renderer import (
    _draw_cue_indicator,
    _check_mid_song_progress,
    _get_lines_to_display,
    _check_cue_indicator,
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
    before = np.array(image)
    draw = ImageDraw.Draw(image)

    _draw_cue_indicator(draw, x=100, y=50, time_until_start=0.2, font_size=24)
    after = np.array(image)
    assert np.any(after != before)


def test_draw_cue_indicator_multiple_active_dots():
    image = Image.new("RGB", (200, 100), "black")
    before = np.array(image)
    draw = ImageDraw.Draw(image)

    _draw_cue_indicator(draw, x=100, y=50, time_until_start=1.2, font_size=24)
    after = np.array(image)
    assert np.any(after != before)


def test_check_mid_song_progress_no_break():
    lines = [
        _line("a", 1.0, 2.0),
        _line("b", 3.0, 4.0),
    ]
    show, progress = _check_mid_song_progress(lines, 0, 2.5)
    assert show is False
    assert progress == 0.0


def test_get_lines_to_display_skips_to_next_after_break():
    lines = [
        _line("a", 0.0, 1.0),
        _line("b", 20.0, 21.0),
        _line("c", 22.0, 23.0),
    ]
    lines_to_show, start_idx = _get_lines_to_display(
        lines, current_line_idx=0, current_time=19.2, activation_time=19.2
    )
    assert start_idx == 1
    assert lines_to_show[0][0].text == "b"


def test_get_lines_to_display_breaks_before_future_section():
    lines = [
        _line("a", 0.0, 1.0),
        _line("b", 2.0, 3.0),
        _line("c", 20.0, 21.0),
        _line("d", 22.0, 23.0),
    ]
    lines_to_show, _ = _get_lines_to_display(
        lines, current_line_idx=1, current_time=19.2, activation_time=19.2
    )
    assert len(lines_to_show) == 2


def test_check_cue_indicator_with_previous_gap():
    lines = [
        _line("a", 0.0, 1.0),
        _line("b", 10.0, 11.0),
    ]
    lines_to_show = [(lines[1], False)]
    show, time_until = _check_cue_indicator(
        lines, lines_to_show, display_start_idx=1, current_time=8.5
    )
    assert show is True
    assert time_until > 0


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

    assert any(
        call.kwargs.get("fill") == (1, 2, 3) for call in draw.text.call_args_list
    )


def test_draw_line_text_uses_default_color():
    line = Line(words=[Word(text="hi", start_time=0.0, end_time=1.0, singer=None)])
    draw = Mock()
    words_with_spaces = ["hi"]
    word_widths = [10]

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

    assert any(
        call.kwargs.get("fill") == (255, 255, 255) for call in draw.text.call_args_list
    )


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
    assert draw.text.call_count > 0


def test_draw_highlight_sweep_no_highlight():
    line = Line(words=[Word(text="hi", start_time=0.0, end_time=1.0)])
    draw = Mock()

    _draw_highlight_sweep(
        draw=draw,
        line=line,
        y=10,
        line_x=5,
        total_width=20,
        words_with_spaces=["hi"],
        word_widths=[20],
        font=Mock(),
        highlight_width=0,
        is_duet=False,
    )

    assert draw.text.call_count == 0


def test_render_frame_draws_cue_indicator(monkeypatch):
    from PIL import ImageFont

    called = {"cue": False}

    def fake_cue(*_args, **_kwargs):
        called["cue"] = True

    monkeypatch.setattr("y2karaoke.core.frame_renderer._draw_cue_indicator", fake_cue)

    font = ImageFont.load_default()
    background = np.zeros((100, 200, 3), dtype=np.uint8)
    line = Line(words=[Word(text="hi", start_time=5.0, end_time=6.0)])

    render_frame(
        lines=[line],
        current_time=3.0,
        font=font,
        background=background,
        title=None,
        artist=None,
    )

    assert called["cue"] is True


def test_render_frame_zero_duration_highlight(monkeypatch):
    from PIL import ImageFont

    captured = {}

    def fake_highlight(*args, **_kwargs):
        captured["total_width"] = args[4]
        captured["highlight_width"] = args[8]

    monkeypatch.setattr(
        "y2karaoke.core.frame_renderer._draw_highlight_sweep", fake_highlight
    )

    font = ImageFont.load_default()
    background = np.zeros((100, 200, 3), dtype=np.uint8)
    line = Line(words=[Word(text="hi", start_time=1.0, end_time=1.0)])

    render_frame(
        lines=[line],
        current_time=0.9,
        font=font,
        background=background,
        title=None,
        artist=None,
    )

    assert captured["highlight_width"] == captured["total_width"]


def test_render_frame_draws_splash(monkeypatch):
    called = {"splash": False}

    def fake_splash(*args, **kwargs):
        called["splash"] = True

    monkeypatch.setattr("y2karaoke.core.frame_renderer.draw_splash_screen", fake_splash)
    monkeypatch.setattr(
        "y2karaoke.core.frame_renderer.draw_progress_bar", lambda *a, **k: None
    )

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

    monkeypatch.setattr(
        "y2karaoke.core.frame_renderer.draw_progress_bar", fake_progress
    )
    monkeypatch.setattr(
        "y2karaoke.core.frame_renderer.draw_splash_screen", lambda *a, **k: None
    )

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
