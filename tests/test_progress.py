import types

import y2karaoke.core.progress as progress


def test_render_progress_bar_prints_every_5_percent(capsys):
    bar = progress.RenderProgressBar(total_frames=20)
    bar(lambda t: None, 0)
    captured = capsys.readouterr().out
    assert "5%" in captured
    assert "Rendering:" in captured

    bar(lambda t: None, 0)
    captured = capsys.readouterr().out
    assert "10%" in captured


def test_render_progress_bar_skips_non_5_percent(capsys):
    bar = progress.RenderProgressBar(total_frames=30)
    bar(lambda t: None, 0)
    captured = capsys.readouterr().out
    assert captured == ""


def test_progress_logger_handles_zero_total_frames(capsys):
    logger = progress.ProgressLogger(total_duration=0, fps=24)
    logger.bars_callback(bar=None, attr="index", value=10, old_value=None)
    captured = capsys.readouterr().out
    assert "0%" in captured


def test_progress_logger_updates_when_percent_changes(capsys):
    logger = progress.ProgressLogger(total_duration=1, fps=10)
    logger.bars_callback(bar=None, attr="index", value=1, old_value=None)
    captured = capsys.readouterr().out
    assert "10%" in captured

    logger.bars_callback(bar=None, attr="index", value=1, old_value=None)
    captured = capsys.readouterr().out
    assert captured == ""


def test_progress_logger_ignores_non_index_attr(capsys):
    logger = progress.ProgressLogger(total_duration=1, fps=10)
    logger.bars_callback(bar=None, attr="foo", value=1, old_value=None)
    captured = capsys.readouterr().out
    assert captured == ""


def test_draw_progress_bar_draws_background_and_fill():
    calls = []

    class FakeDraw:
        def rounded_rectangle(self, box, radius, fill):
            calls.append({"box": box, "radius": radius, "fill": fill})

    fake_draw = FakeDraw()
    Colors = types.SimpleNamespace(PROGRESS_BG=(1, 2, 3), PROGRESS_FG=(4, 5, 6))

    progress.draw_progress_bar(
        fake_draw,
        progress=0.5,
        width=100,
        height=50,
        bar_width=40,
        bar_height=10,
        border_radius=2,
        Colors=Colors,
    )

    assert len(calls) == 2
    assert calls[0]["fill"] == Colors.PROGRESS_BG
    assert calls[1]["fill"] == Colors.PROGRESS_FG


def test_draw_progress_bar_skips_fill_when_zero_progress():
    calls = []

    class FakeDraw:
        def rounded_rectangle(self, box, radius, fill):
            calls.append({"box": box, "radius": radius, "fill": fill})

    fake_draw = FakeDraw()
    Colors = types.SimpleNamespace(PROGRESS_BG=(1, 2, 3), PROGRESS_FG=(4, 5, 6))

    progress.draw_progress_bar(
        fake_draw,
        progress=0.0,
        width=100,
        height=50,
        bar_width=40,
        bar_height=10,
        border_radius=2,
        Colors=Colors,
    )

    assert len(calls) == 1
    assert calls[0]["fill"] == Colors.PROGRESS_BG
