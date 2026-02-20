import types

import y2karaoke.core.components.render.progress as progress


def test_console_progress_bar_prints_updates(capsys):
    bar = progress.ConsoleProgressBar(total=100)

    # 1%
    bar.update()
    captured = capsys.readouterr().out
    assert captured == ""  # 2% threshold

    # 2%
    bar.update()
    captured = capsys.readouterr().out
    assert "2%" in captured
    assert "Rendering:" in captured


def test_console_progress_bar_handles_zero_total(capsys):
    bar = progress.ConsoleProgressBar(total=0)
    bar.update()
    captured = capsys.readouterr().out
    assert "0%" in captured


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
