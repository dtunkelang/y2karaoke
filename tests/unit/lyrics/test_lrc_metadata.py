import pytest

from y2karaoke.core import lrc
from y2karaoke.core.models import Line, Word


def test_has_metadata_keyword_detects_colon_name():
    assert lrc._has_metadata_keyword("Composer: John Doe") is True


def test_is_promo_line_detects_patterns_and_phone():
    assert lrc._is_promo_line("Download mp3 now") is True
    assert lrc._is_promo_line("Call +33 6 12 34 56 78") is True
    assert lrc._is_promo_line("") is False


def test_is_promo_like_title_line_repeated_title():
    assert lrc._is_promo_like_title_line("Hello hello", "hello") is True
    assert lrc._is_promo_like_title_line("just a line", "hello") is False


def test_is_promo_like_title_line_allows_title_start():
    text = "Aucun Express, je reviens encore."
    assert lrc._is_promo_like_title_line(text, "Aucun Express") is False


def test_create_lines_from_lrc_timings_caps_long_gap():
    lines = lrc.create_lines_from_lrc_timings(
        lrc_timings=[(0.0, "hello"), (20.0, "world")],
        genius_lines=["hello", "world"],
    )
    assert len(lines) == 2
    assert lines[0].words[-1].end_time == pytest.approx(4.75)


def test_create_lines_from_lrc_timings_skips_empty_text():
    lines = lrc.create_lines_from_lrc_timings(
        lrc_timings=[(0.0, ""), (1.0, "hello")],
        genius_lines=["", "hello"],
    )
    assert len(lines) == 1
    assert lines[0].text == "hello"


def test_split_long_lines_mid_zero_keeps_line():
    long_word = "x" * 100
    line = Line(words=[Word(text=long_word, start_time=0.0, end_time=1.0)])
    result = lrc.split_long_lines([line], max_width_ratio=0.2)
    assert result == [line]
