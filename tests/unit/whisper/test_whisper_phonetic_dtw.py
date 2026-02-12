from y2karaoke.core.components.whisper import whisper_phonetic_dtw as wpd
from y2karaoke.core.models import Line, Word


def test_extract_lrc_words_all_skips_nonlexical_adlibs():
    lines = [
        Line(
            words=[
                Word("mm-mm", start_time=0.0, end_time=0.2),
                Word("(oh)", start_time=0.2, end_time=0.4),
                Word("love", start_time=0.4, end_time=0.8),
                Word("go", start_time=0.8, end_time=1.0),
            ]
        )
    ]

    words = wpd._extract_lrc_words_all(lines)

    assert [w["text"] for w in words] == ["love", "go"]


def test_extract_lrc_words_all_keeps_plain_short_vocables():
    lines = [Line(words=[Word("oh", start_time=0.0, end_time=0.2)])]

    words = wpd._extract_lrc_words_all(lines)

    assert [w["text"] for w in words] == ["oh"]


def test_extract_lrc_words_all_keeps_short_lexical_words():
    lines = [
        Line(
            words=[
                Word("we", start_time=0.0, end_time=0.2),
                Word("do", start_time=0.2, end_time=0.4),
                Word("it", start_time=0.4, end_time=0.6),
                Word("now", start_time=0.6, end_time=1.0),
            ]
        )
    ]

    words = wpd._extract_lrc_words_all(lines)

    assert [w["text"] for w in words] == ["we", "do", "it", "now"]
