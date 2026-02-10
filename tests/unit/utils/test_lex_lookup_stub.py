from y2karaoke.utils import lex_lookup_stub as stub


def test_tokenize_keeps_words_and_apostrophes() -> None:
    assert stub._tokenize("I'm fine, y'all.") == ["I'm", "fine", "y'all"]


def test_lex_lookup_uses_cmudict_first_pronunciation(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        stub,
        "_CMU_DICT",
        {
            "read": [["R", "EH1", "D"], ["R", "IY1", "D"]],
        },
    )

    exit_code = stub.lex_lookup_main(["read"])
    out = capsys.readouterr().out.strip()

    assert exit_code == 0
    assert out == "[r eh1 d]"


def test_lex_lookup_falls_back_to_literal_for_unknown_word(capsys, monkeypatch) -> None:
    monkeypatch.setattr(stub, "_CMU_DICT", {})

    exit_code = stub.lex_lookup_main(["zzzzword"])
    out = capsys.readouterr().out.strip()

    assert exit_code == 0
    assert out == "[ZZZZWORD]"
