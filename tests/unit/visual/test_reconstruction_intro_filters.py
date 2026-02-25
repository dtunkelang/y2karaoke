from y2karaoke.core.visual.reconstruction_intro_filters import _is_metadata_keyword_line


def test_metadata_keyword_line_does_not_match_fun_inside_funk() -> None:
    assert _is_metadata_keyword_line("uptown funk you up") is False


def test_metadata_keyword_line_still_matches_true_short_keyword_token() -> None:
    assert _is_metadata_keyword_line("mca records ltd") is True
