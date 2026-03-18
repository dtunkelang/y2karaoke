from y2karaoke.core.components.lyrics import api


def test_api_all_exposes_only_public_surface():
    assert "_fetch_lrc_text_and_timings" not in api.__all__
    assert "_apply_whisper_alignment" not in api.__all__
    assert "get_lyrics" in api.__all__
    assert "get_lyrics_simple" in api.__all__
    assert "Line" in api.__all__


def test_api_lazy_legacy_helper_resolution_matches_source_function():
    from y2karaoke.core.components.lyrics.helpers import (
        _apply_whisper_alignment as source_apply_whisper_alignment,
    )

    resolved = api._apply_whisper_alignment

    assert resolved is source_apply_whisper_alignment
