import y2karaoke.core.components.whisper.whisper_integration as wi
import y2karaoke.core.components.whisper.whisper_mapping as wm
import y2karaoke.core.components.whisper.whisper_mapping_post as wmp


def test_whisper_integration_all_only_advertises_public_entrypoints():
    assert wi.__all__ == [
        "transcribe_vocals",
        "correct_timing_with_whisper",
        "align_lrc_text_to_whisper_timings",
    ]
    assert "_assess_lrc_quality" not in wi.__all__


def test_whisper_integration_keeps_legacy_alias_attributes_accessible():
    assert wi._assess_lrc_quality is not None
    assert wi._build_segment_text_overlap_assignments is not None


def test_whisper_mapping_modules_do_not_advertise_private_helpers():
    assert wm.__all__ == []
    assert wmp.__all__ == []
