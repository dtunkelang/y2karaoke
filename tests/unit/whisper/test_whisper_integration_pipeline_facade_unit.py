import y2karaoke.core.components.whisper.whisper_integration_pipeline as facade
import y2karaoke.core.components.whisper.whisper_integration_pipeline_align as align
import y2karaoke.core.components.whisper.whisper_integration_pipeline_correct as correct


def test_pipeline_facade_re_exports_split_orchestration_symbols():
    assert facade._build_alignment_pass_kwargs is align._build_alignment_pass_kwargs
    assert (
        facade.align_lrc_text_to_whisper_timings_impl
        is align.align_lrc_text_to_whisper_timings_impl
    )
    assert facade._build_correct_timing_kwargs is correct._build_correct_timing_kwargs
    assert (
        facade.correct_timing_with_whisper_impl
        is correct.correct_timing_with_whisper_impl
    )
