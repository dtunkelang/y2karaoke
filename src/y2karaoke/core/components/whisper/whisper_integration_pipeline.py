"""Facade exports for Whisper integration pipeline entry points."""

from typing import Any, Callable, List, Optional, Tuple

from ..alignment import timing_models
from .whisper_integration_pipeline_align import (
    _build_alignment_pass_kwargs as _build_alignment_pass_kwargs_impl,
    align_lrc_text_to_whisper_timings_impl as _align_lrc_text_to_whisper_timings_impl,
)
from .whisper_integration_pipeline_correct import (
    _build_correct_timing_kwargs as _build_correct_timing_kwargs_impl,
    correct_timing_with_whisper_impl as _correct_timing_with_whisper_impl,
)
from .whisper_integration_transcribe import (
    transcribe_vocals_impl as _transcribe_vocals_impl,
)

_build_alignment_pass_kwargs = _build_alignment_pass_kwargs_impl
align_lrc_text_to_whisper_timings_impl = _align_lrc_text_to_whisper_timings_impl
_build_correct_timing_kwargs = _build_correct_timing_kwargs_impl
correct_timing_with_whisper_impl = _correct_timing_with_whisper_impl


def transcribe_vocals_impl(
    vocals_path: str,
    language: Optional[str],
    model_size: str,
    aggressive: bool,
    temperature: float,
    *,
    get_whisper_cache_path_fn: Callable[..., Optional[str]],
    find_best_cached_whisper_model_fn: Callable[..., Optional[Tuple[str, str]]],
    load_whisper_cache_fn: Callable[..., Optional[Tuple[Any, Any, str]]],
    save_whisper_cache_fn: Callable[..., None],
    load_whisper_model_class_fn: Callable[[], Any],
    logger,
) -> Tuple[
    List[timing_models.TranscriptionSegment],
    List[timing_models.TranscriptionWord],
    str,
    str,
]:
    return _transcribe_vocals_impl(
        vocals_path,
        language,
        model_size,
        aggressive,
        temperature,
        get_whisper_cache_path_fn=get_whisper_cache_path_fn,
        find_best_cached_whisper_model_fn=find_best_cached_whisper_model_fn,
        load_whisper_cache_fn=load_whisper_cache_fn,
        save_whisper_cache_fn=save_whisper_cache_fn,
        load_whisper_model_class_fn=load_whisper_model_class_fn,
        logger=logger,
    )
