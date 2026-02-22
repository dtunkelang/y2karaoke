"""Lyrics timing evaluation against audio analysis.

This module compares lyrics timing (from LRC files) against actual audio
characteristics to evaluate timing quality and identify inconsistencies.
"""

from ....utils.logging import get_logger
from . import timing_evaluator_core
from . import timing_evaluator_comparison
from . import timing_evaluator_correction
from . import timing_models
from ..whisper import whisper_integration
from ... import audio_analysis

logger = get_logger(__name__)

# Re-export data models from timing_models
TimingIssue = timing_models.TimingIssue
AudioFeatures = timing_models.AudioFeatures
TimingReport = timing_models.TimingReport
TranscriptionWord = timing_models.TranscriptionWord
TranscriptionSegment = timing_models.TranscriptionSegment

# Re-export evaluation core functions
evaluate_timing = timing_evaluator_core.evaluate_timing

# Re-export comparison functions
compare_sources = timing_evaluator_comparison.compare_sources
select_best_source = timing_evaluator_comparison.select_best_source
print_comparison_report = timing_evaluator_comparison.print_comparison_report

# Re-export correction functions
correct_line_timestamps = timing_evaluator_correction.correct_line_timestamps
fix_spurious_gaps = timing_evaluator_correction.fix_spurious_gaps

# Re-export audio analysis functions
extract_audio_features = audio_analysis.extract_audio_features

# Re-export whisper integration functions
transcribe_vocals = whisper_integration.transcribe_vocals
correct_timing_with_whisper = whisper_integration.correct_timing_with_whisper
align_lyrics_to_transcription = whisper_integration.align_lyrics_to_transcription
align_dtw_whisper = whisper_integration.align_dtw_whisper
align_hybrid_lrc_whisper = whisper_integration.align_hybrid_lrc_whisper
align_words_to_whisper = whisper_integration.align_words_to_whisper
