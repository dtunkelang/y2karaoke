"""Confidence helpers for upstream LRC-to-Whisper word assignments."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Tuple

from ..alignment import timing_models
from ... import models
from . import whisper_utils

_PLACEHOLDER_TOKENS = {"[vocal]"}
_LOW_CONF_MIN_ASSIGNED_WORDS = 4
_LOW_CONF_MAX_LEXICAL_OVERLAP_RATIO = 0.2
_LOW_CONF_MIN_PLACEHOLDER_RATIO = 0.5
_LOW_CONF_MIN_MEDIAN_DRIFT_SEC = 6.0
_LOW_CONF_MIN_MAX_DRIFT_SEC = 10.0


@dataclass(frozen=True)
class AssignmentConfidenceProfile:
    total_assigned: int
    lexical_overlap_ratio: float
    placeholder_ratio: float
    median_start_drift_sec: float
    max_start_drift_sec: float
    unique_segment_count: int

    @property
    def low_confidence(self) -> bool:
        lexical_sparse = (
            self.lexical_overlap_ratio <= _LOW_CONF_MAX_LEXICAL_OVERLAP_RATIO
        )
        placeholder_heavy = self.placeholder_ratio >= _LOW_CONF_MIN_PLACEHOLDER_RATIO
        drift_bad = (
            self.median_start_drift_sec >= _LOW_CONF_MIN_MEDIAN_DRIFT_SEC
            or self.max_start_drift_sec >= _LOW_CONF_MIN_MAX_DRIFT_SEC
        )
        return (
            self.total_assigned >= _LOW_CONF_MIN_ASSIGNED_WORDS
            and lexical_sparse
            and placeholder_heavy
            and drift_bad
        )


def build_assignment_confidence_profile(
    *,
    line_idx: int,
    line: models.Line,
    lrc_index_by_loc: Dict[Tuple[int, int], int],
    lrc_assignments: Dict[int, List[int]],
    all_words: List[timing_models.TranscriptionWord],
    word_segment_idx: Dict[int, int],
) -> AssignmentConfidenceProfile:
    lyric_words = [
        whisper_utils._normalize_word(word.text)
        for word in line.words
        if whisper_utils._normalize_word(word.text)
    ]
    lyric_vocab = set(lyric_words)

    assigned_words: List[timing_models.TranscriptionWord] = []
    unique_segments = set()
    for word_idx in range(len(line.words)):
        lrc_idx = lrc_index_by_loc.get((line_idx, word_idx))
        if lrc_idx is None:
            continue
        for assigned_idx in lrc_assignments.get(lrc_idx, []):
            if assigned_idx < 0 or assigned_idx >= len(all_words):
                continue
            assigned_word = all_words[assigned_idx]
            assigned_words.append(assigned_word)
            seg = word_segment_idx.get(assigned_idx)
            if seg is not None:
                unique_segments.add(seg)

    if not assigned_words:
        return AssignmentConfidenceProfile(
            total_assigned=0,
            lexical_overlap_ratio=1.0,
            placeholder_ratio=0.0,
            median_start_drift_sec=0.0,
            max_start_drift_sec=0.0,
            unique_segment_count=0,
        )

    lexical_matches = 0
    placeholder_count = 0
    drifts: List[float] = []
    line_start = line.start_time
    for assigned_word in assigned_words:
        normalized = whisper_utils._normalize_word(assigned_word.text)
        if normalized in _PLACEHOLDER_TOKENS:
            placeholder_count += 1
        if normalized and normalized in lyric_vocab:
            lexical_matches += 1
        drifts.append(abs(assigned_word.start - line_start))

    total_assigned = len(assigned_words)
    return AssignmentConfidenceProfile(
        total_assigned=total_assigned,
        lexical_overlap_ratio=lexical_matches / total_assigned,
        placeholder_ratio=placeholder_count / total_assigned,
        median_start_drift_sec=median(drifts),
        max_start_drift_sec=max(drifts),
        unique_segment_count=len(unique_segments),
    )
