"""Post-processing helpers for Whisper mapping output."""

from typing import Dict, List, Set, Tuple

from ... import models
from ..alignment import timing_models
from .whisper_mapping_post_text import (
    _interjection_similarity,
    _is_interjection_line,
    _is_placeholder_whisper_token,
    _normalize_interjection_token,
    _normalize_match_token,
    _soft_token_match,
    _soft_token_overlap_ratio,
)
from .whisper_mapping_post_repetition import (
    _extend_line_to_trailing_whisper_matches,
)
from .whisper_mapping_post_interjections import (
    _retime_short_interjection_lines as _retime_short_interjection_lines_impl,
)
from .whisper_mapping_post_overlaps import (
    _resolve_line_overlaps as _resolve_line_overlaps_impl,
)
from .whisper_mapping_post_segment_pull import (
    _pull_late_lines_to_matching_segments as _pull_late_lines_to_matching_segments_impl,
)
from . import whisper_mapping_post_repeat_shift as _repeat_shift_helpers


def _build_word_assignments_from_phoneme_path(
    path: List[Tuple[int, int]],
    lrc_phonemes: List[Dict],
    whisper_phonemes: List[Dict],
) -> Dict[int, List[int]]:
    """Convert phoneme-level DTW matches back to word-level assignments."""
    assignments: Dict[int, Set[int]] = {}
    for lpc_idx, wpc_idx in path:
        lrc_token = lrc_phonemes[lpc_idx]
        whisper_token = whisper_phonemes[wpc_idx]
        word_idx = lrc_token["word_idx"]
        whisper_word_idx = whisper_token["parent_idx"]
        assignments.setdefault(word_idx, set()).add(whisper_word_idx)
    return {
        word_idx: sorted(list(indices)) for word_idx, indices in assignments.items()
    }


def _shift_repeated_lines_to_next_whisper(
    mapped_lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
) -> List[models.Line]:
    return _repeat_shift_helpers._shift_repeated_lines_to_next_whisper(
        mapped_lines,
        all_words,
        is_placeholder_whisper_token_fn=_is_placeholder_whisper_token,
    )


def _enforce_monotonic_line_starts_whisper(
    mapped_lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
) -> List[models.Line]:
    return _repeat_shift_helpers._enforce_monotonic_line_starts_whisper(
        mapped_lines, all_words
    )


def _resolve_line_overlaps(lines: List[models.Line]) -> List[models.Line]:  # noqa: C901
    return _resolve_line_overlaps_impl(lines)


def _pull_late_lines_to_matching_segments(  # noqa: C901
    mapped_lines: List[models.Line],
    segments: List[timing_models.TranscriptionSegment],
    language: str,  # retained for call compatibility
    min_similarity: float = 0.4,
    min_late: float = 1.0,
    max_late: float = 3.0,
    strong_match_max_late: float = 6.0,
    min_early: float = 0.8,
    max_early: float = 3.5,
    max_early_push: float = 2.5,
    early_min_similarity: float = 0.2,
    contain_similarity_margin: float = 0.1,
    min_start_gain: float = 0.5,
    min_gap: float = 0.05,
    max_time_window: float = 15.0,
) -> List[models.Line]:
    return _pull_late_lines_to_matching_segments_impl(
        mapped_lines,
        segments,
        language=language,
        min_similarity=min_similarity,
        min_late=min_late,
        max_late=max_late,
        strong_match_max_late=strong_match_max_late,
        min_early=min_early,
        max_early=max_early,
        max_early_push=max_early_push,
        early_min_similarity=early_min_similarity,
        contain_similarity_margin=contain_similarity_margin,
        min_start_gain=min_start_gain,
        min_gap=min_gap,
        max_time_window=max_time_window,
    )


def _retime_short_interjection_lines(
    mapped_lines: List[models.Line],
    segments: List[timing_models.TranscriptionSegment],
    min_similarity: float = 0.8,
    max_shift: float = 8.0,
    min_gap: float = 0.05,
) -> List[models.Line]:
    return _retime_short_interjection_lines_impl(
        mapped_lines,
        segments,
        is_interjection_line_fn=_is_interjection_line,
        interjection_similarity_fn=_interjection_similarity,
        min_similarity=min_similarity,
        max_shift=max_shift,
        min_gap=min_gap,
    )


def _snap_first_word_to_whisper_onset(  # noqa: C901
    mapped_lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
    *,
    early_threshold: float = 0.12,
    max_shift: float = 0.8,
    min_gap: float = 0.05,
) -> List[models.Line]:
    """Shift lines later when the first word starts clearly before matching Whisper."""
    if not mapped_lines or not all_words:
        return mapped_lines

    adjusted = list(mapped_lines)

    for idx, line in enumerate(adjusted):
        if not line.words:
            continue
        if line.text.strip().endswith("?"):
            cur_tokens = [
                _normalize_match_token(w.text)
                for w in line.words
                if _normalize_match_token(w.text)
            ]
            prev_overlap = 0.0
            if idx > 0 and adjusted[idx - 1].words:
                prev_tokens = [
                    _normalize_match_token(w.text)
                    for w in adjusted[idx - 1].words
                    if _normalize_match_token(w.text)
                ]
                prev_overlap = _soft_token_overlap_ratio(cur_tokens, prev_tokens)
            if prev_overlap >= 0.4 and len(line.words) <= 6:
                # Preserve segment-window cadence for repeated question runs.
                continue

        fw = line.words[0]
        fw_norm = _normalize_interjection_token(fw.text)
        if not fw_norm:
            fw_norm = "".join(ch for ch in fw.text.lower() if ch.isalpha())
        if not fw_norm:
            continue

        prev_end = (
            adjusted[idx - 1].end_time if idx > 0 and adjusted[idx - 1].words else None
        )
        next_start = (
            adjusted[idx + 1].start_time
            if idx + 1 < len(adjusted) and adjusted[idx + 1].words
            else None
        )

        search_after = max(2.0, max_shift + 0.4)
        candidates = [
            w
            for w in all_words
            if (fw.start_time - 1.0) <= w.start <= (fw.start_time + search_after)
        ]
        best = None
        repetitive_following = False
        if idx + 1 < len(adjusted) and adjusted[idx + 1].words:
            cur_tokens = [
                _normalize_match_token(w.text)
                for w in line.words
                if _normalize_match_token(w.text)
            ]
            next_tokens = [
                _normalize_match_token(w.text)
                for w in adjusted[idx + 1].words
                if _normalize_match_token(w.text)
            ]
            if cur_tokens and next_tokens:
                repetitive_following = (
                    _soft_token_overlap_ratio(cur_tokens, next_tokens) >= 0.4
                )
        later_match_starts: List[float] = []
        for w in candidates:
            wn = _normalize_interjection_token(w.text)
            if not wn:
                wn = "".join(ch for ch in w.text.lower() if ch.isalpha())
            if not wn:
                continue
            match_score = 0
            if wn == fw_norm:
                match_score = 3
            elif wn.startswith(fw_norm) or fw_norm.startswith(wn):
                match_score = 2
            elif fw_norm in wn or wn in fw_norm:
                match_score = 1
            if match_score == 0:
                continue
            if (
                repetitive_following
                and match_score >= 2
                and w.start >= fw.start_time + 0.8
            ):
                later_match_starts.append(w.start)
            score = (match_score, -abs(w.start - fw.start_time))
            if best is None or score > best[0]:
                best = (score, w)

        if best is None:
            continue

        target_start = best[1].start
        if repetitive_following and later_match_starts:
            target_start = min(later_match_starts)
        delta = fw.start_time - target_start
        if delta >= -early_threshold:
            continue

        desired_shift = min(-delta, max_shift)
        max_allowed = desired_shift
        if prev_end is not None:
            max_allowed = min(
                max_allowed, max(0.0, fw.start_time - (prev_end + min_gap))
            )
        if next_start is not None:
            max_allowed = min(
                max_allowed,
                max(0.0, (next_start - min_gap) - line.end_time),
            )
        run_end = idx + 1
        base_tokens = [
            _normalize_match_token(w.text)
            for w in line.words
            if _normalize_match_token(w.text)
        ]
        while run_end < len(adjusted):
            nxt = adjusted[run_end]
            if not nxt.words:
                break
            nxt_tokens = [
                _normalize_match_token(w.text)
                for w in nxt.words
                if _normalize_match_token(w.text)
            ]
            if not base_tokens or not nxt_tokens:
                break
            if _soft_token_overlap_ratio(base_tokens, nxt_tokens) < 0.4:
                break
            run_end += 1

        should_shift_run = run_end > idx + 1 and desired_shift - max_allowed >= 0.1
        if should_shift_run:
            desired_run_shift = min(-delta, 2.5)
            # Prefer per-line first-word onset matching for repetitive runs.
            run_lines = adjusted[idx:run_end]
            run_start = run_lines[0].start_time - 1.0
            run_end_time = run_lines[-1].end_time + 6.0
            onset_candidates = [
                ww.start
                for ww in all_words
                if run_start <= ww.start <= run_end_time
                and _soft_token_match(
                    fw_norm,
                    (
                        _normalize_interjection_token(ww.text)
                        or "".join(ch for ch in ww.text.lower() if ch.isalpha())
                    ),
                )
            ]
            onset_candidates.sort()
            if len(onset_candidates) >= len(run_lines):
                assigned_starts: List[float] = []
                cand_idx = 0
                for run_line in run_lines:
                    min_start = run_line.start_time + 0.15
                    if assigned_starts:
                        min_spacing = 1.0 if len(run_line.words) <= 4 else 0.35
                        min_start = max(min_start, assigned_starts[-1] + min_spacing)
                    while (
                        cand_idx < len(onset_candidates)
                        and onset_candidates[cand_idx] < min_start
                    ):
                        cand_idx += 1
                    if cand_idx >= len(onset_candidates):
                        assigned_starts = []
                        break
                    # Guardrail: don't jump too far in one step.
                    if onset_candidates[cand_idx] - run_line.start_time > 3.0:
                        assigned_starts = []
                        break
                    assigned_starts.append(onset_candidates[cand_idx])
                    cand_idx += 1
                if assigned_starts and len(assigned_starts) == len(run_lines):
                    shifted_any = False
                    for rel_idx, run_line in enumerate(run_lines):
                        target = assigned_starts[rel_idx]
                        delta_run = target - run_line.start_time
                        if delta_run <= 0.1:
                            continue
                        shifted_any = True
                        shifted_words = [
                            models.Word(
                                text=w.text,
                                start_time=w.start_time + delta_run,
                                end_time=w.end_time + delta_run,
                                singer=w.singer,
                            )
                            for w in run_line.words
                        ]
                        adjusted[idx + rel_idx] = models.Line(
                            words=shifted_words, singer=run_line.singer
                        )
                    if shifted_any:
                        continue

            after_run_start = (
                adjusted[run_end].start_time
                if run_end < len(adjusted) and adjusted[run_end].words
                else None
            )
            if after_run_start is not None:
                available = max(
                    0.0,
                    after_run_start - min_gap - adjusted[run_end - 1].end_time,
                )
                run_shift = min(desired_run_shift, available)
                if run_shift > 0:
                    for run_idx in range(idx, run_end):
                        run_line = adjusted[run_idx]
                        shifted_words = [
                            models.Word(
                                text=w.text,
                                start_time=w.start_time + run_shift,
                                end_time=w.end_time + run_shift,
                                singer=w.singer,
                            )
                            for w in run_line.words
                        ]
                        adjusted[run_idx] = models.Line(
                            words=shifted_words, singer=run_line.singer
                        )
                    continue

        if max_allowed <= 0:
            tightly_packed = False
            if next_start is not None and (next_start - line.end_time) <= 0.12:
                packed_count = 0
                probe = idx
                while probe + 1 < len(adjusted) and packed_count < 5:
                    cur_probe = adjusted[probe]
                    nxt_probe = adjusted[probe + 1]
                    if not cur_probe.words or not nxt_probe.words:
                        break
                    if nxt_probe.start_time - cur_probe.end_time > 0.12:
                        break
                    packed_count += 1
                    probe += 1
                tightly_packed = packed_count >= 2
            best_match_strength = best[0][0] if best is not None else 0
            if (
                desired_shift >= 0.8
                and tightly_packed
                and best_match_strength >= 2
                and idx >= len(adjusted) // 2
                and not line.text.strip().endswith("?")
            ):
                # Late-song drift fallback: if the line has a clear later onset
                # but is blocked by a packed suffix, shift a local packed run.
                suffix_shift = min(desired_shift, max_shift)
                local_end = idx + 1
                while local_end < len(adjusted):
                    prev_local = adjusted[local_end - 1]
                    cur_local = adjusted[local_end]
                    if not prev_local.words or not cur_local.words:
                        break
                    if (cur_local.start_time - prev_local.end_time) > 0.2:
                        break
                    if local_end - idx >= 10:
                        break
                    local_end += 1
                for tail_idx in range(idx, local_end):
                    tail_line = adjusted[tail_idx]
                    if not tail_line.words:
                        break
                    shifted_words = [
                        models.Word(
                            text=w.text,
                            start_time=w.start_time + suffix_shift,
                            end_time=w.end_time + suffix_shift,
                            singer=w.singer,
                        )
                        for w in tail_line.words
                    ]
                    adjusted[tail_idx] = models.Line(
                        words=shifted_words, singer=tail_line.singer
                    )
                continue
            continue

        shifted_words = [
            models.Word(
                text=w.text,
                start_time=w.start_time + max_allowed,
                end_time=w.end_time + max_allowed,
                singer=w.singer,
            )
            for w in line.words
        ]
        adjusted[idx] = models.Line(words=shifted_words, singer=line.singer)

    return adjusted


__all__ = [
    "_build_word_assignments_from_phoneme_path",
    "_enforce_monotonic_line_starts_whisper",
    "_extend_line_to_trailing_whisper_matches",
    "_pull_late_lines_to_matching_segments",
    "_resolve_line_overlaps",
    "_retime_short_interjection_lines",
    "_shift_repeated_lines_to_next_whisper",
    "_snap_first_word_to_whisper_onset",
]
