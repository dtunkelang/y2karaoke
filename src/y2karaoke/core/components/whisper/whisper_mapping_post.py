"""Post-processing helpers for Whisper mapping output."""

from typing import Dict, List, Optional, Set, Tuple

from ... import models
from ..alignment import timing_models
from .whisper_mapping_post_text import (
    _contains_token_sequence,
    _interjection_similarity,
    _is_interjection_line,
    _is_placeholder_whisper_token,
    _light_text_similarity,
    _max_contiguous_soft_match_run,
    _normalize_interjection_token,
    _normalize_match_token,
    _normalize_text_tokens,
    _overlap_suffix_prefix,
    _soft_text_similarity,
    _soft_token_match,
    _soft_token_overlap_ratio,
)
from .whisper_mapping_post_repetition import (
    _extend_line_to_trailing_whisper_matches,
    _pull_adjacent_similar_lines_across_long_gaps,
    _realign_repetitive_runs_to_matching_segments,
    _rebalance_short_question_pairs,
    _retime_repetitive_question_runs_to_segment_windows,
    _smooth_adjacent_duplicate_line_cadence,
)
from .whisper_mapping_post_interjections import (
    _retime_short_interjection_lines as _retime_short_interjection_lines_impl,
)
from .whisper_mapping_post_overlaps import (
    _resolve_line_overlaps as _resolve_line_overlaps_impl,
)


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
    """Ensure repeated lines reserve later Whisper words when they reappear."""
    adjusted_lines: List[models.Line] = []
    last_idx_by_text: Dict[str, int] = {}
    last_end_time: Dict[str, float] = {}
    lexical_indices = [
        i
        for i, ww in enumerate(all_words)
        if not _is_placeholder_whisper_token(ww.text)
    ]

    for idx, line in enumerate(mapped_lines):
        if not line.words:
            adjusted_lines.append(line)
            continue

        text_norm = line.text.strip().lower() if getattr(line, "text", "") else ""
        prev_idx = last_idx_by_text.get(text_norm)
        prev_end = last_end_time.get(text_norm)
        assigned_end_idx: Optional[int] = None

        if prev_idx is not None and prev_end is not None:
            required_time = max(prev_end + 0.4, line.start_time)
            start_idx = next(
                (
                    wi
                    for wi in lexical_indices
                    if wi > prev_idx and all_words[wi].start >= required_time
                ),
                None,
            )
            if start_idx is None:
                start_idx = next(
                    (wi for wi in lexical_indices if wi > prev_idx),
                    None,
                )
            max_repeat_jump = 4.0
            if start_idx is not None and (
                all_words[start_idx].start - prev_end > max_repeat_jump
            ):
                start_idx = None
            if start_idx is not None:
                adjusted_words: List[models.Word] = []
                for word_idx, w in enumerate(line.words):
                    new_idx = min(start_idx + word_idx, len(all_words) - 1)
                    ww = all_words[new_idx]
                    adjusted_words.append(
                        models.Word(
                            text=w.text,
                            start_time=ww.start,
                            end_time=ww.end,
                            singer=w.singer,
                        )
                    )
                line = models.Line(words=adjusted_words, singer=line.singer)
                assigned_end_idx = min(
                    start_idx + len(line.words) - 1, len(all_words) - 1
                )

        # Fallback for adjacent duplicate short lines: if Whisper matching could
        # not place the second occurrence and it drifts too late, pull it closer
        # to the previous duplicate to preserve natural refrain cadence.
        if (
            assigned_end_idx is None
            and prev_end is not None
            and len(line.words) <= 5
            and adjusted_lines
            and text_norm
            and adjusted_lines[-1].words
            and adjusted_lines[-1].text.strip().lower() == text_norm
        ):
            gap = line.start_time - prev_end
            if 1.0 < gap <= 4.0:
                duration = max(0.25, min(line.end_time - line.start_time, 1.8))
                next_start = (
                    mapped_lines[idx + 1].start_time
                    if idx + 1 < len(mapped_lines) and mapped_lines[idx + 1].words
                    else None
                )
                target_start = prev_end + 0.18
                if next_start is not None:
                    target_start = min(
                        target_start,
                        max(prev_end + 0.05, next_start - 0.05 - duration),
                    )
                if target_start < line.start_time - 0.2:
                    shift = target_start - line.start_time
                    shifted_words = [
                        models.Word(
                            text=w.text,
                            start_time=w.start_time + shift,
                            end_time=w.end_time + shift,
                            singer=w.singer,
                        )
                        for w in line.words
                    ]
                    line = models.Line(words=shifted_words, singer=line.singer)

        adjusted_lines.append(line)
        if line.words:
            if assigned_end_idx is None:
                assigned_end_idx = next(
                    (
                        wi
                        for wi in lexical_indices
                        if abs(all_words[wi].start - line.words[-1].start_time) < 0.05
                    ),
                    len(all_words) - 1,
                )
            last_idx_by_text[text_norm] = assigned_end_idx
            last_end_time[text_norm] = line.end_time

    return adjusted_lines


def _enforce_monotonic_line_starts_whisper(
    mapped_lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
) -> List[models.Line]:
    """Ensure line starts are monotonic by shifting backwards lines forward."""
    prev_start = None
    prev_end = None
    monotonic_lines: List[models.Line] = []
    for line in mapped_lines:
        if not line.words:
            monotonic_lines.append(line)
            continue

        if prev_start is not None and line.start_time < prev_start:
            required_time = (prev_end or line.start_time) + 0.01
            start_idx = next(
                (idx for idx, ww in enumerate(all_words) if ww.start >= required_time),
                None,
            )
            if start_idx is not None and (
                all_words[start_idx].start - required_time <= 10.0
            ):
                adjusted_words_2: List[models.Word] = []
                for word_idx, w in enumerate(line.words):
                    new_idx = min(start_idx + word_idx, len(all_words) - 1)
                    ww = all_words[new_idx]
                    adjusted_words_2.append(
                        models.Word(
                            text=w.text,
                            start_time=ww.start,
                            end_time=ww.end,
                            singer=w.singer,
                        )
                    )
                line = models.Line(words=adjusted_words_2, singer=line.singer)
            else:
                shift = required_time - line.start_time
                shifted_words: List[models.Word] = [
                    models.Word(
                        text=w.text,
                        start_time=w.start_time + shift,
                        end_time=w.end_time + shift,
                        singer=w.singer,
                    )
                    for w in line.words
                ]
                line = models.Line(words=shifted_words, singer=line.singer)

        monotonic_lines.append(line)
        if line.words:
            prev_start = line.start_time
            prev_end = line.end_time

    return monotonic_lines


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
    """Pull late line starts toward matching Whisper segments within neighbor bounds."""
    _ = language
    if not mapped_lines or not segments:
        return mapped_lines

    adjusted = list(mapped_lines)
    sorted_segments = sorted(segments, key=lambda s: s.start)

    for idx, line in enumerate(adjusted):
        if not line.words or not line.text.strip():
            continue

        cur_tokens_all = [
            _normalize_match_token(w.text)
            for w in line.words
            if _normalize_match_token(w.text)
        ]
        prev_overlap = 0
        next_overlap = 0
        if idx > 0 and adjusted[idx - 1].words:
            prev_tokens = [
                _normalize_match_token(w.text)
                for w in adjusted[idx - 1].words[-3:]
                if _normalize_match_token(w.text)
            ]
            prev_overlap = _overlap_suffix_prefix(prev_tokens, cur_tokens_all, 3)
        if idx + 1 < len(adjusted) and adjusted[idx + 1].words:
            next_tokens = [
                _normalize_match_token(w.text)
                for w in adjusted[idx + 1].words[:3]
                if _normalize_match_token(w.text)
            ]
            next_overlap = _overlap_suffix_prefix(cur_tokens_all, next_tokens, 3)
        if idx > 0 and adjusted[idx - 1].words and len(line.words) <= 4:
            if prev_overlap >= 2:
                # Repeated phrase handoff is ambiguous; avoid pulling into an
                # earlier segment and let trailing-match logic decide.
                continue

        prev_end = (
            adjusted[idx - 1].end_time if idx > 0 and adjusted[idx - 1].words else None
        )
        next_start = (
            adjusted[idx + 1].start_time
            if idx + 1 < len(adjusted) and adjusted[idx + 1].words
            else None
        )

        best_seg: Optional[timing_models.TranscriptionSegment] = None
        best_sim = 0.0
        best_contains = False
        best_rank: Tuple[float, float] = (-1.0, float("-inf"))
        best_contain_seg: Optional[timing_models.TranscriptionSegment] = None
        best_contain_sim = -1.0
        best_contain_dist = float("inf")
        repetitive_in_first_pass = (
            prev_overlap >= 2
            or next_overlap >= 2
            or (len(cur_tokens_all) >= 5 and len(set(cur_tokens_all)) <= 3)
        )
        for seg in sorted_segments:
            if abs(seg.start - line.start_time) > max_time_window:
                continue
            if (
                repetitive_in_first_pass
                and prev_end is not None
                and seg.start <= (prev_end + min_gap)
            ):
                # For repeated refrains, enforce forward segment progression so
                # adjacent identical lines don't keep reusing the same segment.
                continue
            seg_tokens = _normalize_text_tokens(seg.text)
            sim = max(
                _light_text_similarity(line.text, seg.text),
                _soft_text_similarity(line.text, seg.text),
            )
            if len(cur_tokens_all) >= 3 and len(seg_tokens) <= 1 and sim < 0.55:
                # Avoid anchoring full lyric lines to tiny interjection segments
                # (e.g., "Hé !"), which can shift subsequent repeated lines early.
                continue
            contains = _contains_token_sequence(line.text, seg.text)
            if not contains and len(line.words) <= 6:
                contains = _max_contiguous_soft_match_run(line.text, seg.text) >= 3
            dist = abs(seg.start - line.start_time)
            rank = (sim, -dist)
            if rank > best_rank:
                best_rank = rank
                best_sim = sim
                best_seg = seg
                best_contains = contains
            if contains and (
                sim > best_contain_sim
                or (sim == best_contain_sim and dist < best_contain_dist)
            ):
                best_contain_seg = seg
                best_contain_sim = sim
                best_contain_dist = dist

        if (
            best_seg is not None
            and best_contain_seg is not None
            and best_contain_sim >= (best_sim - contain_similarity_margin)
        ):
            best_seg = best_contain_seg
            best_sim = best_contain_sim
            best_contains = True

        min_sim_required = min_similarity
        if best_contains and len(line.words) <= 5:
            min_sim_required = min(min_similarity, 0.2)
        if best_seg is None or best_sim < min_sim_required:
            continue
        best_seg_tokens = _normalize_text_tokens(best_seg.text)
        best_token_overlap = _soft_token_overlap_ratio(cur_tokens_all, best_seg_tokens)
        if (
            len(line.words) <= 6
            and not repetitive_in_first_pass
            and not best_contains
            and best_token_overlap < 0.35
            and best_sim < 0.65
        ):
            # Guardrail: don't pull short non-repetitive lines toward generic
            # repeated Whisper segments on very weak lexical evidence.
            continue

        late_by = line.start_time - best_seg.start
        allowed_max_late = strong_match_max_late if best_contains else max_late
        if best_contains and len(line.words) <= 5:
            # Short continuation/refrain lines can legitimately be much later
            # than segment start when prior text shares the same segment.
            allowed_max_late += 2.0
        if late_by < min_late or late_by > allowed_max_late:
            continue

        window_start = best_seg.start
        if prev_end is not None:
            window_start = max(window_start, prev_end + min_gap)

        window_end = best_seg.end
        if next_start is not None:
            window_end = min(window_end, next_start - min_gap)

        if window_end - window_start <= 0.1:
            continue

        shift = window_start - line.start_time
        if shift >= -0.2:
            continue
        if (line.start_time - window_start) < min_start_gain:
            continue

        shifted_words = [
            models.Word(
                text=w.text,
                start_time=w.start_time + shift,
                end_time=w.end_time + shift,
                singer=w.singer,
            )
            for w in line.words
        ]
        shifted_line = models.Line(words=shifted_words, singer=line.singer)

        if shifted_line.end_time > window_end:
            total_duration = max(window_end - window_start, 0.2)
            spacing = total_duration / len(shifted_line.words)
            fitted_words = []
            for word_idx, w in enumerate(shifted_line.words):
                start = window_start + word_idx * spacing
                end = start + spacing * 0.9
                fitted_words.append(
                    models.Word(
                        text=w.text,
                        start_time=start,
                        end_time=end,
                        singer=w.singer,
                    )
                )
            shifted_line = models.Line(words=fitted_words, singer=line.singer)

        adjusted[idx] = shifted_line

        continue

        # NOTE: keep this as an explicit second branch for clarity. For short
        # refrain-like lines that are too early, push them later toward the
        # nearest matching segment start.
    for idx, line in enumerate(adjusted):
        if not line.words or not line.text.strip():
            continue

        prev_end = (
            adjusted[idx - 1].end_time if idx > 0 and adjusted[idx - 1].words else None
        )
        next_start = (
            adjusted[idx + 1].start_time
            if idx + 1 < len(adjusted) and adjusted[idx + 1].words
            else None
        )

        cur_tokens_all = [
            _normalize_match_token(w.text)
            for w in line.words
            if _normalize_match_token(w.text)
        ]
        prev_overlap = 0
        next_overlap = 0
        if idx > 0 and adjusted[idx - 1].words:
            prev_tokens = [
                _normalize_match_token(w.text)
                for w in adjusted[idx - 1].words[-3:]
                if _normalize_match_token(w.text)
            ]
            prev_overlap = _overlap_suffix_prefix(prev_tokens, cur_tokens_all, 3)
        if idx + 1 < len(adjusted) and adjusted[idx + 1].words:
            next_tokens = [
                _normalize_match_token(w.text)
                for w in adjusted[idx + 1].words[:3]
                if _normalize_match_token(w.text)
            ]
            next_overlap = _overlap_suffix_prefix(cur_tokens_all, next_tokens, 3)
        prev_set_overlap = 0.0
        next_set_overlap = 0.0
        if idx > 0 and adjusted[idx - 1].words and cur_tokens_all:
            prev_tokens_all = [
                _normalize_match_token(w.text)
                for w in adjusted[idx - 1].words
                if _normalize_match_token(w.text)
            ]
            prev_set_overlap = _soft_token_overlap_ratio(
                cur_tokens_all, prev_tokens_all
            )
        if idx + 1 < len(adjusted) and adjusted[idx + 1].words and cur_tokens_all:
            next_tokens_all = [
                _normalize_match_token(w.text)
                for w in adjusted[idx + 1].words
                if _normalize_match_token(w.text)
            ]
            next_set_overlap = _soft_token_overlap_ratio(
                cur_tokens_all, next_tokens_all
            )
        is_repetitive_phrase = (
            prev_overlap >= 2
            or next_overlap >= 2
            or prev_set_overlap >= 0.5
            or next_set_overlap >= 0.5
            or (len(cur_tokens_all) >= 5 and len(set(cur_tokens_all)) <= 3)
        )

        early_best_seg: Optional[timing_models.TranscriptionSegment] = None
        early_best_sim = 0.0
        early_best_rank: Tuple[float, float] = (-1.0, float("-inf"))
        for seg in sorted_segments:
            if abs(seg.start - line.start_time) > max_time_window:
                continue
            if (
                is_repetitive_phrase
                and prev_end is not None
                and seg.start <= (prev_end + min_gap)
            ):
                continue
            seg_tokens = _normalize_text_tokens(seg.text)
            sim = max(
                _light_text_similarity(line.text, seg.text),
                _soft_text_similarity(line.text, seg.text),
            )
            if len(cur_tokens_all) >= 3 and len(seg_tokens) <= 1 and sim < 0.55:
                continue
            dist = abs(seg.start - line.start_time)
            rank = (sim, -dist)
            if rank > early_best_rank:
                early_best_rank = rank
                early_best_sim = sim
                early_best_seg = seg

        if early_best_seg is None or early_best_sim < early_min_similarity:
            continue
        early_seg_tokens = _normalize_text_tokens(early_best_seg.text)
        early_token_overlap = _soft_token_overlap_ratio(
            cur_tokens_all, early_seg_tokens
        )
        if (
            len(line.words) <= 6
            and not is_repetitive_phrase
            and early_token_overlap < 0.35
            and early_best_sim < 0.65
        ):
            # Guardrail for the "push-early-lines-later" branch: require
            # stronger lexical support before retiming short standalone lines.
            continue

        early_by = early_best_seg.start - line.start_time
        if early_by < min_early or early_by > max_early:
            continue
        if not (is_repetitive_phrase or len(line.words) <= 4):
            continue

        line_duration = max(0.12, line.end_time - line.start_time)
        min_line_duration = max(0.24, min(0.8, 0.12 * len(line.words)))
        upper_start = line.start_time + max_early_push
        if next_start is not None:
            upper_start = min(upper_start, next_start - min_gap - min_line_duration)
        target_start = min(early_best_seg.start, upper_start)
        if prev_end is not None:
            target_start = max(target_start, prev_end + min_gap)
        if target_start <= line.start_time + 0.2:
            continue

        shift = target_start - line.start_time
        target_end = line.end_time + shift
        if next_start is not None:
            target_end = min(target_end, next_start - min_gap)
        if is_repetitive_phrase:
            target_end = min(target_end, early_best_seg.end)
        if target_end <= target_start + min_line_duration:
            continue

        if (target_end - target_start) < (line_duration * 0.85):
            spacing = (target_end - target_start) / len(line.words)
            compressed_words = []
            for word_idx, w in enumerate(line.words):
                ws = target_start + word_idx * spacing
                we = ws + spacing * 0.9
                if word_idx == len(line.words) - 1:
                    we = target_end
                compressed_words.append(
                    models.Word(
                        text=w.text,
                        start_time=ws,
                        end_time=we,
                        singer=w.singer,
                    )
                )
            adjusted[idx] = models.Line(words=compressed_words, singer=line.singer)
        else:
            shifted_words = [
                models.Word(
                    text=w.text,
                    start_time=w.start_time + shift,
                    end_time=w.end_time + shift,
                    singer=w.singer,
                )
                for w in line.words
            ]
            adjusted[idx] = models.Line(words=shifted_words, singer=line.singer)

    adjusted = _realign_repetitive_runs_to_matching_segments(adjusted, sorted_segments)
    adjusted = _smooth_adjacent_duplicate_line_cadence(adjusted)
    adjusted = _rebalance_short_question_pairs(adjusted)
    adjusted = _retime_repetitive_question_runs_to_segment_windows(
        adjusted, sorted_segments
    )
    return _pull_adjacent_similar_lines_across_long_gaps(adjusted)


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
