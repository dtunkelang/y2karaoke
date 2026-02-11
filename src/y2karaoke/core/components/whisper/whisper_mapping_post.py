"""Post-processing helpers for Whisper mapping output."""

import re
from typing import Dict, List, Optional, Set, Tuple

from ... import models
from ..alignment import timing_models

_INTERJECTION_TOKENS = {
    "ooh",
    "oh",
    "ah",
    "aah",
    "mmm",
    "mm",
    "uh",
    "uhh",
    "la",
    "na",
}


def _normalize_interjection_token(token: str) -> str:
    cleaned = "".join(ch for ch in token.lower() if ch.isalpha())
    if not cleaned:
        return ""
    return re.sub(r"(.)\1{2,}", r"\1\1", cleaned)


def _is_interjection_line(text: str, max_tokens: int = 3) -> bool:
    tokens = [_normalize_interjection_token(t) for t in text.split()]
    tokens = [t for t in tokens if t]
    if not tokens or len(tokens) > max_tokens:
        return False
    return all(t in _INTERJECTION_TOKENS for t in tokens)


def _interjection_similarity(line_text: str, seg_text: str) -> float:
    line_tokens = [_normalize_interjection_token(t) for t in line_text.split()]
    seg_tokens = [_normalize_interjection_token(t) for t in seg_text.split()]
    line_tokens = [t for t in line_tokens if t]
    seg_tokens = [t for t in seg_tokens if t]
    if not line_tokens or not seg_tokens:
        return 0.0
    if len(line_tokens) == 1 and len(seg_tokens) == 1:
        if line_tokens[0] == seg_tokens[0]:
            return 1.0
        if line_tokens[0] in seg_tokens[0] or seg_tokens[0] in line_tokens[0]:
            return 0.9
    overlap = len(set(line_tokens) & set(seg_tokens))
    return overlap / max(len(set(line_tokens)), len(set(seg_tokens)))


def _normalize_text_tokens(text: str) -> List[str]:
    tokens = []
    for raw in text.lower().split():
        tok = "".join(ch for ch in raw if ch.isalpha())
        if tok:
            tokens.append(re.sub(r"(.)\1{2,}", r"\1\1", tok))
    return tokens


def _light_text_similarity(a: str, b: str) -> float:
    a_tokens = _normalize_text_tokens(a)
    b_tokens = _normalize_text_tokens(b)
    if not a_tokens or not b_tokens:
        return 0.0
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union else 0.0


def _soft_text_similarity(a: str, b: str) -> float:
    a_tokens = _normalize_text_tokens(a)
    b_tokens = _normalize_text_tokens(b)
    if not a_tokens or not b_tokens:
        return 0.0
    used = [False] * len(b_tokens)
    matched = 0
    for at in a_tokens:
        best_idx = None
        for idx, bt in enumerate(b_tokens):
            if used[idx]:
                continue
            if _soft_token_match(at, bt):
                best_idx = idx
                if at == bt:
                    break
        if best_idx is not None:
            used[best_idx] = True
            matched += 1
    return matched / max(len(a_tokens), len(b_tokens))


def _contains_token_sequence(
    needle_text: str,
    haystack_text: str,
    *,
    min_tokens: int = 3,
) -> bool:
    needle = _normalize_text_tokens(needle_text)
    haystack = _normalize_text_tokens(haystack_text)
    if len(needle) < min_tokens or len(haystack) < len(needle):
        return False
    for start in range(0, len(haystack) - len(needle) + 1):
        ok = True
        for offset, tok in enumerate(needle):
            if not _soft_token_match(tok, haystack[start + offset]):
                ok = False
                break
        if ok:
            return True
    return False


def _max_contiguous_soft_match_run(needle_text: str, haystack_text: str) -> int:
    needle = _normalize_text_tokens(needle_text)
    haystack = _normalize_text_tokens(haystack_text)
    if not needle or not haystack:
        return 0
    best = 0
    for ni in range(len(needle)):
        for hi in range(len(haystack)):
            run = 0
            while (
                ni + run < len(needle)
                and hi + run < len(haystack)
                and _soft_token_match(needle[ni + run], haystack[hi + run])
            ):
                run += 1
            if run > best:
                best = run
    return best


def _is_placeholder_whisper_token(text: str) -> bool:
    cleaned = "".join(ch for ch in text.lower() if ch.isalpha())
    return cleaned in {"vocal", "silence", "gap"}


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
    """Ensure consecutive lines never overlap in time."""
    from ...models import Line as LineModel

    def _rebalance_degenerate_word_timings(
        line: models.Line,
        *,
        min_word_duration: float,
    ) -> models.Line:
        if not line.words or len(line.words) <= 1:
            return line
        if line.end_time <= line.start_time + (min_word_duration * len(line.words)):
            return line

        starts = [w.start_time for w in line.words]
        is_stacked = any(starts[i] >= starts[i + 1] for i in range(len(starts) - 1))
        if not is_stacked:
            return line

        span = line.end_time - line.start_time
        step = span / len(line.words)
        new_words: List[models.Word] = []
        for idx, w in enumerate(line.words):
            start = line.start_time + idx * step
            end = line.start_time + (idx + 1) * step
            if idx == len(line.words) - 1:
                end = line.end_time
            if end - start < min_word_duration:
                end = min(line.end_time, start + min_word_duration)
            new_words.append(
                models.Word(
                    text=w.text,
                    start_time=start,
                    end_time=end,
                    singer=w.singer,
                )
            )
        return LineModel(words=new_words, singer=line.singer)

    min_word_duration = 0.06
    resolved: List[models.Line] = []
    for line in lines:
        if not line.words:
            resolved.append(line)
            continue
        ordered = sorted(line.words, key=lambda w: (w.start_time, w.end_time))
        resolved.append(LineModel(words=ordered, singer=line.singer))
    for i in range(len(resolved) - 1):
        cur = resolved[i]
        if not cur.words:
            continue
        next_idx = None
        for j in range(i + 1, len(resolved)):
            if resolved[j].words:
                next_idx = j
                break
        if next_idx is None:
            continue
        nxt = resolved[next_idx]
        if cur.end_time > nxt.start_time:
            gap_point = nxt.start_time
            new_words: List[models.Word] = []
            for w in cur.words:
                if w.start_time >= gap_point:
                    new_words.append(
                        models.Word(
                            text=w.text,
                            start_time=max(
                                cur.start_time, gap_point - min_word_duration
                            ),
                            end_time=gap_point,
                            singer=w.singer,
                        )
                    )
                elif w.end_time > gap_point:
                    clipped_start = min(w.start_time, gap_point - min_word_duration)
                    new_words.append(
                        models.Word(
                            text=w.text,
                            start_time=clipped_start,
                            end_time=gap_point,
                            singer=w.singer,
                        )
                    )
                else:
                    new_words.append(w)
            if new_words:
                last = new_words[-1]
                if (last.end_time - last.start_time) < min_word_duration:
                    new_start = max(cur.start_time, last.end_time - min_word_duration)
                    if len(new_words) >= 2 and new_words[-2].end_time > new_start:
                        prev = new_words[-2]
                        new_words[-2] = models.Word(
                            text=prev.text,
                            start_time=prev.start_time,
                            end_time=max(
                                prev.start_time + min_word_duration, new_start
                            ),
                            singer=prev.singer,
                        )
                        new_start = max(new_words[-2].end_time, new_start)
                    new_words[-1] = models.Word(
                        text=last.text,
                        start_time=new_start,
                        end_time=last.end_time,
                        singer=last.singer,
                    )
            resolved[i] = LineModel(words=new_words, singer=cur.singer)
    for i, line in enumerate(resolved):
        if not line.words:
            continue
        resolved[i] = _rebalance_degenerate_word_timings(
            line,
            min_word_duration=min_word_duration,
        )
    return resolved


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
    """Retarget short interjection lines (for example, 'Oooh') to matching segments."""
    if not mapped_lines or not segments:
        return mapped_lines

    adjusted = list(mapped_lines)
    sorted_segments = sorted(segments, key=lambda s: s.start)

    for idx, line in enumerate(adjusted):
        if not line.words or not _is_interjection_line(line.text):
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
        best_score = 0.0
        for seg in sorted_segments:
            if abs(seg.start - line.start_time) > max_shift:
                continue
            score = _interjection_similarity(line.text, seg.text)
            if score > best_score:
                best_score = score
                best_seg = seg

        if best_seg is None or best_score < min_similarity:
            continue

        window_start = best_seg.start
        if prev_end is not None:
            window_start = max(window_start, prev_end + min_gap)

        window_end = best_seg.end
        if next_start is not None:
            window_end = min(window_end, next_start - min_gap)

        if window_end - window_start <= 0.05:
            continue

        total_duration = max(window_end - window_start, 0.2)
        spacing = total_duration / len(line.words)
        new_words = []
        for word_idx, w in enumerate(line.words):
            start = window_start + word_idx * spacing
            end = start + spacing * 0.9
            new_words.append(
                models.Word(
                    text=w.text,
                    start_time=start,
                    end_time=end,
                    singer=w.singer,
                )
            )
        adjusted[idx] = models.Line(words=new_words, singer=line.singer)

    return adjusted


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


def _normalize_match_token(token: str) -> str:
    base = _normalize_interjection_token(token)
    if not base:
        base = "".join(ch for ch in token.lower() if ch.isalpha())
    if base.endswith("s") and len(base) > 3:
        base = base[:-1]
    return base


def _soft_token_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    return a in b or b in a


def _overlap_suffix_prefix(
    left_tokens: List[str],
    right_tokens: List[str],
    max_overlap: int = 3,
) -> int:
    if not left_tokens or not right_tokens:
        return 0
    upper = min(max_overlap, len(left_tokens), len(right_tokens))
    for size in range(upper, 0, -1):
        ok = True
        for i in range(size):
            if not _soft_token_match(left_tokens[-size + i], right_tokens[i]):
                ok = False
                break
        if ok:
            return size
    return 0


def _soft_token_overlap_ratio(left_tokens: List[str], right_tokens: List[str]) -> float:
    if not left_tokens or not right_tokens:
        return 0.0
    matched = 0
    used = [False] * len(right_tokens)
    for lt in left_tokens:
        for idx, rt in enumerate(right_tokens):
            if used[idx]:
                continue
            if _soft_token_match(lt, rt):
                used[idx] = True
                matched += 1
                break
    return matched / max(len(left_tokens), len(right_tokens))


def _realign_repetitive_runs_to_matching_segments(  # noqa: C901
    lines: List[models.Line],
    segments: List[timing_models.TranscriptionSegment],
    *,
    min_gap: float = 0.05,
    min_run_len: int = 3,
    min_overlap: float = 0.4,
    min_seg_similarity: float = 0.15,
    min_shift: float = 0.8,
    max_shift: float = 3.0,
) -> List[models.Line]:
    """Shift early repetitive runs later to segment starts as a block."""
    if not lines or not segments:
        return lines

    adjusted = list(lines)
    sorted_segments = sorted(segments, key=lambda s: s.start)
    idx = 0
    while idx < len(adjusted):
        line = adjusted[idx]
        if not line.words:
            idx += 1
            continue

        run_end = idx + 1
        exact_duplicates = 0
        while run_end < len(adjusted):
            prev = adjusted[run_end - 1]
            cur = adjusted[run_end]
            if not prev.words or not cur.words:
                break
            prev_tokens = [
                _normalize_match_token(w.text)
                for w in prev.words
                if _normalize_match_token(w.text)
            ]
            cur_tokens = [
                _normalize_match_token(w.text)
                for w in cur.words
                if _normalize_match_token(w.text)
            ]
            if not prev_tokens or not cur_tokens:
                break
            if _soft_token_overlap_ratio(prev_tokens, cur_tokens) < min_overlap:
                break
            if prev.text.strip().lower() == cur.text.strip().lower():
                exact_duplicates += 1
            run_end += 1

        run_len = run_end - idx
        if run_len < min_run_len or exact_duplicates == 0:
            idx = run_end
            continue

        run_lines = adjusted[idx:run_end]
        run_window_start = run_lines[0].start_time - 1.0
        run_window_end = run_lines[-1].end_time + 8.0
        first_text = run_lines[0].text
        matching_segments: List[timing_models.TranscriptionSegment] = []
        for seg in sorted_segments:
            if seg.start < run_window_start or seg.start > run_window_end:
                continue
            if len(_normalize_text_tokens(seg.text)) < 2:
                continue
            sim = max(
                _light_text_similarity(first_text, seg.text),
                _soft_text_similarity(first_text, seg.text),
            )
            if sim < min_seg_similarity:
                continue
            matching_segments.append(seg)

        if len(matching_segments) < run_len:
            idx = run_end
            continue

        deltas = [
            run_lines[k].start_time - matching_segments[k].start for k in range(run_len)
        ]
        strongly_early = sum(1 for d in deltas if d <= -min_shift)
        if strongly_early < max(1, run_len // 3):
            idx = run_end
            continue

        median_delta = sorted(deltas)[run_len // 2]
        if median_delta >= -min_shift:
            idx = run_end
            continue

        shift = min(max_shift, -median_delta)
        shift_end = run_end
        next_start = (
            adjusted[shift_end].start_time
            if shift_end < len(adjusted) and adjusted[shift_end].words
            else None
        )
        if (
            next_start is not None
            and next_start - run_lines[-1].end_time < min_shift
            and shift_end + 1 < len(adjusted)
            and adjusted[shift_end].words
            and adjusted[shift_end + 1].words
        ):
            boundary_prev_tokens = [
                _normalize_match_token(w.text)
                for w in adjusted[shift_end - 1].words
                if _normalize_match_token(w.text)
            ]
            boundary_tokens = [
                _normalize_match_token(w.text)
                for w in adjusted[shift_end].words
                if _normalize_match_token(w.text)
            ]
            boundary_overlap = _soft_token_overlap_ratio(
                boundary_prev_tokens, boundary_tokens
            )
            trailing_gap = (
                adjusted[shift_end + 1].start_time - adjusted[shift_end].end_time
            )
            if boundary_overlap >= 0.3 and trailing_gap > 3.0:
                shift_end += 1
                next_start = adjusted[shift_end].start_time

        if next_start is not None:
            shift = min(
                shift,
                max(0.0, next_start - min_gap - adjusted[shift_end - 1].end_time),
            )
        if shift < 0.25:
            idx = run_end
            continue

        for run_idx in range(idx, shift_end):
            cur_line = adjusted[run_idx]
            shifted_words = [
                models.Word(
                    text=w.text,
                    start_time=w.start_time + shift,
                    end_time=w.end_time + shift,
                    singer=w.singer,
                )
                for w in cur_line.words
            ]
            adjusted[run_idx] = models.Line(words=shifted_words, singer=cur_line.singer)
        idx = shift_end

    return adjusted


def _retime_repetitive_question_runs_to_segment_windows(  # noqa: C901
    lines: List[models.Line],
    segments: List[timing_models.TranscriptionSegment],
    *,
    min_gap: float = 0.05,
    min_run_len: int = 3,
    min_overlap: float = 0.3,
    min_seg_similarity: float = 0.3,
) -> List[models.Line]:
    """Retarget repeated short question runs to successive segment windows."""
    if not lines or not segments:
        return lines

    adjusted = list(lines)
    sorted_segments = sorted(segments, key=lambda s: s.start)
    idx = 0
    while idx < len(adjusted):
        cur = adjusted[idx]
        if not cur.words or not cur.text.strip().endswith("?"):
            idx += 1
            continue

        run_end = idx + 1
        while run_end < len(adjusted):
            prev = adjusted[run_end - 1]
            nxt = adjusted[run_end]
            if not prev.words or not nxt.words or not nxt.text.strip().endswith("?"):
                break
            prev_tokens = [
                _normalize_match_token(w.text)
                for w in prev.words
                if _normalize_match_token(w.text)
            ]
            nxt_tokens = [
                _normalize_match_token(w.text)
                for w in nxt.words
                if _normalize_match_token(w.text)
            ]
            overlap = _soft_token_overlap_ratio(prev_tokens, nxt_tokens)
            shared_long_token = any(
                len(tok) >= 4 and tok in set(nxt_tokens) for tok in prev_tokens
            )
            if overlap < min_overlap or (overlap < 0.5 and not shared_long_token):
                break
            run_end += 1

        run_len = run_end - idx
        if run_len < min_run_len:
            idx = run_end
            continue

        run_lines = adjusted[idx:run_end]
        run_text = run_lines[0].text
        run_start = run_lines[0].start_time - 1.0
        run_stop = run_lines[-1].end_time + 10.0
        match_segments: List[timing_models.TranscriptionSegment] = []
        for seg in sorted_segments:
            if seg.start < run_start or seg.start > run_stop:
                continue
            if len(_normalize_text_tokens(seg.text)) < 2:
                continue
            sim = max(
                _light_text_similarity(run_text, seg.text),
                _soft_text_similarity(run_text, seg.text),
            )
            if sim < min_seg_similarity:
                continue
            match_segments.append(seg)

        if len(match_segments) < min_run_len:
            idx = run_end
            continue
        # Don't retime runs beyond Whisper segment coverage; this can pull
        # end-of-song refrains (after transcript tail) incorrectly early.
        if match_segments[-1].end < run_lines[-1].start_time - 0.8:
            idx = run_end
            continue

        assigned_seg_idx: List[int] = []
        seg_ptr = 0
        prev_seg_start: Optional[float] = None
        for run_line in run_lines:
            best_idx: Optional[int] = None
            best_cost = float("inf")
            for cand_idx in range(seg_ptr, len(match_segments)):
                seg = match_segments[cand_idx]
                if prev_seg_start is not None and seg.start < prev_seg_start + 0.6:
                    continue
                cost = abs(seg.start - run_line.start_time)
                if cost < best_cost:
                    best_cost = cost
                    best_idx = cand_idx
                if best_idx is not None and seg.start > run_line.start_time + 4.0:
                    break
            if best_idx is None:
                break
            assigned_seg_idx.append(best_idx)
            prev_seg_start = match_segments[best_idx].start
            seg_ptr = best_idx + 1

        apply_count = min(run_len, len(assigned_seg_idx))
        if apply_count < min_run_len:
            idx = run_end
            continue

        for rel_idx in range(apply_count):
            line = run_lines[rel_idx]
            seg = match_segments[assigned_seg_idx[rel_idx]]
            next_start = (
                match_segments[assigned_seg_idx[rel_idx + 1]].start
                if rel_idx + 1 < apply_count
                else (
                    adjusted[run_end].start_time
                    if run_end < len(adjusted) and adjusted[run_end].words
                    else None
                )
            )
            target_start = seg.start
            if rel_idx > 0:
                target_start = max(
                    target_start, adjusted[idx + rel_idx - 1].end_time + min_gap
                )
            target_start = max(target_start, line.start_time - 0.5)
            base_duration = max(0.9, min(1.8, 0.65 + 0.18 * len(line.words)))
            if len(line.words) >= 5:
                base_duration = min(2.2, base_duration + 0.25)
            target_end = target_start + base_duration
            if next_start is not None:
                target_end = min(target_end, next_start - min_gap)
            target_end = min(target_end, seg.end + 0.4)
            target_end = max(target_end, target_start + 0.7)
            if target_end <= target_start + 0.2:
                continue
            spacing = (target_end - target_start) / len(line.words)
            new_words: List[models.Word] = []
            for w_idx, w in enumerate(line.words):
                ws = target_start + w_idx * spacing
                we = ws + spacing * 0.9
                if w_idx == len(line.words) - 1:
                    we = target_end
                new_words.append(
                    models.Word(
                        text=w.text,
                        start_time=ws,
                        end_time=we,
                        singer=w.singer,
                    )
                )
            adjusted[idx + rel_idx] = models.Line(words=new_words, singer=line.singer)

        idx = run_end

    return adjusted


def _enforce_min_duration_for_short_question_lines(  # noqa: C901
    lines: List[models.Line],
    *,
    min_duration: float = 1.0,
    min_gap: float = 0.05,
) -> List[models.Line]:
    """Prevent short question lines from flashing too quickly."""
    if not lines:
        return lines
    adjusted = list(lines)
    for idx, line in enumerate(adjusted):
        if not line.words or not line.text.strip().endswith("?"):
            continue
        if len(line.words) > 6:
            continue
        duration = line.end_time - line.start_time
        if duration >= min_duration:
            continue
        next_start = (
            adjusted[idx + 1].start_time
            if idx + 1 < len(adjusted) and adjusted[idx + 1].words
            else None
        )
        target_end = line.start_time + min_duration
        if next_start is not None:
            target_end = min(target_end, next_start - min_gap)
            if target_end <= line.end_time + 0.05 and idx + 1 < len(adjusted):
                next_line = adjusted[idx + 1]
                if next_line.words and next_line.text.strip().endswith("?"):
                    need = min_duration - duration
                    if need > 0:
                        after_next_start = (
                            adjusted[idx + 2].start_time
                            if idx + 2 < len(adjusted) and adjusted[idx + 2].words
                            else None
                        )
                        available = (
                            max(0.0, after_next_start - next_line.end_time)
                            if after_next_start is not None
                            else 0.0
                        )
                        shift = min(need, available)
                        if shift > 0.05:
                            j = idx + 1
                            while j < len(adjusted):
                                run_line = adjusted[j]
                                if not run_line.words:
                                    break
                                if j > idx + 1 and not run_line.text.strip().endswith(
                                    "?"
                                ):
                                    break
                                shifted_words = [
                                    models.Word(
                                        text=w.text,
                                        start_time=w.start_time + shift,
                                        end_time=w.end_time + shift,
                                        singer=w.singer,
                                    )
                                    for w in run_line.words
                                ]
                                adjusted[j] = models.Line(
                                    words=shifted_words, singer=run_line.singer
                                )
                                j += 1
                            next_start = adjusted[idx + 1].start_time
                            target_end = min(
                                line.start_time + min_duration, next_start - min_gap
                            )
        if target_end <= line.end_time + 0.05:
            continue
        spacing = (target_end - line.start_time) / len(line.words)
        new_words: List[models.Word] = []
        for w_idx, w in enumerate(line.words):
            ws = line.start_time + w_idx * spacing
            we = ws + spacing * 0.9
            if w_idx == len(line.words) - 1:
                we = target_end
            new_words.append(
                models.Word(
                    text=w.text,
                    start_time=ws,
                    end_time=we,
                    singer=w.singer,
                )
            )
        adjusted[idx] = models.Line(words=new_words, singer=line.singer)
    return adjusted


def _pull_adjacent_similar_lines_across_long_gaps(
    lines: List[models.Line],
    *,
    min_gap: float = 0.05,
    long_gap_threshold: float = 8.0,
    min_similarity: float = 0.5,
) -> List[models.Line]:
    """Pull a line forward when a near-duplicate follows after a long gap."""
    if not lines:
        return lines
    adjusted = list(lines)
    for idx in range(len(adjusted) - 1):
        cur = adjusted[idx]
        nxt = adjusted[idx + 1]
        if not cur.words or not nxt.words:
            continue
        gap = nxt.start_time - cur.end_time
        if gap < long_gap_threshold:
            continue
        cur_tokens = [
            _normalize_match_token(w.text)
            for w in cur.words
            if _normalize_match_token(w.text)
        ]
        nxt_tokens = [
            _normalize_match_token(w.text)
            for w in nxt.words
            if _normalize_match_token(w.text)
        ]
        if not cur_tokens or not nxt_tokens:
            continue
        if _soft_token_overlap_ratio(cur_tokens, nxt_tokens) < min_similarity:
            continue
        if len(cur.words) > 8 or len(nxt.words) > 8:
            continue
        prev_end = (
            adjusted[idx - 1].end_time if idx > 0 and adjusted[idx - 1].words else None
        )
        duration = max(0.7, min(cur.end_time - cur.start_time, 2.5))
        new_end = nxt.start_time - min_gap
        new_start = new_end - duration
        if prev_end is not None:
            new_start = max(new_start, prev_end + min_gap)
            new_end = max(new_end, new_start + 0.4)
        if new_start <= cur.start_time + 0.5:
            continue
        shift = new_start - cur.start_time
        shifted_words = [
            models.Word(
                text=w.text,
                start_time=w.start_time + shift,
                end_time=w.end_time + shift,
                singer=w.singer,
            )
            for w in cur.words
        ]
        adjusted[idx] = models.Line(words=shifted_words, singer=cur.singer)
    return adjusted


def _smooth_adjacent_duplicate_line_cadence(  # noqa: C901
    lines: List[models.Line],
    *,
    min_gap: float = 0.05,
    target_gap: float = 0.18,
    min_adjust_gap: float = 0.75,
    max_adjust_gap: float = 4.0,
    max_pull: float = 2.5,
) -> List[models.Line]:
    if not lines:
        return lines
    adjusted = list(lines)
    for idx in range(1, len(adjusted)):
        prev = adjusted[idx - 1]
        cur = adjusted[idx]
        if not prev.words or not cur.words:
            continue
        prev_text = prev.text.strip().lower()
        cur_text = cur.text.strip().lower()
        if not prev_text or not cur_text:
            continue
        prev_tokens = [
            _normalize_match_token(w.text)
            for w in prev.words
            if _normalize_match_token(w.text)
        ]
        cur_tokens = [
            _normalize_match_token(w.text)
            for w in cur.words
            if _normalize_match_token(w.text)
        ]
        overlap = _soft_token_overlap_ratio(prev_tokens, cur_tokens)
        exact_duplicate = prev_text == cur_text
        question_pair = (
            prev_text.endswith("?")
            and cur_text.endswith("?")
            and len(prev.words) <= 3
            and len(cur.words) <= 3
        )
        similar_refrain = overlap >= 0.5 and min(len(prev_tokens), len(cur_tokens)) >= 2
        short_similar_refrain = (
            similar_refrain
            and len(cur.words) <= 4
            and len(prev.words) <= 4
            and overlap >= 0.75
        )
        if not (exact_duplicate or similar_refrain or question_pair):
            continue
        gap = cur.start_time - prev.end_time
        if (
            exact_duplicate
            and len(cur.words) <= 3
            and cur_text.endswith("?")
            and gap >= 1.3
        ):
            # Keep naturally wider spacing for short duplicate refrain lines.
            continue
        if question_pair:
            min_gap_for_adjust = 0.0
        elif exact_duplicate:
            min_gap_for_adjust = min_adjust_gap
        elif short_similar_refrain:
            min_gap_for_adjust = min_adjust_gap
        elif len(cur.words) >= 5:
            min_gap_for_adjust = 0.15
        else:
            min_gap_for_adjust = 1.9
        if gap < min_gap_for_adjust or gap > max_adjust_gap:
            continue
        repetitive_tokens = len(set(cur_tokens)) <= max(3, len(cur_tokens) // 2)
        max_duration = 1.8 if exact_duplicate else 2.2
        duration = max(0.25, min(cur.end_time - cur.start_time, max_duration))
        next_start = (
            adjusted[idx + 1].start_time
            if idx + 1 < len(adjusted) and adjusted[idx + 1].words
            else None
        )
        desired_gap = (
            0.9
            if question_pair
            else (target_gap if (exact_duplicate or similar_refrain) else 1.2)
        )
        pull_limit = (
            3.2
            if question_pair
            else (max_pull if (exact_duplicate or similar_refrain) else 2.8)
        )
        target_start = prev.end_time + desired_gap
        target_start = max(target_start, cur.start_time - pull_limit)
        if next_start is not None:
            target_start = min(
                target_start,
                max(prev.end_time + min_gap, next_start - min_gap - duration),
            )
        if target_start >= cur.start_time - 0.2:
            continue
        if exact_duplicate and (cur.end_time - cur.start_time) > 1.8:
            target_duration = min(cur.end_time - target_start, 1.6)
            if next_start is not None:
                target_duration = min(
                    target_duration, next_start - min_gap - target_start
                )
            target_duration = max(0.8, target_duration)
            spacing = target_duration / len(cur.words)
            rebuilt_words_exact: List[models.Word] = []
            for word_idx, w in enumerate(cur.words):
                ws = target_start + word_idx * spacing
                we = ws + spacing * 0.9
                if word_idx == len(cur.words) - 1:
                    we = target_start + target_duration
                rebuilt_words_exact.append(
                    models.Word(
                        text=w.text,
                        start_time=ws,
                        end_time=we,
                        singer=w.singer,
                    )
                )
            adjusted[idx] = models.Line(words=rebuilt_words_exact, singer=cur.singer)
        elif (similar_refrain and not exact_duplicate) or question_pair:
            if repetitive_tokens and len(cur.words) >= 5:
                similar_max_duration = 1.7
            elif short_similar_refrain:
                similar_max_duration = 1.6
            else:
                similar_max_duration = 2.2
            target_duration = min(cur.end_time - target_start, similar_max_duration)
            if next_start is not None:
                target_duration = min(
                    target_duration, next_start - min_gap - target_start
                )
            target_duration = max(1.0, target_duration)
            spacing = target_duration / len(cur.words)
            rebuilt_words_refrain: List[models.Word] = []
            for word_idx, w in enumerate(cur.words):
                ws = target_start + word_idx * spacing
                we = ws + spacing * 0.9
                if word_idx == len(cur.words) - 1:
                    we = target_start + target_duration
                rebuilt_words_refrain.append(
                    models.Word(
                        text=w.text,
                        start_time=ws,
                        end_time=we,
                        singer=w.singer,
                    )
                )
            adjusted[idx] = models.Line(words=rebuilt_words_refrain, singer=cur.singer)
        else:
            shift = target_start - cur.start_time
            shifted_words = [
                models.Word(
                    text=w.text,
                    start_time=w.start_time + shift,
                    end_time=w.end_time + shift,
                    singer=w.singer,
                )
                for w in cur.words
            ]
            adjusted[idx] = models.Line(words=shifted_words, singer=cur.singer)
    return adjusted


def _rebalance_short_question_pairs(
    lines: List[models.Line],
    *,
    min_gap: float = 0.05,
) -> List[models.Line]:
    """Rebalance collapsed adjacent short question lines into separate windows."""
    if not lines:
        return lines
    adjusted = list(lines)
    for idx in range(len(adjusted) - 1):
        prev = adjusted[idx]
        cur = adjusted[idx + 1]
        if not prev.words or not cur.words:
            continue
        if len(prev.words) > 4 or len(cur.words) > 4:
            continue
        prev_text = prev.text.strip()
        cur_text = cur.text.strip()
        if not prev_text.endswith("?") or not cur_text.endswith("?"):
            continue
        prev_first = _normalize_match_token(prev.words[0].text)
        cur_first = _normalize_match_token(cur.words[0].text)
        if not _soft_token_match(prev_first, cur_first):
            continue

        prev_dur = prev.end_time - prev.start_time
        cur_dur = cur.end_time - cur.start_time
        if prev_dur >= 0.9 or cur_dur <= 2.4:
            continue

        target_cur_duration = 2.0 if len(cur.words) <= 2 else 2.2
        new_cur_start = max(prev.start_time + 0.8, cur.end_time - target_cur_duration)
        if new_cur_start <= cur.start_time + 0.2:
            continue

        next_start = (
            adjusted[idx + 2].start_time
            if idx + 2 < len(adjusted) and adjusted[idx + 2].words
            else None
        )
        target_cur_duration = 2.0 if len(cur.words) <= 2 else 2.2
        if next_start is not None:
            target_cur_duration = min(
                target_cur_duration, next_start - min_gap - new_cur_start
            )
        target_cur_duration = max(0.8, target_cur_duration)
        spacing = target_cur_duration / len(cur.words)
        rebuilt_cur_words: List[models.Word] = []
        for w_idx, w in enumerate(cur.words):
            ws = new_cur_start + w_idx * spacing
            we = ws + spacing * 0.9
            if w_idx == len(cur.words) - 1:
                we = new_cur_start + target_cur_duration
            rebuilt_cur_words.append(
                models.Word(
                    text=w.text,
                    start_time=ws,
                    end_time=we,
                    singer=w.singer,
                )
            )
        adjusted[idx + 1] = models.Line(words=rebuilt_cur_words, singer=cur.singer)

        target_prev_end = min(
            adjusted[idx + 1].start_time - min_gap,
            prev.start_time + max(1.6, prev_dur),
        )
        if target_prev_end > prev.end_time + 0.2:
            duration = max(target_prev_end - prev.start_time, 0.3)
            spacing = duration / len(prev.words)
            stretched_prev_words: List[models.Word] = []
            for w_idx, w in enumerate(prev.words):
                ws = prev.start_time + w_idx * spacing
                we = ws + spacing * 0.9
                if w_idx == len(prev.words) - 1:
                    we = target_prev_end
                stretched_prev_words.append(
                    models.Word(
                        text=w.text,
                        start_time=ws,
                        end_time=we,
                        singer=w.singer,
                    )
                )
            adjusted[idx] = models.Line(words=stretched_prev_words, singer=prev.singer)

    return adjusted


def _extend_line_to_trailing_whisper_matches(  # noqa: C901
    mapped_lines: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
    *,
    min_extension: float = 0.35,
    min_gap: float = 0.05,
) -> List[models.Line]:
    """Extend line tails when repeated-word ambiguity caused premature completion."""
    if not mapped_lines or not all_words:
        return mapped_lines

    adjusted = list(mapped_lines)

    for idx, line in enumerate(adjusted):
        if not line.words:
            continue

        token_pairs = [
            (word_idx, _normalize_match_token(w.text))
            for word_idx, w in enumerate(line.words)
        ]
        token_pairs = [(word_idx, tok) for word_idx, tok in token_pairs if tok]
        if len(token_pairs) < 2:
            continue

        next_start = (
            adjusted[idx + 1].start_time
            if idx + 1 < len(adjusted) and adjusted[idx + 1].words
            else float("inf")
        )
        window_end = (
            next_start + 3.0 if next_start != float("inf") else line.end_time + 3.0
        )
        candidates = [
            w for w in all_words if (line.start_time - 0.25) <= w.start <= window_end
        ]
        if not candidates:
            continue

        best_end = line.end_time
        best_match_count = 0
        best_pairs: List[Tuple[int, timing_models.TranscriptionWord]] = []
        for start_i in range(len(candidates)):
            wi = start_i
            matched = 0
            last_end = None
            matched_pairs: List[Tuple[int, timing_models.TranscriptionWord]] = []
            for word_idx, tok in token_pairs:
                found = False
                while wi < len(candidates):
                    ww_tok = _normalize_match_token(candidates[wi].text)
                    if _soft_token_match(tok, ww_tok):
                        matched += 1
                        last_end = candidates[wi].end
                        matched_pairs.append((word_idx, candidates[wi]))
                        wi += 1
                        found = True
                        break
                    wi += 1
                if wi >= len(candidates):
                    break
                if not found:
                    continue

            if last_end is None:
                continue
            min_required = max(2, int(len(token_pairs) * 0.66))
            if matched < min_required:
                continue
            if (matched > best_match_count) or (
                matched == best_match_count and last_end > best_end
            ):
                best_match_count = matched
                best_end = last_end
                best_pairs = matched_pairs

        target_end = best_end

        if next_start != float("inf"):
            target_end = min(target_end, next_start - min_gap)

        if target_end <= line.end_time + min_extension:
            continue

        new_words = list(line.words)

        # Prefer real Whisper word timings where available to preserve natural pauses.
        if best_pairs:
            anchor_times = {word_idx: ww for word_idx, ww in best_pairs}
            first_anchor_idx = min(anchor_times)
            last_anchor_idx = max(anchor_times)
            for word_idx, ww in anchor_times.items():
                new_words[word_idx] = models.Word(
                    text=new_words[word_idx].text,
                    start_time=ww.start,
                    end_time=ww.end,
                    singer=new_words[word_idx].singer,
                )

            for word_idx in range(first_anchor_idx - 1, -1, -1):
                right = new_words[word_idx + 1].start_time
                duration = max(
                    0.12,
                    new_words[word_idx].end_time - new_words[word_idx].start_time,
                )
                end = max(line.start_time, right - 0.02)
                start = max(line.start_time, end - duration)
                new_words[word_idx] = models.Word(
                    text=new_words[word_idx].text,
                    start_time=start,
                    end_time=end,
                    singer=new_words[word_idx].singer,
                )

            for word_idx in range(last_anchor_idx + 1, len(new_words)):
                left = new_words[word_idx - 1].end_time
                end_cap = target_end if word_idx == len(new_words) - 1 else left + 0.28
                duration = max(
                    0.12,
                    min(0.25, end_cap - left - 0.02),
                )
                start = max(left + 0.02, end_cap - duration)
                end = max(start + 0.08, end_cap)
                new_words[word_idx] = models.Word(
                    text=new_words[word_idx].text,
                    start_time=start,
                    end_time=end,
                    singer=new_words[word_idx].singer,
                )
        else:
            start = line.start_time
            duration = max(target_end - start, 0.2)
            spacing = duration / len(line.words)
            rebuilt_words: List[models.Word] = []
            for word_idx, w in enumerate(line.words):
                ws = start + word_idx * spacing
                we = ws + spacing * 0.9
                if word_idx == len(line.words) - 1:
                    we = target_end
                rebuilt_words.append(
                    models.Word(
                        text=w.text,
                        start_time=ws,
                        end_time=we,
                        singer=w.singer,
                    )
                )
            new_words = rebuilt_words

        if new_words[-1].end_time < target_end:
            last = new_words[-1]
            new_words[-1] = models.Word(
                text=last.text,
                start_time=min(last.start_time, target_end - 0.08),
                end_time=target_end,
                singer=last.singer,
            )

        for word_idx in range(1, len(new_words)):
            prev = new_words[word_idx - 1]
            cur = new_words[word_idx]
            if cur.start_time <= prev.end_time:
                shift = prev.end_time + 0.01 - cur.start_time
                new_words[word_idx] = models.Word(
                    text=cur.text,
                    start_time=cur.start_time + shift,
                    end_time=max(cur.end_time + shift, cur.start_time + shift + 0.06),
                    singer=cur.singer,
                )

        if next_start != float("inf") and new_words[-1].end_time >= next_start:
            last = new_words[-1]
            clipped_end = max(last.start_time + 0.06, next_start - min_gap)
            new_words[-1] = models.Word(
                text=last.text,
                start_time=last.start_time,
                end_time=clipped_end,
                singer=last.singer,
            )

        adjusted[idx] = models.Line(words=new_words, singer=line.singer)

    adjusted = _smooth_adjacent_duplicate_line_cadence(adjusted)
    return _rebalance_short_question_pairs(adjusted)
