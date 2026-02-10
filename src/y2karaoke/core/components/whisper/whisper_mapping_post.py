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

    for line in mapped_lines:
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
                    idx
                    for idx, ww in enumerate(all_words)
                    if idx > prev_idx and ww.start >= required_time
                ),
                None,
            )
            if start_idx is None:
                start_idx = next(
                    (idx for idx, _ww in enumerate(all_words) if idx > prev_idx),
                    None,
                )
            if start_idx is not None and (all_words[start_idx].start - prev_end > 10.0):
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

        adjusted_lines.append(line)
        if line.words:
            if assigned_end_idx is None:
                assigned_end_idx = next(
                    (
                        idx
                        for idx, ww in enumerate(all_words)
                        if abs(ww.start - line.words[-1].start_time) < 0.05
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


def _resolve_line_overlaps(lines: List[models.Line]) -> List[models.Line]:
    """Ensure consecutive lines never overlap in time."""
    from ...models import Line as LineModel

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
    return resolved


def _pull_late_lines_to_matching_segments(  # noqa: C901
    mapped_lines: List[models.Line],
    segments: List[timing_models.TranscriptionSegment],
    language: str,  # retained for call compatibility
    min_similarity: float = 0.4,
    min_late: float = 1.0,
    max_late: float = 3.0,
    strong_match_max_late: float = 6.0,
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

        if idx > 0 and adjusted[idx - 1].words and len(line.words) <= 4:
            prev_tokens = [
                _normalize_match_token(w.text)
                for w in adjusted[idx - 1].words[-3:]
                if _normalize_match_token(w.text)
            ]
            cur_tokens = [
                _normalize_match_token(w.text)
                for w in line.words[:3]
                if _normalize_match_token(w.text)
            ]
            if _overlap_suffix_prefix(prev_tokens, cur_tokens, max_overlap=3) >= 2:
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
        for seg in sorted_segments:
            if abs(seg.start - line.start_time) > max_time_window:
                continue
            sim = _light_text_similarity(line.text, seg.text)
            contains = _contains_token_sequence(line.text, seg.text)
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

        if best_seg is None or best_sim < min_similarity:
            continue

        late_by = line.start_time - best_seg.start
        allowed_max_late = strong_match_max_late if best_contains else max_late
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

    return adjusted


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

        candidates = [
            w
            for w in all_words
            if (fw.start_time - 1.0) <= w.start <= (fw.start_time + 2.0)
        ]
        best = None
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
            score = (match_score, -abs(w.start - fw.start_time))
            if best is None or score > best[0]:
                best = (score, w)

        if best is None:
            continue

        target_start = best[1].start
        delta = fw.start_time - target_start
        if delta >= -early_threshold:
            continue

        shift = min(-delta, max_shift)
        max_allowed = shift
        if prev_end is not None:
            max_allowed = min(
                max_allowed, max(0.0, fw.start_time - (prev_end + min_gap))
            )
        if next_start is not None:
            max_allowed = min(
                max_allowed,
                max(0.0, (next_start - min_gap) - line.end_time),
            )
        if max_allowed <= 0:
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

    return adjusted
