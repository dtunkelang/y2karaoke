"""Pipeline implementations for timing evaluator correction flows."""

from typing import Callable, List, Optional, Tuple

from ...models import Line, Word
from .timing_models import AudioFeatures


def correct_line_timestamps_impl(
    lines: List[Line],
    audio_features: Optional[AudioFeatures],
    max_correction: float,
    *,
    check_vocal_activity_fn: Callable[[float, float, AudioFeatures], float],
    find_best_onset_for_phrase_end_fn: Callable[..., Optional[float]],
    find_best_onset_proximity_fn: Callable[..., Optional[float]],
    find_best_onset_during_silence_fn: Callable[..., Optional[float]],
    find_phrase_end_fn: Callable[..., float],
) -> Tuple[List[Line], List[str]]:
    """Correct line timestamps to align with detected vocal onsets."""
    if not lines:
        return lines, []
    if audio_features is None:
        return lines, ["Audio features unavailable; skipping onset corrections"]

    corrected_lines: List[Line] = []
    corrections: List[str] = []
    onset_times = audio_features.onset_times
    prev_line_audio_end = 0.0

    first_line = next((line for line in lines if line.words), None)
    if first_line is not None:
        first_start = first_line.start_time
        vocal_start = audio_features.vocal_start
        if vocal_start > 0 and first_start < vocal_start - 0.5:
            global_offset = vocal_start - first_start
            shifted_lines: List[Line] = []
            for line in lines:
                if not line.words:
                    shifted_lines.append(line)
                    continue
                new_words = [
                    Word(
                        text=word.text,
                        start_time=word.start_time + global_offset,
                        end_time=word.end_time + global_offset,
                        singer=word.singer,
                    )
                    for word in line.words
                ]
                shifted_lines.append(Line(words=new_words, singer=line.singer))
            lines = shifted_lines
            corrections.append(
                f"Global shift {global_offset:+.1f}s to align with vocal start"
            )

    for i, line in enumerate(lines):
        if not line.words:
            corrected_lines.append(line)
            continue

        line_start = line.start_time
        singing_at_lrc_time = (
            check_vocal_activity_fn(line_start - 0.5, line_start + 0.5, audio_features)
            > 0.3
        )

        best_onset = None
        if len(onset_times) > 0:
            if singing_at_lrc_time:
                silence_after = (
                    check_vocal_activity_fn(
                        line_start + 0.1, line_start + 0.6, audio_features
                    )
                    < 0.3
                )
                silence_before = (
                    check_vocal_activity_fn(
                        max(0, line_start - 0.6), line_start - 0.1, audio_features
                    )
                    < 0.3
                )

                if silence_after and not silence_before:
                    best_onset = find_best_onset_for_phrase_end_fn(
                        onset_times, line_start, prev_line_audio_end, audio_features
                    )
                else:
                    best_onset = find_best_onset_proximity_fn(
                        onset_times, line_start, max_correction, audio_features
                    )
            else:
                best_onset = find_best_onset_during_silence_fn(
                    onset_times,
                    line_start,
                    prev_line_audio_end,
                    max_correction,
                    audio_features,
                )

            if best_onset is not None:
                offset = best_onset - line_start
                if abs(offset) > 0.3:
                    new_words = [
                        Word(
                            text=word.text,
                            start_time=word.start_time + offset,
                            end_time=word.end_time + offset,
                            singer=word.singer,
                        )
                        for word in line.words
                    ]

                    corrected_lines.append(Line(words=new_words, singer=line.singer))
                    prev_line_audio_end = find_phrase_end_fn(
                        best_onset,
                        best_onset + 30.0,
                        audio_features,
                        min_silence_duration=0.3,
                    )

                    line_text = " ".join(w.text for w in line.words)[:30]
                    corrections.append(
                        f'Line {i+1} shifted {offset:+.1f}s: "{line_text}..."'
                    )
                    continue

        corrected_lines.append(line)
        prev_line_audio_end = find_phrase_end_fn(
            line_start, line_start + 30.0, audio_features, min_silence_duration=0.3
        )

    return corrected_lines, corrections


def fix_spurious_gaps_impl(
    lines: List[Line],
    audio_features: AudioFeatures,
    activity_threshold: float,
    *,
    collect_lines_to_merge_fn: Callable[..., Tuple[List[Line], int]],
    merge_lines_with_audio_fn: Callable[..., Tuple[Line, float]],
) -> Tuple[List[Line], List[str]]:
    """Fix spurious gaps by merging lines that should be continuous."""
    if not lines:
        return lines, []

    fixed_lines: List[Line] = []
    fixes: List[str] = []
    i = 0

    while i < len(lines):
        current_line = lines[i]
        if not current_line.words:
            fixed_lines.append(current_line)
            i += 1
            continue

        lines_to_merge, j = collect_lines_to_merge_fn(
            lines, i, audio_features, activity_threshold
        )
        if len(lines_to_merge) > 1:
            next_line_start = (
                lines[j].start_time if j < len(lines) and lines[j].words else None
            )
            merged_line, phrase_end = merge_lines_with_audio_fn(
                lines_to_merge, next_line_start, audio_features
            )
            fixed_lines.append(merged_line)

            phrase_start = lines_to_merge[0].start_time
            merged_texts = [
                " ".join(w.text for w in line.words)[:20] for line in lines_to_merge
            ]
            fixes.append(
                f"Merged {len(lines_to_merge)} lines ({i+1}-{i+len(lines_to_merge)}): "
                f"duration {phrase_end - phrase_start:.1f}s - "
                f"\"{' + '.join(merged_texts)}...\""
            )

            i = j
            continue

        fixed_lines.append(current_line)
        i += 1

    return fixed_lines, fixes
