from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal, Optional

VisualExtractorMode = Literal["auto", "line-first", "block-first"]
ResolvedVisualExtractorMode = Literal["line-first", "block-first"]


@dataclass(frozen=True)
class VisualExtractorSelection:
    requested_mode: VisualExtractorMode
    resolved_mode: ResolvedVisualExtractorMode
    reason: str
    diagnostics: dict[str, Any] | None = None


def _frame_row_centers(words: list[dict[str, Any]]) -> list[float]:
    ys = sorted(float(w.get("y", 0.0)) for w in words if isinstance(w, dict))
    if not ys:
        return []
    groups: list[list[float]] = [[ys[0]]]
    for y in ys[1:]:
        if y - groups[-1][-1] < 22.0:
            groups[-1].append(y)
        else:
            groups.append([y])
    centers: list[float] = []
    for g in groups:
        if len(g) >= 2:
            centers.append(sum(g) / float(len(g)))
    return centers


def _collect_block_style_signals(  # noqa: C901
    raw_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    total_frames = len(raw_frames)
    if total_frames <= 0:
        return {"total_frames": 0, "eligible_frames": 0}

    eligible: list[tuple[float, int, list[float]]] = []
    sigs: list[tuple[int, tuple[int, ...], float]] = []
    for fr in raw_frames:
        words = [w for w in fr.get("words", []) if isinstance(w, dict)]
        if len(words) < 4:
            continue
        centers = _frame_row_centers(words)
        if len(centers) < 2:
            continue
        row_count = len(centers)
        if row_count > 6:
            continue
        t = float(fr.get("time", 0.0))
        eligible.append((t, row_count, centers))
        sig = (row_count, tuple(int(round(c / 12.0)) for c in centers), t)
        sigs.append(sig)

    if not eligible:
        return {"total_frames": total_frames, "eligible_frames": 0}

    row_counts = Counter(rc for _, rc, _ in eligible)
    modal_row_count, modal_count = row_counts.most_common(1)[0]
    modal_frames = [
        (t, centers) for t, rc, centers in eligible if rc == modal_row_count
    ]
    modal_ratio = modal_count / float(len(eligible))
    eligible_ratio = len(eligible) / float(total_frames)

    y_stds: list[float] = []
    if 2 <= modal_row_count <= 6 and modal_frames:
        per_idx: list[list[float]] = [[] for _ in range(modal_row_count)]
        for _, centers in modal_frames:
            if len(centers) != modal_row_count:
                continue
            for i, c in enumerate(centers):
                per_idx[i].append(c)
        for vals in per_idx:
            if len(vals) >= 3:
                mean_v = sum(vals) / float(len(vals))
                var = sum((v - mean_v) ** 2 for v in vals) / float(len(vals))
                y_stds.append(var**0.5)
    mean_row_y_std = sum(y_stds) / float(len(y_stds)) if y_stds else None

    # Stable layout run detection by exact quantized row signature.
    long_run_count = 0
    long_run_total_frames = 0
    if sigs:
        prev_key = None
        run_len = 0
        for rc, qcenters, _t in sigs:
            key = (rc, qcenters)
            if key == prev_key:
                run_len += 1
            else:
                if prev_key is not None and run_len >= 8:  # ~0.8s at 10fps
                    long_run_count += 1
                    long_run_total_frames += run_len
                prev_key = key
                run_len = 1
        if prev_key is not None and run_len >= 8:
            long_run_count += 1
            long_run_total_frames += run_len
    long_run_ratio = long_run_total_frames / float(len(eligible)) if eligible else 0.0

    return {
        "total_frames": total_frames,
        "eligible_frames": len(eligible),
        "eligible_ratio": round(eligible_ratio, 4),
        "modal_row_count": modal_row_count,
        "modal_row_ratio": round(modal_ratio, 4),
        "mean_row_y_std": None if mean_row_y_std is None else round(mean_row_y_std, 3),
        "long_stable_run_count": long_run_count,
        "long_stable_run_ratio": round(long_run_ratio, 4),
    }


def resolve_visual_extractor_mode(
    requested_mode: VisualExtractorMode,
    *,
    raw_frames: Optional[list[dict[str, Any]]] = None,
) -> VisualExtractorSelection:
    """Resolve visual extractor mode.

    `auto` intentionally resolves to `line-first` while the block-first backend
    remains under active development. This function is the seam for a future
    confidence-based mode selector.
    """
    if requested_mode == "block-first":
        return VisualExtractorSelection(
            requested_mode=requested_mode,
            resolved_mode="block-first",
            reason="manual_override",
            diagnostics=None,
        )
    if requested_mode == "line-first":
        return VisualExtractorSelection(
            requested_mode=requested_mode,
            resolved_mode="line-first",
            reason="manual_override",
            diagnostics=None,
        )
    if os.environ.get("Y2K_VISUAL_BLOCK_FIRST_PROTOTYPE", "0") == "1":
        return VisualExtractorSelection(
            requested_mode=requested_mode,
            resolved_mode="block-first",
            reason="env_compat_block_first_prototype",
            diagnostics=None,
        )
    diagnostics = None
    if raw_frames is not None:
        diagnostics = _collect_block_style_signals(raw_frames)
        modal_row_count = int(diagnostics.get("modal_row_count", 0) or 0)
        modal_row_ratio = float(diagnostics.get("modal_row_ratio", 0.0) or 0.0)
        eligible_ratio = float(diagnostics.get("eligible_ratio", 0.0) or 0.0)
        long_run_ratio = float(diagnostics.get("long_stable_run_ratio", 0.0) or 0.0)
        mean_row_y_std = diagnostics.get("mean_row_y_std")
        mean_row_y_std_f = (
            float(mean_row_y_std) if isinstance(mean_row_y_std, (int, float)) else None
        )
        y_ok = mean_row_y_std_f is not None and mean_row_y_std_f <= 10.0
        if (
            2 <= modal_row_count <= 5
            and modal_row_ratio >= 0.74
            and eligible_ratio >= 0.90
            and long_run_ratio >= 0.93
            and y_ok
            and mean_row_y_std_f is not None
            and mean_row_y_std_f >= 3.0
        ):
            return VisualExtractorSelection(
                requested_mode=requested_mode,
                resolved_mode="block-first",
                reason="auto_block_style_layout_stable",
                diagnostics=diagnostics,
            )
    return VisualExtractorSelection(
        requested_mode=requested_mode,
        resolved_mode="line-first",
        reason="auto_default_line_first",
        diagnostics=diagnostics,
    )
