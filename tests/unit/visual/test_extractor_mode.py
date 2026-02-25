from __future__ import annotations

from y2karaoke.core.visual.extractor_mode import resolve_visual_extractor_mode


def _mk_frame(t: float, row_centers: list[float], *, words_per_row: int = 2):
    words = []
    x = 10.0
    for r_i, y in enumerate(row_centers):
        for w_i in range(words_per_row):
            words.append(
                {
                    "text": f"w{r_i}_{w_i}",
                    "x": x + 20.0 * w_i,
                    "y": y + (w_i % 2) * 2.0,
                    "w": 18.0,
                    "h": 8.0,
                }
            )
    return {"time": t, "words": words}


def test_resolve_visual_extractor_mode_manual_modes(monkeypatch):
    monkeypatch.delenv("Y2K_VISUAL_BLOCK_FIRST_PROTOTYPE", raising=False)
    sel = resolve_visual_extractor_mode("line-first")
    assert sel.resolved_mode == "line-first"
    assert sel.reason == "manual_override"

    sel = resolve_visual_extractor_mode("block-first")
    assert sel.resolved_mode == "block-first"
    assert sel.reason == "manual_override"


def test_resolve_visual_extractor_mode_auto_defaults_to_line_first(monkeypatch):
    monkeypatch.delenv("Y2K_VISUAL_BLOCK_FIRST_PROTOTYPE", raising=False)
    sel = resolve_visual_extractor_mode("auto")
    assert sel.resolved_mode == "line-first"
    assert sel.reason == "auto_default_line_first"


def test_resolve_visual_extractor_mode_auto_honors_env_compat(monkeypatch):
    monkeypatch.setenv("Y2K_VISUAL_BLOCK_FIRST_PROTOTYPE", "1")
    sel = resolve_visual_extractor_mode("auto")
    assert sel.resolved_mode == "block-first"
    assert sel.reason == "env_compat_block_first_prototype"


def test_resolve_visual_extractor_mode_auto_selects_block_first_for_stable_blocks(
    monkeypatch,
):
    monkeypatch.delenv("Y2K_VISUAL_BLOCK_FIRST_PROTOTYPE", raising=False)
    frames = []
    # Simulate a basic karaoke video with repeated stable 4-row screens.
    t = 0.0
    for _ in range(30):
        frames.append(_mk_frame(t, [100.0, 130.0, 160.0, 190.0]))
        t += 0.1
    for _ in range(4):
        frames.append({"time": t, "words": []})
        t += 0.1
    for _ in range(15):
        frames.append(_mk_frame(t, [104.0, 134.0, 164.0, 194.0]))
        t += 0.1
    for _ in range(15):
        # A later stable block at a shifted vertical position creates non-trivial row
        # variance while still preserving long stable runs.
        frames.append(_mk_frame(t, [110.0, 140.0, 170.0, 200.0]))
        t += 0.1

    sel = resolve_visual_extractor_mode("auto", raw_frames=frames)
    assert sel.resolved_mode == "block-first"
    assert sel.reason == "auto_block_style_layout_stable"
    assert sel.diagnostics is not None
    assert sel.diagnostics["modal_row_count"] == 4


def test_resolve_visual_extractor_mode_auto_keeps_line_first_for_sparse_or_unstable(
    monkeypatch,
):
    monkeypatch.delenv("Y2K_VISUAL_BLOCK_FIRST_PROTOTYPE", raising=False)
    frames = []
    t = 0.0
    # Sparse and jittery rows should not trigger block-first.
    for i in range(40):
        if i % 3 == 0:
            frames.append({"time": t, "words": []})
        else:
            base = 80.0 + (i * 7.0)
            frames.append(_mk_frame(t, [base, base + 40.0], words_per_row=2))
        t += 0.1
    sel = resolve_visual_extractor_mode("auto", raw_frames=frames)
    assert sel.resolved_mode == "line-first"
    assert sel.reason == "auto_default_line_first"
