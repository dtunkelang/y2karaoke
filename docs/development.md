# Development Workflow

Last updated: 2026-03-19

## Bootstrap

Create or refresh the local environment with:

```bash
./tools/bootstrap_dev.sh
source venv/bin/activate
```

Core system dependencies:
- `ffmpeg`
- `tesseract` for the visual bootstrap workflow

## Daily Commands

Primary local gate:

```bash
make check
```

Useful focused commands:

```bash
make fmt
make dep-check
make lint
make type
make test-fast
make test-full
make perf-smoke
make quality-guardrails
```

Notes:
- `make check` starts with `pip check`.
- Normal pytest runs do not require `PYTHONPATH=src`; the test configuration prefers the local source tree automatically.
- For direct one-off script or interpreter commands where the editable install may be stale, prefer `PYTHONPATH=src`.
- For benchmark scripts and ad hoc helper scripts, prefer `PYTHONPATH=src ./.venv/bin/python ...` unless you are sure the editable install is current.

## CI Expectations

Fast lane:
- formatting
- lint
- types
- fast unit tests
- perf smoke
- `tools/quality_guardrails.py`

Full lane:
- broader non-network test coverage with coverage reporting

The custom guardrail script is expected to catch:
- oversized source files
- excessive complexity
- certain test-seam / production-boundary regressions

## Generated Artifacts

Keep these out of version control:
- `src/y2karaoke.egg-info/`
- benchmark results under `benchmarks/results/`
- scratch experiments under `benchmarks/experiments/`
- transient logs such as `*.log`

If the editable install or metadata is stale:

```bash
pip install -e ".[dev]"
```

Cleanup example:

```bash
rm -rf benchmarks/experiments
find . -maxdepth 1 -name "*.log" -delete
```

## Documentation Maintenance

When structural work lands, update:
- `ARCHITECTURE.md`
- `docs/tech_debt_backlog.md`
- `docs/README.md`

Use `NEXT_SESSION_TODO.md` only as a lightweight handoff, not as a second backlog.

When manual curated clip work lands, also update:
- `docs/curated_clips.md`
- `benchmarks/curated_clip_songs.yaml` if clip windows or tags changed

Manual curation is high-cost user work, so protect it immediately:
- verify the saved gold JSON on disk
- verify the canonical clip audio path and duration
- reopen the saved gold file through `tools/curated_clip_helper.py`
- commit and push before doing any more engineering work

For quality work on curated clips:
- prefer `PYTHONPATH=src ./.venv/bin/python ...` so ad hoc script runs use the repo source, not an installed package copy
- if a hard clip is still failing after a plausible fix, compare the real helper-generated seed, the accepted forced-alignment output, and the final timing report before adding more heuristics
- treat 2-line clips as their own family; they can expose structural assumptions that never appear in longer repeated-hook packs
- use a narrow iteration loop even when time pressure is low:
  - pick one clip, one line, and one likely code path
  - keep the exact artifact, rerun command, and local checks next to the current hypothesis
  - define success and failure criteria before broadening a fix
  - commit/push after a real behavioral win or an expensive-to-rediscover diagnostic elimination, but batch tiny note-only updates in normal mode
