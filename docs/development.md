# Development Workflow

## Bootstrap

Use the project bootstrap script to create/update `venv` and install dependencies from `pyproject.toml`:

```bash
./tools/bootstrap_dev.sh
source venv/bin/activate
```

**System Dependencies:**
- ffmpeg (for audio/video processing)
- Tesseract OCR (required for the Karaoke bootstrap tool)

## Daily Commands

Run the standard checks:

```bash
make check
```

`make check` now starts with `pip check` to catch dependency conflicts early.

Run individual steps:

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

## CI Lanes

- Fast lane: formatting/linting/types + fast unit tests + perf smoke.
- Fast lane also enforces `tools/quality_guardrails.py` to prevent oversized files
  and monkeypatch-specific production seams.
- Full lane: full non-network test suite with coverage.

## Generated Artifacts

- `src/y2karaoke.egg-info/` is generated packaging metadata.
- Keep it out of version control; it is ignored via `.gitignore`.
- If you need to refresh metadata locally, reinstall with `pip install -e .`.

## Perf Smoke Thresholds

- `tools/perf_smoke.py` has conservative per-test time budgets.
- Tune thresholds only after reviewing at least a week of CI runtimes.
- Prefer widening by small increments (for example, `+0.5s`) to avoid masking regressions.
