# Development Workflow

## Bootstrap

Use the project bootstrap script to create/update `venv` and install dependencies from `pyproject.toml`:

```bash
./tools/bootstrap_dev.sh
source venv/bin/activate
```

## Daily Commands

Run the standard checks:

```bash
make check
```

Run individual steps:

```bash
make fmt
make lint
make type
make test-fast
make test-full
make perf-smoke
```

## CI Lanes

- Fast lane: formatting/linting/types + fast unit tests + perf smoke.
- Full lane: full non-network test suite with coverage.
