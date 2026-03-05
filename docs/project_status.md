# Project Status

Last updated: 2026-03-04

## Canonical Sources

- Product and usage: `README.md`
- Architecture: `ARCHITECTURE.md`
- Development workflow and quality gates: `docs/development.md`
- Pipeline-specific behavior: `docs/pipelines/`
- Benchmark manifests and guardrails:
  - `benchmarks/main_benchmark_songs.yaml`
  - `benchmarks/main_dev_songs.yaml`
  - `benchmarks/visual_benchmark_songs.yaml`
  - `benchmarks/visual_dev_songs.yaml`
  - `benchmarks/main_benchmark_guardrails.json`
  - `benchmarks/visual_eval_guardrails.json`

## Current Focus

1. Maintain green CI/CD on every push (`black`, `flake8`, `mypy`, unit tests, quality guardrails).
2. Continue quality improvements in timing/alignment and visual extraction.
3. Keep benchmark guardrails stable while running riskier experiments in development manifests.

## Artifact Policy

- Keep generated benchmark runs under `benchmarks/results/` (ignored in git).
- Keep ad-hoc experiments under `benchmarks/experiments/` (ignored in git).
- Keep transient logs out of version control (`*.log` ignored).
- Do not add one-off snapshots to repo root; place reusable docs under `docs/`.
