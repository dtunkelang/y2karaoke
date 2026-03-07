# Human Correction Delta Report

- Baseline: `/tmp/two_song_curated_baseline_report.json`
- Corrected: `benchmarks/results/20260306T_two_song_curated_fresh/benchmark_report.json`
- Songs compared: `2`
- Net song outcomes: improved=`1`, regressed=`0`, unchanged=`1`

## Aggregate Deltas

| Metric | Baseline | Corrected | Delta | Improved |
|---|---:|---:|---:|---|
| timing_quality | 0.8075 | 0.7538 | -0.0537 | no |
| agreement_coverage | 0.2206 | 0.2236 | +0.0030 | yes |
| agreement_start_p95_abs_sec | 0.8800 | 0.8473 | -0.0327 | yes |
| agreement_bad_ratio | 0.1177 | 0.0732 | -0.0445 | yes |
| dtw_word_coverage | 0.9422 | 0.7887 | -0.1535 | no |
| gold_start_abs_word | 2.5700 | 1.0190 | -1.5510 | yes |

## Top Net Improvements

| Song | Net | Improved | Regressed | timing_quality_delta | agreement_cov_delta | p95_delta_sec |
|---|---:|---:|---:|---:|---:|---:|
| The Weeknd - Blinding Lights | 1 | 3 | 2 | -0.0436 | +0.0571 | +0.0095 |
| Indila - Derniere danse | 0 | 3 | 3 | -0.0594 | -0.0510 | -0.0750 |

## Top Net Regressions

| Song | Net | Improved | Regressed | timing_quality_delta | agreement_cov_delta | p95_delta_sec |
|---|---:|---:|---:|---:|---:|---:|
| Indila - Derniere danse | 0 | 3 | 3 | -0.0594 | -0.0510 | -0.0750 |
| The Weeknd - Blinding Lights | 1 | 3 | 2 | -0.0436 | +0.0571 | +0.0095 |
