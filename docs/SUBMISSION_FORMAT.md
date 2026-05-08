# Submission Format

A valid AIRisk-2030 submission is a single JSON file. This page is the
authoritative spec; the validator at
`airisk/evaluation/leaderboard.py:validate_submission` enforces it
mechanically.

## Top-level structure

```json
{
  "schema_version": 1,
  "scenario": "baseline",
  "n_sims": 500,
  "quarters": ["2026Q3", "2026Q4", ..., "2030Q4"],
  "forecasts": {
    "track_a_metr_task_horizon_hours":         { ... },
    "track_b_aggregate_automation":            { ... },
    "track_b_unemployment_rate":               { ... },
    "track_c_cumulative_catastrophic":         { ... },
    "track_a3_threshold_crossing_probabilities": { ... }
  }
}
```

- `schema_version` **must** be `1`.
- `scenario` is the policy scenario the forecast applies to. Use
  `"baseline"` unless you are submitting to Track C3 counterfactuals.
- `quarters` lists the forecast horizons. The official schedule is **18
  quarters from 2026Q3 through 2030Q4**.

## Continuous targets

For each continuous target, supply a quantile representation of the
predictive distribution at every quarter:

```json
{
  "quantile_levels": [0.05, 0.10, 0.15, ..., 0.95],
  "values_by_quarter": [
    [v_q1_at_5pct, v_q1_at_10pct, ..., v_q1_at_95pct],
    [v_q2_at_5pct, v_q2_at_10pct, ..., v_q2_at_95pct],
    ...
  ]
}
```

Constraints:

- `quantile_levels` must be strictly in `(0, 1)` and strictly increasing.
- We recommend **19 levels at 5% increments** (`0.05, 0.10, ..., 0.95`)
  for compatibility with the reference scorer; other meshes are accepted
  but interpolation may add noise.
- For each row in `values_by_quarter`, values must be non-decreasing
  (a CDF inverse cannot decrease).

## Threshold-crossing target (Track A3)

```json
"track_a3_threshold_crossing_probabilities": {
  "task_horizon_ge_168":  [0.05, 0.12, ..., 0.91],
  "task_horizon_ge_720":  [...],
  "task_horizon_ge_2160": [...]
}
```

Each list has length `len(quarters)`. Each value is the *cumulative*
probability that the threshold has been crossed by the end of the
corresponding quarter, so the lists must be **non-decreasing** and lie in
`[0, 1]`.

## Concrete example

Run `scripts/make_example_submission.py` to write a complete, valid
example to `examples/baseline_submission/submission.json`. You can use
that file as a starting template.

## Validation locally

```bash
python scripts/score_submission.py path/to/your/submission.json
```

This performs structural validation only. To also score against the
synthetic public ground-truth file, first build it:

```bash
python scripts/build_ground_truth_public.py
python scripts/score_submission.py path/to/your/submission.json \
    --ground-truth data/ground_truth_public.json
```
