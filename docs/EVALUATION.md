# Evaluation Protocol

This document describes the official scoring rules for the AIRisk-2030
leaderboard. The implementation lives in `airisk/evaluation/`; see
`scoring.py` and `leaderboard.py` for code-level details.

## Scoring rules

All forecasts are scored using **strictly proper scoring rules**: rules
under which the unique optimal strategy is to report your true
predictive distribution. By construction these rules reward calibration
and penalize both over- and under-confidence.

| Track | Target | Primary score | Auxiliary scores |
|-------|--------|---------------|------------------|
| A | METR task-horizon (continuous) | CRPS | tail-weighted CRPS, sharpness |
| A3 | Threshold-crossing probabilities | log-loss | Brier, ECE |
| B | Sectoral automation, unemployment, productivity | CRPS | tail-weighted CRPS, sharpness |
| C | Cumulative catastrophic-risk probability | CRPS | tail-weighted CRPS |
| C2 | Annual incident counts | Poisson log-score | — |
| C3 | Counterfactual scenarios | Reviewer rubric (15% of Track C) | — |

### CRPS

Implemented via the quantile-decomposition identity
   CRPS(F, y) = ∫₀¹ 2·(𝟙{y ≤ F⁻¹(α)} − α)·(F⁻¹(α) − y) dα,
discretized with the trapezoidal rule on the submitted quantile mesh.
CRPS reduces to mean absolute error for a degenerate (point) forecast,
so deterministic submissions remain comparable but cannot win
calibration-sensitive tracks.

### Tail-weighted CRPS

Catastrophic-risk forecasting is dominated by the upper tail; uniform
CRPS over-rewards models that nail the median while ignoring rare
extreme outcomes. We follow Gneiting & Ranjan (2011) and apply weight
`w(α) = 𝟙{α ≥ 0.90}` for the upper tail. Both standard and
tail-weighted CRPS are reported on the leaderboard.

### Threshold-crossing scores

For binary outcomes (e.g. "task horizon ≥ 720h has been crossed by
2028Q4") we report log-loss as the primary score and Brier as
auxiliary. Expected Calibration Error (ECE) with 10 bins is reported
alongside.

### Counterfactual scenarios (Track C3)

Three external reviewers score each Track C3 submission against a
structured rubric (internal consistency, parameter justification,
robustness to perturbation). The rubric will be published at competition
launch. Reviewer scores are normalized to a 0–10 scale and contribute
15% of the Track C aggregate.

## Public / private leaderboards

- **Public leaderboard** (updated weekly): scores submissions against a
  rotating subset of historical evaluation data. Used for participant
  feedback and ranking visibility during the competition.
- **Private leaderboard** (final): scores submissions against
  - data that materializes during the competition window
    (capability measurements, incident logs, BLS releases), and
  - a held-out historical split released only at competition close.

Submissions are versioned and timestamped at upload. Final-ranking
submissions are taken from a participant-designated finalist set
(maximum 5 submissions per team per track) submitted before the freeze
date.

## Calibration diagnostics

Beyond aggregate scores, we publish for every submission:

- **PIT histogram** with 10 bins, tested for uniformity using
  Kolmogorov–Smirnov (`scipy.stats.kstest`). A well-calibrated forecaster
  produces a uniform histogram.
- **Reliability diagram** for binary forecasts (Track A3, C1).
- **Sharpness**: mean width of the central 80% predictive interval. We
  report this *alongside* calibration so that deceptively narrow
  forecasts cannot appear better than they are.
- **ECE** at 10 bins for all binary tracks.

## Aggregation

Per-track score is the (uniform-weighted) mean of per-target scores
within the track. The aggregate competition score is the (uniform-
weighted) mean of per-track scores. Track and target weights for the
final leaderboard will be confirmed in the competition launch
announcement and may differ from uniform if reviewer feedback
identifies a more informative weighting; any change will be announced
at least 30 days before the freeze date.

## Anti-gaming measures

- LLM/agentic submissions run in a standardized 4-A100 inference
  container with a 24-hour budget per submission. The harness logs the
  system prompt and any retrieved documents to permit ex-post leakage
  audits.
- Submissions demonstrably trained on post-cutoff data (including
  retrieval over post-cutoff web pages) are disqualified. We will
  publicly document any disqualifications.
- Maximum 5 finalist submissions per team per track. Submission
  metadata (timestamp, environment, dependency versions) is preserved
  for audit.

## References

- Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules,
  prediction, and estimation. *JASA*, 102(477), 359–378.
- Gneiting, T. & Ranjan, R. (2011). Comparing density forecasts using
  threshold- and quantile-weighted scoring rules. *JBES*, 29(3),
  411–422.
- Bröcker, J. (2009). Reliability, sufficiency, and the decomposition
  of proper scores. *Quarterly Journal of the Royal Meteorological
  Society*, 135(643), 1512–1519.
