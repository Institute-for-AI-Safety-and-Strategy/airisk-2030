"""
Strictly proper scoring rules for AIRisk-2030.

All functions return per-target scores with the standard
"lower is better" sign convention. The leaderboard aggregates these
across forecast horizons and tracks.

References
----------
Gneiting & Raftery (2007), Strictly proper scoring rules, prediction,
    and estimation. JASA 102(477), 359-378.
Gneiting & Ranjan (2011), Comparing density forecasts using threshold-
    and quantile-weighted scoring rules. JBES 29(3), 411-422.
"""
from __future__ import annotations

import numpy as np


def crps_from_quantiles(quantile_levels: np.ndarray,
                        quantile_values: np.ndarray,
                        observation: float) -> float:
    """
    Approximate Continuous Ranked Probability Score from a sorted
    quantile representation of the predictive CDF.

    Uses the formula
        CRPS(F, y) = integral_0^1 QS_alpha(F^{-1}(alpha), y) d alpha
    discretized via the trapezoidal rule, where
        QS_alpha(q, y) = 2 * (1{y <= q} - alpha) * (q - y).

    Parameters
    ----------
    quantile_levels : array-like, shape (k,), strictly increasing in (0, 1)
    quantile_values : array-like, shape (k,), the CDF inverse at those levels
    observation     : float, realized value

    Returns
    -------
    float : CRPS (lower is better; reduces to MAE for a degenerate forecast).
    """
    a = np.asarray(quantile_levels, dtype=float)
    q = np.asarray(quantile_values, dtype=float)
    if a.shape != q.shape:
        raise ValueError("quantile_levels and quantile_values must match")
    indicator = (observation <= q).astype(float)
    integrand = 2.0 * (indicator - a) * (q - observation)
    return float(np.trapezoid(integrand, a))


def tail_weighted_crps(quantile_levels: np.ndarray,
                       quantile_values: np.ndarray,
                       observation: float,
                       tail: str = "upper",
                       cutoff: float = 0.90) -> float:
    """
    Tail-weighted CRPS variant emphasizing one tail.

    Multiplies the per-quantile QS contribution by an indicator weight
    that is 1 inside the tail and 0 outside. For `tail="upper"` the
    weight is `1{alpha >= cutoff}`; for `tail="lower"` it is
    `1{alpha <= 1 - cutoff}`.
    """
    a = np.asarray(quantile_levels, dtype=float)
    q = np.asarray(quantile_values, dtype=float)
    if tail == "upper":
        w = (a >= cutoff).astype(float)
    elif tail == "lower":
        w = (a <= 1 - cutoff).astype(float)
    else:
        raise ValueError("tail must be 'upper' or 'lower'")
    if w.sum() == 0:
        return 0.0
    indicator = (observation <= q).astype(float)
    integrand = w * 2.0 * (indicator - a) * (q - observation)
    return float(np.trapezoid(integrand, a))


def log_loss(probability: float, observation: int, eps: float = 1e-9) -> float:
    """
    Binary log-loss (negative log-likelihood, base e) of a single
    probability against a {0, 1} observation. Lower is better.
    """
    p = float(np.clip(probability, eps, 1 - eps))
    return -(observation * np.log(p) + (1 - observation) * np.log(1 - p))


def brier_score(probability: float, observation: int) -> float:
    """Mean squared error between a probability and a {0, 1} observation."""
    return float((probability - observation) ** 2)


def poisson_log_score(rate: float, k: int) -> float:
    """
    Negative Poisson log-likelihood for incident-count outcomes.

    Returns -log P(N = k | lambda = rate). Lower is better.
    """
    if rate <= 0:
        return float("inf") if k > 0 else 0.0
    from math import lgamma, log
    return -(k * log(rate) - rate - lgamma(k + 1))


def aggregate_track_score(per_target_scores: dict[str, float],
                          weights: dict[str, float] | None = None) -> float:
    """
    Aggregate target-level scores into a single track-level score.

    Default weights are uniform; the official leaderboard uses
    documented weights specified in docs/EVALUATION.md.
    """
    if weights is None:
        weights = {k: 1.0 for k in per_target_scores}
    total_weight = sum(weights.get(k, 1.0) for k in per_target_scores)
    weighted_sum = sum(weights.get(k, 1.0) * v
                       for k, v in per_target_scores.items())
    return weighted_sum / total_weight if total_weight > 0 else float("nan")
