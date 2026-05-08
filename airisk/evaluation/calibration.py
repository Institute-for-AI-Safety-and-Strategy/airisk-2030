"""
Calibration diagnostics for AIRisk-2030.

The leaderboard publishes these for every submission alongside aggregate
scores so that participants and reviewers can distinguish "lucky"
submissions from genuinely well-calibrated ones.
"""
from __future__ import annotations

import numpy as np


def _interp_cdf(quantile_levels: np.ndarray,
                quantile_values: np.ndarray,
                x: float) -> float:
    """
    Estimate F(x) by linear interpolation through the quantile mesh.
    Returns 0 below the smallest quantile, 1 above the largest.
    """
    q = np.asarray(quantile_values)
    a = np.asarray(quantile_levels)
    if x <= q[0]:
        return float(a[0]) if x == q[0] else 0.0
    if x >= q[-1]:
        return float(a[-1]) if x == q[-1] else 1.0
    return float(np.interp(x, q, a))


def pit_histogram(forecasts: list[dict], observations: list[float],
                  n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Probability Integral Transform histogram. A well-calibrated forecaster
    produces PIT values uniform on [0, 1].

    Parameters
    ----------
    forecasts : list of dicts each with `quantile_levels` and `values`.
    observations : list of realized values, same length as forecasts.

    Returns
    -------
    bin_centers, counts : numpy arrays of length n_bins.
    """
    pits = [_interp_cdf(np.array(f["quantile_levels"]),
                        np.array(f["values"]), y)
            for f, y in zip(forecasts, observations)]
    counts, edges = np.histogram(pits, bins=n_bins, range=(0, 1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def reliability_diagram(probabilities: np.ndarray,
                        outcomes: np.ndarray,
                        n_bins: int = 10) -> dict:
    """
    Reliability (calibration) diagram for binary forecasts.

    Returns dict with bin centers, mean predicted probability per bin,
    observed frequency per bin, and bin counts.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    centers, mean_p, mean_y, counts = [], [], [], []
    for i in range(n_bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1
                                  else p <= edges[i + 1])
        if mask.sum() == 0:
            continue
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        mean_p.append(float(p[mask].mean()))
        mean_y.append(float(y[mask].mean()))
        counts.append(int(mask.sum()))
    return {"bin_center": np.array(centers),
            "mean_predicted": np.array(mean_p),
            "observed_frequency": np.array(mean_y),
            "count": np.array(counts)}


def expected_calibration_error(probabilities: np.ndarray,
                               outcomes: np.ndarray,
                               n_bins: int = 10) -> float:
    """
    Expected Calibration Error: weighted absolute deviation between mean
    predicted probability and observed frequency, across bins.
    """
    diag = reliability_diagram(probabilities, outcomes, n_bins=n_bins)
    if len(diag["count"]) == 0:
        return float("nan")
    weights = diag["count"] / diag["count"].sum()
    return float(np.sum(weights * np.abs(diag["mean_predicted"]
                                          - diag["observed_frequency"])))


def sharpness(forecasts: list[dict], interval: float = 0.80) -> float:
    """
    Mean width of the central `interval` predictive interval across
    forecasts. Smaller is sharper; sharpness should always be reported
    alongside calibration to prevent overconfident submissions from
    appearing better than they are.
    """
    lo_q = (1 - interval) / 2
    hi_q = 1 - lo_q
    widths = []
    for f in forecasts:
        a = np.array(f["quantile_levels"])
        v = np.array(f["values"])
        lo = float(np.interp(lo_q, a, v))
        hi = float(np.interp(hi_q, a, v))
        widths.append(hi - lo)
    return float(np.mean(widths))
