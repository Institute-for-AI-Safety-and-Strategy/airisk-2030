"""
Capability forecasting model.

Jump-diffusion stochastic process for AI task-horizon capability:

    dC_t = mu * C_t * dt + sigma * C_t * dW_t + J_t * dN_t

The continuous component captures incremental scaling progress; the jump
component captures discrete breakthroughs (transformer, GPT-3 scaling,
RLHF, reasoning traces, MoE-at-scale, ...). Jump magnitudes follow a
Generalized Extreme Value (Frechet) distribution to capture the empirical
heavy-tailed distribution of capability advances.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CapabilityParams:
    c0: float = 30.0           # initial task-horizon (hours), 2026 Q3
    mu: float = 0.10           # baseline annual drift (continuous part)
    sigma: float = 0.25        # diffusion volatility
    lambda_jump: float = 0.05  # jump intensity (per month)
    gev_xi:  float = 0.15      # GEV shape (heavy right tail)
    gev_loc: float = 0.30      # location of log-magnitude
    gev_scale: float = 0.30    # scale
    # Compute coupling (how much frontier compute boosts capability drift)
    compute_elasticity: float = 0.35


class CapabilityModel:
    """Quarterly Monte Carlo simulator for AI task-horizon capability."""

    def __init__(self, params: CapabilityParams | None = None,
                 rng: np.random.Generator | None = None):
        self.p = params or CapabilityParams()
        self.rng = rng or np.random.default_rng(43)

    def simulate(self, n_quarters: int,
                 compute_log10: np.ndarray | None = None) -> np.ndarray:
        """
        Returns an array of shape (n_quarters,) of task-horizon hours.

        If compute_log10 is provided (length n_quarters), capability drift
        is amplified proportional to compute growth above the 2026 anchor.
        """
        p = self.p
        dt = 0.25
        c = p.c0
        path = np.zeros(n_quarters)
        anchor_compute = 26.0

        for t in range(n_quarters):
            # Compute-amplified drift
            if compute_log10 is not None:
                excess = max(0.0, compute_log10[t] - anchor_compute)
                eff_mu = p.mu + p.compute_elasticity * excess
            else:
                eff_mu = p.mu

            # Continuous component: geometric Brownian motion
            dW = self.rng.normal(0, np.sqrt(dt))
            c *= np.exp((eff_mu - 0.5 * p.sigma ** 2) * dt + p.sigma * dW)

            # Jump component (Poisson arrivals, GEV-distributed magnitudes)
            jumps_in_quarter = self.rng.poisson(p.lambda_jump * 3)  # per-quarter
            for _ in range(jumps_in_quarter):
                u = self.rng.uniform(0.001, 0.999)
                # GEV (Frechet) inverse CDF, shape xi > 0
                x = p.gev_loc + p.gev_scale / p.gev_xi * \
                    ((-np.log(u)) ** (-p.gev_xi) - 1)
                magnitude = np.exp(x)
                c *= magnitude

            path[t] = c
        return path

    def threshold_probability(self, paths: np.ndarray,
                              threshold_hours: float) -> np.ndarray:
        """
        Given (n_sims, n_quarters), return per-quarter cumulative probability
        that the trajectory has crossed `threshold_hours`.
        """
        crossed = paths >= threshold_hours
        ever_crossed = np.cumsum(crossed, axis=1) > 0
        return ever_crossed.mean(axis=0)
