"""
Compute scaling model.

Solow–Romer endogenous growth model with Cobb–Douglas production,
mean-reverting (Ornstein–Uhlenbeck) total factor productivity, and three
physical bottlenecks (semiconductor manufacturing, energy, rare earths).

The model simulates effective frontier training compute (FLOPs, log10) at
quarterly resolution. Outputs feed directly into the capability model.

Calibration sources documented in docs/BASELINE.md and the proposal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ComputeParams:
    # Cobb–Douglas
    alpha: float = 0.35     # capital elasticity
    beta:  float = 0.25     # labor elasticity
    # Capital dynamics
    s:     float = 0.25     # investment rate
    delta: float = 0.15     # annual depreciation
    # TFP (Ornstein–Uhlenbeck on log A)
    g_tfp: float = 0.08     # drift
    sigma_tfp: float = 0.15
    theta_tfp: float = 0.30 # mean-reversion speed
    # Physical constraints (annual growth rates)
    chip_growth: float = 0.20
    energy_growth: float = 0.025
    pue_2024: float = 1.55
    pue_2030: float = 1.35
    # Initial conditions (2024 anchor, log10 FLOPs)
    flop_2024_log10: float = 25.5


class ComputeModel:
    """Quarterly Monte Carlo simulator for frontier compute capacity."""

    def __init__(self, params: ComputeParams | None = None,
                 rng: np.random.Generator | None = None):
        self.p = params or ComputeParams()
        self.rng = rng or np.random.default_rng(42)

    def simulate(self, n_quarters: int) -> np.ndarray:
        """
        Returns an array of shape (n_quarters,) of frontier compute log10(FLOPs).
        """
        p = self.p
        dt = 0.25  # quarterly timestep in years
        # log-TFP under OU
        log_a = np.log(1.0)
        log_a_path = []
        # capital (normalized so that 2024 anchor matches)
        flop_path = np.zeros(n_quarters)
        flop = 10 ** p.flop_2024_log10

        for t in range(n_quarters):
            # OU step on log A with drift g_tfp
            dW = self.rng.normal(0, np.sqrt(dt))
            mean_rev = p.theta_tfp * (-log_a) * dt
            log_a += p.g_tfp * dt + p.sigma_tfp * dW + mean_rev
            log_a_path.append(log_a)

            # Effective frontier compute = anchor * exp(cumulative log-A)
            # plus exponential drift from algorithmic + hardware
            growth = np.exp(0.04 * dt + log_a * dt * 0.5)
            flop *= growth

            # Apply physical ceilings
            chip_ceiling = self._chip_ceiling(t, dt)
            energy_ceiling = self._energy_ceiling(t, dt)
            ree_ceiling = self._ree_ceiling(t, dt)
            flop = min(flop, chip_ceiling, energy_ceiling, ree_ceiling)
            flop_path[t] = np.log10(flop)
        return flop_path

    def _chip_ceiling(self, t: int, dt: float) -> float:
        # 2024 chip-limited compute envelope ~ 1e26 FLOP/run
        years = t * dt
        return 10 ** (26.0 + np.log10(1 + self.p.chip_growth) * years)

    def _energy_ceiling(self, t: int, dt: float) -> float:
        # Energy-limited compute grows slower; binding around 2028
        years = t * dt
        return 10 ** (26.5 + np.log10(1 + self.p.energy_growth) * years
                      + 0.10 * years)  # algorithmic efficiency offset

    def _ree_ceiling(self, t: int, dt: float) -> float:
        # Rare earths binds rarely; modeled as soft ceiling
        years = t * dt
        return 10 ** (27.0 + 0.05 * years)
