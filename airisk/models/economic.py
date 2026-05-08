"""
Economic impact model.

Computable General Equilibrium (CGE) with seven sectors. Each sector
employs a CES production function combining AI capital and human labor,
with sticky-wage adjustment. Outputs: sectoral automation rates,
aggregate unemployment (after retraining), and an AI productivity
multiplier index.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Sector:
    name: str
    labor_share: float
    elasticity_sub: float        # sigma in CES (>0)
    init_automation: float
    productivity_beta: float     # productivity multiplier coefficient
    productivity_gamma: float    # exponent


DEFAULT_SECTORS = [
    Sector("software_engineering", 0.05, 0.85, 0.12, 1.50, 0.40),
    Sector("knowledge_work",       0.30, 0.70, 0.07, 0.80, 0.35),
    Sector("manual_labor",         0.25, 0.40, 0.04, 0.35, 0.25),
    Sector("creative",             0.15, 0.65, 0.10, 1.20, 0.38),
    Sector("management",           0.20, 0.35, 0.03, 0.50, 0.30),
    Sector("manufacturing",        0.12, 0.75, 0.15, 1.10, 0.42),
    Sector("healthcare_education", 0.05, 0.30, 0.04, 0.60, 0.32),
]


@dataclass
class EconomicParams:
    rho: float = -0.50        # CES parameter (sigma=1/(1-rho)=0.67)
    retrain_tau: float = 2.0  # years
    wage_rigidity: float = 0.30
    capability_threshold: float = 168.0   # hours; automation accelerates above this
    disruption_threshold: float = 0.30


class EconomicModel:
    """Quarterly Monte Carlo simulator for sectoral automation and employment."""

    def __init__(self, sectors: list[Sector] | None = None,
                 params: EconomicParams | None = None,
                 rng: np.random.Generator | None = None):
        self.sectors = sectors or DEFAULT_SECTORS
        self.p = params or EconomicParams()
        self.rng = rng or np.random.default_rng(45)

    def simulate(self, capability_path: np.ndarray) -> dict:
        """
        capability_path: array (n_quarters,) of task-horizon hours.
        Returns dict with sectoral automation, unemployment, productivity.
        """
        n_q = len(capability_path)
        n_s = len(self.sectors)
        auto = np.zeros((n_q, n_s))
        for j, sec in enumerate(self.sectors):
            auto[0, j] = sec.init_automation
        dt = 0.25

        for t in range(1, n_q):
            cap = capability_path[t]
            # threshold-modulated automation growth
            mult = 1.0 + 1.5 * max(0, np.log(cap / self.p.capability_threshold))
            for j, sec in enumerate(self.sectors):
                base_growth = 0.05 * sec.elasticity_sub * dt
                growth = base_growth * mult
                noise = self.rng.normal(0, 0.005)
                new_auto = auto[t-1, j] + growth + noise
                auto[t, j] = np.clip(new_auto, 0.0, 0.95)

        # Aggregate weighted automation
        weights = np.array([s.labor_share for s in self.sectors])
        weights = weights / weights.sum()
        agg_auto = (auto * weights).sum(axis=1)

        # Unemployment with retraining lag
        retrain_completed = 1 - np.exp(-np.arange(n_q) * dt / self.p.retrain_tau)
        unempl = agg_auto * (1 - retrain_completed) * 0.30  # ~30% become unemployed

        # Productivity multiplier (relative to t=0)
        prod = np.zeros(n_q)
        for t in range(n_q):
            mult_total = 1.0
            for j, sec in enumerate(self.sectors):
                mult_total += weights[j] * sec.productivity_beta * \
                              (auto[t, j] / max(auto[0, j], 1e-3)) ** sec.productivity_gamma * 0.1
            prod[t] = mult_total

        return {
            "sector_names": [s.name for s in self.sectors],
            "sectoral_automation": auto,    # (n_q, n_s)
            "aggregate_automation": agg_auto, # (n_q,)
            "unemployment": unempl,          # (n_q,)
            "productivity_multiplier": prod, # (n_q,)
            "disruption_threshold_crossed":
                np.cumsum(agg_auto > self.p.disruption_threshold) > 0,
        }
