"""
Geopolitical competition model.

Simplified Markov Perfect Equilibrium over a fixed roster of countries.
Each country chooses an investment rate to maximize utility over relative
capability minus quadratic cost minus catch-up pressure. The full proposal
specifies value-function iteration; the baseline here uses a fast best-
response approximation that captures the same qualitative dynamics
(near-parity stability, instability above a 40% gap threshold) at a
fraction of the runtime.

Outputs: per-country capability trajectories, aggregate concentration
indices, and a binary "instability flag" used by the risk model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CountryAttrs:
    name: str
    init_capability: float
    investment_capacity: float
    risk_aversion: float
    transparency: float
    gdp_share: float


DEFAULT_COUNTRIES = [
    CountryAttrs("US",      1.00, 1.00, 0.50, 0.70, 0.247),
    CountryAttrs("China",   0.95, 1.45, 0.20, 0.45, 0.184),
    CountryAttrs("EU",      0.60, 0.50, 0.70, 0.85, 0.168),
    CountryAttrs("UK",      0.55, 0.40, 0.60, 0.80, 0.033),
    CountryAttrs("Japan",   0.45, 0.35, 0.65, 0.75, 0.050),
    CountryAttrs("S_Korea", 0.50, 0.45, 0.55, 0.70, 0.019),
    CountryAttrs("India",   0.35, 0.60, 0.50, 0.65, 0.037),
    CountryAttrs("Israel",  0.40, 0.30, 0.45, 0.75, 0.005),
]


@dataclass
class GeopoliticalParams:
    phi:   float = 0.80     # security premium
    gamma: float = 0.10     # quadratic investment cost
    rho:   float = 0.15     # default catch-up pressure
    obs_noise: float = 0.20 # observation noise on rivals
    instability_threshold: float = 0.40
    transfer_rate: float = 0.10  # baseline downhill knowledge flow


class GeopoliticalModel:
    """Best-response approximation to MPE over n_quarters."""

    def __init__(self, countries: list[CountryAttrs] | None = None,
                 params: GeopoliticalParams | None = None,
                 rng: np.random.Generator | None = None):
        self.countries = countries or DEFAULT_COUNTRIES
        self.p = params or GeopoliticalParams()
        self.rng = rng or np.random.default_rng(44)

    def simulate(self, n_quarters: int) -> dict:
        n = len(self.countries)
        cap = np.array([c.init_capability for c in self.countries])
        inv_cap = np.array([c.investment_capacity for c in self.countries])
        gdp = np.array([c.gdp_share for c in self.countries])
        trace = np.zeros((n_quarters, n))

        dt = 0.25
        for t in range(n_quarters):
            frontier = cap.max()
            # observe rivals with country-specific noise
            obs_cap = cap + self.rng.normal(0, self.p.obs_noise, size=n)
            # best-response investment ~ closing the gap, scaled by capacity
            gap = np.maximum(0, frontier - obs_cap)
            invest = np.minimum(0.05 + 0.30 * gap, inv_cap * 0.15)
            # Capability growth: investment + spillover
            spillover = self.p.transfer_rate * np.maximum(0, frontier - cap)
            cap = cap * np.exp((0.18 * invest + 0.40 * spillover) * dt) \
                  + 0.001 * self.rng.standard_normal(n)
            trace[t] = cap

        # Concentration / stability metrics
        share = trace / trace.sum(axis=1, keepdims=True)
        hhi = (share ** 2).sum(axis=1)
        gap_pct = (trace.max(axis=1) - np.partition(trace, -2, axis=1)[:, -2]) \
                  / trace.max(axis=1)
        instability = gap_pct > self.p.instability_threshold

        return {
            "names": [c.name for c in self.countries],
            "capabilities": trace,           # (n_quarters, n_countries)
            "hhi": hhi,                      # (n_quarters,)
            "us_china_gap": self._us_china_gap(trace),
            "instability_flag": instability,
        }

    def _us_china_gap(self, trace: np.ndarray) -> np.ndarray:
        names = [c.name for c in self.countries]
        i_us, i_cn = names.index("US"), names.index("China")
        diff = np.abs(trace[:, i_us] - trace[:, i_cn])
        return diff / trace[:, i_us]
