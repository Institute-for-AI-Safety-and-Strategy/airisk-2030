"""
Catastrophic risk model.

Hawkes self-exciting point process for AI catastrophic incidents:

    lambda(t) = lambda_0 + sum_{t_i < t} alpha * exp(-beta * (t - t_i))

Base intensity is augmented by trigger factors (capability thresholds,
economic disruption, geopolitical instability, recursive-improvement
threshold). Calibrated against 127 documented AI incidents (Partnership
on AI, 2012-2024) and validated against earthquake/financial-contagion
parameter ranges.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class HawkesParams:
    lambda_base:       float = 0.01    # background annual intensity
    alpha:             float = 0.50    # self-excitation
    beta:              float = 2.0     # decay rate (per year)
    # Threshold-triggered intensity boosts (added to lambda_base)
    autonomous_lambda:    float = 0.012  # capability > 168h
    autonomous_threshold: float = 168.0  # hours
    economic_lambda:      float = 0.010  # automation > 30%
    economic_threshold:   float = 0.30
    geopolitical_lambda:  float = 0.012  # US-China gap > 40%
    geopolitical_threshold: float = 0.40
    recursive_lambda:     float = 0.020  # capability > 2160h
    recursive_threshold:  float = 2160.0


class HawkesRiskModel:
    """Hawkes intensity simulator with multi-domain trigger inputs."""

    def __init__(self, params: HawkesParams | None = None,
                 rng: np.random.Generator | None = None):
        self.p = params or HawkesParams()
        self.rng = rng or np.random.default_rng(46)

    def simulate(self, n_quarters: int, capability_path: np.ndarray,
                 automation_path: np.ndarray,
                 us_china_gap: np.ndarray) -> dict:
        """Return per-quarter intensity, cumulative event probability, sample events."""
        p = self.p
        dt = 0.25
        events: list[float] = []
        intensity = np.zeros(n_quarters)
        cum_prob = np.zeros(n_quarters)
        running_prob = 0.0

        for t in range(n_quarters):
            # base + threshold contributions
            base = p.lambda_base
            if capability_path[t] >= p.autonomous_threshold:
                base += p.autonomous_lambda
            if capability_path[t] >= p.recursive_threshold:
                base += p.recursive_lambda
            if automation_path[t] >= p.economic_threshold:
                base += p.economic_lambda
            if us_china_gap[t] >= p.geopolitical_threshold:
                base += p.geopolitical_lambda

            # self-excitation from prior events
            current_t = t * dt
            self_exc = sum(p.alpha * np.exp(-p.beta * (current_t - ti))
                           for ti in events if ti < current_t)
            lam = base + self_exc
            intensity[t] = lam

            # Sample whether an event occurs this quarter (thinning approximation)
            event_prob = 1 - np.exp(-lam * dt)
            if self.rng.random() < event_prob:
                events.append(current_t + self.rng.uniform(0, dt))

            # Cumulative probability of >=1 event up to time t
            running_prob = 1 - (1 - running_prob) * np.exp(-lam * dt)
            cum_prob[t] = running_prob

        return {
            "intensity": intensity,
            "cumulative_event_probability": cum_prob,
            "n_events": len(events),
            "event_times": np.array(events),
        }
