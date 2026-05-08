"""
Policy scenarios for counterfactual forecasting (Track C3).

Each scenario perturbs the default parameter set in well-documented ways:
investment reductions affect capability drift; coordination reduces
geopolitical observation noise; coalition exclusion widens capability gaps
and amplifies arms-race dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass

from .models.capability import CapabilityParams
from .models.risk import HawkesParams


@dataclass
class Scenario:
    name: str
    description: str
    cap_drift_multiplier: float = 1.0
    risk_self_excitation_multiplier: float = 1.0
    risk_base_multiplier: float = 1.0

    def capability_params(self) -> CapabilityParams:
        p = CapabilityParams()
        p.mu = p.mu * self.cap_drift_multiplier
        return p

    def risk_params(self) -> HawkesParams:
        p = HawkesParams()
        p.alpha = p.alpha * self.risk_self_excitation_multiplier
        p.lambda_base = p.lambda_base * self.risk_base_multiplier
        return p


BASELINE = Scenario(
    name="baseline",
    description="Current trajectory; no new international coordination.",
)

GLOBAL_COORDINATION = Scenario(
    name="global_coordination",
    description="All 8 major powers implement aligned safety standards.",
    cap_drift_multiplier=0.70,
    risk_self_excitation_multiplier=0.55,
    risk_base_multiplier=0.65,
)

WESTERN_COALITION = Scenario(
    name="western_coalition",
    description="US/EU/UK/Japan/S.Korea/Canada coordinate; China and India out.",
    cap_drift_multiplier=0.95,
    risk_self_excitation_multiplier=1.20,
    risk_base_multiplier=1.10,
)

US_CHINA_BILATERAL = Scenario(
    name="us_china_bilateral",
    description="Direct US-China safety agreement; others follow or remain independent.",
    cap_drift_multiplier=0.80,
    risk_self_excitation_multiplier=0.70,
    risk_base_multiplier=0.75,
)

UNILATERAL_US = Scenario(
    name="unilateral_us",
    description="US implements strong AI safety regulation; others continue baseline.",
    cap_drift_multiplier=0.97,
    risk_self_excitation_multiplier=0.95,
    risk_base_multiplier=0.92,
)

CHINA_LED_GLOBAL_SOUTH = Scenario(
    name="china_led_global_south",
    description="China establishes alternative governance framework (BRICS+).",
    cap_drift_multiplier=1.02,
    risk_self_excitation_multiplier=1.10,
    risk_base_multiplier=1.05,
)

ALL_SCENARIOS = {
    s.name: s for s in [BASELINE, GLOBAL_COORDINATION, WESTERN_COALITION,
                        US_CHINA_BILATERAL, UNILATERAL_US,
                        CHINA_LED_GLOBAL_SOUTH]
}
