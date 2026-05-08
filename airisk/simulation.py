"""
Monte Carlo simulation orchestrator.

Couples the five domain models, runs `n_sims` independent trajectories,
and produces per-quarter predictive distributions in the canonical
submission format.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import (CapabilityModel, ComputeModel, EconomicModel,
                     GeopoliticalModel, HawkesRiskModel)
from .scenarios import Scenario, BASELINE


def _quarter_labels(start_year: int, start_q: int, n_quarters: int) -> list[str]:
    out = []
    y, q = start_year, start_q
    for _ in range(n_quarters):
        out.append(f"{y}Q{q}")
        q += 1
        if q == 5:
            q = 1
            y += 1
    return out


def run(n_sims: int = 500,
        n_quarters: int = 18,
        scenario: Scenario = BASELINE,
        seed: int = 42) -> dict:
    """
    Run the integrated baseline simulation under a given policy scenario.

    Returns a dict with the keys required by the submission format
    (capability, automation, unemployment, threshold-crossing probs,
    cumulative catastrophic-risk).
    """
    rng = np.random.default_rng(seed)

    # Apply scenario perturbations to default parameters
    cap_params = scenario.capability_params()
    risk_params = scenario.risk_params()

    # Pre-allocate
    cap_paths     = np.zeros((n_sims, n_quarters))
    auto_paths    = np.zeros((n_sims, n_quarters))
    unempl_paths  = np.zeros((n_sims, n_quarters))
    cumprob_paths = np.zeros((n_sims, n_quarters))
    gap_paths     = np.zeros((n_sims, n_quarters))

    for s in range(n_sims):
        sub_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        compute = ComputeModel(rng=sub_rng).simulate(n_quarters)
        cap = CapabilityModel(params=cap_params,
                              rng=sub_rng).simulate(n_quarters,
                                                    compute_log10=compute)
        geo = GeopoliticalModel(rng=sub_rng).simulate(n_quarters)
        econ = EconomicModel(rng=sub_rng).simulate(cap)
        risk = HawkesRiskModel(params=risk_params, rng=sub_rng).simulate(
            n_quarters, capability_path=cap,
            automation_path=econ["aggregate_automation"],
            us_china_gap=geo["us_china_gap"])

        cap_paths[s]     = cap
        auto_paths[s]    = econ["aggregate_automation"]
        unempl_paths[s]  = econ["unemployment"]
        cumprob_paths[s] = risk["cumulative_event_probability"]
        gap_paths[s]     = geo["us_china_gap"]

    quarters = _quarter_labels(2026, 3, n_quarters)
    return {
        "scenario": scenario.name,
        "n_sims": n_sims,
        "quarters": quarters,
        "capability_hours": cap_paths,
        "aggregate_automation": auto_paths,
        "unemployment": unempl_paths,
        "us_china_gap": gap_paths,
        "cumulative_catastrophic_probability": cumprob_paths,
    }


def to_submission_format(run_output: dict) -> dict:
    """
    Convert simulator output into the canonical submission JSON structure.

    Each forecast is represented as 19 quantiles (5%, 10%, ..., 95%) across
    the empirical distribution of Monte Carlo trajectories.
    """
    quantiles = np.linspace(0.05, 0.95, 19)
    out = {
        "schema_version": 1,
        "scenario": run_output["scenario"],
        "n_sims": run_output["n_sims"],
        "quarters": run_output["quarters"],
        "forecasts": {}
    }
    targets = {
        "track_a_metr_task_horizon_hours":  run_output["capability_hours"],
        "track_b_aggregate_automation":     run_output["aggregate_automation"],
        "track_b_unemployment_rate":        run_output["unemployment"],
        "track_c_cumulative_catastrophic":  run_output["cumulative_catastrophic_probability"],
    }
    for name, paths in targets.items():
        q = np.quantile(paths, quantiles, axis=0)  # (19, n_quarters)
        out["forecasts"][name] = {
            "quantile_levels": [round(float(x), 3) for x in quantiles],
            "values_by_quarter": [
                [float(q[i, t]) for i in range(len(quantiles))]
                for t in range(paths.shape[1])
            ]
        }

    # Threshold-crossing probabilities (Track A3 / C1)
    cap_paths = run_output["capability_hours"]
    threshold_probs = {}
    for thr in [168.0, 720.0, 2160.0]:
        ever_cross = np.cumsum(cap_paths >= thr, axis=1) > 0
        threshold_probs[f"task_horizon_ge_{int(thr)}h"] = \
            [float(p) for p in ever_cross.mean(axis=0)]
    out["forecasts"]["track_a3_threshold_crossing_probabilities"] = threshold_probs
    return out
