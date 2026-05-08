"""
Submission validator and scorer.

Given a submission JSON file conforming to docs/SUBMISSION_FORMAT.md and
a ground-truth dataset, produces the per-track and aggregate scores that
the leaderboard reports.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calibration import expected_calibration_error, sharpness
from .scoring import (aggregate_track_score, brier_score,
                      crps_from_quantiles, log_loss, tail_weighted_crps)


REQUIRED_TARGETS = [
    "track_a_metr_task_horizon_hours",
    "track_b_aggregate_automation",
    "track_b_unemployment_rate",
    "track_c_cumulative_catastrophic",
    "track_a3_threshold_crossing_probabilities",
]


class SubmissionError(Exception):
    pass


def validate_submission(path: str | Path) -> dict:
    """
    Load and validate a submission JSON file. Raises SubmissionError on
    any structural problem; returns the parsed dict on success.
    """
    path = Path(path)
    if not path.exists():
        raise SubmissionError(f"submission file not found: {path}")
    try:
        sub = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise SubmissionError(f"invalid JSON: {e}")

    if sub.get("schema_version") != 1:
        raise SubmissionError("schema_version must be 1")
    if "quarters" not in sub or not isinstance(sub["quarters"], list):
        raise SubmissionError("missing or invalid 'quarters' field")
    if "forecasts" not in sub:
        raise SubmissionError("missing 'forecasts' field")

    n_q = len(sub["quarters"])

    for tgt in REQUIRED_TARGETS:
        if tgt not in sub["forecasts"]:
            raise SubmissionError(f"missing forecast target: {tgt}")
        f = sub["forecasts"][tgt]
        if tgt.endswith("threshold_crossing_probabilities"):
            # dict of threshold -> list of probs
            for thr, probs in f.items():
                if len(probs) != n_q:
                    raise SubmissionError(
                        f"{tgt}/{thr} length {len(probs)} != n_quarters {n_q}")
                for v in probs:
                    if not (0.0 <= v <= 1.0):
                        raise SubmissionError(
                            f"{tgt}/{thr} contains value outside [0,1]")
        else:
            if "quantile_levels" not in f or "values_by_quarter" not in f:
                raise SubmissionError(
                    f"{tgt} missing quantile_levels / values_by_quarter")
            ql = f["quantile_levels"]
            vbq = f["values_by_quarter"]
            if not (0 < ql[0] and ql[-1] < 1):
                raise SubmissionError(f"{tgt} quantile_levels must be in (0,1)")
            if any(ql[i] >= ql[i+1] for i in range(len(ql)-1)):
                raise SubmissionError(f"{tgt} quantile_levels must be strictly increasing")
            if len(vbq) != n_q:
                raise SubmissionError(
                    f"{tgt} has {len(vbq)} quarters, expected {n_q}")
            for t, row in enumerate(vbq):
                if len(row) != len(ql):
                    raise SubmissionError(
                        f"{tgt} quarter {t} has {len(row)} values, expected {len(ql)}")
                if any(row[i] > row[i+1] for i in range(len(row)-1)):
                    raise SubmissionError(
                        f"{tgt} quarter {t} quantile values not non-decreasing "
                        f"(violates monotonicity of CDF)")
    return sub


def _score_continuous_target(forecast: dict,
                             observations: list[float]) -> dict:
    ql = np.array(forecast["quantile_levels"])
    vbq = forecast["values_by_quarter"]
    crps_vals, twcrps_vals = [], []
    for t, y in enumerate(observations):
        if y is None or (isinstance(y, float) and np.isnan(y)):
            continue
        v = np.array(vbq[t])
        crps_vals.append(crps_from_quantiles(ql, v, y))
        twcrps_vals.append(tail_weighted_crps(ql, v, y, tail="upper", cutoff=0.90))
    return {
        "n_resolved": len(crps_vals),
        "crps_mean": float(np.mean(crps_vals)) if crps_vals else float("nan"),
        "tail_crps_mean": float(np.mean(twcrps_vals)) if twcrps_vals else float("nan"),
        "sharpness_80pct": sharpness(
            [{"quantile_levels": ql, "values": np.array(v)} for v in vbq]),
    }


def _score_threshold_target(forecast: dict,
                            observations: dict[str, list[int]]) -> dict:
    """
    forecast: {threshold_label: [probs per quarter]}
    observations: {threshold_label: [{0,1} per quarter]}
    """
    out = {}
    for label, probs in forecast.items():
        if label not in observations:
            continue
        obs = observations[label]
        ll, br = [], []
        for p, y in zip(probs, obs):
            if y is None:
                continue
            ll.append(log_loss(p, int(y)))
            br.append(brier_score(p, int(y)))
        if not ll:
            continue
        out[label] = {
            "n_resolved": len(ll),
            "log_loss_mean": float(np.mean(ll)),
            "brier_mean": float(np.mean(br)),
            "ece": float(expected_calibration_error(
                np.array(probs[:len(obs)]),
                np.array([y if y is not None else 0 for y in obs])))
        }
    return out


def score_submission(submission_path: str | Path,
                     ground_truth_path: str | Path) -> dict:
    """
    Score a submission against a ground-truth file.

    The ground-truth JSON has the structure
        {
          "quarters": [...],
          "observations": {
            "track_a_metr_task_horizon_hours": [float|null, ...],
            "track_b_aggregate_automation":    [...],
            "track_b_unemployment_rate":       [...],
            "track_c_cumulative_catastrophic": [...],
            "track_a3_threshold_crossing_probabilities": {
                "task_horizon_ge_168h":  [0|1|null, ...],
                ...
            }
          }
        }
    """
    sub = validate_submission(submission_path)
    gt = json.loads(Path(ground_truth_path).read_text())
    obs_root = gt.get("observations", {})

    results = {"scenario": sub.get("scenario"), "tracks": {}}

    # Track A: capability
    a_obs = obs_root.get("track_a_metr_task_horizon_hours", [])
    if a_obs:
        results["tracks"]["A_capability"] = _score_continuous_target(
            sub["forecasts"]["track_a_metr_task_horizon_hours"], a_obs)

    # Track A3: threshold crossing
    a3_obs = obs_root.get("track_a3_threshold_crossing_probabilities", {})
    if a3_obs:
        results["tracks"]["A3_thresholds"] = _score_threshold_target(
            sub["forecasts"]["track_a3_threshold_crossing_probabilities"],
            a3_obs)

    # Track B: economic
    b_results = {}
    for tgt in ["track_b_aggregate_automation", "track_b_unemployment_rate"]:
        b_obs = obs_root.get(tgt, [])
        if b_obs:
            b_results[tgt] = _score_continuous_target(
                sub["forecasts"][tgt], b_obs)
    if b_results:
        results["tracks"]["B_economic"] = b_results

    # Track C: catastrophic risk
    c_obs = obs_root.get("track_c_cumulative_catastrophic", [])
    if c_obs:
        results["tracks"]["C_catastrophic"] = _score_continuous_target(
            sub["forecasts"]["track_c_cumulative_catastrophic"], c_obs)

    # Aggregate (only over resolved data)
    track_means = []
    for tname, tdata in results["tracks"].items():
        if isinstance(tdata, dict) and "crps_mean" in tdata:
            track_means.append(tdata["crps_mean"])
        elif isinstance(tdata, dict):
            for sub_name, sub_data in tdata.items():
                if isinstance(sub_data, dict) and "crps_mean" in sub_data:
                    track_means.append(sub_data["crps_mean"])
                elif isinstance(sub_data, dict) and "log_loss_mean" in sub_data:
                    track_means.append(sub_data["log_loss_mean"])
    results["aggregate_score"] = float(np.mean(track_means)) if track_means \
                                 else float("nan")
    return results
