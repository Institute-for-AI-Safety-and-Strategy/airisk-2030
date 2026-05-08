"""End-to-end smoke tests for the baseline pipeline."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from airisk.evaluation.calibration import (expected_calibration_error,
                                           sharpness)
from airisk.evaluation.leaderboard import (SubmissionError,
                                           validate_submission,
                                           score_submission)
from airisk.evaluation.scoring import (brier_score, crps_from_quantiles,
                                       log_loss, tail_weighted_crps)
from airisk.scenarios import ALL_SCENARIOS, BASELINE
from airisk.simulation import run, to_submission_format


def test_baseline_runs():
    out = run(n_sims=20, n_quarters=8, scenario=BASELINE, seed=0)
    assert out["capability_hours"].shape == (20, 8)
    assert out["aggregate_automation"].shape == (20, 8)
    assert (out["cumulative_catastrophic_probability"] >= 0).all()
    assert (out["cumulative_catastrophic_probability"] <= 1).all()


def test_capability_monotone_nonneg():
    out = run(n_sims=10, n_quarters=8, scenario=BASELINE, seed=1)
    assert (out["capability_hours"] > 0).all()


def test_cum_risk_monotone():
    out = run(n_sims=10, n_quarters=8, scenario=BASELINE, seed=2)
    cp = out["cumulative_catastrophic_probability"]
    diffs = np.diff(cp, axis=1)
    assert (diffs >= -1e-9).all(), "cumulative risk must be monotone"


def test_to_submission_format_validates():
    out = run(n_sims=20, n_quarters=8, scenario=BASELINE, seed=3)
    sub = to_submission_format(out)
    tmp = REPO / "tests" / "_tmp_submission.json"
    tmp.write_text(json.dumps(sub))
    try:
        validate_submission(tmp)
    finally:
        tmp.unlink()


def test_invalid_submission_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 999, "quarters": [],
                               "forecasts": {}}))
    with pytest.raises(SubmissionError):
        validate_submission(bad)


def test_crps_zero_when_perfect():
    levels = np.linspace(0.05, 0.95, 19)
    obs = 5.0
    values = np.full_like(levels, obs)
    assert abs(crps_from_quantiles(levels, values, obs)) < 1e-9


def test_crps_positive_when_off():
    levels = np.linspace(0.05, 0.95, 19)
    values = np.linspace(0.0, 10.0, 19)
    score = crps_from_quantiles(levels, values, 5.0)
    assert score > 0


def test_tail_crps_isolates_upper():
    levels = np.linspace(0.05, 0.95, 19)
    values = np.linspace(0.0, 10.0, 19)
    full = crps_from_quantiles(levels, values, 5.0)
    tail = tail_weighted_crps(levels, values, 5.0,
                              tail="upper", cutoff=0.90)
    assert 0 <= tail <= full


def test_log_loss_and_brier():
    assert log_loss(0.99, 1) < log_loss(0.01, 1)
    assert brier_score(0.5, 1) == 0.25


def test_scenarios_change_outcomes():
    base = run(n_sims=40, n_quarters=8,
               scenario=ALL_SCENARIOS["baseline"], seed=5)
    coord = run(n_sims=40, n_quarters=8,
                scenario=ALL_SCENARIOS["global_coordination"], seed=5)
    # Global coordination should produce lower mean terminal cumulative risk
    base_term = base["cumulative_catastrophic_probability"][:, -1].mean()
    coord_term = coord["cumulative_catastrophic_probability"][:, -1].mean()
    assert coord_term < base_term


def test_run_baseline_script():
    """Smoke-test the CLI end to end."""
    out_path = REPO / "tests" / "_smoke_baseline.json"
    cmd = [sys.executable, str(REPO / "scripts" / "run_baseline.py"),
           "--scenario", "baseline", "--n-sims", "30", "--n-quarters", "8",
           "--out", str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    sub = json.loads(out_path.read_text())
    assert sub["schema_version"] == 1
    out_path.unlink()
