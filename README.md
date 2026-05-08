# AIRisk-2030

**A Multi-Domain Probabilistic Forecasting Challenge for AI Catastrophic Risk**
*NeurIPS 2026 Competition Track — Starter Kit*

[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

This repository contains the official starter kit for the AIRisk-2030
NeurIPS 2026 competition: datasets, baseline implementation, and the
evaluation infrastructure used to score the public and private leaderboards.

## What is AIRisk-2030?

AIRisk-2030 challenges participants to produce calibrated **probabilistic
forecasts** of AI-related systemic risks through 2030 across three tracks:

| Track | Focus | Targets |
|-------|-------|---------|
| **A — Capability** | Frontier-model progress | Quarterly distributions over METR task-horizon, benchmark scores, threshold-crossing probabilities |
| **B — Economic** | Labor-market disruption | Sectoral automation rates, unemployment, productivity multipliers |
| **C — Integrated Risk** | Catastrophic outcomes | Cumulative incident probability, annual incident counts, counterfactual policy scenarios |

All forecasts are scored using strictly proper scoring rules (CRPS, Brier,
log-loss) with explicit tail-risk weighting. Calibration diagnostics
(reliability diagrams, PIT histograms, sharpness) are published for every
submission.

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/Institute-for-AI-Safety-and-Strategy/airisk-2030
pip install -e .

# 2. Run the baseline simulation
python scripts/run_baseline.py --scenario baseline --n-sims 500 --out outputs/baseline.json

# 3. Score the baseline against held-out historical data
python scripts/score_submission.py outputs/baseline.json --split public

# 4. Inspect calibration diagnostics
python scripts/calibration_report.py outputs/baseline.json
```

A typical baseline run takes **3–5 minutes on a single CPU core** with
500 Monte Carlo trajectories. No GPU is required for the classical baseline.

## Repository Structure

```
.
├── LICENSE              MIT (code)
├── LICENSE-DATA         CC BY 4.0 (datasets)
├── README.md            this file
├── pyproject.toml       package metadata
├── requirements.txt     pinned dependencies
│
├── data/                Curated public datasets (CC BY 4.0)
│   ├── capability/      METR task-horizons, benchmarks, breakthroughs
│   ├── compute/         Compute trends, fab capacity, energy demand
│   ├── economic/        Sectoral labor share, automation, productivity
│   ├── geopolitical/    Country capabilities, investment, transfer
│   ├── incidents/       AI incident database (curated subset)
│   └── policy/          Regulatory milestone log
│
├── airisk/              Reference baseline (Python package)
│   ├── models/
│   │   ├── compute.py       Solow–Romer endogenous growth
│   │   ├── capability.py    Jump-diffusion + GEV
│   │   ├── geopolitical.py  Markov Perfect Equilibrium
│   │   ├── economic.py      CES / sticky-wage CGE
│   │   └── risk.py          Hawkes self-exciting process
│   ├── simulation.py    Monte Carlo integrator
│   ├── scenarios.py     Policy scenario specifications
│   └── evaluation/
│       ├── scoring.py       CRPS, Brier, log-loss, tail-weighted CRPS
│       ├── calibration.py   PIT, reliability, ECE, sharpness
│       └── leaderboard.py   Submission-format validator + scorer
│
├── scripts/             Command-line entry points
│   ├── run_baseline.py
│   ├── score_submission.py
│   ├── calibration_report.py
│   └── make_example_submission.py
│
├── examples/
│   └── baseline_submission/  Reference submission in the official format
│
├── docs/
│   ├── DATASETS.md      Per-file schemas and provenance
│   ├── BASELINE.md      Mathematical specification of the baseline
│   ├── EVALUATION.md    Scoring rules, leaderboard mechanics
│   └── SUBMISSION_FORMAT.md
│
└── tests/               Smoke tests for the baseline + evaluators
```

## Submission Format

A valid submission is a **single JSON file** with predictive distributions
over each forecast target at each quarterly horizon (Q3 2026 → Q4 2030).
See [`docs/SUBMISSION_FORMAT.md`](docs/SUBMISSION_FORMAT.md) for the full
spec, [`examples/baseline_submission/submission.json`](examples/baseline_submission/submission.json)
for a concrete example, and run

```bash
python scripts/score_submission.py examples/baseline_submission/submission.json
```

to verify your submission is well-formed before uploading.

## Data Provenance and Licensing

All datasets are aggregations of publicly available sources, harmonized to
quarterly resolution and a uniform schema. Each row carries a `source`
column linking it to its upstream provider. Data is released under
[CC BY 4.0](LICENSE-DATA); code under [MIT](LICENSE).

For the full provenance of every parameter and dataset, see
[`docs/DATASETS.md`](docs/DATASETS.md).

## Computational Constraints

- **Classical-statistical track**: no compute restriction; submit any
  probabilistic model.
- **LLM/agentic track**: bounded inference budget of USD 500 per submission,
  audited via a standardized API harness (a `harness/` directory will be
  added at competition launch).
- **Hybrid track**: same constraints as LLM/agentic.

A **low-resource sub-track** scores submissions that ran end-to-end on a
single CPU-only Codabench worker (no GPU during development).

## Citation

If you use this starter kit or build on the baseline, please cite:

```bibtex
@inproceedings{airisk2030,
  title  = {AIRisk-2030: A Multi-Domain Probabilistic Forecasting
            Challenge for AI Catastrophic Risk},
  author = {El-chami, Ibrahim and others},
  booktitle = {NeurIPS 2026 Competition Track},
  year   = {2026},
  url    = {https://github.com/Institute-for-AI-Safety-and-Strategy/airisk-2030}
}
```

## Contact

- **Email**: info@ai-rd.ca
- **Discord**: invite link in competition portal at launch
- **Office hours**: weekly during Phase 1 (June–September 2026), schedule
  posted on the competition portal

## Code of Conduct

This competition follows the
[NeurIPS Code of Conduct](https://neurips.cc/public/CodeOfConduct) and the
NeurIPS 2026 Main Track Handbook for plagiarism, anti-collusion, and
LLM-use policies.
