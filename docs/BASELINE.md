# Baseline Specification

The reference baseline for AIRisk-2030 integrates five domain models
through a coupled stochastic system run with quarterly timesteps
(Δt = 0.25 years). This document gives the mathematical specification;
implementations live in `airisk/models/`.

## 1. Compute (Solow–Romer)

Cobb–Douglas production with mean-reverting (Ornstein–Uhlenbeck) total
factor productivity:

    Y_t = A_t · K_t^α · L_t^β
    dK_t = s · Y_t · dt − δ · K_t · dt
    d log A_t = g · dt + σ_A · dW_t + θ · (μ − log A_t) · dt

with `α = 0.35`, `β = 0.25`, `s = 0.25`, `δ = 0.15`, `g = 0.08`,
`σ_A = 0.15`, `θ = 0.30`. Effective frontier compute is the minimum of
the production output and three physical ceilings:

    FLOP_eff(t) = min(Y_t · k_chip,
                      C_chip(t),
                      C_energy(t),
                      C_REE(t))

Calibration: TSMC/Samsung/Intel filings; IEA World Energy Outlook 2024;
USGS Mineral Commodity Summaries.

## 2. Capability (jump-diffusion + GEV)

Task-horizon `C_t` (hours) evolves under

    dC_t = μ · C_t · dt + σ · C_t · dW_t + J_t · dN_t

with continuous drift `μ = 0.10`, volatility `σ = 0.25`, Poisson jump
intensity `λ_jump = 0.05/month`, and jump magnitudes following a
Generalized Extreme Value (Frechet) distribution with shape `ξ = 0.15`,
location `log(7)` (METR doubling time), scale `0.30`. The drift is
amplified by frontier compute through a compute elasticity of `0.35`
above the 2026 anchor (10²⁶ FLOPs).

Three thresholds are tracked:
- **168 hours** — sustained one-week autonomous operation
- **720 hours** — one-month autonomous projects
- **2160 hours** — three-month research cycles (recursive improvement)

## 3. Geopolitical (best-response approximation to MPE)

Eight countries `i` choose investment `a_{i,t}` to maximize utility

    u_i(s, a_i) = φ · (c_{i,t} / max_j c_{j,t})
                  − γ · a_{i,t}^2
                  − ρ · max(0, c_max − c_{i,t})

with `φ = 0.80`, `γ = 0.10`, country-specific `ρ ∈ [0.10, 0.25]`. Each
country observes rivals with noise `ε ~ N(0, σ_obs · (1 − t_j))`,
`σ_obs = 0.20`, `t_j ∈ [0, 1]` the country's transparency. The full
proposal solves Markov Perfect Equilibrium via value-function iteration;
the reference baseline uses a fast best-response approximation that
matches MPE on the metrics relevant to the leaderboard.

Technology transfer follows
    Δ c_{i,t} ← Σ_j τ_{ji} · max(0, c_{j,t} − c_{i,t})
with `τ` calibrated from patent-citation networks, researcher mobility,
and arXiv co-authorship.

## 4. Economic (CES + sticky wages)

Each sector `s` produces

    Y_s = [α_s (AI_s)^ρ + (1 − α_s) (L_s)^ρ]^(1/ρ)

with `ρ = −0.50` (substitution elasticity σ = 0.67, Acemoglu &
Restrepo 2020). Automation grows above the **168-hour capability
threshold** at sector-specific rates calibrated to elasticity of
substitution. Unemployment results from automation minus retraining,
with `τ_retrain = 2.0` years and 30% wage rigidity.

Sectors and their elasticities:

| Sector | Labor share | σ | Productivity β | γ |
|--------|------------:|---:|---:|---:|
| Software engineering | 0.05 | 0.85 | 1.50 | 0.40 |
| Knowledge work | 0.30 | 0.70 | 0.80 | 0.35 |
| Manual labor | 0.25 | 0.40 | 0.35 | 0.25 |
| Creative | 0.15 | 0.65 | 1.20 | 0.38 |
| Management | 0.20 | 0.35 | 0.50 | 0.30 |
| Manufacturing | 0.12 | 0.75 | 1.10 | 0.42 |
| Healthcare/Education | 0.05 | 0.30 | 0.60 | 0.32 |

## 5. Catastrophic risk (Hawkes self-exciting process)

Intensity

    λ(t) = λ_0(t) + Σ_{t_i < t} α · exp(−β · (t − t_i))

with `λ_base = 0.01/year`, `α = 0.50`, `β = 2.0/year`. The base intensity
is augmented by domain triggers:

| Trigger | Threshold | Δλ |
|---------|-----------|---:|
| Autonomous systems | C ≥ 168 h | 0.012 |
| Recursive improvement | C ≥ 2,160 h | 0.020 |
| Economic disruption | aggregate automation ≥ 30% | 0.010 |
| Geopolitical instability | US–China gap ≥ 40% | 0.012 |

Calibration: 127 documented AI incidents from the Partnership on AI
Incident Database (2012–2024), validated against earthquake aftershock
and financial-contagion parameter ranges (`α ∈ [0.30, 0.80]`,
`β ∈ [1.0, 3.0]`).

## What the baseline does NOT do

- Treats parameters as independent except where coupling is explicit.
- Smooth stochastic processes; potential regime shifts not modeled.
- Alignment-research progress is exogenous.
- Non-state actors and individual frontier labs not represented.
- Does not adapt online as new information arrives during forecast horizon.

The leaderboard rewards methods that improve on these explicit
limitations.
