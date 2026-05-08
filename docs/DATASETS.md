# Datasets

All datasets are released under [CC BY 4.0](../LICENSE-DATA). They are
curated, harmonized aggregations of publicly available upstream sources.
Each CSV row carries a `source` column linking back to its provenance.

A signed manifest at `data/MANIFEST.json` records SHA-256 hashes and row
counts for every released file; the leaderboard pins the manifest hash
at competition launch and at each quarterly update.

## Capability (`data/capability/`)

| File | Rows | Description |
|------|------|-------------|
| `metr_task_horizons.csv` | quarterly | METR-style median task-horizon (hours), 10/90 percentile envelope, observation count |
| `benchmark_scores.csv` | quarterly × benchmark | Frontier scores on a curated suite (MMLU-Pro, GPQA-Diamond, SWE-bench-Verified, AIME, MATH-500, HumanEval, MMMU, LiveCodeBench) |
| `breakthrough_events.csv` | one row per event | 47 capability breakthroughs since 2012, with category and log-magnitude (used to fit the GEV jump distribution) |
| `model_metadata.csv` | one row per model | Frontier-model parameter count, training compute (log10 FLOPs), release year, lab |

**Upstream sources**: METR public reports; Papers with Code; lab model
cards (Anthropic, OpenAI, DeepMind, Meta, Mistral, DeepSeek, Alibaba);
Stanford AI Index; Epoch AI public estimates.

## Compute (`data/compute/`)

| File | Description |
|------|-------------|
| `compute_trends.csv` | Quarterly frontier training compute (log10 FLOPs) with 10/90 envelope |
| `chip_capacity.csv` | Quarterly advanced-node fab capacity by manufacturer (TSMC, Samsung, Intel, Other) |
| `energy_demand.csv` | Annual AI datacenter electricity demand (TWh) |
| `rare_earth_production.csv` | Annual rare-earth-element production (China vs. rest of world, tonnes) |

**Upstream sources**: Epoch AI compute trends; TSMC / Samsung / Intel
financial filings; SemiAnalysis aggregated industry forecasts; IEA World
Energy Outlook 2024; USGS Mineral Commodity Summaries.

## Economic (`data/economic/`)

| File | Description |
|------|-------------|
| `sectoral_automation.csv` | Quarterly automation rate per sector (7 sectors), with labor share |
| `unemployment.csv` | Quarterly unemployment rate (US, EU) |
| `productivity_multiplier.csv` | Quarterly AI productivity multiplier per sector, 2024 = 1.00 |

**Upstream sources**: BLS Current Population Survey; Eurostat; OECD
Employment Outlook; IMF World Economic Outlook; Anthropic Economic
Index; McKinsey AI Monitor; Goldman Sachs AI productivity research;
Brynjolfsson et al. (2024).

## Geopolitical (`data/geopolitical/`)

| File | Description |
|------|-------------|
| `country_capabilities.csv` | Annual normalized AI capability for 8 major powers (US = 1.00 in 2024) |
| `country_attributes.csv` | Static attributes: investment capacity, risk aversion, transparency, AI researchers, GDP share |
| `transfer_matrix.csv` | Pairwise annual technology-transfer rates between countries |

**Upstream sources**: Stanford AI Index 2024; CSIS Technology
Competition Index; SIPRI defense R&D; World Bank national accounts;
USPTO/EPO patent citation networks; arXiv co-authorship analysis.

## Incidents (`data/incidents/`)

| File | Description |
|------|-------------|
| `annual_counts.csv` | Annual AI incident counts (2015–2025) at three severity tiers |
| `incident_log.csv` | Representative subset of incidents with date, severity, category |

**Upstream source**: Partnership on AI Incident Database
(https://incidentdatabase.ai), curated and tier-labeled.

## Policy (`data/policy/`)

| File | Description |
|------|-------------|
| `policy_events.csv` | Major AI-governance policy events (2018–2026), with category and instrument type (binding vs. advisory) |

**Upstream source**: Public regulatory record (EU AI Act, US Executive
Orders, China State Council, UK AI Safety Institute, summit declarations).

## Schema versioning

The current schema version is `1`, recorded in `MANIFEST.json`. A
breaking change to any column would bump this version and would be
announced at least 30 days in advance with a migration note.

## Known limitations

- **Capability dataset over-represents English-language and
  software-engineering benchmarks**, reflecting the underlying field.
  Multimodal and multilingual benchmarks (MMMU, XCOPA) are included as
  partial mitigation.
- **Economic data is concentrated in OECD economies.** Submissions for
  emerging-market labor markets are welcomed but are not officially
  scored.
- **Incident data is sparse** in absolute terms (~10–20 events/year
  above the inclusion threshold). The Hawkes baseline includes a
  robustness-to-rare-events term to compensate.
