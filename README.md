# Methodological appraisal of AI studies in bone tumour imaging

Appraisal dataset, operational scoring rubric and sensitivity-analysis code for a
systematic review of artificial intelligence in bone tumour imaging
(**N = 78 studies**, published 2019–2026).

> **Status:** the accompanying manuscript is under review (2026). The archived
> version of this repository will receive a DOI upon acceptance.

This repository supports verification and reuse of the methodological quality
appraisal reported in the manuscript. It contains the per-study domain scores,
the full scoring rubric, the analysis script, and every table the script generates.

## Repository structure


## Files

| File | Rows | Description |
|---|---|---|
| `Table/appraisal_rubric.csv` | 23 | 18 scoring anchors (6 domains × 3 levels), 4 category thresholds and the framework notes |
| `Table/scores.csv` | 78 | Per-study domain scores (DR, SS, VS, RP, CR, AD) and the derived total |
| `Table/included_studies.csv` | 78 | Study ID, publication year, venue and title of each included study |
| `Table/sensitivity_analysis.csv` | 15 | **Table S2**: category distribution under the primary model, four alternative cut-offs, three weighting schemes, one non-compensatory rule and six leave-one-domain-out models, with agreement statistics |
| `Table/sensitivity_agreement_statistics.csv` | 15 | Exact agreement, unweighted / linear / quadratic weighted kappa and Spearman rho versus the primary model |
| `Table/sensitivity_bootstrap_CI.csv` | 15 | Percentile bootstrap confidence intervals (10,000 resamples) for each category proportion |
| `Table/sensitivity_conclusion_checks.csv` | 15 | Whether each headline conclusion survives each scenario |
| `Table/domain_descriptive_statistics.csv` | 6 | Mean, SD, variance, share of item-level variance and proportion scoring 2 for each domain |
| `Table/study_classifications.csv` | 78 | Per-study category labels under the primary model and all 15 scenarios |

## Data dictionary: `Table/scores.csv`

| Column | Meaning |
|---|---|
| `id` | Study identifier (S1–S78); matches `Table/included_studies.csv` |
| `DR` | Data representativeness, 0–2 |
| `SS` | Sample size, 0–2 |
| `VS` | Validation strategy, 0–2 |
| `RP` | Reproducibility, 0–2 |
| `CR` | Clinical relevance, 0–2 |
| `AD` | Analytical depth, 0–2 |
| `total` | Sum of the six domains, 0–12 |

A score of 0 is annotated in the manuscript as **M** (criterion explicitly not met)
or **U** (not reported / unverifiable). The numeric value used in every analysis is
0 in both cases; the M/U distinction is recorded in the manuscript tables only.

## Appraisal framework

Six domains, each scored 0–2 (maximum 12 points, equal weighting):
data representativeness (**DR**), sample size (**SS**), validation strategy (**VS**),
reproducibility (**RP**), clinical relevance (**CR**), analytical depth (**AD**).

| Category | Total score |
|---|---|
| Low | 0–4 |
| Moderate | 5–8 |
| High | 9–12 |

Primary distribution (N = 78): **Low 34 (43.6%), Moderate 30 (38.5%), High 14 (17.9%)**.

## Reproducing the analysis

```bash
pip install -r requirements.txt
python sensitivity_appraisal.py
Higher category across scenarios: 9-21 studies (11.5-26.9%)

All 15 scenarios match the documented expected counts.
