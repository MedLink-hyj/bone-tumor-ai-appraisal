# Methodological appraisal of AI studies in bone tumour imaging

Appraisal dataset, operational scoring rubric and sensitivity-analysis code for a
systematic review of artificial intelligence in bone tumour imaging (N = 79).

## Contents
| File | Description |
|---|---|
| `scores.csv` | Per-study domain scores (DR, SS, VS, RP, CR, AD); `0(M)` = criterion explicitly not met, `0(U)` = not reported / unverifiable |
| `appraisal_rubric.pdf` | Operational scoring criteria for the six domains |
| `sensitivity_appraisal.py` | Cut-point, weighting, leave-one-domain-out, agreement and bootstrap analyses |
| `tbl_*.csv` | Output tables (scenarios, LODO, agreement, bootstrap CIs) |

## Reproducing the analysis
```bash
pip install numpy pandas
python sensitivity_appraisal.py
```
The script prints and writes every count reported in the paper and includes
integrity checks: domain scores must sum to each study's total, and column sums
must equal 438. Baseline categories: Low 35 / Moderate 29 / High 15.

## Appraisal framework (summary)
Six domains scored 0–2 (maximum 12): Data Representativeness (DP), Sample Size (SS),
Validation Strategy (VS), Reproducibility (RP), Clinical Relevance (CR), Analytical Depth (AD).
Categories: 0–4 Low, 5–8 Moderate, 9–12 High.

## Citation
This review is under review in Information Fusion.

## License
Code: MIT. Data tables: CC BY 4.0.

## Contact
Mingyang Jiang, The First Affiliated Hospital of Guangxi Medical University — 202120576@sr.gxmu.edu.cn
