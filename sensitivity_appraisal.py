#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity analysis for a 6-domain / 12-point methodological appraisal framework
(bone-tumour AI systematic review, N = 79).

Reproduces:
  tbl_scenarios.csv     cut-point and weighting scenarios
  tbl_lodo.csv          leave-one-domain-out analysis
  tbl_agreement.csv     exact agreement / kappa (unweighted, linear, quadratic) / Spearman rho
  tbl_bootstrap_ci.csv  bootstrap percentile CIs for category proportions
  tbl_domain_stats.csv  domain means, SDs, variance shares, Cronbach alpha
  scores.csv            cleaned per-study scores (input for the R companion)

Usage:  python sensitivity_appraisal.py
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

DOMAINS = ["DR", "SS", "VS", "RP", "CR", "AD"]
CATS = ["Low", "Moderate", "High"]

# (study_id, DR, SS, VS, RP, CR, AD, total_as_published)   -- S54 AD = 1
RAW = [
    ("S1", 0, 0, 0, 1, 1, 1, 3),
    ("S2", 1, 0, 0, 1, 2, 1, 5),
    ("S3", 2, 1, 2, 1, 1, 2, 9),
    ("S4", 0, 2, 2, 1, 2, 2, 9),
    ("S5", 1, 0, 1, 1, 1, 1, 5),
    ("S6", 1, 1, 1, 1, 1, 1, 6),
    ("S7", 1, 1, 1, 1, 1, 1, 6),
    ("S8", 2, 1, 2, 1, 1, 1, 8),
    ("S9", 0, 0, 0, 1, 1, 0, 2),
    ("S10", 2, 0, 2, 1, 2, 2, 9),
    ("S11", 1, 1, 1, 1, 1, 1, 6),
    ("S12", 1, 2, 1, 1, 1, 2, 8),
    ("S13", 1, 0, 0, 1, 1, 2, 5),
    ("S14", 2, 2, 2, 1, 1, 2, 10),
    ("S15", 2, 1, 2, 1, 2, 1, 9),
    ("S16", 1, 1, 1, 1, 1, 2, 7),
    ("S17", 1, 0, 0, 1, 1, 0, 3),
    ("S18", 0, 0, 0, 1, 1, 2, 4),
    ("S19", 0, 0, 0, 1, 1, 2, 4),
    ("S20", 0, 0, 0, 1, 1, 2, 4),
    ("S21", 0, 0, 0, 1, 1, 2, 4),
    ("S22", 0, 1, 0, 1, 1, 2, 5),
    ("S23", 1, 1, 1, 1, 1, 2, 7),
    ("S24", 0, 1, 1, 1, 1, 2, 6),
    ("S25", 2, 1, 1, 1, 1, 2, 8),
    ("S26", 0, 0, 0, 1, 1, 1, 3),
    ("S27", 0, 1, 1, 1, 1, 0, 4),
    ("S28", 0, 1, 0, 1, 1, 1, 4),
    ("S29", 0, 1, 1, 1, 1, 0, 4),
    ("S30", 0, 0, 0, 1, 1, 2, 4),
    ("S31", 0, 0, 0, 1, 1, 1, 3),
    ("S32", 0, 2, 1, 1, 1, 2, 7),
    ("S33", 1, 2, 1, 1, 1, 1, 7),
    ("S34", 0, 0, 0, 1, 0, 1, 2),
    ("S35", 0, 0, 0, 1, 1, 0, 2),
    ("S36", 0, 0, 0, 1, 1, 2, 4),
    ("S37", 0, 2, 0, 0, 0, 0, 2),
    ("S38", 0, 0, 0, 1, 1, 1, 3),
    ("S39", 1, 0, 0, 1, 1, 2, 5),
    ("S40", 0, 0, 0, 0, 1, 0, 1),
    ("S41", 2, 1, 1, 1, 1, 2, 8),
    ("S42", 2, 1, 2, 1, 1, 2, 9),
    ("S43", 0, 1, 0, 1, 1, 1, 4),
    ("S44", 1, 2, 2, 1, 2, 2, 10),
    ("S45", 0, 2, 0, 1, 0, 0, 3),
    ("S46", 0, 2, 0, 1, 1, 0, 4),
    ("S47", 1, 1, 0, 0, 1, 0, 3),
    ("S48", 0, 1, 1, 1, 0, 0, 3),
    ("S49", 2, 2, 2, 1, 1, 2, 10),
    ("S50", 0, 0, 0, 1, 0, 0, 1),
    ("S51", 0, 0, 0, 1, 1, 0, 2),
    ("S52", 2, 2, 2, 1, 2, 2, 11),
    ("S53", 1, 0, 0, 1, 1, 0, 3),
    ("S54", 2, 1, 2, 1, 1, 1, 8),
    ("S55", 0, 0, 0, 1, 0, 0, 1),
    ("S56", 1, 1, 1, 1, 0, 2, 6),
    ("S57", 1, 0, 1, 1, 1, 2, 6),
    ("S58", 0, 2, 2, 1, 1, 0, 6),
    ("S59", 1, 1, 1, 1, 1, 2, 7),
    ("S60", 0, 0, 0, 1, 1, 1, 3),
    ("S61", 0, 0, 0, 1, 0, 1, 2),
    ("S62", 2, 2, 2, 1, 1, 2, 10),
    ("S63", 0, 0, 0, 1, 1, 0, 2),
    ("S64", 2, 2, 2, 1, 2, 2, 11),
    ("S65", 0, 0, 0, 1, 1, 2, 4),
    ("S66", 0, 0, 0, 1, 1, 2, 4),
    ("S67", 2, 2, 2, 1, 1, 2, 10),
    ("S68", 0, 0, 1, 1, 1, 2, 5),
    ("S69", 1, 1, 2, 1, 2, 1, 8),
    ("S70", 2, 2, 2, 1, 2, 2, 11),
    ("S71", 2, 1, 2, 1, 1, 2, 9),
    ("S72", 1, 0, 1, 1, 1, 2, 6),
    ("S73", 0, 1, 1, 1, 1, 2, 6),
    ("S74", 1, 0, 1, 1, 1, 1, 5),
    ("S75", 0, 0, 0, 1, 1, 1, 3),
    ("S76", 0, 0, 0, 1, 1, 2, 4),
    ("S77", 0, 0, 1, 1, 1, 2, 5),
    ("S78", 2, 2, 2, 1, 1, 2, 10),
    ("S79", 2, 1, 2, 1, 1, 1, 8),
]

df = pd.DataFrame(RAW, columns=["id"] + DOMAINS + ["total"]).set_index("id")

# ---- integrity checks -------------------------------------------------------
assert df[DOMAINS].sum(axis=1).eq(df["total"]).all(), "domain scores do not sum to total"
assert df[DOMAINS].sum().sum() == 438, "column sums must equal sum of totals (438)"


# ------------------------------------------------------------------ helpers
def classify(score, low=4.0, high=9.0) -> pd.Categorical:
    """Low: score <= low | High: score >= high | Moderate otherwise."""
    s = np.asarray(score, dtype=float)
    lab = np.where(s <= low, "Low", np.where(s >= high, "High", "Moderate"))
    return pd.Categorical(lab, categories=CATS, ordered=True)


def classify_prop(score, maximum, low_frac=1 / 3, high_frac=2 / 3) -> pd.Categorical:
    """Category = thirds of the maximum attainable score (used for LODO)."""
    p = np.asarray(score, dtype=float) / float(maximum)
    lab = np.where(p <= low_frac, "Low", np.where(p > high_frac, "High", "Moderate"))
    return pd.Categorical(lab, categories=CATS, ordered=True)


def weighted_scores(df, weights):
    """Domain weights rescaled to sum to 6, preserving the 0-12 metric."""
    w = np.asarray(weights, dtype=float)
    w = w * (len(DOMAINS) / w.sum())
    return df[DOMAINS].to_numpy(dtype=float) @ w


def classify_noncomp(df, total) -> pd.Categorical:
    """Non-compensatory rule:
    Low  = total <= 4  OR  VS = 0 (absent patient-level validation is disqualifying)
    High = total >= 9 AND DR >= 1 AND SS >= 1 AND VS = 2
    """
    lab = []
    for i, t in enumerate(np.asarray(total, dtype=float)):
        d, s, v = df["DR"].iat[i], df["SS"].iat[i], df["VS"].iat[i]
        if t <= 4 or v == 0:
            lab.append("Low")
        elif t >= 9 and d >= 1 and s >= 1 and v == 2:
            lab.append("High")
        else:
            lab.append("Moderate")
    return pd.Categorical(lab, categories=CATS, ordered=True)


def tertile_cuts(score):
    """Cut after the ceil(n/3)-th and ceil(2n/3)-th order statistics
    (ties at a cut are assigned to the lower category)."""
    s = np.sort(np.asarray(score, dtype=float))
    n = s.size
    return s[math.ceil(n / 3) - 1], s[math.ceil(2 * n / 3) - 1]


def cohen_kappa(a, b, weights=None) -> float:
    ia = pd.Categorical(a, CATS, ordered=True).codes
    ib = pd.Categorical(b, CATS, ordered=True).codes
    k = len(CATS)
    O = np.zeros((k, k))
    np.add.at(O, (ia, ib), 1.0)
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    D = np.abs(np.subtract.outer(np.arange(k), np.arange(k))).astype(float)
    if weights is None:
        D = (D > 0).astype(float)
    elif weights == "linear":
        D = D / (k - 1)
    elif weights == "quadratic":
        D = (D / (k - 1)) ** 2
    else:
        raise ValueError(weights)
    return 1.0 - (D * O).sum() / (D * E).sum()


def bootstrap_ci(labels, B=10_000, seed=20261126):
    rng = np.random.default_rng(seed)
    lab = np.asarray(labels)
    n = lab.size
    draws = rng.integers(0, n, size=(B, n))
    out = {}
    for c in CATS:
        p = (lab[draws] == c).mean(axis=1) * 100
        out[c] = tuple(np.round(np.percentile(p, [2.5, 50, 97.5]), 1))
    return out


# ---------------------------------------------------------------- scenarios
total = df[DOMAINS].sum(axis=1)
sd_pop = df[DOMAINS].std(ddof=0)
c1, c2 = tertile_cuts(total)

scen = {
    "1 baseline (<=4 / 5-8 / >=9)":
        (total, classify(total, 4, 9)),
    "2 cutpoints shifted -1 (<=3 / 4-7 / >=8)":
        (total, classify(total, 3, 8)),
    "3 cutpoints shifted +1 (<=5 / 6-9 / >=10)":
        (total, classify(total, 5, 10)),
    "4 higher category >=8":
        (total, classify(total, 4, 8)),
    "5 higher category >=10":
        (total, classify(total, 4, 10)),
    f"6 empirical tertiles (<= {c1:.0f} / {c2:.0f} +)":
        (total, classify(total, c1, c2)),
    "7 evidence-oriented weights (1.5/1/2/.5/.5/.5)":
        (weighted_scores(df, [1.5, 1, 2, .5, .5, .5]), None),
    "8 variance-based weights (proportional to SD)":
        (weighted_scores(df, sd_pop), None),
    "9 validity-only, four domains equally weighted":
        (weighted_scores(df, [1, 1, 1, 0, 0, 1]), None),
    "10 reproducibility-emphasised (RP x2)":
        (weighted_scores(df, [1, 1, 1, 2, 1, 1]), None),
    "11 non-compensatory rule (VS=0 disqualifying)":
        (total, classify_noncomp(df, total)),
}
for key in list(scen)[6:10]:
    scen[key] = (scen[key][0], classify(scen[key][0], 4, 9))

base_lab = scen["1 baseline (<=4 / 5-8 / >=9)"][1]


# ------------------------------------------------------------------ tables
rows = []
for name, (score, lab) in scen.items():
    vc = pd.Series(lab).value_counts().reindex(CATS, fill_value=0)
    rows.append({
        "scenario": name,
        "Low": int(vc["Low"]),
        "Moderate": int(vc["Moderate"]),
        "High": int(vc["High"]),
        "pct_Low": round(100 * vc["Low"] / len(lab), 1),
        "pct_High": round(100 * vc["High"] / len(lab), 1),
        "reclassified_n": int((np.asarray(lab) != np.asarray(base_lab)).sum()),
        "rho_vs_baseline": round(float(pd.Series(score).corr(pd.Series(total), method="spearman")), 3),
    })
tbl_scen = pd.DataFrame(rows)
assert tbl_scen.loc[0, ["Low", "Moderate", "High"]].tolist() == [35, 29, 15]

lodo_rows = []
for d in DOMAINS:
    s = total - df[d]
    lab = classify_prop(s, 10)
    vc = pd.Series(lab).value_counts().reindex(CATS, fill_value=0)
    lodo_rows.append({
        "left_out": f"-{d}",
        "Low": int(vc["Low"]),
        "Moderate": int(vc["Moderate"]),
        "High": int(vc["High"]),
        "exact_agreement_pct": round(100 * (np.asarray(lab) == np.asarray(base_lab)).mean(), 1),
        "kappa_unweighted": round(cohen_kappa(base_lab, lab), 3),
        "kappa_linear": round(cohen_kappa(base_lab, lab, "linear"), 3),
        "kappa_quadratic": round(cohen_kappa(base_lab, lab, "quadratic"), 3),
        "rho_vs_baseline": round(float(pd.Series(s).corr(pd.Series(total), method="spearman")), 3),
    })
tbl_lodo = pd.DataFrame(lodo_rows)

agree = []
for name, (score, lab) in scen.items():
    agree.append({
        "scenario": name,
        "exact_agreement_pct": round(100 * (np.asarray(lab) == np.asarray(base_lab)).mean(), 1),
        "kappa_unweighted": round(cohen_kappa(base_lab, lab), 3),
        "kappa_linear": round(cohen_kappa(base_lab, lab, "linear"), 3),
        "kappa_quadratic": round(cohen_kappa(base_lab, lab, "quadratic"), 3),
        "rho_vs_baseline": round(float(pd.Series(score).corr(pd.Series(total), method="spearman")), 3),
    })
tbl_agree = pd.DataFrame(agree)

boot_rows = []
for name, (score, lab) in scen.items():
    ci = bootstrap_ci(lab)
    boot_rows.append({
        "scenario": name,
        **{f"{c}_% (95% CI)": f"{ci[c][1]} ({ci[c][0]}-{ci[c][2]})" for c in CATS},
    })
tbl_boot = pd.DataFrame(boot_rows)

k = len(DOMAINS)
var_item = df[DOMAINS].var(ddof=0)
tbl_dom = pd.DataFrame({
    "mean": df[DOMAINS].mean().round(2),
    "sd": sd_pop.round(2),
    "variance": var_item.round(3),
    "pct_of_item_variance": (100 * var_item / var_item.sum()).round(1),
    "pct_scoring_2": (100 * (df[DOMAINS] == 2).mean()).round(1),
})
alpha = k / (k - 1) * (1 - var_item.sum() / total.var(ddof=0))


# ------------------------------------------------------------------- output
pd.set_option("display.width", 200, "display.max_columns", 30)
print("\n== Table S-A. Domain-level descriptive statistics ==")
print(tbl_dom.to_string())
print(f"\nSum of item variances = {var_item.sum():.3f} | Var(total) = {total.var(ddof=0):.3f} "
      f"| Cronbach alpha = {alpha:.3f}")

print("\n== Table S-B. Cut-point and weighting scenarios ==")
print(tbl_scen.to_string(index=False))

print("\n== Table S-C. Leave-one-domain-out ==")
print(tbl_lodo.to_string(index=False))

print("\n== Table S-D. Agreement with baseline classification ==")
print(tbl_agree.to_string(index=False))

print("\n== Table S-E. Bootstrap percentile CIs (10,000 resamples) ==")
print(tbl_boot.to_string(index=False))

df.reset_index().to_csv("scores.csv", index=False)
for name, t in {"tbl_scenarios": tbl_scen, "tbl_lodo": tbl_lodo, "tbl_agreement": tbl_agree,
                "tbl_bootstrap_ci": tbl_boot, "tbl_domain_stats": tbl_dom}.items():
    t.to_csv(f"{name}.csv", index=False)
print("\nCSV files written: scores.csv, tbl_*.csv")
