#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity analysis for a six-domain / 12-point methodological appraisal
framework (bone-tumour AI systematic review, N = 78).

Analyses
--------
1. Alternative cut-offs       5 scenarios (2 stricter + 2 lenient + primary)
2. Alternative weighting      3 scenarios (VS x2, RP x2, variance-proportional)
3. Non-compensatory rule      1 scenario (absent patient-level validation disqualifies)
4. Leave-one-domain-out       6 scenarios (score rescaled to 0-12, primary cut-offs)

The cut-off scenarios form a closed two-sided envelope around the primary
thresholds so that the distribution can move in both directions.

Outputs (written to the working directory)
------------------------------------------
scores.csv                            cleaned per-study scores            (utf-8)
study_classifications.csv             per-study labels for every scenario (utf-8)
Table_S2_sensitivity_analysis.csv     publication-oriented summary        (utf-8-sig)
sensitivity_agreement_statistics.csv  agreement / kappa / rho             (utf-8-sig)
sensitivity_bootstrap_CI.csv          percentile bootstrap CIs            (utf-8-sig)
sensitivity_conclusion_checks.csv     does each conclusion survive?       (utf-8-sig)
domain_descriptive_statistics.csv     domain means, SDs, variance shares  (utf-8-sig)

Usage
-----
    python sensitivity_appraisal.py

Requires numpy and pandas only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DOMAINS = ["DR", "SS", "VS", "RP", "CR", "AD"]
CATS = ["Low", "Moderate", "High"]

PRIMARY_LOW = 4.0
PRIMARY_HIGH = 9.0
N_EXPECTED = 78
TOTAL_SUM_EXPECTED = 433
N_BOOT = 10_000
SEED = 20261126

# (study_id, DR, SS, VS, RP, CR, AD, total_as_published)
RAW = [
    ("S1", 0, 0, 0, 1, 1, 1, 3),
    ("S2", 1, 0, 0, 1, 2, 1, 5),
    ("S3", 1, 0, 1, 1, 2, 1, 6),
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
    ("S34", 0, 0, 0, 1, 1, 0, 2),
    ("S35", 0, 0, 0, 1, 1, 2, 4),
    ("S36", 0, 2, 0, 0, 0, 0, 2),
    ("S37", 0, 0, 0, 1, 1, 1, 3),
    ("S38", 1, 0, 0, 1, 1, 2, 5),
    ("S39", 0, 0, 0, 0, 1, 0, 1),
    ("S40", 2, 1, 1, 1, 1, 2, 8),
    ("S41", 2, 1, 2, 1, 1, 2, 9),
    ("S42", 0, 1, 0, 1, 1, 1, 4),
    ("S43", 1, 2, 2, 1, 2, 2, 10),
    ("S44", 0, 2, 0, 1, 0, 0, 3),
    ("S45", 0, 2, 0, 1, 1, 0, 4),
    ("S46", 1, 1, 0, 0, 1, 0, 3),
    ("S47", 0, 1, 1, 1, 0, 0, 3),
    ("S48", 2, 2, 2, 1, 1, 2, 10),
    ("S49", 0, 0, 0, 1, 0, 0, 1),
    ("S50", 0, 0, 0, 1, 1, 0, 2),
    ("S51", 2, 2, 2, 1, 2, 2, 11),
    ("S52", 1, 0, 0, 1, 1, 0, 3),
    ("S53", 2, 1, 2, 1, 1, 1, 8),
    ("S54", 0, 0, 0, 1, 0, 0, 1),
    ("S55", 1, 1, 1, 1, 0, 2, 6),
    ("S56", 1, 0, 1, 1, 1, 2, 6),
    ("S57", 0, 2, 2, 1, 1, 0, 6),
    ("S58", 1, 1, 1, 1, 1, 2, 7),
    ("S59", 0, 0, 0, 1, 1, 1, 3),
    ("S60", 0, 0, 0, 1, 0, 1, 2),
    ("S61", 2, 2, 2, 1, 1, 2, 10),
    ("S62", 0, 0, 0, 1, 1, 0, 2),
    ("S63", 2, 2, 2, 1, 2, 2, 11),
    ("S64", 0, 0, 0, 1, 1, 2, 4),
    ("S65", 0, 0, 0, 1, 1, 2, 4),
    ("S66", 2, 2, 2, 1, 1, 2, 10),
    ("S67", 0, 0, 1, 1, 1, 2, 5),
    ("S68", 1, 1, 2, 1, 2, 1, 8),
    ("S69", 2, 2, 2, 1, 2, 2, 11),
    ("S70", 2, 1, 2, 1, 1, 2, 9),
    ("S71", 1, 0, 1, 1, 1, 2, 6),
    ("S72", 0, 1, 1, 1, 1, 2, 6),
    ("S73", 1, 0, 1, 1, 1, 1, 5),
    ("S74", 0, 0, 0, 1, 1, 1, 3),
    ("S75", 0, 0, 0, 1, 1, 2, 4),
    ("S76", 0, 0, 1, 1, 1, 2, 5),
    ("S77", 2, 2, 2, 1, 1, 2, 10),
    ("S78", 2, 1, 2, 1, 1, 1, 8),
]

# Documented expected counts. Mismatches are reported as warnings (not errors)
# so the script stays usable if a single data point is later corrected; the
# primary distribution itself is enforced by an assertion.
EXPECTED_COUNTS = {
    "Cutoff: primary (<=4 / 5-8 / >=9)": (34, 30, 14),
    "Cutoff: alternative A (Low <=3)": (20, 44, 14),
    "Cutoff: alternative B (High >=10)": (34, 35, 9),
    "Cutoff: alternative C (High >=8, lenient)": (34, 23, 21),
    "Cutoff: alternative D (Low <=5, High >=8, lenient)": (42, 15, 21),
    "Weighting: validation-weighted (VS x2)": (32, 32, 14),
    "Weighting: reproducibility-weighted (RP x2)": (20, 49, 9),
    "Weighting: variance-based (proportional to SD)": (34, 30, 14),
    "Rule: non-compensatory (VS=0 disqualifying)": (38, 28, 12),
    "LODO: excluding DR": (26, 42, 10),
    "LODO: excluding SS": (34, 31, 13),
    "LODO: excluding VS": (30, 39, 9),
    "LODO: excluding RP": (34, 30, 14),
    "LODO: excluding CR": (35, 32, 11),
    "LODO: excluding AD": (36, 32, 10),
}


# --------------------------------------------------------------------- data
def load_data() -> pd.DataFrame:
    """Build the score matrix and run integrity checks."""
    columns = ["id"] + DOMAINS + ["total"]
    if len({len(row) for row in RAW}) != 1:
        raise ValueError("All RAW rows must have the same number of fields.")
    if len(RAW[0]) != len(columns):
        raise ValueError(
            f"Each RAW row must contain {len(columns)} fields "
            f"(id + {len(DOMAINS)} domains + total); found {len(RAW[0])}."
        )
    data = pd.DataFrame(RAW, columns=columns).set_index("id")
    validate(data)
    return data


def validate(data: pd.DataFrame) -> None:
    if len(data) != N_EXPECTED:
        raise ValueError(f"Expected {N_EXPECTED} studies, found {len(data)}.")
    if not data.index.is_unique:
        raise ValueError("Study IDs must be unique.")
    if not data[DOMAINS].apply(lambda c: c.between(0, 2).all()).all():
        raise ValueError("Each domain score must be between 0 and 2.")
    if not data[DOMAINS].apply(lambda c: np.equal(c, np.floor(c)).all()).all():
        raise ValueError("Domain scores must be integer values (0, 1, 2).")
    if not data[DOMAINS].sum(axis=1).eq(data["total"]).all():
        bad = data.index[~data[DOMAINS].sum(axis=1).eq(data["total"])].tolist()
        raise ValueError(f"Published total != sum of domains for: {bad}")
    if int(data[DOMAINS].sum().sum()) != TOTAL_SUM_EXPECTED:
        raise ValueError(
            f"Sum of all domain scores is {int(data[DOMAINS].sum().sum())}, "
            f"expected {TOTAL_SUM_EXPECTED}."
        )


# ------------------------------------------------------------------ helpers
def classify(score, low: float = PRIMARY_LOW, high: float = PRIMARY_HIGH) -> pd.Categorical:
    """Classify scores on a 0-12 metric.

    Low      : score <= low
    Moderate : low < score < high
    High     : score >= high
    """
    values = np.asarray(score, dtype=float)
    labels = np.where(values <= low, "Low", np.where(values >= high, "High", "Moderate"))
    return pd.Categorical(labels, categories=CATS, ordered=True)


def weighted_scores(data: pd.DataFrame, weights) -> np.ndarray:
    """Weighted score rescaled to the original 0-12 metric.

    S_w = 12 * sum(w_i * s_i) / (2 * sum(w_i))

    With all weights equal this returns the unweighted total exactly.
    """
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(DOMAINS),):
        raise ValueError(f"Expected {len(DOMAINS)} domain weights; got {w.shape}.")
    if np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("Weights must be finite, non-negative, with positive sum.")
    scores = data[DOMAINS].to_numpy(dtype=float)
    return 12.0 * (scores @ w) / (2.0 * w.sum())


def noncomp_labels(data: pd.DataFrame, total) -> pd.Categorical:
    """Non-compensatory rule.

    Low      : total <= 4  OR  VS = 0 (no patient-level validation separation)
    High     : total >= 9 AND DR >= 1 AND SS >= 1 AND VS = 2
    Moderate : otherwise

    Rationale: deficiencies in data provenance or validation cannot be offset
    by strength in analytical depth.
    """
    labels = []
    for i, t in enumerate(np.asarray(total, dtype=float)):
        dr, ss, vs = data["DR"].iat[i], data["SS"].iat[i], data["VS"].iat[i]
        if t <= PRIMARY_LOW or vs == 0:
            labels.append("Low")
        elif t >= PRIMARY_HIGH and dr >= 1 and ss >= 1 and vs == 2:
            labels.append("High")
        else:
            labels.append("Moderate")
    return pd.Categorical(labels, categories=CATS, ordered=True)


def category_summary(labels, reference=None) -> dict:
    """Counts, percentages, and agreement against a reference classification."""
    lab = np.asarray(labels).astype(str)
    n = lab.size
    out = {"N": n}
    for cat in CATS:
        k = int(np.sum(lab == cat))
        out[f"{cat}_n"] = k
        out[f"{cat}_pct"] = round(100.0 * k / n, 1)
    if reference is None:
        out["agreement_pct"] = 100.0
        out["reclassified_n"] = 0
    else:
        ref = np.asarray(reference).astype(str)
        if ref.size != n:
            raise ValueError("Reference and scenario labels differ in length.")
        out["agreement_pct"] = round(100.0 * np.mean(lab == ref), 1)
        out["reclassified_n"] = int(np.sum(lab != ref))
    return out


def cohen_kappa(a, b, weighting: str = "linear") -> float:
    """Weighted Cohen's kappa for three ordered categories."""
    ac = pd.Categorical(a, categories=CATS, ordered=True).codes
    bc = pd.Categorical(b, categories=CATS, ordered=True).codes
    if np.any(ac < 0) or np.any(bc < 0):
        raise ValueError("Unexpected or missing category label.")

    k = len(CATS)
    observed = np.zeros((k, k), dtype=float)
    np.add.at(observed, (ac, bc), 1.0)
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))

    distance = np.abs(np.subtract.outer(np.arange(k), np.arange(k))).astype(float)
    if weighting == "linear":
        disagreement = distance / (k - 1)
    elif weighting == "quadratic":
        disagreement = (distance / (k - 1)) ** 2
    elif weighting == "unweighted":
        disagreement = (distance > 0).astype(float)
    else:
        raise ValueError("weighting must be 'linear', 'quadratic' or 'unweighted'.")

    denom = float(np.sum(disagreement * expected))
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum(disagreement * observed) / denom)


def spearman_rho(x, y) -> float:
    """Spearman correlation; NaN if either vector is constant."""
    r = pd.Series(np.asarray(x, dtype=float)).corr(
        pd.Series(np.asarray(y, dtype=float)), method="spearman"
    )
    return float(r) if pd.notna(r) else float("nan")


def bootstrap_category_ci(labels, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Percentile bootstrap CIs for category proportions.

    These describe resampling variability of the included studies; they are not
    inferential intervals for a target population, and they do not quantify
    uncertainty in the appraisal thresholds themselves.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(labels).astype(str)
    n = values.size
    draws = rng.integers(0, n, size=(n_boot, n))
    out = {}
    for cat in CATS:
        props = (values[draws] == cat).mean(axis=1) * 100.0
        lo, med, hi = np.percentile(props, [2.5, 50, 97.5])
        out[cat] = (round(float(lo), 1), round(float(med), 1), round(float(hi), 1))
    return out


# ---------------------------------------------------------------- scenarios
def build_scenarios(data: pd.DataFrame) -> dict:
    """Return {label: (score_vector, labels)} for all 15 scenarios."""
    total = data[DOMAINS].sum(axis=1).astype(float)
    total_np = total.to_numpy(dtype=float)
    sd_pop = data[DOMAINS].std(ddof=0)

    scen = {}

    # --- 1. alternative cut-offs (two-sided envelope) ---------------------
    scen["Cutoff: primary (<=4 / 5-8 / >=9)"] = (total_np, classify(total))
    scen["Cutoff: alternative A (Low <=3)"] = (total_np, classify(total, low=3, high=9))
    scen["Cutoff: alternative B (High >=10)"] = (total_np, classify(total, low=4, high=10))
    scen["Cutoff: alternative C (High >=8, lenient)"] = (
        total_np, classify(total, low=4, high=8))
    scen["Cutoff: alternative D (Low <=5, High >=8, lenient)"] = (
        total_np, classify(total, low=5, high=8))

    # --- 2. alternative weighting ----------------------------------------
    vs_weighted = weighted_scores(data, [1, 1, 2, 1, 1, 1])
    rp_weighted = weighted_scores(data, [1, 1, 1, 2, 1, 1])
    sd_weighted = weighted_scores(data, sd_pop.to_numpy())
    scen["Weighting: validation-weighted (VS x2)"] = (
        vs_weighted, classify(vs_weighted))
    scen["Weighting: reproducibility-weighted (RP x2)"] = (
        rp_weighted, classify(rp_weighted))
    scen["Weighting: variance-based (proportional to SD)"] = (
        sd_weighted, classify(sd_weighted))

    # --- 3. non-compensatory rule ----------------------------------------
    scen["Rule: non-compensatory (VS=0 disqualifying)"] = (
        total_np, noncomp_labels(data, total))

    # --- 4. leave-one-domain-out -----------------------------------------
    # Remove one 0-2 domain (leaving a 0-10 score), rescale to 0-12, then
    # apply the primary cut-offs. This preserves each study's position
    # relative to the maximum attainable score.
    for domain in DOMAINS:
        reduced = total_np - data[domain].to_numpy(dtype=float)
        standardized = (12.0 / 10.0) * reduced
        scen[f"LODO: excluding {domain}"] = (
            standardized, classify(standardized, PRIMARY_LOW, PRIMARY_HIGH))

    return scen


# ------------------------------------------------------------------- tables
def build_tables(data: pd.DataFrame, scen: dict, base_labels):
    total_np = data[DOMAINS].sum(axis=1).to_numpy(dtype=float)
    primary_labels = np.asarray(base_labels).astype(str)

    s2_rows, agree_rows, boot_rows, check_rows = [], [], [], []
    for name, (score, labels) in scen.items():
        summary = category_summary(labels, reference=primary_labels)
        s2_rows.append({
            "Analysis": name,
            "Low n (%)": f"{summary['Low_n']} ({summary['Low_pct']:.1f})",
            "Moderate n (%)": f"{summary['Moderate_n']} ({summary['Moderate_pct']:.1f})",
            "High n (%)": f"{summary['High_n']} ({summary['High_pct']:.1f})",
            "Agreement with primary (%)": summary["agreement_pct"],
            "Reclassified n": summary["reclassified_n"],
            "Linear weighted kappa": round(cohen_kappa(primary_labels, labels, "linear"), 3),
            "Spearman rho": round(spearman_rho(score, total_np), 3),
        })

        agree_rows.append({
            "Analysis": name,
            "Exact agreement (%)": summary["agreement_pct"],
            "Reclassified n": summary["reclassified_n"],
            "Unweighted kappa": round(cohen_kappa(primary_labels, labels, "unweighted"), 3),
            "Linear weighted kappa": round(cohen_kappa(primary_labels, labels, "linear"), 3),
            "Quadratic weighted kappa": round(cohen_kappa(primary_labels, labels, "quadratic"), 3),
            "Spearman rho": round(spearman_rho(score, total_np), 3),
        })

        ci = bootstrap_category_ci(labels)
        row = {"Analysis": name}
        for cat in CATS:
            lo, med, hi = ci[cat]
            row[f"{cat} %"] = med
            row[f"{cat} 95% CI"] = f"{lo:.1f}-{hi:.1f}"
        boot_rows.append(row)

        counts = [summary[f"{c}_n"] for c in CATS]
        check_rows.append({
            "Analysis": name,
            "High < 50%": counts[2] / summary["N"] < 0.5,
            "High <= 1/3": counts[2] / summary["N"] <= 1 / 3,
            "Low is modal category": counts[0] == max(counts),
            "Low + Moderate >= 2/3": (counts[0] + counts[1]) / summary["N"] >= 2 / 3,
        })

    return (pd.DataFrame(s2_rows), pd.DataFrame(agree_rows),
            pd.DataFrame(boot_rows), pd.DataFrame(check_rows))


def build_domain_table(data: pd.DataFrame) -> pd.DataFrame:
    var_item = data[DOMAINS].var(ddof=0)
    tbl = pd.DataFrame({
        "Mean": data[DOMAINS].mean().round(2),
        "SD (population)": data[DOMAINS].std(ddof=0).round(2),
        "Variance (population)": var_item.round(3),
        "% of item variance": (100.0 * var_item / var_item.sum()).round(1),
        "Studies scoring 2 (%)": (100.0 * (data[DOMAINS] == 2).mean()).round(1),
    })
    tbl.index.name = "Domain"
    # Cronbach's alpha is deliberately not reported: the six domains are
    # formative indicators of methodological rigour, not a unidimensional
    # reflective psychometric scale.
    return tbl


# --------------------------------------------------------------------- main
def main(out_dir: str | Path = ".") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = load_data()
    total = data[DOMAINS].sum(axis=1).astype(float)
    base_labels = classify(total)
    primary = category_summary(base_labels)

    # ---- enforce the primary distribution -------------------------------
    assert [primary[f"{c}_n"] for c in CATS] == [34, 30, 14], (
        "Primary distribution changed: check the data or the cut-offs before "
        f"publication. Got {[primary[f'{c}_n'] for c in CATS]}."
    )

    scen = build_scenarios(data)
    tbl_s2, tbl_agree, tbl_boot, tbl_checks = build_tables(data, scen, base_labels)
    tbl_dom = build_domain_table(data)

    # ---- per-study label matrix -----------------------------------------
    labels_df = pd.DataFrame(
        {name: np.asarray(lab).astype(str) for name, (_, lab) in scen.items()},
        index=data.index,
    )
    labels_df.insert(0, "primary", np.asarray(base_labels).astype(str))

    # ---- verify documented counts (warn, do not crash) -------------------
    mismatches = []
    for _, row in tbl_s2.iterrows():
        got = tuple(int(row[c].split()[0]) for c in
                    ["Low n (%)", "Moderate n (%)", "High n (%)"])
        exp = EXPECTED_COUNTS.get(row["Analysis"])
        if exp is not None and got != exp:
            mismatches.append((row["Analysis"], exp, got))
        assert sum(got) == N_EXPECTED, f"counts do not sum to N for {row['Analysis']}"

    # ---- console output --------------------------------------------------
    pd.set_option("display.width", 240, "display.max_columns", 40)
    print(f"\n=== Primary classification (N = {N_EXPECTED}) ===")
    for cat in CATS:
        print(f"{cat:<9}: {primary[f'{cat}_n']:>2} ({primary[f'{cat}_pct']:.1f}%)")

    print("\n=== Table S2. Sensitivity analysis of the six-domain appraisal ===")
    print(tbl_s2.to_string(index=False))

    print("\n=== Agreement statistics ===")
    print(tbl_agree.to_string(index=False))

    print("\n=== Conclusion robustness ===")
    print(tbl_checks.to_string(index=False))

    print(f"\n=== Bootstrap percentile CIs ({N_BOOT:,} resamples) ===")
    print(tbl_boot.to_string(index=False))

    print("\n=== Domain-level descriptive statistics ===")
    print(tbl_dom.to_string())

    hi = [int(r.split()[0]) for r in tbl_s2["High n (%)"]]
    print(f"\nHigher category across scenarios: {min(hi)}-{max(hi)} studies "
          f"({100*min(hi)/N_EXPECTED:.1f}-{100*max(hi)/N_EXPECTED:.1f}%)")

    if mismatches:
        print("\nWARNING: the following scenarios differ from the documented "
              "expected counts. Check the data and the scenario definitions:")
        for name, exp, got in mismatches:
            print(f"  - {name}: expected {exp}, got {got}")
    else:
        print(f"\nAll {len(EXPECTED_COUNTS)} scenarios match the documented expected counts.")

    # ---- exports ---------------------------------------------------------
    data.reset_index().to_csv(out / "scores.csv", index=False, encoding="utf-8")
    labels_df.to_csv(out / "study_classifications.csv", encoding="utf-8")
    tbl_s2.to_csv(out / "Table_S2_sensitivity_analysis.csv", index=False, encoding="utf-8-sig")
    tbl_agree.to_csv(out / "sensitivity_agreement_statistics.csv", index=False, encoding="utf-8-sig")
    tbl_boot.to_csv(out / "sensitivity_bootstrap_CI.csv", index=False, encoding="utf-8-sig")
    tbl_checks.to_csv(out / "sensitivity_conclusion_checks.csv", index=False, encoding="utf-8-sig")
    tbl_dom.to_csv(out / "domain_descriptive_statistics.csv", encoding="utf-8-sig")

    print("\nFiles written:")
    for f in sorted(out.glob("*.csv")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
