#!/usr/bin/env python3
"""
Variance-corrected resampled t-test (Nadeau & Bengio, 2003, Mach Learn 52:239-281).

For N repeated random train/test splits, paired per-seed differences d_1..d_N of a metric are
NOT independent because training sets overlap. The corrected statistic is

    t = mean(d) / sqrt( (1/N + n_test/n_train) * var(d, ddof=1) ),   df = N - 1

with a two-sided p-value from the t distribution and a (1-alpha) interval
mean(d) +/- t_{1-alpha/2, N-1} * sqrt( (1/N + n_test/n_train) * var(d, ddof=1) ).

Usage (command line, reads the wide CSV written by h3_resplit_baseline.py):
  python stats_corrected.py --wide results_h3/per_seed_wide_for_spss.csv \
                            --out results_h3/table6_corrected_tests.csv [--test-size 0.2]
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["auroc", "balanced_accuracy", "f1_malignant"]
METRIC_LABELS = {"auroc": "AUROC", "balanced_accuracy": "Balanced accuracy",
                 "f1_malignant": "F1 (malignant)"}
RULE_LABELS = {"patient": "vs patient-level", "inferred": "vs inferred groups"}


def corrected_resampled_ttest(diff, test_train_ratio=0.25, conf=0.95):
    """Nadeau-Bengio corrected resampled t-test on paired per-seed differences.

    Parameters
    ----------
    diff : array-like, one paired difference per repeated split.
    test_train_ratio : n_test / n_train of each split (0.25 for an 80/20 split).
    conf : confidence level of the interval.
    """
    d = np.asarray(diff, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 3:
        raise ValueError("need at least 3 paired differences")
    if test_train_ratio < 0:
        raise ValueError("test_train_ratio must be >= 0")
    mean = float(d.mean())
    var = float(d.var(ddof=1))
    se = float(np.sqrt((1.0 / n + test_train_ratio) * var))
    df = n - 1
    if se == 0:
        t = float("inf") if mean != 0 else 0.0
        p = 0.0 if mean != 0 else 1.0
    else:
        t = mean / se
        p = float(2.0 * stats.t.sf(abs(t), df))
    tcrit = float(stats.t.ppf(0.5 + conf / 2.0, df))
    return {"n": n, "mean_diff": mean, "sd_diff": float(np.sqrt(var)), "se_corrected": se,
            "t": float(t), "df": df, "p": p,
            "ci_low": mean - tcrit * se, "ci_high": mean + tcrit * se,
            "seeds_first_higher": int((d > 0).sum())}


def table6_from_wide(wide, test_size=0.2, first="image", others=("patient", "inferred")):
    """Six (or fewer) contrasts: first-rule minus each other rule, for each metric."""
    ratio = test_size / (1.0 - test_size)
    rows = []
    for rule in others:
        for m in METRICS:
            a, b = f"{m}_{first}", f"{m}_{rule}"
            if a not in wide.columns or b not in wide.columns:
                continue
            r = corrected_resampled_ttest(wide[a] - wide[b], ratio)
            r.update({"comparison": RULE_LABELS.get(rule, rule), "metric": METRIC_LABELS[m],
                      "test_train_ratio": ratio})
            rows.append(r)
    cols = ["comparison", "metric", "mean_diff", "ci_low", "ci_high", "t", "df", "p",
            "seeds_first_higher", "n", "sd_diff", "se_corrected", "test_train_ratio"]
    return pd.DataFrame(rows)[cols]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wide", required=True, help="per_seed_wide_for_spss.csv")
    ap.add_argument("--out", default="table6_corrected_tests.csv")
    ap.add_argument("--test-size", type=float, default=0.2, help="test fraction used in the splits")
    args = ap.parse_args()
    wide = pd.read_csv(args.wide)
    tab = table6_from_wide(wide, args.test_size)
    tab.to_csv(args.out, index=False)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(tab.to_string(index=False))
    print("\nWrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
