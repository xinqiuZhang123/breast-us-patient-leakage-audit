#!/usr/bin/env python3
"""Compare the numbers printed in the manuscript with the freshly generated outputs.

Usage: python check_manuscript.py --h1 results/BUS-BRA/summary.json --h3 results_h3/summary.json \
                                  [--h1-uclm results/BUS-UCLM/summary.json] [--out discrepancy_report.csv]

Every manuscript value below is the one currently printed in 'Haibin Article revised 16 Nov.docx'. A row marked
MISMATCH means the TEXT/TABLE must be changed to the new output (never the other way round).
Exit code 0 only if every row passes.
"""
import argparse
import json
import sys

import pandas as pd

# (label, source, path-in-json, manuscript value, absolute tolerance = half unit of the printed precision)
H1 = [
    ("BUS-BRA held-out F1 (size rule)", "h1", "heldout_f1_mean", 0.69, 0.005),
    ("BUS-BRA held-out precision", "h1", "heldout_precision_mean", 0.75, 0.005),
    ("BUS-BRA held-out recall", "h1", "heldout_recall_mean", 0.64, 0.005),
    ("BUS-BRA fragmentation ratio", "h1", "fragmentation_ratio_mean", 1.20, 0.005),
    ("BUS-BRA single images wrongly linked (%)", "h1", "single_image_wrongly_linked_rate_mean", 0.073, 0.0005),
    ("BUS-BRA n inferred groups", "h1", "n_inferred_groups", 1290, 0.5),
    ("BUS-BRA leakage true groups (%)", "h1", "leak_true_groups_mean", 0.696, 0.0005),
    ("BUS-BRA leakage detected by inferred (%)", "h1", "leak_inferred_mean", 0.487, 0.0005),
    ("BUS-BRA no-size F1", "h1", "nosize_heldout_f1_mean", 0.46, 0.005),
    ("BUS-BRA no-size precision", "h1", "nosize_heldout_precision_mean", 0.50, 0.005),
    ("BUS-BRA no-size leakage inferred (%)", "h1", "nosize_leak_inferred_mean", 0.438, 0.0005),
]
UCLM = [
    ("BUS-UCLM held-out F1", "uclm", "heldout_f1_mean", 0.18, 0.005),
    ("BUS-UCLM n inferred groups", "uclm", "n_inferred_groups", 66, 0.5),
    ("BUS-UCLM leakage true groups (%)", "uclm", "leak_true_groups_mean", 1.00, 0.0005),
    ("BUS-UCLM leakage detected by inferred (%)", "uclm", "leak_inferred_mean", 0.971, 0.0005),
]
H3_MEANS = {  # rule -> metric -> manuscript mean (Table 5)
    "image": {"auroc": 0.773, "balanced_accuracy": 0.697, "f1_malignant": 0.588, "leak_rate_true_patients": 0.690},
    "inferred": {"auroc": 0.757, "balanced_accuracy": 0.685, "f1_malignant": 0.570, "leak_rate_true_patients": 0.266},
    "patient": {"auroc": 0.751, "balanced_accuracy": 0.680, "f1_malignant": 0.561, "leak_rate_true_patients": 0.0},
}
TABLE6 = [  # comparison, metric, mean_diff, ci_low, ci_high, t, p, seeds
    ("vs patient-level", "AUROC", 0.0221, -0.0113, 0.0554, 1.31, 0.192, 75),
    ("vs patient-level", "Balanced accuracy", 0.0167, -0.0186, 0.0519, 0.94, 0.350, 71),
    ("vs patient-level", "F1 (malignant)", 0.0270, -0.0261, 0.0801, 1.01, 0.315, 73),
    ("vs inferred groups", "AUROC", 0.0163, -0.0188, 0.0514, 0.92, 0.360, 66),
    ("vs inferred groups", "Balanced accuracy", 0.0123, -0.0244, 0.0489, 0.66, 0.508, 60),
    ("vs inferred groups", "F1 (malignant)", 0.0177, -0.0381, 0.0736, 0.63, 0.530, 60),
]
MS_N_TEST = {"patient": (365, 390), "inferred": (365, 390)}   # Section 2.5 as printed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h1", required=True)
    ap.add_argument("--h3", required=True)
    ap.add_argument("--h1-uclm", default=None)
    ap.add_argument("--out", default="discrepancy_report.csv")
    a = ap.parse_args()
    h1 = json.load(open(a.h1))
    h3 = json.load(open(a.h3))
    uclm = json.load(open(a.h1_uclm)) if a.h1_uclm else None
    rows = []

    def add(label, ms, new, tol):
        ok = new is not None and abs(new - ms) <= tol
        rows.append({"item": label, "manuscript": ms, "new_output": new, "status": "OK" if ok else "MISMATCH"})

    for label, src, key, ms, tol in H1 + (UCLM if uclm else []):
        d = h1 if src == "h1" else uclm
        add(label, ms, d.get(key), tol)
    for rule, mm in H3_MEANS.items():
        for m, ms in mm.items():
            v = h3["per_rule"].get(rule, {}).get(m, {}).get("mean")
            add(f"Table 5 {rule} {m}", ms, v, 0.0005)
    tab6 = {(r["comparison"], r["metric"]): r for r in h3.get("corrected_tests", [])}
    for comp, met, md, lo, hi, t, p, k in TABLE6:
        r = tab6.get((comp, met), {})
        add(f"Table 6 {comp} {met} mean diff", md, r.get("mean_diff"), 0.00005)
        add(f"Table 6 {comp} {met} CI low", lo, r.get("ci_low"), 0.00005)
        add(f"Table 6 {comp} {met} CI high", hi, r.get("ci_high"), 0.00005)
        add(f"Table 6 {comp} {met} t", t, r.get("t"), 0.005)
        add(f"Table 6 {comp} {met} p", p, r.get("p"), 0.0005)
        add(f"Table 6 {comp} {met} seeds image higher", k, r.get("seeds_first_higher"), 0.5)
    sa = h3.get("test_size_audit", {})
    for rule, (lo, hi) in MS_N_TEST.items():
        s = sa.get(rule, {})
        add(f"n_test min ({rule})", lo, s.get("n_test_min"), 0.5)
        add(f"n_test max ({rule})", hi, s.get("n_test_max"), 0.5)
    add("seeds dropped from paired analysis", 0, h3.get("seeds_dropped_from_paired_analysis"), 0.5)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    bad = out[out["status"] != "OK"]
    print(out.to_string(index=False))
    print(f"\n{len(out) - len(bad)}/{len(out)} manuscript values reproduced; {len(bad)} MISMATCH")
    return 0 if bad.empty else 1


if __name__ == "__main__":
    sys.exit(main())
