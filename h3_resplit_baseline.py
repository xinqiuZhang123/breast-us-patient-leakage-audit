#!/usr/bin/env python3
"""
H3 baseline experiment: does patient-level splitting lower reported performance?

Data  : BUS-BRA (1,875 images, 1,064 patients, benign vs malignant). Needs BUSBRA.zip only.
Model : a standard, fixed baseline - frozen ImageNet ResNet-50 features + logistic regression.
        (Features are extracted once and cached, so the experiment itself runs in seconds.)
Design: for each of N random seeds, draw a test set of the SAME SIZE under each split rule:
          image   : random image-level split (images of one patient can land on both sides)
          patient : patient-level split (all images of a patient on one side)  <- the correct design
          inferred: optional split by groups reconstructed by the audit pipeline (--inferred-groups)
        The same model, hyper-parameters and seed are used in every condition, so the only thing
        that changes is the split rule.
Output: per-seed metrics, paired differences with 95% intervals and tests, a leakage measure for
        each rule, a wide CSV ready for SPSS, and a figure.

Usage:
  python h3_resplit_baseline.py --data-dir data --out-dir results_h3
  python h3_resplit_baseline.py --data-dir data --out-dir results_h3 \\
         --inferred-groups results/BUS-BRA/groups.csv
"""
import argparse
import io
import json
import os
import platform
import sys
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from stats_corrected import table6_from_wide

SEED = 42


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_busbra(data_dir):
    path = os.path.join(data_dir, "BUSBRA.zip")
    if not os.path.exists(path):
        sys.exit(f"Cannot find {path}")
    z = zipfile.ZipFile(path)
    csv_member = [n for n in z.namelist() if n.endswith("bus_data.csv")][0]
    root = csv_member.rsplit("/", 1)[0]
    df = pd.read_csv(io.BytesIO(z.read(csv_member)))
    df["member"] = [f"{root}/Images/{i}.png" for i in df["ID"]]
    df["y"] = (df["Pathology"] == "malignant").astype(int)
    return df, z


# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------
def features_resnet50(df, z, batch_size=64):
    """Frozen ImageNet ResNet-50, 2048-d global-pooled features. Needs torch + torchvision and
    an internet connection the first time (to download the pretrained weights)."""
    import torch
    import torchvision
    from torchvision import transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Extracting ResNet-50 features on {device} ...")
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    out = []
    members = list(df["member"])
    with torch.no_grad():
        for i in range(0, len(members), batch_size):
            batch = []
            for m in members[i:i + batch_size]:
                im = Image.open(io.BytesIO(z.read(m))).convert("L").convert("RGB")
                batch.append(tf(im))
            x = torch.stack(batch).to(device)
            out.append(model(x).cpu().numpy())
            print(f"  {min(i + batch_size, len(members))}/{len(members)}", end="\r")
    print()
    return np.vstack(out)


def features_handcrafted(df, z):
    """Cheap fallback (no deep-learning libraries): 32x32 thumbnail + 32-bin histogram.
    Use it only to test that the pipeline runs; report ResNet-50 results in the paper."""
    print("Extracting handcrafted features ...")
    out = []
    for m in df["member"]:
        im = Image.open(io.BytesIO(z.read(m))).convert("L")
        t = np.asarray(im.resize((32, 32), Image.BILINEAR), dtype=np.float32).ravel()
        t = (t - t.mean()) / (t.std() + 1e-6)
        h = np.histogram(np.asarray(im), bins=32, range=(0, 255))[0].astype(np.float32)
        out.append(np.concatenate([t, h / h.sum()]))
    return np.array(out)


def get_features(df, z, backbone, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"features_{backbone}.npy")
    if os.path.exists(cache):
        X = np.load(cache)
        if len(X) == len(df):
            print(f"Loaded cached features: {cache}")
            return X
    X = features_resnet50(df, z) if backbone == "resnet50" else features_handcrafted(df, z)
    np.save(cache, X)
    return X


# --------------------------------------------------------------------------------------
# Splits and metrics
# --------------------------------------------------------------------------------------
def matched_group_split(groups, n_test_target, seed):
    """Group-level split whose test set has EXACTLY n_test_target images when possible.

    Groups are visited in a seeded random order and added to the test set whenever they still fit
    under the target. Exact matching needs enough small groups (singletons) to fill the gap; the
    achieved size is returned so any residual mismatch is recorded, never hidden.
    """
    groups = np.asarray(groups)
    uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(seed)
    chosen, total = [], 0
    for gi in rng.permutation(len(uniq)):
        if total + counts[gi] <= n_test_target:
            chosen.append(gi)
            total += counts[gi]
            if total == n_test_target:
                break
    is_test = np.isin(inv, chosen)
    return np.where(~is_test)[0], np.where(is_test)[0]


def make_splits(groups, seed, n_images, test_size, size_matching="greedy"):
    """Return {rule: (train_idx, test_idx)} for one seed.

    The patient-level split (sklearn GroupShuffleSplit) fixes the target test size n_test.
    image    : random image split with exactly n_test test images.
    inferred : size_matching='greedy' -> group split forced to n_test images (matched_group_split);
               size_matching='none'   -> plain GroupShuffleSplit (size NOT matched; original behaviour).
    """
    out = {}
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    out["patient"] = next(gss.split(np.zeros(n_images), groups=groups["patient"]))
    n_test = len(out["patient"][1])
    for rule, g in groups.items():
        if rule == "patient":
            continue
        if size_matching == "greedy":
            out[rule] = matched_group_split(g, n_test, seed)
        else:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            out[rule] = next(gss.split(np.zeros(n_images), groups=g))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_images)
    out["image"] = (np.sort(perm[n_test:]), np.sort(perm[:n_test]))
    return out


def leak_rate(case, tr, te):
    """Share of test images whose TRUE patient also has an image in the training set."""
    train_cases = set(case[tr])
    return float(np.mean([c in train_cases for c in case[te]]))


def fit_eval(X, y, tr, te, C):
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    clf.fit(sc.transform(X[tr]), y[tr])
    p = clf.predict_proba(sc.transform(X[te]))[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "auroc": roc_auc_score(y[te], p),
        "balanced_accuracy": balanced_accuracy_score(y[te], pred),
        "f1_malignant": f1_score(y[te], pred, zero_division=0),
    }


def interval(v, n_boot=10000, seed=SEED):
    """Mean and 95% percentile bootstrap interval over seeds."""
    v = np.asarray(v, dtype=float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="folder containing BUSBRA.zip")
    ap.add_argument("--out-dir", default="results_h3")
    ap.add_argument("--backbone", choices=["resnet50", "handcrafted"], default="resnet50")
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--C", type=float, default=0.1, help="logistic-regression inverse regularisation")
    ap.add_argument("--inferred-groups", default=None,
                    help="groups.csv from bus_audit_pipeline.py (adds an 'inferred' split rule)")
    ap.add_argument("--size-matching", choices=["greedy", "none"], default="greedy",
                    help="greedy: inferred-group test set forced to the patient-level test size; "
                         "none: original behaviour (sizes may differ, recorded per seed)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df, z = load_busbra(args.data_dir)
    X = get_features(df, z, args.backbone, args.out_dir)
    y = df["y"].values
    case = df["Case"].astype(str).values
    n = len(df)
    print(f"{n} images, {len(set(case))} patients, {y.mean():.1%} malignant")

    groups = {"patient": case}
    if args.inferred_groups:
        g = pd.read_csv(args.inferred_groups)
        g = g.set_index("image_id").loc[df["ID"]]
        groups["inferred"] = g["inferred_group"].astype(str).values
        print("Using inferred groups:", len(set(groups["inferred"])), "groups")

    rows, skipped, manifest = [], [], []
    for seed in range(args.n_seeds):
        splits = make_splits(groups, seed, n, args.test_size, args.size_matching)
        for rule, (tr, te) in splits.items():
            # split manifest: test image IDs per seed and rule (train = all other images)
            manifest.extend((seed, rule, i) for i in df["ID"].values[te])
            if len(set(y[te])) < 2:  # AUROC undefined
                skipped.append({"seed": seed, "rule": rule, "n_test": len(te),
                                "reason": "single-class test set"})
                continue
            m = fit_eval(X, y, tr, te, args.C)
            m.update({"seed": seed, "rule": rule, "n_train": len(tr), "n_test": len(te),
                      "test_malignant_share": float(y[te].mean()),
                      "leak_rate_true_patients": leak_rate(case, tr, te)})
            rows.append(m)
    pd.DataFrame(manifest, columns=["seed", "rule", "test_image_id"]).to_csv(
        os.path.join(args.out_dir, "split_manifest_test_ids.csv"), index=False)
    pd.DataFrame(skipped, columns=["seed", "rule", "n_test", "reason"]).to_csv(
        os.path.join(args.out_dir, "skipped_seeds.csv"), index=False)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out_dir, "per_seed_results.csv"), index=False)

    # Test-size audit: min/max n_test per rule, and seed-level mismatch against the patient rule
    size_audit = {}
    for rule, grp in res.groupby("rule"):
        size_audit[rule] = {"n_test_min": int(grp["n_test"].min()), "n_test_max": int(grp["n_test"].max()),
                            "n_train_min": int(grp["n_train"].min()), "n_train_max": int(grp["n_train"].max())}
    pt = res[res["rule"] == "patient"].set_index("seed")["n_test"]
    for rule in [r for r in res["rule"].unique() if r != "patient"]:
        other = res[res["rule"] == rule].set_index("seed")["n_test"]
        diff = (other - pt).dropna()
        size_audit[rule]["max_abs_n_test_difference_vs_patient"] = int(diff.abs().max())
        size_audit[rule]["seeds_with_size_mismatch"] = int((diff != 0).sum())

    # Wide format for SPSS: one row per seed, one column per rule and metric
    metrics = ["auroc", "balanced_accuracy", "f1_malignant", "leak_rate_true_patients"]
    wide = res.pivot(index="seed", columns="rule", values=metrics)
    wide.columns = [f"{m}_{r}" for m, r in wide.columns]
    wide = wide.dropna().reset_index()
    wide.to_csv(os.path.join(args.out_dir, "per_seed_wide_for_spss.csv"), index=False)

    # Summary: mean and 95% bootstrap interval per rule, and paired differences
    summary = {"backbone": args.backbone, "n_seeds_used": int(len(wide)), "n_images": n,
               "n_patients": int(len(set(case))), "per_rule": {}, "paired_vs_image": {}}
    for rule in sorted(res["rule"].unique()):
        d = {}
        for m in metrics:
            mean, lo, hi = interval(wide[f"{m}_{rule}"])
            d[m] = {"mean": mean, "ci_low": lo, "ci_high": hi}
        summary["per_rule"][rule] = d
    for rule in [r for r in res["rule"].unique() if r != "image"]:
        d = {}
        for m in ["auroc", "balanced_accuracy", "f1_malignant"]:
            diff = wide[f"{m}_image"] - wide[f"{m}_{rule}"]  # positive = image-level split looks better
            mean, lo, hi = interval(diff)
            try:
                p_w = float(stats.wilcoxon(diff).pvalue)
            except ValueError:
                p_w = float("nan")
            d[m] = {"mean_difference_image_minus_" + rule: mean, "ci_low": lo, "ci_high": hi,
                    "wilcoxon_p": p_w,
                    "cohens_dz": float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else float("nan"),
                    "seeds_where_image_split_higher": int((diff > 0).sum())}
        summary["paired_vs_image"][rule] = d
    summary["test_size_audit"] = size_audit
    summary["size_matching"] = args.size_matching
    summary["n_seeds_requested"] = args.n_seeds
    summary["skipped_rule_seed_pairs"] = len(skipped)
    summary["seeds_dropped_from_paired_analysis"] = int(args.n_seeds - len(wide))
    meta = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__, "scipy": __import__("scipy").__version__,
            "logreg_C": args.C, "test_size": args.test_size,
            "weights": "torchvision ResNet50_Weights.IMAGENET1K_V2" if args.backbone == "resnet50" else "none"}
    try:
        import torch
        import torchvision
        meta.update({"torch": torch.__version__, "torchvision": torchvision.__version__,
                     "device": "cuda" if torch.cuda.is_available() else "cpu"})
    except Exception:
        pass
    summary["environment"] = meta
    # Nadeau-Bengio corrected resampled t-tests (Table 6) from the saved per-seed data
    tab6 = table6_from_wide(wide, args.test_size)
    tab6.to_csv(os.path.join(args.out_dir, "table6_corrected_tests.csv"), index=False)
    summary["corrected_tests"] = tab6.to_dict(orient="records")
    with open(os.path.join(args.out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # Readable table
    lines = []
    for rule, d in summary["per_rule"].items():
        for m, v in d.items():
            lines.append({"rule": rule, "metric": m, "mean": v["mean"], "ci_low": v["ci_low"],
                          "ci_high": v["ci_high"]})
    pd.DataFrame(lines).to_csv(os.path.join(args.out_dir, "summary_table.csv"), index=False)

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rules = [r for r in ["image", "inferred", "patient"] if r in res["rule"].unique()]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, m, title in zip(axes, ["auroc", "balanced_accuracy"], ["AUROC", "Balanced accuracy"]):
            data = [wide[f"{m}_{r}"].values for r in rules]
            ax.boxplot(data, showmeans=True)
            ax.set_xticks(range(1, len(rules) + 1))
            ax.set_xticklabels(rules)
            ax.set_title(title)
            ax.set_xlabel("Split rule")
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("BUS-BRA baseline performance by split rule (" + str(len(wide)) + " seeds)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "fig_split_rule_performance.png"), dpi=300)
    except Exception as e:  # figure is optional
        print("Figure skipped:", e)

    print("\n=== Mean (95% bootstrap interval over seeds) ===")
    for rule, d in summary["per_rule"].items():
        a, b = d["auroc"], d["balanced_accuracy"]
        print(f"{rule:9s} AUROC {a['mean']:.3f} ({a['ci_low']:.3f}-{a['ci_high']:.3f})   "
              f"Bal.acc {b['mean']:.3f} ({b['ci_low']:.3f}-{b['ci_high']:.3f})   "
              f"leak {d['leak_rate_true_patients']['mean']:.1%}")
    for rule, d in summary["paired_vs_image"].items():
        a = d["auroc"]
        print(f"\nAUROC image-level minus {rule}: {a['mean_difference_image_minus_' + rule]:+.3f} "
              f"({a['ci_low']:+.3f} to {a['ci_high']:+.3f}), Wilcoxon p={a['wilcoxon_p']:.2g}, "
              f"image-level higher in {a['seeds_where_image_split_higher']}/{len(wide)} seeds")
    print("\n=== Test-set sizes (n_test) per rule ===")
    for rule, d in size_audit.items():
        print(f"{rule:9s} min {d['n_test_min']} max {d['n_test_max']}", {k: v for k, v in d.items() if k.startswith(('max_abs', 'seeds_with'))})
    print("\n=== Corrected resampled t-tests (Nadeau-Bengio), image-level minus other rule ===")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(tab6[["comparison", "metric", "mean_diff", "ci_low", "ci_high", "t", "p", "seeds_first_higher"]].to_string(index=False))
    if skipped:
        print(f"\nNOTE: {len(skipped)} rule/seed pairs skipped (single-class test set); see skipped_seeds.csv")
    print("\nFiles written to", args.out_dir)


if __name__ == "__main__":
    sys.exit(main())