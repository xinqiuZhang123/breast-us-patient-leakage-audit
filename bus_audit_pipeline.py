#!/usr/bin/env python3
"""
Patient-level provenance and leakage audit pipeline for public breast ultrasound datasets.

What it does, per dataset:
  1. Inventory   - counts, image sizes, labels, which identifier fields the release carries.
  2. Duplicates  - exact duplicates (MD5) and near-duplicates (difference hash), flagging
                   duplicates that carry different labels.
  3. Grouping    - reconstructs probable patient groups from image content only
                   (it never reads patient IDs or file-name numbers when grouping).
  4. Validation  - where a ground-truth patient ID exists (BUS-BRA; BUS-UCLM, whose file-name
                   prefix is a documented anonymised patient ID), scores the reconstruction against it on HELD-OUT patients,
                   repeated over many random patient-level splits to give 95% intervals.
  5. Leakage     - simulates random image-level train/test splits and measures how many test
                   images share a patient group with a training image.

Inputs are the zip files exactly as downloaded; nothing needs unzipping.

Usage (see README steps):
  python bus_audit_pipeline.py --data-dir ./data --out-dir ./results

Every dataset is grouped under two variants, reported side by side:
  "size"   : images may only be linked if they have the same image height (default rule)
  "nosize" : no size rule (sensitivity analysis; needed for BUSI, whose images vary in size)
"""
import argparse
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering

SEED = 42

# --------------------------------------------------------------------------------------
# Dataset loaders. Each returns (records DataFrame, zipfile, member-path function, info)
# records columns: image_id, member, label, gt_group (ground-truth patient id or None)
# --------------------------------------------------------------------------------------


def _find(z, pattern):
    rx = re.compile(pattern)
    hits = [n for n in z.namelist() if rx.search(n)]
    if not hits:
        raise FileNotFoundError(f"No member matching {pattern!r} in {z.filename}")
    return hits[0]


def load_busbra(path):
    z = zipfile.ZipFile(path)
    csv_member = _find(z, r"bus_data\.csv$")
    df = pd.read_csv(io.BytesIO(z.read(csv_member)))
    root = csv_member.rsplit("/", 1)[0]
    rec = pd.DataFrame({
        "image_id": df["ID"],
        "member": [f"{root}/Images/{i}.png" for i in df["ID"]],
        "label": df["Pathology"],
        "gt_group": df["Case"].astype(str),
    })
    info = {"identifier_field": "Case (explicit patient ID column)", "gt_is_confirmed": True}
    return rec, z, info


def load_busuclm(path):
    z = zipfile.ZipFile(path)
    csv_member = _find(z, r"INFO\.csv$")
    df = pd.read_csv(io.BytesIO(z.read(csv_member)), sep=";", encoding="utf-8-sig")
    root = csv_member.rsplit("/", 1)[0]
    rec = pd.DataFrame({
        "image_id": df["Image"],
        "member": [f"{root}/images/{i}" for i in df["Image"]],
        "label": df["Label"],
        # File-name pattern XXXX_YYY.png: XXXX is an anonymised patient identifier and YYY the image
        # number within that patient's study (Vallez et al., Sci Data 2025;12:242, Data Records).
        # The release has no separate ID column; the ID is encoded in the file name.
        "gt_group": df["Image"].str.split("_").str[0],
    })
    info = {"identifier_field": "file-name prefix XXXX (anonymised patient ID, documented in Vallez et al. 2025)",
            "gt_is_confirmed": True}
    return rec, z, info


def load_busi(path):
    z = zipfile.ZipFile(path)
    rows = []
    for n in z.namelist():
        if n.endswith(".png") and "_mask" not in n:
            parts = n.split("/")
            rows.append((parts[-1], n, parts[-2]))
    rec = pd.DataFrame(rows, columns=["image_id", "member", "label"])
    rec["gt_group"] = None
    info = {"identifier_field": "none (running number in file name only)", "gt_is_confirmed": False}
    return rec.sort_values("member").reset_index(drop=True), z, info


def load_curated(path):
    z = zipfile.ZipFile(path)
    rows = []
    for n in z.namelist():
        if n.endswith(".png") and "/masks/" not in n and "/images/" in n:
            f = n.split("/")[-1]
            rows.append((f, n, f.split("_id_")[0]))
    rec = pd.DataFrame(rows, columns=["image_id", "member", "label"])
    rec["gt_group"] = None
    info = {"identifier_field": "none (re-numbered ids)", "gt_is_confirmed": False}
    return rec.sort_values("member").reset_index(drop=True), z, info


DATASETS = {
    # name: (zip file name, loader)
    "BUS-BRA": ("BUSBRA.zip", load_busbra),
    "BUS-UCLM": ("BUS-UCLM Breast ultrasound lesion segmentation dataset.zip", load_busuclm),
    "BUSI": ("Dataset_BUSI.zip", load_busi),
    "Curated-BUSI": ("Curated BUSI.zip", load_curated),
}

# --------------------------------------------------------------------------------------
# Feature extraction (image content only)
# --------------------------------------------------------------------------------------


def dhash(im, size=8):
    a = np.asarray(im.resize((size + 1, size), Image.BILINEAR), dtype=np.float32)
    return (a[:, 1:] > a[:, :-1]).ravel()


def extract_features(rec, z):
    md5, dh, thumbs, hists, sizes, t64 = [], [], [], [], [], []
    for member in rec["member"]:
        raw = z.read(member)
        md5.append(hashlib.md5(raw).hexdigest())
        im = Image.open(io.BytesIO(raw)).convert("L")
        sizes.append(im.size)
        dh.append(dhash(im))
        t = np.asarray(im.resize((32, 32), Image.BILINEAR), dtype=np.float32).ravel()
        thumbs.append((t - t.mean()) / (t.std() + 1e-6))
        h = np.histogram(np.asarray(im), bins=32, range=(0, 255))[0].astype(np.float32)
        hists.append(h / h.sum())
        t2 = np.asarray(im.resize((64, 64), Image.BILINEAR), dtype=np.float32).ravel()
        t64.append((t2 - t2.mean()) / (t2.std() + 1e-6))
    return {
        "thumb64": np.array(t64),
        "md5": np.array(md5), "dhash": np.array(dh), "thumb": np.array(thumbs),
        "hist": np.array(hists), "size": np.array(sizes),
    }


# --------------------------------------------------------------------------------------
# Stage 1-2: inventory and duplicates
# --------------------------------------------------------------------------------------


def duplicate_report(rec, feat, near_thresh=8, corr_thresh=0.98):
    """Exact duplicates by MD5. Near-duplicates: candidates from a 64-bit difference hash
    (Hamming <= near_thresh), then VERIFIED by Pearson correlation of 64x64 thumbnails
    (>= corr_thresh). The verification step removes the false positives that a hash alone
    produces on ultrasound images, which look alike at very low resolution."""
    rows = []
    md5 = feat["md5"]
    groups = defaultdict(list)
    for i, h in enumerate(md5):
        groups[h].append(i)
    for idx in groups.values():
        if len(idx) > 1:
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    rows.append((idx[a], idx[b], "exact", 0))
    ham = cdist(feat["dhash"].astype(np.uint8), feat["dhash"].astype(np.uint8), "hamming") * 64
    ia, ib = np.where(np.triu(ham <= near_thresh, k=1))
    seen = {(a, b) for a, b, _, _ in rows}
    t64 = feat["thumb64"]
    for a, b in zip(ia, ib):
        if (a, b) in seen:
            continue
        corr = float(np.dot(t64[a], t64[b]) / t64.shape[1])
        if corr >= corr_thresh:
            rows.append((a, b, "near", int(ham[a, b])))
    out = pd.DataFrame(rows, columns=["i", "j", "type", "hamming"])
    if out.empty:
        return pd.DataFrame(columns=["image_a", "image_b", "type", "hamming", "label_a", "label_b",
                                     "label_conflict"])
    out["image_a"] = rec["image_id"].values[out["i"]]
    out["image_b"] = rec["image_id"].values[out["j"]]
    out["label_a"] = rec["label"].values[out["i"]]
    out["label_b"] = rec["label"].values[out["j"]]
    out["label_conflict"] = out["label_a"] != out["label_b"]
    return out.drop(columns=["i", "j"])


# --------------------------------------------------------------------------------------
# Stage 3: grouping from image content
# --------------------------------------------------------------------------------------


def distance_matrix(feat, same_size_only=True):
    dt = cdist(feat["thumb"], feat["thumb"], "euclidean")
    dh = cdist(feat["hist"], feat["hist"], "cityblock")
    n = len(dt)
    mask = ~np.eye(n, dtype=bool)
    D = dt / dt[mask].mean() + dh / dh[mask].mean()
    if same_size_only:
        # Views of one patient from one session share the image height.
        diff = feat["size"][:, 1][:, None] != feat["size"][:, 1][None, :]
        D = D + diff * 10.0
    np.fill_diagonal(D, 0.0)
    return D


def cluster(D, threshold):
    if len(D) < 2:
        return np.zeros(len(D), dtype=int)
    model = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                    distance_threshold=threshold)
    return model.fit_predict(D)


def pair_set(labels):
    """All unordered same-group index pairs."""
    groups = defaultdict(list)
    for i, g in enumerate(labels):
        groups[g].append(i)
    pairs = set()
    for idx in groups.values():
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                pairs.add((idx[a], idx[b]))
    return pairs


def pair_scores(pred, truth, subset=None):
    p, t = pair_set(pred), pair_set(truth)
    if subset is not None:
        p = {x for x in p if x[0] in subset and x[1] in subset}
        t = {x for x in t if x[0] in subset and x[1] in subset}
    tp = len(p & t)
    prec = tp / len(p) if p else float("nan")
    rec = tp / len(t) if t else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and (prec + rec) > 0 else float("nan")
    return {"pairs_pred": len(p), "pairs_true": len(t), "precision": prec, "recall": rec, "f1": f1}


def tune_threshold(D, truth, tune_idx):
    """Choose the clustering threshold that maximises pair F1 on the tuning images only."""
    best, best_f1 = None, -1
    sub = np.array(sorted(tune_idx))
    Dsub = D[np.ix_(sub, sub)]
    tsub = np.asarray(truth)[sub]
    for thr in np.linspace(0.3, 2.5, 23):
        pred = cluster(Dsub, thr)
        s = pair_scores(pred, tsub)
        if s["f1"] == s["f1"] and s["f1"] > best_f1:
            best, best_f1 = float(thr), s["f1"]
    return best, best_f1


# --------------------------------------------------------------------------------------
# Stage 5: leakage simulation
# --------------------------------------------------------------------------------------


def leakage_simulation(groups, n_repeats=100, test_frac=0.2, seed=SEED):
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    n = len(groups)
    out = []
    for _ in range(n_repeats):
        perm = rng.permutation(n)
        n_test = int(round(n * test_frac))
        test, train = perm[:n_test], perm[n_test:]
        train_groups = set(groups[train])
        out.append(np.mean([g in train_groups for g in groups[test]]))
    return float(np.mean(out)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# --------------------------------------------------------------------------------------
# Main per-dataset run
# --------------------------------------------------------------------------------------


VARIANTS = {"": True, "nosize_": False}  # column prefix -> same-size constraint on/off


def _interval(values):
    v = np.asarray([x for x in values if x == x], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def repeated_heldout(D, truth, n_splits, seed=SEED):
    """Repeat n_splits times: split PATIENTS in half, tune the clustering threshold on one half,
    score the other half. Returns one row per split. Intervals computed from these rows show how
    much the result varies with which patients land in the test half."""
    truth = np.asarray(truth)
    uniq = np.array(sorted(set(truth)))
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_splits):
        tune_groups = set(rng.permutation(uniq)[: len(uniq) // 2])
        mask = np.array([g in tune_groups for g in truth])
        tune_idx = set(np.where(mask)[0].tolist())
        sub = np.where(~mask)[0]
        thr, tune_f1 = tune_threshold(D, truth, tune_idx)
        pred = cluster(D[np.ix_(sub, sub)], thr)
        t = truth[sub]
        sc = pair_scores(pred, t)
        single = pd.Series(t).map(pd.Series(t).value_counts()).values == 1
        pred_sizes = pd.Series(pred).map(pd.Series(pred).value_counts()).values
        n_single = int(single.sum())
        rows.append({
            "threshold": thr, "tuning_f1": tune_f1,
            "precision": sc["precision"], "recall": sc["recall"], "f1": sc["f1"],
            "single_rate": (float((pred_sizes[single] > 1).sum()) / n_single) if n_single else float("nan"),
            "fragmentation": len(set(pred)) / len(set(t)),
        })
    return pd.DataFrame(rows)


def run_dataset(name, data_dir, out_dir, fallback, n_splits):
    zip_name, loader = DATASETS[name]
    path = os.path.join(data_dir, zip_name)
    if not os.path.exists(path):
        print(f"[skip] {name}: {path} not found")
        return None
    print(f"[{name}] loading ...")
    rec, z, info = loader(path)
    print(f"[{name}] {len(rec)} images; extracting features ...")
    feat = extract_features(rec, z)
    ddir = os.path.join(out_dir, name)
    os.makedirs(ddir, exist_ok=True)

    # Stage 1: inventory
    inv = rec.copy()
    inv["width"], inv["height"] = feat["size"][:, 0], feat["size"][:, 1]
    inv["md5"] = feat["md5"]
    inv.to_csv(os.path.join(ddir, "inventory.csv"), index=False)

    # Stage 2: duplicates
    dup = duplicate_report(rec, feat)
    dup.to_csv(os.path.join(ddir, "duplicates.csv"), index=False)

    summary = {
        "dataset": name,
        "n_images": int(len(rec)),
        "label_counts": rec["label"].value_counts().to_dict(),
        "identifier_available": info["identifier_field"],
        "n_distinct_image_sizes": int(len({tuple(s) for s in feat["size"]})),
        "exact_duplicate_pairs": int((dup["type"] == "exact").sum()) if len(dup) else 0,
        "near_duplicate_pairs": int((dup["type"] == "near").sum()) if len(dup) else 0,
        "duplicate_pairs_with_label_conflict": int(dup["label_conflict"].sum()) if len(dup) else 0,
    }

    has_gt = bool(rec["gt_group"].notna().all())
    truth = rec["gt_group"].values if has_gt else None
    if has_gt:
        summary["n_true_groups"] = int(pd.Series(truth).nunique())
        summary["ground_truth_confirmed"] = info["gt_is_confirmed"]
        m, lo, hi = leakage_simulation(truth)
        summary["leak_true_groups_mean"], summary["leak_true_groups_ci_low"], summary["leak_true_groups_ci_high"] = m, lo, hi

    # Stage 3-4: grouping from image content, under each variant
    for prefix, same_size in VARIANTS.items():
        print(f"[{name}] grouping variant: {'same-size rule' if same_size else 'no size rule'} ...")
        D = distance_matrix(feat, same_size_only=same_size)
        if has_gt:
            res = repeated_heldout(D, truth, n_splits)
            res.to_csv(os.path.join(ddir, f"{prefix}heldout_splits.csv"), index=False)
            thr = float(res["threshold"].median())
            summary[prefix + "threshold"] = thr
            summary[prefix + "n_splits"] = n_splits
            for col, label in [("precision", "heldout_precision"), ("recall", "heldout_recall"),
                               ("f1", "heldout_f1"), ("single_rate", "single_image_wrongly_linked_rate"),
                               ("fragmentation", "fragmentation_ratio")]:
                mean, lo, hi = _interval(res[col])
                summary[f"{prefix}{label}_mean"] = mean
                summary[f"{prefix}{label}_ci_low"] = lo
                summary[f"{prefix}{label}_ci_high"] = hi
        else:
            thr = fallback[prefix]
            summary[prefix + "threshold"] = thr
            summary[prefix + "threshold_source"] = "median threshold from the BUS-BRA splits"
        pred_all = cluster(D, thr)
        rec[prefix + "inferred_group"] = pred_all
        summary[prefix + "n_inferred_groups"] = int(len(set(pred_all)))
        summary[prefix + "mean_images_per_inferred_group"] = float(len(rec) / len(set(pred_all)))
        m, lo, hi = leakage_simulation(pred_all)
        summary[prefix + "leak_inferred_mean"] = m
        summary[prefix + "leak_inferred_ci_low"], summary[prefix + "leak_inferred_ci_high"] = lo, hi

    rec.drop(columns=["member"]).to_csv(os.path.join(ddir, "groups.csv"), index=False)
    with open(os.path.join(ddir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"[{name}] done -> {ddir}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="folder holding the downloaded zip files")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--n-splits", type=int, default=30,
                    help="number of random patient-level half splits for the held-out intervals")
    ap.add_argument("--fallback-threshold", type=float, default=None,
                    help="clustering threshold for datasets without ground truth; "
                         "default = median tuned threshold from BUS-BRA")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Ground-truth datasets first so BUS-BRA's tuned thresholds can be reused on the others.
    order = sorted(args.datasets, key=lambda d: d not in ("BUS-BRA", "BUS-UCLM"))
    fb = {"": args.fallback_threshold or 1.0, "nosize_": args.fallback_threshold or 1.0}
    results = []
    for name in order:
        s = run_dataset(name, args.data_dir, args.out_dir, fb, args.n_splits)
        if s is None:
            continue
        if name == "BUS-BRA" and args.fallback_threshold is None:
            fb = {"": s["threshold"], "nosize_": s["nosize_threshold"]}
        results.append(s)
    if results:
        out = os.path.join(args.out_dir, "all_datasets_summary.csv")
        pd.DataFrame(results).to_csv(out, index=False)
        print("\nSummary written to", out)


if __name__ == "__main__":
    sys.exit(main())