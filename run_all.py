#!/usr/bin/env python3
"""One-command replication: hash inputs -> unit tests -> H1 pipeline -> H3 experiment -> manuscript check.

  python run_all.py --data-dir ./data --out-dir ./outputs            # full run (ResNet-50)
  python run_all.py --data-dir ./data --out-dir ./outputs_smoke --backbone handcrafted --n-splits 3   # quick test only

Needs in --data-dir (as downloaded, not unzipped): BUSBRA.zip, "BUS-UCLM Breast ultrasound lesion
segmentation dataset.zip", Dataset_BUSI.zip, "Curated BUSI.zip".  Image archives are third-party data:
download them from the original sources (see README); do not redistribute them.
Writes Run_Log.txt and hashes.sha256 into --out-dir.
"""
import argparse
import hashlib
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ZIPS = ["BUSBRA.zip", "BUS-UCLM Breast ultrasound lesion segmentation dataset.zip",
        "Dataset_BUSI.zip", "Curated BUSI.zip"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, log):
    log.write(f"\n$ {' '.join(cmd)}\n")
    log.flush()
    t0 = time.time()
    p = subprocess.run(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log.write(p.stdout)
    log.write(f"[exit {p.returncode}, {time.time() - t0:.0f}s]\n")
    log.flush()
    print(f"{' '.join(cmd[:3])} ... exit {p.returncode}")
    return p.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--backbone", choices=["resnet50", "handcrafted"], default="resnet50")
    ap.add_argument("--n-splits", type=int, default=30)
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--size-matching", choices=["greedy", "none"], default="greedy")
    a = ap.parse_args()
    data, out = os.path.abspath(a.data_dir), os.path.abspath(a.out_dir)
    os.makedirs(out, exist_ok=True)
    py = sys.executable
    with open(os.path.join(out, "Run_Log.txt"), "w") as log:
        log.write(f"run started {time.strftime('%Y-%m-%d %H:%M:%S')} | python {platform.python_version()} | "
                  f"{platform.platform()}\nbackbone={a.backbone} n_splits={a.n_splits} n_seeds={a.n_seeds} "
                  f"size_matching={a.size_matching}\n")
        with open(os.path.join(out, "hashes.sha256"), "w") as hf:
            for z in ZIPS:
                p = os.path.join(data, z)
                if os.path.exists(p):
                    hf.write(f"{sha256(p)}  {z}\n")
                else:
                    log.write(f"[missing input] {z}\n")
        codes = [run([py, "-m", "pytest", "-q", "-p", "no:cacheprovider"], log)]
        codes.append(run([py, "bus_audit_pipeline.py", "--data-dir", data, "--out-dir", os.path.join(out, "h1"),
                          "--n-splits", str(a.n_splits)], log))
        groups = os.path.join(out, "h1", "BUS-BRA", "groups.csv")
        codes.append(run([py, "h3_resplit_baseline.py", "--data-dir", data, "--out-dir", os.path.join(out, "h3"),
                          "--backbone", a.backbone, "--n-seeds", str(a.n_seeds), "--inferred-groups", groups,
                          "--size-matching", a.size_matching], log))
        cmd = [py, "check_manuscript.py", "--h1", os.path.join(out, "h1", "BUS-BRA", "summary.json"),
               "--h3", os.path.join(out, "h3", "summary.json"), "--out", os.path.join(out, "discrepancy_report.csv")]
        u = os.path.join(out, "h1", "BUS-UCLM", "summary.json")
        if os.path.exists(u):
            cmd += ["--h1-uclm", u]
        check = run(cmd, log)   # non-zero = the manuscript text/tables must be updated to the new outputs
        log.write(f"\nfinished {time.strftime('%Y-%m-%d %H:%M:%S')}; step exit codes {codes}; "
                  f"manuscript check exit {check}\n")
    print("See", os.path.join(out, "Run_Log.txt"), "and discrepancy_report.csv")
    return 0 if all(c == 0 for c in codes) else 1


if __name__ == "__main__":
    sys.exit(main())
