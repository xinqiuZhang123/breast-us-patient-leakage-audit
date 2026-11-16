# Patient-level evaluation in public breast ultrasound research - analysis code

## What is here
| File | Purpose |
|---|---|
| `bus_audit_pipeline.py` | H1: inventory, duplicates, patient-group recovery from image content, held-out validation, leakage simulation |
| `h3_resplit_baseline.py` | H3: frozen ResNet-50 + logistic regression on BUS-BRA under image / patient / inferred-group splits; writes per-seed results, split manifest, skipped-seed log, test-size audit, corrected tests |
| `stats_corrected.py` | Nadeau-Bengio variance-corrected resampled t-test (Table 6) |
| `check_manuscript.py` | Compares every number printed in the manuscript with the new outputs; any MISMATCH means the manuscript must be changed |
| `run_all.py` | Hashes inputs, runs tests, H1, H3 and the manuscript check; writes `Run_Log.txt` |
| `test_stats_corrected.py`, `test_splits.py` | Unit tests (`python -m pytest -q`) |

## Data (not included - third-party licences)
Download the archives from their original sources and put them, unzipped-as-downloaded, in one folder:
`BUSBRA.zip` (Gomez-Flores et al., doi:10.1002/mp.16812), `BUS-UCLM Breast ultrasound lesion segmentation dataset.zip`
(Vallez et al., doi:10.1038/s41597-025-04562-3; Mendeley doi:10.17632/7fvgj4jsp7.3), `Dataset_BUSI.zip`
(Al-Dhabyani et al., doi:10.1016/j.dib.2019.104863), `Curated BUSI.zip` (Zenodo doi:10.5281/zenodo.19047974).
`run_all.py` writes SHA-256 hashes of the archives you used to `hashes.sha256`.

## Run
    pip install -r requirements.txt     # pin exact versions with pip freeze before archiving
    python run_all.py --data-dir ./data --out-dir ./outputs

ResNet-50 uses torchvision `ResNet50_Weights.IMAGENET1K_V2` (downloaded on first run). Use
`--backbone handcrafted --n-splits 3 --n-seeds 20` only for a quick pipeline test, never for reported results.

## Changes in this version
* BUS-UCLM prefix is treated as a documented anonymised patient ID (Vallez et al. 2025, Data Records).
* `--size-matching greedy` (default) forces the inferred-group test set to the patient-level test size per seed;
  `--size-matching none` reproduces the previous behaviour. Per-seed `n_train`/`n_test` and a size audit are always saved.
* Corrected resampled t-tests are computed in code (`table6_corrected_tests.csv`) from the saved per-seed data.
* Skipped (single-class) seeds, the test-image manifest per seed and rule, and the software environment are saved.

## Status
Unit tests and a smoke test pass. The full ResNet-50 run has NOT yet been executed with these scripts; until it has,
every manuscript value must be treated as unverified. Interpret the BUS-BRA inferred-group rule with care: its
clustering threshold was tuned on BUS-BRA identifiers, so it is optimistic on that release.
