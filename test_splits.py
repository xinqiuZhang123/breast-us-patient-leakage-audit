"""Tests for the split rules in h3_resplit_baseline.py.   Run:  python -m pytest -q"""
import numpy as np
import pytest

from h3_resplit_baseline import make_splits, matched_group_split


def _toy(n_patients=300, seed=0):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 5, n_patients)
    patient = np.repeat(np.arange(n_patients), sizes)
    # inferred groups: merge some patients and fragment others, keep many singletons
    inferred = patient.copy()
    inferred[rng.random(len(inferred)) < 0.3] += 10_000
    return patient.astype(str), inferred.astype(str)


def test_patient_split_has_no_patient_overlap():
    patient, inferred = _toy()
    n = len(patient)
    for seed in range(20):
        s = make_splits({"patient": patient, "inferred": inferred}, seed, n, 0.2)
        tr, te = s["patient"]
        assert not set(patient[tr]) & set(patient[te])
        assert len(tr) + len(te) == n and not set(tr) & set(te)


def test_inferred_split_has_no_group_overlap_and_matches_size():
    patient, inferred = _toy()
    n = len(patient)
    for seed in range(50):
        s = make_splits({"patient": patient, "inferred": inferred}, seed, n, 0.2, "greedy")
        tr, te = s["inferred"]
        assert not set(inferred[tr]) & set(inferred[te])
        assert len(te) == len(s["patient"][1]) == len(s["image"][1])
        assert len(tr) + len(te) == n


def test_image_split_matches_patient_size_and_is_disjoint():
    patient, inferred = _toy()
    n = len(patient)
    s = make_splits({"patient": patient, "inferred": inferred}, 3, n, 0.2)
    tr, te = s["image"]
    assert len(te) == len(s["patient"][1]) and not set(tr) & set(te) and len(tr) + len(te) == n


def test_size_matching_none_can_differ():
    patient, inferred = _toy()
    n = len(patient)
    diffs = [len(make_splits({"patient": patient, "inferred": inferred}, s, n, 0.2, "none")["inferred"][1])
             - len(make_splits({"patient": patient, "inferred": inferred}, s, n, 0.2, "none")["patient"][1])
             for s in range(30)]
    assert any(d != 0 for d in diffs)  # documents why matching is needed


def test_matched_split_is_deterministic_and_seed_dependent():
    _, inferred = _toy()
    a = matched_group_split(inferred, 120, 7)[1]
    b = matched_group_split(inferred, 120, 7)[1]
    c = matched_group_split(inferred, 120, 8)[1]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_unreachable_target_is_reported_not_hidden():
    g = np.array(["a"] * 5 + ["b"] * 5)       # only sizes 5 and 5; target 7 is unreachable
    tr, te = matched_group_split(g, 7, 0)
    assert len(te) in (0, 5, 5) and len(te) != 7
