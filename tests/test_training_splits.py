"""Fold partition and run seeds (needs torch, which pvc_dlif_pipeline.training imports)."""

import pytest

torch = pytest.importorskip("torch")

from pvc_dlif_pipeline.training import make_folds, run_seed_of  # noqa: E402


def test_folds_partition_the_scans():
    scan_ids = [f"S{i}" for i in range(70)]
    folds = make_folds(scan_ids, 10, seed=42)
    test_sets = [set(test_ids) for _, _, test_ids in folds]
    assert [fold for fold, _, _ in folds] == list(range(1, 11))
    assert set().union(*test_sets) == set(scan_ids)
    assert sum(len(test_set) for test_set in test_sets) == 70
    for _, train_ids, test_ids in folds:
        assert not set(train_ids) & set(test_ids)


def test_folds_do_not_depend_on_input_order():
    scan_ids = [f"S{i}" for i in range(70)]
    assert make_folds(scan_ids, 10, 42) == make_folds(list(reversed(scan_ids)), 10, 42)


def test_run_seed():
    assert run_seed_of(42, 3, 7) == 3049
