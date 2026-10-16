"""Paired statistics on cases with known answers."""

import numpy as np
import pandas as pd

from pvc_dlif_pipeline.statistics import (bootstrap_interval_of_median, compare_families, holm,
                                          network_permutation_p, rank_biserial, wilcoxon_p)


def test_holm_by_hand():
    # Sorted: 0.01*3 = 0.03, 0.02*2 = 0.04, 0.04*1 = 0.04 (kept monotone).
    adjusted = holm([0.04, 0.01, 0.02])
    assert np.allclose(adjusted, [0.04, 0.03, 0.04])


def test_holm_caps_at_one_and_keeps_nan():
    adjusted = holm([0.6, np.nan, 0.9])
    assert np.isnan(adjusted[1])
    assert np.allclose(adjusted[[0, 2]], [1.0, 1.0])


def test_rank_biserial_extremes():
    assert rank_biserial(np.array([1.0, 2.0, 3.0])) == 1.0
    assert rank_biserial(np.array([-1.0, -2.0, 0.0])) == -1.0
    assert rank_biserial(np.zeros(4)) == 0.0


def test_wilcoxon_identical_values_give_one():
    values = np.array([1.0, 2.0, 3.0])
    assert wilcoxon_p(values, values) == 1.0


def test_bootstrap_is_seeded_and_brackets_the_median():
    differences = np.random.default_rng(1).normal(0.5, 1.0, 70)
    first = bootstrap_interval_of_median(differences, 2000, seed=42)
    second = bootstrap_interval_of_median(differences, 2000, seed=42)
    assert first == second
    assert first[0] < np.median(differences) < first[1]


def test_permutation_without_effect_and_with_a_large_effect():
    generator = np.random.default_rng(0)
    # 20 scans, 10 networks, each scan predicted by 5 of them (as with folds).
    table = np.full((20, 10), np.nan)
    for scan in range(20):
        networks = [scan % 2 * 5 + k for k in range(5)]
        table[scan, networks] = generator.normal(0.0, 1.0, 5)
    assert network_permutation_p(table, 2000, seed=42) > 0.05
    assert network_permutation_p(table + 5.0, 2000, seed=42) < 0.01


def test_compare_families_finds_a_shift_and_leaves_no_shift_alone():
    """Three conditions, 30 scans: 'shifted' is the reference plus 0.5, 'same' equals it."""
    generator = np.random.default_rng(3)
    reference = generator.normal(1.0, 0.2, 30)
    rows = []
    for scan in range(30):
        for condition, value in [("reference", reference[scan]), ("same", reference[scan]),
                                 ("shifted", reference[scan] + 0.5)]:
            rows.append({"condition": condition, "scan_id": f"S{scan}", "rmse": value})
    families = [{"family": "test", "reference": "reference", "conditions": ["same", "shifted"]}]
    table = compare_families(pd.DataFrame(rows), ["rmse"], families, 1000, 1000, 42).set_index("condition")
    assert np.isclose(table.loc["shifted", "median_difference"], 0.5)
    assert table.loc["shifted", "p_wilcoxon_holm"] < 1e-4
    assert table.loc["same", "median_difference"] == 0.0
    assert table.loc["same", "p_wilcoxon"] == 1.0
    assert "p_permutation" not in table.columns  # no fold/run columns: ensemble-like input
