"""Paired statistics: Wilcoxon signed-rank test, bootstrap interval, effect size,
Holm correction, and a permutation test with the trained network as the unit.
"""

import numpy as np
from scipy.stats import rankdata, wilcoxon


def bootstrap_interval_of_median(differences, n_bootstrap, seed):
    """95 % percentile bootstrap interval of the median paired difference (pairs resampled)."""
    generator = np.random.default_rng(seed)
    resampled_positions = generator.integers(0, differences.size, size=(n_bootstrap, differences.size))
    medians = np.median(differences[resampled_positions], axis=1)
    low, high = np.percentile(medians, [2.5, 97.5])
    return float(low), float(high)


def rank_biserial(differences):
    """Matched-pairs rank-biserial correlation (W+ - W-) / (W+ + W-), zero differences dropped."""
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    positive, negative = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
    return float((positive - negative) / (positive + negative))


def wilcoxon_p(condition_values, reference_values):
    """Two-sided Wilcoxon signed-rank p-value (1.0 if every difference is zero)."""
    if np.all(condition_values == reference_values):
        return 1.0
    return float(wilcoxon(condition_values, reference_values, alternative="two-sided",
                          zero_method="wilcox").pvalue)


def holm(p_values):
    """Holm-adjusted p-values for one family (NaN values are left out and stay NaN)."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan)
    finite_positions = np.flatnonzero(np.isfinite(p_values))
    order = finite_positions[np.argsort(p_values[finite_positions])]
    n = order.size
    running_maximum = 0.0
    for rank, position in enumerate(order):
        running_maximum = max(running_maximum, (n - rank) * p_values[position])
        adjusted[position] = min(running_maximum, 1.0)
    return adjusted


def network_permutation_p(differences_by_scan_and_network, n_permutations, seed):
    """Sign-flip permutation test with the trained network (fold, run) as the unit.

    differences_by_scan_and_network: array (scans, networks) of paired
    differences, NaN where a network did not predict a scan (each scan is
    predicted only by the networks of its own fold). Under the null hypothesis
    the two members of a network pair are exchangeable, so one permutation
    flips the sign of every difference of a network at once. The statistic is
    the reported one: per scan the mean difference over its networks, then the
    median over scans. Signs are drawn in batches of 2000 to keep memory small.
    """
    present = np.isfinite(differences_by_scan_and_network)
    differences = np.where(present, differences_by_scan_and_network, 0.0)
    networks_per_scan = present.sum(axis=1)
    observed = np.median(differences.sum(axis=1) / networks_per_scan)

    generator = np.random.default_rng(seed)
    n_as_extreme = 0
    remaining = n_permutations
    while remaining > 0:
        batch = min(remaining, 2000)
        signs = generator.integers(0, 2, size=(batch, differences.shape[1])).astype(np.float64) * 2.0 - 1.0
        permuted = np.einsum("bm,sm->bs", signs, differences) / networks_per_scan
        n_as_extreme += int(np.sum(np.abs(np.median(permuted, axis=1)) >= abs(observed) - 1e-12))
        remaining -= batch
    return (n_as_extreme + 1) / (n_permutations + 1)


def differences_by_scan_and_network(scored, condition, reference, column):
    """Array (scans, networks) of condition minus reference, for networks present in both.

    Rows are sorted by scan, fold and run, and networks are numbered in order
    of first appearance; this fixes which column each random sign belongs to,
    and reproduces the original pipeline's permutation p-values exactly.
    """
    import pandas as pd

    keys = ["scan_id", "fold", "run"]
    paired = (scored.loc[scored["condition"] == condition, keys + [column]]
              .merge(scored.loc[scored["condition"] == reference, keys + [column]],
                     on=keys, suffixes=("_condition", "_reference"))
              .sort_values(keys))
    scan_index, scan_names = pd.factorize(paired["scan_id"])
    network_index, network_names = pd.factorize(paired["fold"].astype(str) + "/" + paired["run"].astype(str))
    table = np.full((len(scan_names), len(network_names)), np.nan)
    table[scan_index, network_index] = (paired[f"{column}_condition"] - paired[f"{column}_reference"]).to_numpy()
    return table


def compare_families(scored, columns, families, n_bootstrap, n_permutations, seed):
    """Paired comparisons of every family against its reference, for every column.

    scored: one row per scored curve, with columns condition and scan_id, the
    value columns, and fold and run when there is one row per trained network
    (the single-run estimand). The value of a scan is the mean over its rows.
    When a run column is present, the network-level permutation test is added.
    The Holm correction is applied within each family and column.
    Returns one row per family, column and condition.
    """
    import pandas as pd

    with_permutation = "run" in scored.columns
    per_scan = scored.groupby(["condition", "scan_id"])[columns].mean()
    predicted_conditions = set(per_scan.index.get_level_values("condition"))
    tables = []
    for family in families:
        reference = family["reference"]
        members = [condition for condition in family["conditions"] if condition in predicted_conditions]
        if reference not in predicted_conditions or not members:
            print(f"{family['family']}: not all conditions are predicted yet, skipped")
            continue
        for column in columns:
            rows = []
            for condition in members:
                pairs = pd.concat([per_scan.loc[condition, column].rename("condition"),
                                   per_scan.loc[reference, column].rename("reference")],
                                  axis=1, join="inner").dropna()
                differences = (pairs["condition"] - pairs["reference"]).to_numpy()
                low, high = bootstrap_interval_of_median(differences, n_bootstrap, seed)
                row = {"family": family["family"], "metric": column, "condition": condition,
                       "reference": reference, "n_scans": len(pairs),
                       "median_reference": float(pairs["reference"].median()),
                       "median_condition": float(pairs["condition"].median()),
                       "median_difference": float(np.median(differences)), "ci_low": low, "ci_high": high,
                       "rank_biserial": rank_biserial(differences),
                       "p_wilcoxon": wilcoxon_p(pairs["condition"].to_numpy(), pairs["reference"].to_numpy())}
                if with_permutation:
                    table = differences_by_scan_and_network(scored.dropna(subset=[column]), condition,
                                                            reference, column)
                    row["n_network_pairs"] = table.shape[1]
                    row["p_permutation"] = network_permutation_p(table, n_permutations, seed)
                rows.append(row)
            family_table = pd.DataFrame(rows)
            family_table["p_wilcoxon_holm"] = holm(family_table["p_wilcoxon"])
            if with_permutation:
                family_table["p_permutation_holm"] = holm(family_table["p_permutation"])
            tables.append(family_table)
    return pd.concat(tables, ignore_index=True)
