"""How many training runs per fold does a paired comparison need?

For k = 1 to 10 runs per fold, the same k run numbers are drawn for the
reference and the condition (so the pairs stay matched on seed and split),
each scan's RMSE is averaged over the drawn runs that converged, and the
paired Wilcoxon test is run over the scans, Holm-adjusted within the family.
This is repeated `--draws` times (1000 in the thesis). Reported: the share of draws that reject at
0.05, and the 5th, 50th and 95th percentile of the median paired difference.
With all 10 runs there is one draw, and the result is the reported one.

Exploratory. Wilcoxon only, without the network-level permutation test, so a
rejection here is the less strict half of the decision rule of stage 07.

Reads   <work>/results/<network>/single_run/curve_metrics.csv
Writes  <work>/analyses/<network>/run_subsampling.csv

Usage
    python analyses/run_subsampling.py --config configs/local.yaml --network fcdlif
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from pvc_dlif_pipeline.config import load_study_config
from pvc_dlif_pipeline.statistics import holm

FAMILY = "retrained"


def main():
    parser = argparse.ArgumentParser(description="Paired comparisons with fewer runs per fold.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--network", required=True)
    parser.add_argument("--draws", type=int, default=1000, help="random draws per k (1000 in the thesis)")
    args = parser.parse_args()
    config = load_study_config(args.config)
    work_folder = config["paths"]["work_folder"]
    family = next(family for family in config["evaluation"]["comparison_families"] if family["family"] == FAMILY)
    reference, conditions = family["reference"], family["conditions"]

    metrics = pd.read_csv(work_folder / "results" / args.network / "single_run" / "curve_metrics.csv")
    # One table per condition: rows scans, columns run numbers (each scan is in one fold only).
    tables = {condition: metrics[metrics["condition"] == condition].pivot_table(index="scan_id", columns="run",
                                                                                values="rmse")
              for condition in [reference, *conditions]}
    scans = tables[reference].index
    runs = np.array(sorted(tables[reference].columns))

    generator = np.random.default_rng(config["evaluation"]["random_seed"])
    rows = []
    for k in range(1, len(runs) + 1):
        n_draws = 1 if k == len(runs) else args.draws
        rejections = {condition: 0 for condition in conditions}
        medians = {condition: [] for condition in conditions}
        for _ in range(n_draws):
            drawn = runs if k == len(runs) else generator.choice(runs, size=k, replace=False)
            reference_rmse = tables[reference].loc[scans, drawn].mean(axis=1)
            p_values, median_differences = [], []
            for condition in conditions:
                differences = (tables[condition].loc[scans, drawn].mean(axis=1) - reference_rmse).dropna()
                p_values.append(wilcoxon(differences).pvalue)
                median_differences.append(differences.median())
            for condition, adjusted, median in zip(conditions, holm(p_values), median_differences):
                rejections[condition] += adjusted < 0.05
                medians[condition].append(median)
        for condition in conditions:
            low, middle, high = np.percentile(medians[condition], [5, 50, 95])
            rows.append({"condition": condition, "reference": reference, "runs_per_fold": k, "draws": n_draws,
                         "share_rejecting": rejections[condition] / n_draws,
                         "median_difference_p05": low, "median_difference_p50": middle,
                         "median_difference_p95": high})
    table = pd.DataFrame(rows)

    output_folder = work_folder / "analyses" / args.network
    output_folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_folder / "run_subsampling.csv", index=False)
    print(table.pivot_table(index="runs_per_fold", columns="condition", values="share_rejecting").round(3).to_string())


if __name__ == "__main__":
    main()
