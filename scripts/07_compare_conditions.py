"""Stage 07: curve metrics of every prediction, and the paired comparisons between conditions.

Two estimands, read from the two stage 06 files:
  single_run  every converged run is scored; per scan, the metric is the mean
              over that scan's runs. Tested with the Wilcoxon signed-rank test
              over the 70 scans and with a sign-flip permutation test over the
              trained networks (fold, run); an effect is established only if
              both reject.
  ensemble    the mean curve of a fold's converged runs is scored, one value
              per scan. Wilcoxon only: there is one ensemble per fold, so no
              network-level permutation is possible.

Every family in evaluation.comparison_families is compared against its own
reference, and the Holm correction is applied within each family and metric.

Writes
    <work>/results/<network>/<estimand>/curve_metrics.csv   one row per scored curve
    <work>/results/<network>/<estimand>/comparisons.csv     one row per family, metric and condition

Usage
    python scripts/07_compare_conditions.py --config configs/study.yaml --network fcdlif --estimand single_run
    python scripts/07_compare_conditions.py --config configs/study.yaml --network fcdlif --estimand ensemble
"""

import argparse

import pandas as pd

from pvc_dlif_pipeline.config import load_study_config
from pvc_dlif_pipeline.metrics import curve_metrics
from pvc_dlif_pipeline.statistics import compare_families


def main():
    parser = argparse.ArgumentParser(description="Stage 07: curve metrics and paired comparisons.")
    parser.add_argument("--config", required=True, help="path to configs/study.yaml")
    parser.add_argument("--network", required=True, help="e.g. fcdlif")
    parser.add_argument("--estimand", required=True, choices=["single_run", "ensemble"])
    args = parser.parse_args()

    config = load_study_config(args.config)
    work_folder = config["paths"]["work_folder"]
    settings = config["evaluation"]

    # 1. Score every curve.
    file_name = f"{args.network}.csv" if args.estimand == "single_run" else f"{args.network}_ensemble.csv"
    predictions = pd.read_csv(work_folder / "predictions" / file_name)
    if args.estimand == "single_run":
        predictions = predictions[predictions["converged"]]
        curve_keys = ["condition", "scan_id", "fold", "run"]
    else:
        curve_keys = ["condition", "scan_id", "fold"]
    rows = []
    for key_values, curve in predictions.groupby(curve_keys):
        curve = curve.sort_values("frame")
        rows.append({**dict(zip(curve_keys, key_values)),
                     **curve_metrics(curve["predicted"], curve["measured"], curve["time_min"])})
    scored = pd.DataFrame(rows)

    # 2. Paired comparisons, family by family (one value per scan: the mean over its runs).
    comparisons = compare_families(scored, settings["tested_metrics"], settings["comparison_families"],
                                   settings["n_bootstrap"], settings["n_permutations"], settings["random_seed"])

    output_folder = work_folder / "results" / args.network / args.estimand
    output_folder.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_folder / "curve_metrics.csv", index=False)
    comparisons.to_csv(output_folder / "comparisons.csv", index=False)
    columns = ["family", "condition", "reference", "median_difference", "ci_low", "ci_high", "p_wilcoxon_holm"]
    if args.estimand == "single_run":
        columns.append("p_permutation_holm")
    print(comparisons[comparisons["metric"] == "rmse"][columns].to_string(index=False))
    print(f"Written: {output_folder}")


if __name__ == "__main__":
    main()
