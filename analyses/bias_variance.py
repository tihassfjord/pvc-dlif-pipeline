"""Bias-variance decomposition of the squared error over the training runs.

For one scan and one frame, over the R converged runs that predict the scan
(the 10 runs of its fold, minus any that did not converge):

    mean squared error = bias^2 + variance * (R - 1) / R

    variance  = sample variance of the R predictions (ddof = 1), an unbiased
                estimate of the spread between training runs;
    bias^2    = (mean prediction - measured)^2 - variance / R, an unbiased
                estimate of the squared error of the mean prediction (the plain
                squared difference overstates it by variance / R).

Both are unbiased for the population of training runs, so conditions with
different numbers of converged runs can be compared. A frame's bias^2 can come
out slightly negative when the bias is small against the run spread; it is
kept, so the averages stay unbiased. Frames are averaged within a scan, scans
within a condition.

The paired comparisons test, per scan, the change in bias^2 and in variance of
each condition against its family's reference (Wilcoxon signed-rank, not
adjusted for multiplicity; the decomposition explains the stage 07 results and
is not a separate test of them).

Reads   <work>/predictions/<network>.csv
Writes  <work>/analyses/<network>/bias_variance_scans.csv        per condition and scan
        <work>/analyses/<network>/bias_variance.csv              per condition
        <work>/analyses/<network>/bias_variance_comparisons.csv  per family and condition

Usage
    python analyses/bias_variance.py --config configs/local.yaml --network fcdlif
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from pvc_dlif_pipeline.config import load_study_config


def decompose_scan(predictions_by_run, measured):
    """Bias^2, variance, MSE and signed bias of one scan; predictions_by_run is (runs, frames)."""
    n_runs = predictions_by_run.shape[0]
    mean_prediction = predictions_by_run.mean(axis=0)
    variance = predictions_by_run.var(axis=0, ddof=1)
    bias_squared = (mean_prediction - measured) ** 2 - variance / n_runs
    return {"n_runs": n_runs,
            "mse": float(np.mean((predictions_by_run - measured) ** 2)),
            "bias_squared": float(bias_squared.mean()),
            "variance": float(variance.mean()),
            "signed_bias": float((mean_prediction - measured).mean())}


def main():
    parser = argparse.ArgumentParser(description="Bias-variance decomposition over training runs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--network", required=True)
    args = parser.parse_args()
    config = load_study_config(args.config)
    work_folder = config["paths"]["work_folder"]

    predictions = pd.read_csv(work_folder / "predictions" / f"{args.network}.csv")
    predictions = predictions[predictions["converged"]]

    rows = []
    for (condition, scan_id), scan in predictions.groupby(["condition", "scan_id"]):
        by_run = scan.pivot_table(index=["fold", "run"], columns="frame", values="predicted")
        measured = scan.drop_duplicates("frame").sort_values("frame")["measured"].to_numpy()
        rows.append({"condition": condition, "scan_id": scan_id,
                     **decompose_scan(by_run.to_numpy(), measured)})
    per_scan = pd.DataFrame(rows)

    per_condition = (per_scan.groupby("condition")
                     .agg(n_scans=("scan_id", "size"), mse=("mse", "mean"), bias_squared=("bias_squared", "mean"),
                          variance=("variance", "mean"), signed_bias=("signed_bias", "mean"))
                     .reset_index())
    per_condition["variance_share"] = per_condition["variance"] / per_condition["mse"]

    comparison_rows = []
    by_scan = per_scan.set_index(["condition", "scan_id"])
    present = set(per_scan["condition"])
    for family in config["evaluation"]["comparison_families"]:
        reference = family["reference"]
        if reference not in present:
            continue
        for condition in [name for name in family["conditions"] if name in present]:
            pair = by_scan.loc[condition].join(by_scan.loc[reference], lsuffix="_condition", rsuffix="_reference",
                                               how="inner")
            row = {"family": family["family"], "condition": condition, "reference": reference, "n_scans": len(pair)}
            for term in ["mse", "bias_squared", "variance"]:
                difference = pair[f"{term}_condition"] - pair[f"{term}_reference"]
                row[f"{term}_median_difference"] = float(difference.median())
                row[f"{term}_mean_difference"] = float(difference.mean())
                row[f"{term}_p_wilcoxon"] = float(wilcoxon(difference, zero_method="wilcox").pvalue)
            comparison_rows.append(row)
    comparisons = pd.DataFrame(comparison_rows)

    output_folder = work_folder / "analyses" / args.network
    output_folder.mkdir(parents=True, exist_ok=True)
    per_scan.to_csv(output_folder / "bias_variance_scans.csv", index=False)
    per_condition.to_csv(output_folder / "bias_variance.csv", index=False)
    comparisons.to_csv(output_folder / "bias_variance_comparisons.csv", index=False)
    print(per_condition.round(4).to_string(index=False))
    print(f"Written: {output_folder}")


if __name__ == "__main__":
    main()
