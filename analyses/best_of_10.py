"""Best of 10: the run a user would deploy, chosen on validation loss.

The single-run estimand of stage 07 is the expected performance of one
training run. In practice one network is chosen per training, on validation
loss. This analysis asks whether the conclusions hold for that network.
Secondary and exploratory: it was specified after the first results were seen.

Selection, on the training side only and the same for every condition: per
fold, the run of the trained weights (`weights` in the conditions) with the
lowest best validation loss; ties go to the lowest run number. Every condition
that uses the same weights therefore uses the same run. Each scan is a test
scan of exactly one fold, so each scan gets one curve per condition.

The comparisons are those of stages 07 and 08 (families, Wilcoxon, Holm within
family and metric, bootstrap interval of the median), on one curve per scan.
There is one network per fold, so there is no network-level permutation test.
The bootstrap seed is 20261002, the date the analysis was specified, as in the
original analysis.

Reads   <work>/models/<network>/<weights>/fold_XX/run_YY/summary.json
        <work>/results/<network>/single_run/curve_metrics.csv and kinetics_curves.csv
Writes  <work>/analyses/<network>/best_of_10_selected_runs.csv
        <work>/analyses/<network>/best_of_10_comparisons.csv        curve metrics
        <work>/analyses/<network>/best_of_10_kinetics_comparisons.csv  Patlak Ki errors

Usage
    python analyses/best_of_10.py --config configs/local.yaml --network fcdlif
"""

import argparse
import json

import numpy as np
import pandas as pd

from pvc_dlif_pipeline.config import load_study_config
from pvc_dlif_pipeline.statistics import compare_families

BOOTSTRAP_SEED = 20261002


def select_runs(config, network):
    """Per weights and fold, the run with the lowest best validation loss."""
    models_folder = config["paths"]["work_folder"] / "models" / network
    threshold = config["evaluation"]["converged_val_loss"]
    rows = []
    for weights in sorted({condition["weights"] for condition in config["conditions"]}):
        for fold in range(1, config["training"]["n_folds"] + 1):
            losses = []
            for run in range(1, config["training"]["n_runs"] + 1):
                summary_path = models_folder / weights / f"fold_{fold:02d}" / f"run_{run:02d}" / "summary.json"
                if summary_path.exists():
                    with open(summary_path, "r", encoding="utf-8") as summary_file:
                        losses.append((json.load(summary_file)["best_val_loss"], run))
            if not losses:
                continue
            if len(losses) != config["training"]["n_runs"]:
                raise SystemExit(f"{weights} fold {fold}: {len(losses)} runs, expected {config['training']['n_runs']}")
            best_loss, best_run = min(losses)  # lowest loss; on a tie, the lowest run number
            if best_loss > threshold:
                raise SystemExit(f"{weights} fold {fold}: no run converged")
            rows.append({"weights": weights, "fold": fold, "run": best_run, "best_val_loss": best_loss,
                         "fold_median_val_loss": float(np.median([loss for loss, _ in losses]))})
    return pd.DataFrame(rows)


def keep_selected(table, selected, config):
    """Rows of a per-curve table that belong to the selected run of their condition and fold."""
    weights_of = {condition["name"]: condition["weights"] for condition in config["conditions"]}
    table = table.assign(weights=table["condition"].map(weights_of))
    kept = table.merge(selected[["weights", "fold", "run"]], on=["weights", "fold", "run"], how="inner")
    keys = ["condition", "scan_id"] + (["voi"] if "voi" in kept.columns else [])
    if kept.groupby(keys).size().max() != 1:
        raise SystemExit("more than one selected curve for a condition and scan")
    return kept.drop(columns=["weights", "fold", "run"])


def main():
    parser = argparse.ArgumentParser(description="Best of 10 runs by validation loss.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--network", required=True)
    args = parser.parse_args()
    config = load_study_config(args.config)
    work_folder = config["paths"]["work_folder"]
    evaluation = config["evaluation"]
    results_folder = work_folder / "results" / args.network / "single_run"

    selected = select_runs(config, args.network)

    curves = keep_selected(pd.read_csv(results_folder / "curve_metrics.csv"), selected, config)
    curve_comparisons = compare_families(curves, evaluation["tested_metrics"], evaluation["comparison_families"],
                                         evaluation["n_bootstrap"], evaluation["n_permutations"], BOOTSTRAP_SEED)

    fits = pd.read_csv(results_folder / "kinetics_curves.csv")
    ki = fits[(fits["model"] == "patlak") & np.isfinite(fits["relative_error"])]
    ki = keep_selected(ki, selected, config)
    signed = ki.pivot_table(index=["condition", "scan_id"], columns="voi", values="relative_error") * 100.0
    errors = pd.concat([signed.abs().add_prefix("ki_abs_error_pct_"), signed.add_prefix("ki_error_pct_")],
                       axis=1).reset_index()
    ki_columns = [column for column in errors.columns if column.startswith("ki_")]
    ki_comparisons = compare_families(errors, ki_columns, evaluation["comparison_families"],
                                      evaluation["n_bootstrap"], evaluation["n_permutations"], BOOTSTRAP_SEED)

    output_folder = work_folder / "analyses" / args.network
    output_folder.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_folder / "best_of_10_selected_runs.csv", index=False)
    curve_comparisons.to_csv(output_folder / "best_of_10_comparisons.csv", index=False)
    ki_comparisons.to_csv(output_folder / "best_of_10_kinetics_comparisons.csv", index=False)
    rmse = curve_comparisons[curve_comparisons["metric"] == "rmse"]
    print(rmse[["family", "condition", "reference", "median_difference", "ci_low", "ci_high",
                "p_wilcoxon_holm"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
