"""Stage 08: kinetic parameters with the predicted and with the measured input.

For every scored curve of stage 07 (one per converged run, or one per fold for
the ensemble), the brain and myocardium curves are fitted with Patlak and with
the two-tissue compartment model, once with the predicted input and once with
the measured input. Both inputs are converted to plasma first.

The fits are per curve, and the errors are averaged within a scan only
afterwards: the same order as the curve metrics. Fitting the mean curve of a
scan's runs would score an ensemble instead.

The paired comparisons use the error of Patlak Ki, in per cent, absolute and
signed, with the families and tests of stage 07.

Writes, in results/<network>/<estimand>/:
  kinetics_curves.csv       one row per curve, VOI, model and parameter
  kinetics_scans.csv        the same, averaged over the curves of a scan
  kinetics_comparisons.csv  paired comparisons of the Patlak Ki errors

The single-run estimand has about 2900 curves per network and takes about
10-15 minutes with --jobs 4; the ensemble about one minute.

Usage
    python scripts/08_kinetics.py --config configs/study.yaml --network fcdlif --estimand single_run --jobs 4
    python scripts/08_kinetics.py --config configs/study.yaml --network fcdlif --estimand ensemble
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from pvc_dlif_pipeline.config import load_study_config
from pvc_dlif_pipeline.group_data import load_tissue_curves
from pvc_dlif_pipeline.kinetics import fit_scan
from pvc_dlif_pipeline.statistics import compare_families


def make_scan_tasks(predictions, curve_keys, config, scan_ids):
    """One task per scan for kinetics.fit_scan."""
    settings = config["kinetics"]
    tasks = []
    for scan_id, scan in predictions.groupby("scan_id"):
        if scan_ids and scan_id not in scan_ids:
            continue
        curves, time_min = load_tissue_curves(config["paths"]["group_data_folder"], scan_id)
        measured = None
        predicted_curves = []
        for key_values, curve in scan.groupby(curve_keys):
            curve = curve.sort_values("frame")
            if measured is None:
                measured = curve["measured"].to_numpy()
            elif not np.array_equal(measured, curve["measured"].to_numpy()):
                raise ValueError(f"{scan_id}: the measured curve differs between conditions")
            predicted_curves.append({**dict(zip(curve_keys, key_values)), "scan_id": scan_id,
                                     "predicted": curve["predicted"].to_numpy()})
        tasks.append({"scan_id": scan_id,
                      "pypet_folder": str(config["paths"]["pypet_folder"]),
                      "irreversible": bool(settings["irreversible"]),
                      "plasma_ratio": (settings["plasma_ratio_a"], settings["plasma_ratio_b"],
                                       settings["plasma_ratio_c"]),
                      "time_min": time_min,
                      "tissue": {voi: curves[voi] for voi in settings["vois"]},
                      "measured": measured,
                      "curves": predicted_curves})
    return tasks


def average_over_curves(per_curve):
    """One row per condition, scan, VOI, model and parameter: means over the scan's curves.

    relative_error is the mean of the signed errors, abs_relative_error the mean
    of the absolute errors (not the absolute mean error). n_bounds_hit counts
    the two-tissue fits with the predicted input that ended at a bound.
    """
    table = per_curve.assign(abs_relative_error=per_curve["relative_error"].abs(),
                             bounds_hit_predicted=per_curve["bounds_hit_predicted"].astype(float))
    averaged = (table.groupby(["condition", "scan_id", "voi", "model", "parameter"])
                .agg(value_predicted=("value_predicted", "mean"),
                     value_measured=("value_measured", "first"),
                     relative_error=("relative_error", "mean"),
                     abs_relative_error=("abs_relative_error", "mean"),
                     n_curves=("relative_error", "size"),
                     n_bounds_hit_predicted=("bounds_hit_predicted", "sum"),
                     bounds_hit_measured=("bounds_hit_measured", "first"))
                .reset_index())
    is_patlak = averaged["model"] == "patlak"
    averaged.loc[is_patlak, ["n_bounds_hit_predicted", "bounds_hit_measured"]] = np.nan
    return averaged


def main():
    parser = argparse.ArgumentParser(description="Stage 08: kinetic modelling of the predicted inputs.")
    parser.add_argument("--config", required=True, help="path to configs/study.yaml")
    parser.add_argument("--network", required=True, help="e.g. fcdlif")
    parser.add_argument("--estimand", required=True, choices=["single_run", "ensemble"])
    parser.add_argument("--jobs", type=int, default=1, help="worker processes, one scan at a time each")
    parser.add_argument("--scan-ids", nargs="*", default=None,
                        help="only these scans (for checking; the comparisons need all scans)")
    args = parser.parse_args()

    config = load_study_config(args.config)
    work_folder = config["paths"]["work_folder"]
    evaluation = config["evaluation"]

    # 1. The curves that stage 07 scores.
    file_name = f"{args.network}.csv" if args.estimand == "single_run" else f"{args.network}_ensemble.csv"
    predictions = pd.read_csv(work_folder / "predictions" / file_name)
    if args.estimand == "single_run":
        predictions = predictions[predictions["converged"]]
        curve_keys = ["condition", "fold", "run"]
    else:
        curve_keys = ["condition", "fold"]

    # 2. Fit, one scan per task.
    tasks = make_scan_tasks(predictions, curve_keys, config, args.scan_ids)
    print(f"{len(tasks)} scans, {sum(len(task['curves']) for task in tasks)} curves")
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = [row for scan_rows in pool.map(fit_scan, tasks) for row in scan_rows]
    else:
        rows = [row for task in tasks for row in fit_scan(task)]
    per_curve = pd.DataFrame(rows)
    per_curve.insert(1, "scan_id", per_curve.pop("scan_id"))

    output_folder = work_folder / "results" / args.network / args.estimand
    output_folder.mkdir(parents=True, exist_ok=True)
    per_curve.to_csv(output_folder / "kinetics_curves.csv", index=False)
    average_over_curves(per_curve).to_csv(output_folder / "kinetics_scans.csv", index=False)

    # 3. Paired comparisons of the Patlak Ki error, per VOI, absolute and signed, in per cent.
    ki = per_curve[(per_curve["model"] == "patlak") & np.isfinite(per_curve["relative_error"])]
    signed = ki.pivot_table(index=["scan_id"] + curve_keys, columns="voi",
                            values="relative_error", aggfunc="mean") * 100.0
    errors = pd.concat([signed.abs().add_prefix("ki_abs_error_pct_"),
                        signed.add_prefix("ki_error_pct_")], axis=1).reset_index()
    columns = [column for column in errors.columns if column.startswith("ki_")]
    comparisons = compare_families(errors, columns, evaluation["comparison_families"],
                                   evaluation["n_bootstrap"], evaluation["n_permutations"],
                                   evaluation["random_seed"])
    comparisons.to_csv(output_folder / "kinetics_comparisons.csv", index=False)
    print(f"Written: {output_folder}")


if __name__ == "__main__":
    main()
