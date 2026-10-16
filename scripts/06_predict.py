"""Stage 06: predict the input function of every scan, under every condition.

For each condition in configs/study.yaml, the networks trained on `weights`
(stage 05) are shown the network inputs of `shown` (stage 04). Each scan is
predicted only by the 10 runs of the fold that held it out, so no network
predicts a scan it was trained or validated on.

Two estimands are written:
  - single run: every run's prediction, with a flag saying whether the run
    converged (best validation loss <= evaluation.converged_val_loss);
  - ensemble: per scan, the mean curve over the converged runs of its fold,
    which is how the group uses the network.

Writes
    <work>/predictions/<network>.csv            condition, scan_id, fold, run, converged,
                                                frame, time_min, predicted, measured
    <work>/predictions/<network>_ensemble.csv   condition, scan_id, fold, n_runs,
                                                frame, time_min, predicted, measured

Usage
    python scripts/06_predict.py --config configs/study.yaml --network fcdlif
    python scripts/06_predict.py --config configs/study.yaml --network fcdlif --conditions baseline rl
"""

import argparse
import json

import pandas as pd
import torch

from pvc_dlif_pipeline.config import load_study_config
from pvc_dlif_pipeline.group_data import add_group_code_to_python_path, load_arterial_curve, load_network_input
from pvc_dlif_pipeline.prediction import load_weights, predict_curve
from pvc_dlif_pipeline.training import make_folds


def main():
    parser = argparse.ArgumentParser(description="Stage 06: predictions.")
    parser.add_argument("--config", required=True, help="path to configs/study.yaml")
    parser.add_argument("--network", required=True, help="a key of `networks` in the config, e.g. fcdlif")
    parser.add_argument("--conditions", nargs="*", help="only these conditions (default: all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_study_config(args.config)
    work_folder = config["paths"]["work_folder"]
    group_data_folder = config["paths"]["group_data_folder"]
    input_shape = tuple(config["scans"]["network_input_shape"])
    n_runs = config["training"]["n_runs"]

    # The same folds as stage 05: they depend only on the used scans and the seed.
    manifest = pd.read_csv(work_folder / "manifest.csv")
    scan_ids = manifest.loc[manifest["used"], "scan_id"].tolist()
    folds = make_folds(scan_ids, config["training"]["n_folds"], config["training"]["seed"])

    add_group_code_to_python_path(config["paths"]["group_code_folder"])
    from models import models as group_models
    network_class = getattr(group_models, config["networks"][args.network])

    conditions = config["conditions"]
    if args.conditions:
        conditions = [condition for condition in conditions if condition["name"] in args.conditions]

    rows = []
    for condition in conditions:
        models_folder = work_folder / "models" / args.network / condition["weights"]
        inputs_folder = work_folder / "network_inputs" / condition["shown"]

        n_predicted_runs = 0
        n_missing_runs = 0
        for fold, _, test_ids in folds:
            held_out = {}
            for scan_id in test_ids:
                image, _ = load_network_input(inputs_folder, scan_id, input_shape)
                measured, time_min = load_arterial_curve(group_data_folder, scan_id)
                held_out[scan_id] = (image, measured, time_min)

            for run in range(1, n_runs + 1):
                model_path = models_folder / f"fold_{fold:02d}" / f"run_{run:02d}" / "model.pt"
                if not model_path.exists():
                    n_missing_runs += 1
                    continue
                with open(model_path.parent / "summary.json", "r", encoding="utf-8") as summary_file:
                    best_val_loss = json.load(summary_file)["best_val_loss"]
                converged = best_val_loss <= config["evaluation"]["converged_val_loss"]
                network = load_weights(network_class(), model_path, args.device)  # one input channel by default
                n_predicted_runs += 1
                for scan_id, (image, measured, time_min) in held_out.items():
                    predicted = predict_curve(network, image, args.device)
                    for frame in range(len(measured)):
                        rows.append({"condition": condition["name"], "scan_id": scan_id,
                                     "fold": fold, "run": run, "converged": converged, "frame": frame,
                                     "time_min": time_min[frame], "predicted": predicted[frame],
                                     "measured": measured[frame]})
        print(f"{condition['name']}: {n_predicted_runs} runs, weights {condition['weights']}, "
              f"shown {condition['shown']}"
              + (f" ({n_missing_runs} runs have no trained model yet and were skipped)" if n_missing_runs else ""))

    predictions = pd.DataFrame(rows)

    # Ensemble: mean over the converged runs of the fold, frame by frame.
    converged_runs = predictions[predictions["converged"]]
    ensemble = (converged_runs
                .groupby(["condition", "scan_id", "fold", "frame"], as_index=False)
                .agg(n_runs=("run", "count"), time_min=("time_min", "first"),
                     predicted=("predicted", "mean"), measured=("measured", "first")))
    ensemble = ensemble[["condition", "scan_id", "fold", "n_runs", "frame", "time_min", "predicted", "measured"]]

    output_folder = work_folder / "predictions"
    output_folder.mkdir(parents=True, exist_ok=True)
    for table, file_name in [(predictions, f"{args.network}.csv"), (ensemble, f"{args.network}_ensemble.csv")]:
        output_path = output_folder / file_name
        if args.conditions and output_path.exists():
            # Replace only the conditions predicted now, keep the others.
            previous = pd.read_csv(output_path)
            previous = previous[~previous["condition"].isin(table["condition"].unique())]
            table = pd.concat([previous, table], ignore_index=True)
        table.to_csv(output_path, index=False)
        print(f"Written: {output_path} ({len(table)} rows)")


if __name__ == "__main__":
    main()
