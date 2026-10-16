"""Stage 05: train one network on one input representation, for chosen folds and runs.

Each (fold, run) is trained from scratch with the protocol in the `training`
section of configs/study.yaml (see pvc_dlif_pipeline/training.py). The scans
of the representation are read into memory once and shared by all runs.
A run whose summary.json already exists is skipped.

The network classes come from the group's repository (paths.group_code_folder),
which is imported at run time and not part of this repository.

Writes
    <work>/models/<network>/<representation>/fold_XX/run_YY/model.pt and summary.json
    <work>/models/<network>/<representation>/folds.json   the partition, for every fold

Usage
    python scripts/05_train.py --config configs/study.yaml --network fcdlif --representation uncorrected
    python scripts/05_train.py --config configs/study.yaml --network fcdlif --representation rl_i15 --folds 1 --runs 1
"""

import argparse
import json

import numpy as np
import pandas as pd

from pvc_dlif_pipeline.config import load_study_config
from pvc_dlif_pipeline.group_data import add_group_code_to_python_path, load_arterial_curve, load_network_input
from pvc_dlif_pipeline.training import make_folds, run_seed_of, train_one_run, validation_split


def main():
    parser = argparse.ArgumentParser(description="Stage 05: train a network.")
    parser.add_argument("--config", required=True, help="path to configs/study.yaml")
    parser.add_argument("--network", required=True, help="a key of `networks` in the config, e.g. fcdlif")
    parser.add_argument("--representation", required=True, help="a stage 04 folder name, e.g. rl_i15")
    parser.add_argument("--folds", nargs="*", type=int, help="fold numbers to train (default: all)")
    parser.add_argument("--runs", nargs="*", type=int, help="run numbers to train (default: all)")
    parser.add_argument("--epochs", type=int, help="override training.epochs (for a quick test only)")
    args = parser.parse_args()

    config = load_study_config(args.config)
    work_folder = config["paths"]["work_folder"]
    settings = dict(config["training"])
    if args.epochs:
        settings["epochs"] = args.epochs
    input_shape = tuple(config["scans"]["network_input_shape"])

    add_group_code_to_python_path(config["paths"]["group_code_folder"])
    from models import models as group_models
    network_class = getattr(group_models, config["networks"][args.network])

    def build_network():
        # Both networks take one input channel by default: DLIFNet_MAX has
        # in_channels=1 as its default, CNN3_1dConv3_linear takes no arguments.
        return network_class()

    manifest = pd.read_csv(work_folder / "manifest.csv", keep_default_na=False)
    used = manifest[manifest["used"]]
    scan_ids = used["scan_id"].tolist()
    group_of_scan = dict(zip(used["scan_id"], used["group"]))

    print(f"Reading {len(scan_ids)} scans of {args.representation} into memory ...")
    input_folder = work_folder / "network_inputs" / args.representation
    scans = {}
    for scan_id in scan_ids:
        image, _ = load_network_input(input_folder, scan_id, input_shape)
        arterial, _ = load_arterial_curve(config["paths"]["group_data_folder"], scan_id)
        scans[scan_id] = {"image": image.astype(np.float32), "arterial": arterial.astype(np.float32)}

    output_root = work_folder / "models" / args.network / args.representation
    output_root.mkdir(parents=True, exist_ok=True)
    folds = make_folds(scan_ids, settings["n_folds"], settings["seed"])
    with open(output_root / "folds.json", "w", encoding="utf-8") as folds_file:
        json.dump([{"fold": fold, "train_ids": train_ids, "test_ids": test_ids}
                   for fold, train_ids, test_ids in folds], folds_file, indent=2)

    for fold, fold_train_ids, test_ids in folds:
        if args.folds and fold not in args.folds:
            continue
        for run in range(1, settings["n_runs"] + 1):
            if args.runs and run not in args.runs:
                continue
            run_folder = output_root / f"fold_{fold:02d}" / f"run_{run:02d}"
            if (run_folder / "summary.json").exists():
                print(f"fold {fold} run {run}: already trained, skipped")
                continue
            seed = run_seed_of(settings["seed"], fold, run)
            training_ids, validation_ids = validation_split(fold_train_ids, group_of_scan,
                                                            settings["validation_fraction"], seed)
            print(f"fold {fold} run {run}: {len(training_ids)} train, {len(validation_ids)} validation, "
                  f"{len(test_ids)} test scans, seed {seed}", flush=True)
            summary = train_one_run(build_network, scans, training_ids, validation_ids, settings, seed, run_folder)
            print(f"fold {fold} run {run}: best validation loss {summary['best_val_loss']:.4f} "
                  f"at epoch {summary['best_epoch'] + 1}", flush=True)


if __name__ == "__main__":
    main()
