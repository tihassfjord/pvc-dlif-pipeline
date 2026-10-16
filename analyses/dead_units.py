"""Are the runs that never left the loss plateau dead inside?

The runs that did not converge predict almost the same curve for every scan.
In FC-DLIF (DLIFNet_MAX) the encoder ends in Conv3d -> ReLU -> average
pooling, so if every channel of that ReLU were zero for every input, the
encoder output would be constant and so would the prediction.

For every run that did not converge, and for the first two converged runs of
the same fold for comparison, the network is fed the test scans of its fold
(the inputs it was scored on). For every ReLU layer the analysis records how
many channels are never positive ("dead") and the share of zero outputs, and
for the run how much the prediction varies between scans:
    spread = mean over frames of the standard deviation over scans
             / mean absolute prediction.

Inference only; needs torch, the trained weights and the group's code.

Reads   <work>/models/<network>/<representation>/fold_XX/run_YY/ (model.pt, summary.json)
        <work>/network_inputs/<representation>/
Writes  <work>/analyses/<network>/dead_units.csv

Usage
    python analyses/dead_units.py --config configs/local.yaml --network fcdlif
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch

from pvc_dlif_pipeline.config import load_study_config
from pvc_dlif_pipeline.group_data import add_group_code_to_python_path, load_network_input
from pvc_dlif_pipeline.prediction import load_weights
from pvc_dlif_pipeline.training import make_folds, network_prediction


def layer_activity(network, images, device):
    """Per ReLU layer: (channels, channels never positive, share of zero outputs); and the predictions."""
    ever_positive, zero_shares, hooks = {}, {}, []
    for name, module in network.named_modules():
        if isinstance(module, torch.nn.ReLU):
            def record(module, inputs, output, name=name):
                all_but_channel = tuple(axis for axis in range(output.dim()) if axis != 1)
                positive = (output > 0).any(dim=all_but_channel).cpu()
                ever_positive[name] = positive if name not in ever_positive else ever_positive[name] | positive
                zero_shares.setdefault(name, []).append(float((output <= 0).float().mean()))
            hooks.append(module.register_forward_hook(record))
    predictions = []
    with torch.no_grad():
        for image in images:
            tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)[None, None]).to(device)
            predictions.append(network_prediction(network, tensor).cpu().numpy().reshape(-1))
    for hook in hooks:
        hook.remove()
    layers = {name: (int(ever_positive[name].numel()), int((~ever_positive[name]).sum()),
                     float(np.mean(zero_shares[name]))) for name in ever_positive}
    return layers, np.asarray(predictions)


def main():
    parser = argparse.ArgumentParser(description="Dead ReLU channels in the runs that did not converge.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--network", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_study_config(args.config)
    work_folder = config["paths"]["work_folder"]
    threshold = config["evaluation"]["converged_val_loss"]
    input_shape = tuple(config["scans"]["network_input_shape"])

    manifest = pd.read_csv(work_folder / "manifest.csv")
    folds = make_folds(manifest.loc[manifest["used"], "scan_id"].tolist(), config["training"]["n_folds"],
                       config["training"]["seed"])
    test_ids_of_fold = {fold: test_ids for fold, _, test_ids in folds}
    add_group_code_to_python_path(config["paths"]["group_code_folder"])
    from models import models as group_models
    network_class = getattr(group_models, config["networks"][args.network])

    rows = []
    for representation in config["training"]["trained_representations"]:
        models_folder = work_folder / "models" / args.network / representation
        losses = {}
        for summary_path in sorted(models_folder.glob("fold_*/run_*/summary.json")):
            with open(summary_path, "r", encoding="utf-8") as summary_file:
                fold = int(summary_path.parent.parent.name.split("_")[1])
                run = int(summary_path.parent.name.split("_")[1])
                losses[(fold, run)] = json.load(summary_file)["best_val_loss"]
        failed = sorted(key for key, loss in losses.items() if loss > threshold)
        for fold in sorted({fold for fold, _ in failed}):
            converged_runs = sorted(run for (f, run), loss in losses.items() if f == fold and loss <= threshold)
            images = [load_network_input(work_folder / "network_inputs" / representation, scan_id, input_shape)[0]
                      for scan_id in test_ids_of_fold[fold]]
            to_check = [(run, "not converged") for f, run in failed if f == fold] + \
                       [(run, "converged") for run in converged_runs[:2]]
            for run, status in to_check:
                model_path = models_folder / f"fold_{fold:02d}" / f"run_{run:02d}" / "model.pt"
                network = load_weights(network_class(), model_path, args.device)
                layers, predictions = layer_activity(network, images, args.device)
                spread = float(predictions.std(axis=0).mean() / np.abs(predictions).mean())
                print(f"{representation} fold {fold} run {run} ({status}): spread between scans {spread:.3f}")
                for layer, (channels, dead, zero_share) in layers.items():
                    rows.append({"representation": representation, "fold": fold, "run": run, "status": status,
                                 "best_val_loss": losses[(fold, run)], "layer": layer, "channels": channels,
                                 "dead": dead, "zero_share": zero_share, "spread_between_scans": spread})
    columns = ["representation", "fold", "run", "status", "best_val_loss", "layer", "channels", "dead",
               "zero_share", "spread_between_scans"]
    table = pd.DataFrame(rows, columns=columns)
    if table.empty:
        print("Every run converged; there is nothing to check (the file has only the column names).")
    output_folder = work_folder / "analyses" / args.network
    output_folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_folder / "dead_units.csv", index=False)
    print(f"Written: {output_folder / 'dead_units.csv'}")


if __name__ == "__main__":
    main()
