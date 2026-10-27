"""Train FC-DLIF with image-derived tissue curves (VOI TACs), four arms, same protocol as the thesis.

Arms (--arm):
  control   FC-DLIF (DLIFNet_MAX) unchanged, through this training loop
  hybrid    the four tissue curves (Brain, LV, Liver, Myocardium) are appended to the per-frame image features
            before the time convolution; the curves are needed when the model is used (an upper bound)
  aux       an extra head predicts the four tissue curves from the image features; the curves only enter the
            loss (lambda_aux * MSE), the model needs only the image when used
  kiloss    an extra loss on Patlak Ki: Ki from the predicted input function with the brain and myocardium
            curves against Ki from the measured input function, (relative error, capped at +-1)^2, weight lambda_ki
            faded in linearly between two epochs (after the loss plateau); the model needs only the image when used

Everything else is the thesis protocol of pvc_dlif_pipeline.training (folds, run seeds, validation split,
Adam, batch 8, mixed precision, weighted MSE, Poisson noise and flips, 1000 epochs). The checkpoint is chosen
on the validation weighted MSE of the input function only, in every arm, so the selection rule is the same.
The extra layers of an arm are initialised with the random state saved and restored around them, so the
base network weights, the batch order and the augmentation are the same as in the control arm for a run.

Patlak in the loss: whole blood to plasma with Wu et al. (as stage 08), cumulative trapezoid integral over
the frame mid times, slope over the frames with t >= t_star (default 10 min), no resampling. It is a
differentiable stand-in for the evaluation, which uses pyPET.

After training, the best checkpoint predicts the test scans of the fold: test_predictions.json.

Settings: the `training` section of the configuration (the thesis protocol) and its `kiloss` section
(representation, lambda_aux, lambda_ki, ki_fade_epochs, t_star_min); the plasma ratio from `kinetics`.
Writes <work>/models_kiloss/<arm>/fold_XX/run_YY/{model.pt, summary.json, test_predictions.json}.
Needs a GPU, the group's code (paths.group_code_folder) and the tissue curves (VOI_SUV).

    python analyses/kiloss_train.py --config configs/local.yaml --arm control --folds 1 --runs 1 --epochs 2   (smoke test)
    python analyses/kiloss_train.py --config configs/local.yaml --arm kiloss                                (10 x 10)
In the thesis (Appendix B) control and kiloss were trained 10 folds x 10 runs, hybrid and aux 10 folds x 2 runs.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pvc_dlif_pipeline.config import load_study_config

from pvc_dlif_pipeline.group_data import (add_group_code_to_python_path, load_arterial_curve,
                                          load_network_input, load_tissue_curves)
from pvc_dlif_pipeline.training import (augment_batch, make_folds, run_seed_of, set_all_seeds,
                                        validation_split, weighted_mse)

VOIS = ["Brain", "LV", "Liver", "Myocardium"]
KI_VOIS = ["Brain", "Myocardium"]
ARMS = ["control", "hybrid", "aux", "kiloss"]


# ----------------------------------------------------------------------------------------------- networks
def make_network_classes(base):
    """Arm networks built on the group's DLIFNet_MAX class (base)."""

    def bottleneck(net, x):
        """The per-frame image features of DLIFNet.forward: (batch, 32, frames)."""
        bottle = []
        for i in range(x.shape[0]):
            temp = torch.permute(x[i], (1, 0, 2, 3, 4))        # (frames, channels, z, y, x)
            out = net.conv3d(net.encoder(temp))
            bottle.append(out.permute(1, 0, 2, 3, 4))
        return torch.stack(bottle).squeeze(5).squeeze(4).squeeze(3)

    class Control(base):
        def forward(self, x, curves=None):
            return {"REG": self.regression(self.tcn(bottleneck(self, x))).squeeze(1)}

    class Hybrid(base):
        def __init__(self):
            super().__init__()
            state = torch.get_rng_state()
            self.tcn[0] = nn.Conv1d(32 + len(VOIS), 16, kernel_size=3, stride=1, padding=1,
                                    padding_mode="replicate", bias=False)
            torch.set_rng_state(state)

        def forward(self, x, curves=None):
            features = torch.cat([bottleneck(self, x), curves.to(x.dtype)], dim=1)
            return {"REG": self.regression(self.tcn(features)).squeeze(1)}

    class Aux(base):
        def __init__(self):
            super().__init__()
            state = torch.get_rng_state()
            self.aux_head = nn.Sequential(
                nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1, padding_mode="replicate", bias=False),
                nn.ELU(inplace=True),
                nn.Conv1d(16, len(VOIS), kernel_size=3, stride=1, padding=1, padding_mode="replicate", bias=False))
            torch.set_rng_state(state)

        def forward(self, x, curves=None):
            b = bottleneck(self, x)
            return {"REG": self.regression(self.tcn(b)).squeeze(1), "AUX": self.aux_head(b)}

    return {"control": Control, "hybrid": Hybrid, "aux": Aux, "kiloss": Control}


# ----------------------------------------------------------------------------------------------- Patlak
def plasma_ratio(t_min, a, b, c):
    return a * torch.exp(-b * t_min) + c


def patlak_ki(whole_blood, tissue, t_min, ratio, select, floor=1e-3):
    """Patlak slope. whole_blood (B, T), tissue (B, R, T), t_min (B, T), ratio (B, T), select (T,) bool."""
    cp = whole_blood.clamp_min(floor) * ratio
    dt = t_min[:, 1:] - t_min[:, :-1]
    integral = torch.cat([torch.zeros_like(cp[:, :1]), torch.cumsum(0.5 * (cp[:, 1:] + cp[:, :-1]) * dt, dim=1)], 1)
    x = integral[:, select] / cp[:, select]                         # (B, S)
    y = tissue[:, :, select] / cp[:, None, select]                  # (B, R, S)
    xc = x - x.mean(1, keepdim=True)
    yc = y - y.mean(2, keepdim=True)
    return (xc[:, None, :] * yc).sum(2) / (xc ** 2).sum(1, keepdim=True)   # (B, R)


# ----------------------------------------------------------------------------------------------- data
def load_scans(config, scan_ids, input_shape):
    paths = config["paths"]
    input_folder = Path(paths["work_folder"]) / "network_inputs" / config["kiloss"]["representation"]
    scans = {}
    for scan_id in scan_ids:
        image, _ = load_network_input(input_folder, scan_id, input_shape)
        arterial, t = load_arterial_curve(paths["group_data_folder"], scan_id)
        curves, _ = load_tissue_curves(paths["group_data_folder"], scan_id)
        n = image.shape[0]
        scans[scan_id] = {"image": image.astype(np.float32), "arterial": arterial[:n].astype(np.float32),
                          "t": t[:n].astype(np.float32),
                          "curves": np.stack([curves[v][:n] for v in VOIS]).astype(np.float32)}
    return scans


def stack(scans, ids, device, curve_scale):
    images = torch.from_numpy(np.stack([scans[s]["image"] for s in ids])[:, np.newaxis]).to(device)
    arterial = torch.from_numpy(np.stack([scans[s]["arterial"] for s in ids])).to(device)
    raw = torch.from_numpy(np.stack([scans[s]["curves"] for s in ids])).to(device)
    t = torch.from_numpy(np.stack([scans[s]["t"] for s in ids])).to(device)
    return images, arterial, raw, raw / curve_scale[None, :, None], t


# ----------------------------------------------------------------------------------------------- training
def train_one_run(build, arm, scans, training_ids, validation_ids, test_ids, settings, h, run_seed, out):
    """As pvc_dlif_pipeline.training.train_one_run, with the arm's input and loss added."""
    set_all_seeds(run_seed)
    network = build()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=settings["learning_rate"])
    use_amp = settings["mixed_precision"] and device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    generator = torch.Generator(device=device)
    generator.manual_seed(run_seed)
    bounds, weights = settings["loss_frame_bounds"], settings["loss_weights"]
    batch_size = settings["batch_size"]

    # curve scale: mean peak of each tissue curve over the training scans (curves enter the network as curve / scale)
    curve_scale = torch.tensor(np.mean([scans[s]["curves"].max(1) for s in training_ids], 0),
                               dtype=torch.float32, device=device)
    t_star, a, b, c = h["t_star_min"], h["plasma_a"], h["plasma_b"], h["plasma_c"]
    ki_index = [VOIS.index(v) for v in KI_VOIS]

    def forward(images, scaled):
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            output = network(images, scaled if arm == "hybrid" else None)
        return {k: v.float() for k, v in output.items()}

    def ki_pair(prediction, arterial, raw, t):
        ratio = plasma_ratio(t, a, b, c)
        select = t[0] >= t_star
        tissue = raw[:, ki_index]
        return patlak_ki(prediction, tissue, t, ratio, select), patlak_ki(arterial, tissue, t, ratio, select)

    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "train_extra": [], "val_extra": []}
    best_val, best_epoch = float("inf"), -1
    started = time.perf_counter()
    for epoch in range(settings["epochs"]):
        if arm == "kiloss":
            f0, f1 = h["ki_fade_epochs"]
            lam = h["lambda_ki"] * float(np.clip((epoch - f0) / max(f1 - f0, 1), 0.0, 1.0))
        else:
            lam = h["lambda_aux"] if arm == "aux" else 0.0

        network.train()
        order = torch.randperm(len(training_ids)).tolist()
        losses, extras = [], []
        for first in range(0, len(order), batch_size):
            ids = [training_ids[i] for i in order[first:first + batch_size]]
            images, arterial, raw, scaled, t = stack(scans, ids, device, curve_scale)
            images = augment_batch(images, generator, settings["poisson_noise"], settings["random_flip"])
            output = forward(images, scaled)
            loss = weighted_mse(output["REG"], arterial, bounds, weights)
            extra = torch.zeros((), device=device)
            if arm == "aux":
                extra = torch.mean((output["AUX"] - scaled) ** 2)
            elif arm == "kiloss":
                ki_p, ki_m = ki_pair(output["REG"], arterial, raw, t)
                # relative error capped at +-1, so a scan with a near-zero predicted tail cannot dominate
                extra = torch.mean(torch.clamp((ki_p - ki_m) / ki_m, -1.0, 1.0) ** 2)
            total = loss + lam * extra
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(total).backward(); scaler.step(optimizer); scaler.update()
            else:
                total.backward(); optimizer.step()
            losses.append(float(loss.detach())); extras.append(float(extra.detach()))

        network.eval()
        val_losses, val_extras = [], []
        with torch.no_grad():
            for first in range(0, len(validation_ids), batch_size):
                images, arterial, raw, scaled, t = stack(scans, validation_ids[first:first + batch_size], device, curve_scale)
                output = forward(images, scaled)
                val_losses.append(float(weighted_mse(output["REG"], arterial, bounds, weights)))
                ki_p, ki_m = ki_pair(output["REG"], arterial, raw, t)
                val_extras.append(float(torch.mean(((ki_p - ki_m) / ki_m) ** 2)))
        train_loss, val_loss = float(np.mean(losses)), float(np.mean(val_losses))
        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        history["train_extra"].append(float(np.mean(extras))); history["val_extra"].append(float(np.mean(val_extras)))
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            torch.save(network.state_dict(), out / "model.pt")
        if epoch < 3 or (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch + 1:4d}/{settings['epochs']}  train {train_loss:.4f}  val {val_loss:.4f}  "
                  f"best {best_val:.4f}  extra {history['train_extra'][-1]:.4f}  val Ki^2 {history['val_extra'][-1]:.4f}  "
                  f"lambda {lam:.3g}  ({(time.perf_counter() - started) / (epoch + 1):.1f} s/epoch)", flush=True)

    # test predictions with the best checkpoint
    network.load_state_dict(torch.load(out / "model.pt", map_location=device))
    network.eval()
    predictions = {}
    with torch.no_grad():
        for first in range(0, len(test_ids), batch_size):
            ids = test_ids[first:first + batch_size]
            images, arterial, raw, scaled, t = stack(scans, ids, device, curve_scale)
            for s, p in zip(ids, forward(images, scaled)["REG"].cpu().numpy()):
                predictions[s] = p.tolist()
    with open(out / "test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    summary = {"arm": arm, "run_seed": run_seed, "best_val_loss": best_val, "best_epoch": best_epoch,
               "epochs_trained": settings["epochs"], "seconds": time.perf_counter() - started,
               "curve_scale": curve_scale.cpu().tolist(), "hybrid_settings": h,
               "train_ids": list(training_ids), "val_ids": list(validation_ids), "test_ids": list(test_ids),
               "history": history}
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Train FC-DLIF arms with tissue curves.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--folds", nargs="*", type=int)
    ap.add_argument("--runs", nargs="*", type=int)
    ap.add_argument("--epochs", type=int, help="override (smoke test only)")
    ap.add_argument("--ki-fade", nargs=2, type=int, help="override ki_fade_epochs (smoke test only)")
    args = ap.parse_args()

    config = load_study_config(args.config)
    settings, h = dict(config["training"]), dict(config["kiloss"])
    h.update(plasma_a=config["kinetics"]["plasma_ratio_a"], plasma_b=config["kinetics"]["plasma_ratio_b"],
             plasma_c=config["kinetics"]["plasma_ratio_c"])
    if args.epochs:
        settings["epochs"] = args.epochs
    if args.ki_fade:
        h["ki_fade_epochs"] = args.ki_fade
    input_shape = tuple(config["scans"]["network_input_shape"])

    add_group_code_to_python_path(config["paths"]["group_code_folder"])
    from models import models as group_models
    classes = make_network_classes(group_models.DLIFNet_MAX)
    build = classes[args.arm]

    manifest = pd.read_csv(config["paths"]["work_folder"] / "manifest.csv", keep_default_na=False)
    used = manifest[manifest["used"]]
    scan_ids = used["scan_id"].tolist()
    group_of_scan = dict(zip(used["scan_id"], used["group"]))
    print(f"arm {args.arm}: reading {len(scan_ids)} scans ...", flush=True)
    scans = load_scans(config, scan_ids, input_shape)

    # a smoke test (--epochs) writes elsewhere, so it can never be mistaken for, or block, a real run
    root = config["paths"]["work_folder"] / ("models_kiloss_smoketest" if args.epochs else "models_kiloss") / args.arm
    root.mkdir(parents=True, exist_ok=True)
    folds = make_folds(scan_ids, settings["n_folds"], settings["seed"])
    with open(root / "folds.json", "w", encoding="utf-8") as f:
        json.dump([{"fold": k, "train_ids": tr, "test_ids": te} for k, tr, te in folds], f, indent=2)
    for fold, fold_train, test_ids in folds:
        if args.folds and fold not in args.folds:
            continue
        for run in range(1, settings["n_runs"] + 1):
            if args.runs and run not in args.runs:
                continue
            out = root / f"fold_{fold:02d}" / f"run_{run:02d}"
            if (out / "summary.json").exists():
                print(f"fold {fold} run {run}: done, skipped"); continue
            seed = run_seed_of(settings["seed"], fold, run)
            tr, va = validation_split(fold_train, group_of_scan, settings["validation_fraction"], seed)
            print(f"arm {args.arm} fold {fold} run {run}: {len(tr)} train, {len(va)} val, {len(test_ids)} test, seed {seed}",
                  flush=True)
            s = train_one_run(build, args.arm, scans, tr, va, test_ids, settings, h, seed, out)
            print(f"arm {args.arm} fold {fold} run {run}: best val {s['best_val_loss']:.4f} at epoch {s['best_epoch'] + 1}",
                  flush=True)


if __name__ == "__main__":
    main()
