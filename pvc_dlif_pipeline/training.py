"""Cross-validation splits and the training of one network.

The protocol is the group's: 10-fold cross-validation over scans, 10 runs per
fold, a validation split per run stratified by experimental group, Adam at a
constant learning rate, batch size 8, mixed precision, the weighted MSE loss,
Poisson-noise and flip augmentation, and the checkpoint with the lowest
validation loss kept.

Every random choice of a run follows from one number, the run seed
    run_seed = seed + 1000 * fold + run.
It is set before the network is built, so it fixes the initial weights as well
as the validation split, the batch order and the augmentation.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split


def run_seed_of(seed, fold, run):
    return seed + 1000 * fold + run


def set_all_seeds(seed):
    """Seed Python, NumPy and PyTorch (CPU and GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_folds(scan_ids, n_folds, seed):
    """Return [(fold number, train_ids, test_ids)], fold numbers starting at 1.

    KFold with shuffling over the sorted scan IDs, as in the group's trainer.
    The split depends only on the IDs and the seed, so every input
    representation gets exactly the same folds.
    """
    sorted_ids = sorted(scan_ids)
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for fold_index, (train_positions, test_positions) in enumerate(splitter.split(sorted_ids), start=1):
        folds.append((fold_index,
                      [sorted_ids[i] for i in train_positions],
                      [sorted_ids[i] for i in test_positions]))
    return folds


def validation_split(train_ids, group_of_scan, validation_fraction, seed):
    """Split the training scans of a fold into training and validation scans.

    As in the group's stratified_split: when the validation set has fewer
    scans than there are experimental groups, one scan is drawn from each of a
    random selection of groups; otherwise a split stratified by group is used.
    Returns (training ids, validation ids).
    """
    ids = list(train_ids)
    groups = [group_of_scan.get(scan_id) or "ungrouped" for scan_id in ids]
    scans_in_group = {}
    for scan_id, group in zip(ids, groups):
        scans_in_group.setdefault(group, []).append(scan_id)
    n_validation = int(round(len(ids) * validation_fraction))
    group_names = list(scans_in_group)

    if n_validation < len(group_names):
        generator = random.Random(seed)
        generator.shuffle(group_names)
        validation_ids = []
        remaining_ids = list(ids)
        for group in group_names[:n_validation]:
            chosen = generator.choice(scans_in_group[group])
            validation_ids.append(chosen)
            remaining_ids.remove(chosen)
        return remaining_ids, validation_ids

    every_group_has_two = all(groups.count(group) >= 2 for group in set(groups))
    training_ids, validation_ids = train_test_split(
        ids, test_size=validation_fraction, stratify=groups if every_group_has_two else None,
        random_state=seed)
    return list(training_ids), list(validation_ids)


def weighted_mse(prediction, target, frame_bounds, weights):
    """Sum over frame segments of weight * mean squared error in that segment."""
    total = 0.0
    for segment in range(len(weights)):
        start, stop = frame_bounds[segment], frame_bounds[segment + 1]
        total = total + torch.mean(weights[segment] * (prediction[:, start:stop] - target[:, start:stop]) ** 2)
    return total


def augment_batch(images, generator, poisson_noise, random_flip):
    """The group's augmentation, applied on the GPU to a batch (batch, 1, frames, z, y, x).

    Per scan: a scale s is drawn from U(0, 1), and Poisson(s * x) - s * x is
    added to the image x (zero-mean noise whose size follows the image); then
    both in-plane axes are flipped with probability 0.5.
    """
    if poisson_noise:
        scale = torch.rand(images.shape[0], 1, 1, 1, 1, 1, device=images.device, generator=generator)
        rate = (images * scale).clamp_min(0.0)
        images = images + (torch.poisson(rate, generator=generator) - rate)
    if random_flip:
        flip = torch.rand(images.shape[0], device=images.device, generator=generator) < 0.5
        if bool(flip.any()):
            images = images.clone()
            images[flip] = torch.flip(images[flip], dims=(-1, -2))
    return images


def stack_batch(scans, scan_ids, device):
    """Stack the in-memory scans of a batch into (images, arterial curves) on the device."""
    images = np.stack([scans[scan_id]["image"] for scan_id in scan_ids])[:, np.newaxis]
    curves = np.stack([scans[scan_id]["arterial"] for scan_id in scan_ids])
    return torch.from_numpy(images).to(device), torch.from_numpy(curves).to(device)


def network_prediction(network, images):
    """The group's networks return either a tensor or a dict with key "REG"."""
    output = network(images)
    return output["REG"] if isinstance(output, dict) else output


def train_one_run(build_network, scans, training_ids, validation_ids, settings, run_seed, output_folder):
    """Train one network and keep the weights with the lowest validation loss.

    build_network: function returning a new, untrained network.
    scans: {scan_id: {"image": (frames, z, y, x) float32, "arterial": (frames,) float32}}.
    Writes model.pt (the best weights) and summary.json to output_folder and
    returns the summary as a dictionary.
    """
    set_all_seeds(run_seed)
    network = build_network()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=settings["learning_rate"])
    use_amp = settings["mixed_precision"] and device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    augmentation_generator = torch.Generator(device=device)
    augmentation_generator.manual_seed(run_seed)
    bounds, weights = settings["loss_frame_bounds"], settings["loss_weights"]
    batch_size = settings["batch_size"]

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_loss": []}
    best_val_loss, best_epoch = float("inf"), -1
    started = time.perf_counter()

    for epoch in range(settings["epochs"]):
        # Training: shuffled batches, augmented, one optimiser step per batch.
        network.train()
        order = torch.randperm(len(training_ids)).tolist()
        batch_losses = []
        for first in range(0, len(order), batch_size):
            batch_ids = [training_ids[i] for i in order[first:first + batch_size]]
            images, curves = stack_batch(scans, batch_ids, device)
            images = augment_batch(images, augmentation_generator, settings["poisson_noise"],
                                   settings["random_flip"])
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                prediction = network_prediction(network, images)
            loss = weighted_mse(prediction.float(), curves, bounds, weights)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            batch_losses.append(float(loss.detach()))

        # Validation: no augmentation, loss averaged over batches.
        network.eval()
        validation_losses = []
        with torch.no_grad():
            for first in range(0, len(validation_ids), batch_size):
                images, curves = stack_batch(scans, validation_ids[first:first + batch_size], device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    prediction = network_prediction(network, images)
                validation_losses.append(float(weighted_mse(prediction.float(), curves, bounds, weights)))

        train_loss, val_loss = float(np.mean(batch_losses)), float(np.mean(validation_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            torch.save(network.state_dict(), output_folder / "model.pt")
        if epoch < 3 or (epoch + 1) % 50 == 0:
            seconds_per_epoch = (time.perf_counter() - started) / (epoch + 1)
            print(f"  epoch {epoch + 1:4d}/{settings['epochs']}  train {train_loss:.4f}  val {val_loss:.4f}  "
                  f"best {best_val_loss:.4f}  ({seconds_per_epoch:.1f} s/epoch)", flush=True)

    summary = {
        "run_seed": run_seed,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_trained": settings["epochs"],
        "seconds": time.perf_counter() - started,
        "train_ids": list(training_ids),
        "val_ids": list(validation_ids),
        "history": history,
    }
    with open(output_folder / "summary.json", "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    return summary
