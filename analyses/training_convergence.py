"""Training convergence: when each run leaves the initial loss plateau.

Every run falls from a validation loss of about 12 to a plateau near 3 within
the first epochs, and leaves the plateau at an epoch that varies from run to
run. The escape epoch is the first epoch whose validation loss is below the
convergence threshold (evaluation.converged_val_loss); a run that never gets
there is given the escape epoch epochs + 1.

The representations are compared pairwise on matched (fold, run): the same
run seed, data split, batch order and augmentation, so neither confounds the
comparison. Wilcoxon signed-rank test on the paired escape epochs, against the
uncorrected representation; not adjusted for multiplicity.

Reads   <work>/models/<network>/<representation>/fold_XX/run_YY/summary.json
Writes  <work>/analyses/<network>/training_convergence_runs.csv         one row per run
        <work>/analyses/<network>/training_convergence.csv              one row per representation
        <work>/analyses/<network>/training_convergence_comparisons.csv  each representation vs uncorrected

Usage
    python analyses/training_convergence.py --config configs/local.yaml --network fcdlif
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from pvc_dlif_pipeline.config import load_study_config

REFERENCE = "uncorrected"


def escape_epoch(validation_losses, threshold, epochs):
    """First epoch (counted from 1) with validation loss below threshold; epochs + 1 if none."""
    for epoch, loss in enumerate(validation_losses, start=1):
        if loss < threshold:
            return epoch
    return epochs + 1


def main():
    parser = argparse.ArgumentParser(description="Escape epoch of every training run.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--network", required=True)
    args = parser.parse_args()
    config = load_study_config(args.config)
    work_folder = config["paths"]["work_folder"]
    threshold = config["evaluation"]["converged_val_loss"]
    epochs = config["training"]["epochs"]

    rows = []
    for representation in config["training"]["trained_representations"]:
        folder = work_folder / "models" / args.network / representation
        for summary_path in sorted(folder.glob("fold_*/run_*/summary.json")):
            with open(summary_path, "r", encoding="utf-8") as summary_file:
                summary = json.load(summary_file)
            epoch = escape_epoch(summary["history"]["val_loss"], threshold, epochs)
            rows.append({"representation": representation,
                         "fold": int(summary_path.parent.parent.name.split("_")[1]),
                         "run": int(summary_path.parent.name.split("_")[1]),
                         "escape_epoch": epoch, "escaped": epoch <= epochs,
                         "best_val_loss": summary["best_val_loss"], "best_epoch": summary["best_epoch"]})
    runs = pd.DataFrame(rows)

    summary_rows = []
    for representation, group in runs.groupby("representation", sort=False):
        escaped = group.loc[group["escaped"], "escape_epoch"]
        summary_rows.append({"representation": representation, "runs": len(group),
                             "never_escaped": int((~group["escaped"]).sum()),
                             "median_escape_epoch": float(escaped.median()),
                             "p90_escape_epoch": float(np.percentile(escaped, 90)),
                             "median_best_val_loss": float(group["best_val_loss"].median())})
    per_representation = pd.DataFrame(summary_rows)

    wide = runs.pivot_table(index=["fold", "run"], columns="representation", values="escape_epoch")
    comparison_rows = []
    for representation in [name for name in wide.columns if name != REFERENCE]:
        pair = wide[[REFERENCE, representation]].dropna()
        difference = pair[representation] - pair[REFERENCE]
        comparison_rows.append({"representation": representation, "reference": REFERENCE, "pairs": len(pair),
                                "median_difference_epochs": float(difference.median()),
                                "faster_than_reference": int((difference < 0).sum()),
                                "p_wilcoxon": float(wilcoxon(difference).pvalue) if difference.any() else 1.0})
    comparisons = pd.DataFrame(comparison_rows)

    output_folder = work_folder / "analyses" / args.network
    output_folder.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_folder / "training_convergence_runs.csv", index=False)
    per_representation.to_csv(output_folder / "training_convergence.csv", index=False)
    comparisons.to_csv(output_folder / "training_convergence_comparisons.csv", index=False)
    print(per_representation.to_string(index=False))
    print(comparisons.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
